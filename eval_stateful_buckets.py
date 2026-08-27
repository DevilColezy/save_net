#!/usr/bin/env python3
"""Stateful (LSTM hidden carried across chunks) dataset inference that
compares the student policy against the expert labels in velocity / yaw-rate
buckets.

Purpose: the v25 architecture removed velocity/yaw_rate inputs (7-D state +
visual_scale), so the ONLY honest offline evaluation is to feed the recorded
depth+state sequence through the LSTM with hidden state carried across chunk
boundaries (identical to train.py's stateful TBPTT / rollout inference) and
compare the student's 30 Hz command against the expert's label.

Reports:
  * overall speed mean (student vs expert)
  * speed tiers (low/mid/high) mean speed and coverage
  * yaw-rate buckets: expert yaw-rate signed buckets -> student mean yaw-rate
    (right-turn under-yaw used to show up as student/expert ~0.015 in the
    [0.2, 0.4) bucket; a healthy model is ~1.0)
  * TURN frames (|expert yaw_rate| > 0.3) coverage + mean yaw-rate
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from dataloader import (  # noqa: E402
    COMMAND_SCALE,
    HIERARCHICAL_MODE_TO_INDEX,
    TARGET_FIELDS,
    V25SequenceDataset,
    discover_committed_episodes,
)
from rollout import load_policy_checkpoint  # noqa: E402


def _parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="train.py schema-v25 checkpoint (.pt)")
    p.add_argument("--dataset-root", required=True,
                   help="il_dataset committed-episodes root (e.g. il_data_joint_v2_col_3)")
    p.add_argument("--model-file", default=str(THIS_DIR / "model" / "model.py"))
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--max-episodes", type=int, default=0,
                   help="0 = all committed episodes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = _parse_args().parse_args()
    device = torch.device(args.device)

    episodes = discover_committed_episodes(args.dataset_root)
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
    print(f"episodes: {len(episodes)}")

    model, _, _ = load_policy_checkpoint(
        args.checkpoint, args.model_file, device)
    model.eval()

    dataset = V25SequenceDataset(
        episodes, sequence_length=args.sequence_length, stateful=True,
        augment=False)

    scale = torch.as_tensor(COMMAND_SCALE, device=device).view(1, 4)
    preds = []   # (vx, vy, vz, yaw_rate) unscaled
    targets = []  # expert label, unscaled
    modes = []
    loss_masks = []
    hidden_by_episode = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            depth = item["depth"][None].to(device)
            state = item["state"][None].to(device)
            ep_id = item["episode_id"]
            is_first = bool(item["is_first_chunk"].item())
            if is_first or ep_id not in hidden_by_episode:
                hidden = model.initial_hidden(1, device=device)
            else:
                hidden = hidden_by_episode[ep_id]
            out = model(depth, state, hidden)
            # keep hidden across chunks, detached at boundaries
            hidden_by_episode[ep_id] = (out.hidden[0].detach(), out.hidden[1].detach())
            pred_norm = out.normalized_command[0]              # [T,4] normalized
            tgt_norm = item["target"].to(device)               # [T,4] normalized
            mask = item["loss_mask"]
            mode = item["hierarchical_mode"]
            preds.append((pred_norm * scale).cpu().numpy())
            targets.append((tgt_norm * scale).cpu().numpy())
            modes.append(mode.numpy())
            loss_masks.append(mask.numpy())

    pred = np.concatenate(preds, axis=0)          # [N,4] unscaled
    tgt = np.concatenate(targets, axis=0)
    mode = np.concatenate(modes, axis=0)
    mask = np.concatenate(loss_masks, axis=0) > 0

    sel = mask
    pred, tgt, mode = pred[sel], tgt[sel], mode[sel]

    # --- overall speed ---
    pred_speed = np.hypot(pred[:, 0], pred[:, 1])
    tgt_speed = np.hypot(tgt[:, 0], tgt[:, 1])
    print("\n=== overall ===")
    print(f"rows={len(pred)}  student_speed={pred_speed.mean():.3f} m/s  "
          f"expert_speed={tgt_speed.mean():.3f} m/s  ratio={pred_speed.mean()/max(1e-6,tgt_speed.mean()):.3f}")
    print(f"student yaw_rate mean={np.abs(pred[:,3]).mean():.4f}  "
          f"expert yaw_rate mean={np.abs(tgt[:,3]).mean():.4f}")

    # --- speed tiers ---
    print("\n=== speed tiers (by expert speed) ===")
    for name, lo, hi in (("low", 0.0, 0.8), ("mid", 0.8, 1.6), ("high", 1.6, 99.0)):
        m = (tgt_speed >= lo) & (tgt_speed < hi)
        if m.sum() == 0:
            print(f"{name:4s}: 0 frames"); continue
        r = pred_speed[m].mean() / max(1e-6, tgt_speed[m].mean())
        print(f"{name:4s}: n={m.sum():6d}  student={pred_speed[m].mean():.3f}  "
              f"expert={tgt_speed[m].mean():.3f}  ratio={r:.3f}")

    # --- yaw buckets (signed) ---
    print("\n=== yaw-rate buckets (by expert yaw_rate) ===")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 99.0]
    labels = ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4+)"]
    for sign, sname in ((1, "RIGHT(+yaw)"), (-1, "LEFT(-yaw)")):
        for i in range(len(labels)):
            lo, hi = edges[i], edges[i + 1]
            m = (sign * tgt[:, 3] >= lo) & (sign * tgt[:, 3] < hi)
            if m.sum() == 0:
                print(f"{sname:12s} {labels[i]:10s}: 0 frames")
                continue
            ps = pred[m, 3].mean()
            ts = tgt[m, 3].mean()
            print(f"{sname:12s} {labels[i]:10s}: n={m.sum():6d}  "
                  f"student={sign*ps:+.4f}  expert={sign*ts:+.4f}  ratio={(ps/ts if abs(ts)>1e-3 else float('nan')):+.3f}")

    # --- TURN frames (|expert yaw|>0.3) ---
    turn = np.abs(tgt[:, 3]) > 0.3
    print("\n=== TURN frames (|expert yaw_rate|>0.3) ===")
    print(f"n={turn.sum():6d}  student_yaw={pred[turn,3].mean():+.4f}  "
          f"expert_yaw={tgt[turn,3].mean():+.4f}  "
          f"ratio={pred[turn,3].mean()/max(1e-6,tgt[turn,3].mean()):+.3f}")

    # --- per-mode speed ---
    print("\n=== per-mode (student vs expert speed) ===")
    idx_to_mode = {v: k for k, v in HIERARCHICAL_MODE_TO_INDEX.items()}
    for mid in sorted(set(mode.tolist())):
        m = mode == mid
        if m.sum() == 0:
            continue
        print(f"{idx_to_mode.get(mid,'?'):20s}: n={m.sum():6d}  "
              f"spd student={pred_speed[m].mean():.3f} / expert={tgt_speed[m].mean():.3f}  "
              f"yaw student={pred[m,3].mean():+.4f} / expert={tgt[m,3].mean():+.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
