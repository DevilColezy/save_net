#!/usr/bin/env python3
"""Ablation trainer: 30 Hz student on the ORIGINAL navigation goal.

Normal training (``train.py``) feeds the 30 Hz student the 5 Hz-corrected
EFFECTIVE target (``goal_direction_flu_*`` + ``goal_distance_norm``).  This
ablation instead feeds the ORIGINAL navigation goal
(``navigation_goal_direction_flu_*`` + ``navigation_goal_distance_norm``) so
the 30 Hz policy must plan the whole detour from the raw goal alone — no 5 Hz
corrector.  Every other training parameter is identical to ``train.py``.

Why the checkpoint stays rollout-compatible:
  The 7-D field ORDER is unchanged (gravity 3 + goal direction 3 + distance
  1), so ``student_input_fields`` is kept as ``STATE_FIELDS`` and the released
  ``student30`` rollout path (which feeds the raw goal into the 30 Hz student)
  consumes this model unchanged.  The extra ``goal_source`` /
  ``ablation`` keys mark this model as the original-goal ablation.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

import dataloader as dl
from dataloader import (COMMAND_SCALE, SCHEMA_VERSION, STATE_FIELDS,
                        STATE_SCALE, TARGET_FIELDS, StatefulEpisodeBatchSampler,
                        WeightedRandomSampler, discover_committed_episodes,
                        split_episodes)
from model.model import ViTFlyLSTMPolicy, ViTFlyPolicyConfig

from train import (compute_loss, make_scheduler, print_avoidance_label_coverage,
                   run_epoch, save_checkpoint as _save_checkpoint, set_seed)


class OriginalGoalV25SequenceDataset(dl.V25SequenceDataset):
    """Identical to V25SequenceDataset, but the student state goal fields are
    the ORIGINAL navigation goal instead of the 5 Hz-corrected effective
    target.

    The base ``__getitem__`` already returns ``state_5hz`` =
    STATE_FIELDS_5HZ (``navigation_goal_direction_flu_*`` +
    ``navigation_goal_distance_norm``), mirror-flipped consistently with the
    7-D state, so we simply substitute columns 3..6 of the state with it.
    gravity (0..2) is unchanged.  Mirrored goal_dir_y already carries the same
    sign flip on both sides, so consistency is preserved for free.
    """

    def __getitem__(self, index: int) -> dict:
        batch = super().__getitem__(index)
        s5 = batch["state_5hz"]      # [T,4] nav_dir x,y,z + nav_dist
        state = batch["state"].clone()  # [T,7] gravity 3 + goal 4
        state[:, 3:6] = s5[:, 0:3]   # goal_direction_flu -> navigation_goal_direction_flu
        state[:, 6] = s5[:, 3]       # goal_distance_norm -> navigation_goal_distance_norm
        batch["state"] = state
        return batch


def build_original_goal_dataloaders(dataset_root, batch_size: int = 8,
                                    sequence_length: int = 32, burn_in: int = 8,
                                    stride: Optional[int] = None,
                                    val_fraction: float = 0.2, seed: int = 1337,
                                    workers: int = 0,
                                    balanced_sampling: bool = True,
                                    verify_depth: bool = True,
                                    stateful: bool = True,
                                    mirror_augmentation: bool = False,
                                    depth_noise_std_ratio: float = 0.02):
    """Same signature / behaviour as dataloader.build_dataloaders but with the
    original-goal dataset wrapper."""
    import warnings
    episodes = discover_committed_episodes(dataset_root,
                                           verify_depth=verify_depth)
    train_eps, val_eps = split_episodes(episodes, val_fraction, seed)
    train_ds = OriginalGoalV25SequenceDataset(
        train_eps, sequence_length, burn_in, stride,
        augment=mirror_augmentation, stateful=stateful,
        depth_noise_std_ratio=depth_noise_std_ratio)
    val_ds = OriginalGoalV25SequenceDataset(
        val_eps, sequence_length, burn_in, stride, augment=False,
        stateful=stateful, depth_noise_std_ratio=0.0)
    if stateful:
        if balanced_sampling:
            warnings.warn(
                "stateful TBPTT preserves episode order, so weighted window "
                "sampling is disabled; rare expert modes remain loss-weighted.",
                RuntimeWarning)
        common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=StatefulEpisodeBatchSampler(
                train_ds, batch_size, shuffle=True, seed=seed), **common)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_sampler=StatefulEpisodeBatchSampler(
                val_ds, batch_size, shuffle=False, seed=seed), **common)
        return train_loader, val_loader, train_eps, val_eps
    sampler = None
    shuffle = True
    if balanced_sampling:
        weights = train_ds.sampling_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=shuffle, sampler=sampler, **common)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_eps, val_eps


def save_ablation_checkpoint(path: Path, model: ViTFlyLSTMPolicy, optimizer,
                             scheduler, scaler, epoch: int, best_val: float,
                             train_eps, val_eps, args) -> None:
    """train.save_checkpoint + an explicit ablation marker."""
    _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                     best_val, train_eps, val_eps, args)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    ck["goal_source"] = "original_navigation_goal"
    ck["ablation"] = "30hz_original_goal_no_5hz_correction"
    torch.save(ck, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABLATION: train 30 Hz student on the ORIGINAL navigation "
                    "goal (no 5 Hz effective-target correction)")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="checkpoints/vitfly_origgoal_v3_complete_full_mirror_nonoise")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--mirror-augmentation", action="store_true")
    parser.add_argument("--depth-noise-std-ratio", type=float, default=0.02)
    parser.add_argument("--stateless-windows", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    episodes = discover_committed_episodes(args.dataset_root)
    print("dataset: %d committed v25 episodes, %d scenes, %d frames" % (
        len(episodes), len(set(ep.scene_id for ep in episodes)),
        sum(ep.rows for ep in episodes)))
    print("coverage: avoidance_episodes=%d avoidance_frames=%d macro_frames=%d"
          % (sum(ep.has_avoidance for ep in episodes),
             sum(ep.avoidance_frames for ep in episodes),
             sum(ep.macro_frames for ep in episodes)))
    print_avoidance_label_coverage(episodes)
    print("ABLATION: 30 Hz student state goal fields <- ORIGINAL navigation "
          "goal (navigation_goal_direction_flu_* + "
          "navigation_goal_distance_norm); 5 Hz effective target NOT used.")
    if args.audit_only:
        return
    train_loader, val_loader, train_eps, val_eps = \
        build_original_goal_dataloaders(
            args.dataset_root, batch_size=args.batch_size,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            stride=args.stride, val_fraction=args.val_fraction, seed=args.seed,
            workers=args.workers,
            balanced_sampling=not args.no_balanced_sampling,
            stateful=not args.stateless_windows,
            mirror_augmentation=args.mirror_augmentation,
            depth_noise_std_ratio=args.depth_noise_std_ratio)
    if args.stateless_windows:
        print("training mode: independent windows with %d-frame burn-in" %
              args.burn_in)
    else:
        print("training mode: stateful truncated BPTT; LSTM state is carried "
              "between adjacent chunks and detached at every chunk boundary")
    print("horizontal mirror augmentation: %s" % (
        "enabled" if args.mirror_augmentation else "disabled"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = ViTFlyLSTMPolicy(ViTFlyPolicyConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(
        optimizer, int(args.warmup_epochs * len(train_loader)), total_steps)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("resume checkpoint is not schema-v25")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": [ep.episode_id for ep in train_eps],
                   "val": [ep.episode_id for ep in val_eps]}, handle, indent=2)
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, scaler, scheduler, amp,
            args.grad_clip, args.smoothness_weight,
            stateful=not args.stateless_windows)
        val_metrics = run_epoch(
            model, val_loader, device,
            smoothness_weight=args.smoothness_weight,
            stateful=not args.stateless_windows)
        print("epoch %03d train=%s val=%s" %
              (epoch + 1, json.dumps(train_metrics, sort_keys=True),
               json.dumps(val_metrics, sort_keys=True)))
        save_ablation_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, val_metrics["loss"], train_eps, val_eps, args)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ablation_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_val, train_eps, val_eps, args)
        print("  best val loss so far: %.6f" % best_val)


if __name__ == "__main__":
    main()
