#!/usr/bin/env python3
"""Train the causal 5 Hz upper-planner student for schema-v25 episodes.

This trainer is intentionally separate from ``train.py``.  The 30 Hz policy
learns effective-target-to-control commands; this policy learns the
CORRECTED target directly: given the ORIGINAL goal (FLU direction +
distance) it regresses the corrected FLU direction + normalized distance
(pure regression — no PASS/NORMAL/TURN type, no direction token; the expert
type is used only as a loss weight).  The loader feeds only real macro
decision rows (``macro_update_mask==1``), never the six zero-order-held CSV
copies between decisions.
"""

import argparse
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataloader import (MACRO_STATE_FIELDS, MACRO_STATE_SCALE,
                        MACRO_TYPE_TO_INDEX, SCHEMA_VERSION,
                        build_macro_dataloaders, discover_committed_episodes)
from model.model import MacroPlannerPolicy, MacroPolicyConfig


@contextmanager
def nullcontext():
    yield


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device, non_blocking=True)
            if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def compute_macro_loss(output, batch: Dict[str, torch.Tensor],
                       direction_weight: float = 1.0,
                       distance_weight: float = 0.5,
                       pass_weight: float = 0.15,
                       correction_weight: float = 4.0):
    """Regression loss for the corrected-target student (no type/token).

    Every row is supervised on the corrected FLU direction + normalized
    distance: PASS rows learn to copy the original-goal direction (PASS
    label == original goal), NORMAL/TURN rows learn the correction.  The
    expert macro_correction_type is used ONLY as a loss weight — never a
    network input or output — so the sparse NORMAL/TURN corrections are not
    drowned out by the dominant PASS class.
    R31: PASS ≈ 88% of macro decisions (col_4: 20200/22911).  With the old
    0.5/2.0 weighting the correction rows contributed only ~35% of the
    loss, so the cheapest strategy was to always copy the original goal.
    The weights 0.15/4.0 push corrections to ~78% of the loss so the
    network is forced to learn real corrections from depth.
    """
    mask = batch["loss_mask"] > 0.5
    if not mask.any():
        zero = output.direction.sum() * 0.0
        return zero, {"direction_loss": zero, "distance_loss": zero}
    type_target = batch["macro_type"].long()
    pass_idx = int(MACRO_TYPE_TO_INDEX["PASS_THROUGH"])
    weight = torch.where(
        type_target == pass_idx,
        torch.full_like(type_target, pass_weight, dtype=torch.float32),
        torch.full_like(type_target, correction_weight, dtype=torch.float32))
    weight = weight * mask.float()
    denom = weight.sum().clamp_min(1.0)
    # 1 - cosine distance is stable for unit-normalized network output.
    cos = F.cosine_similarity(
        output.direction, batch["macro_direction"], dim=-1).clamp(-1.0, 1.0)
    direction_loss = (weight * (1.0 - cos)).sum() / denom
    distance_err = F.smooth_l1_loss(
        output.distance_norm, batch["macro_distance"], beta=0.05,
        reduction="none")
    distance_loss = (weight.unsqueeze(-1) * distance_err).sum() / denom
    total = (float(direction_weight) * direction_loss +
             float(distance_weight) * distance_loss)
    return total, {"direction_loss": direction_loss,
                   "distance_loss": distance_loss}


class Metrics:
    def __init__(self) -> None:
        self.total: Dict[str, float] = {}
        self.count: Dict[str, float] = {}

    def add(self, name: str, value: float, count: float = 1.0) -> None:
        self.total[name] = self.total.get(name, 0.0) + value * count
        self.count[name] = self.count.get(name, 0.0) + count

    def result(self) -> Dict[str, float]:
        return {key: self.total[key] / max(1.0, self.count[key])
                for key in sorted(self.total)}


def _initial_hidden(model: MacroPlannerPolicy, batch, hidden_by_episode,
                    device: torch.device):
    first_chunks = batch["is_first_chunk"].detach().cpu().tolist()
    hidden_columns = []
    cell_columns = []
    for episode_id, is_first in zip(batch["episode_id"], first_chunks):
        if bool(is_first) or episode_id not in hidden_by_episode:
            hidden, cell = model.initial_hidden(1, device=device)
        else:
            hidden, cell = hidden_by_episode[episode_id]
        hidden_columns.append(hidden)
        cell_columns.append(cell)
    return torch.cat(hidden_columns, dim=1), torch.cat(cell_columns, dim=1)


def _remember_hidden(batch, hidden, hidden_by_episode) -> None:
    hidden_state, cell_state = hidden
    for column, episode_id in enumerate(batch["episode_id"]):
        hidden_by_episode[episode_id] = (
            hidden_state[:, column:column + 1].detach(),
            cell_state[:, column:column + 1].detach())


def run_epoch(model: MacroPlannerPolicy, loader: Iterable,
              device: torch.device, optimizer=None, scaler=None,
              scheduler=None, amp: bool = False, grad_clip: float = 1.0,
              stateful: bool = True) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = Metrics()
    hidden_by_episode: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for batch in loader:
        batch = _move(batch, device)
        hidden = (_initial_hidden(model, batch, hidden_by_episode, device)
                  if stateful else None)
        context = (torch.cuda.amp.autocast(enabled=amp)
                   if device.type == "cuda" else nullcontext())
        with torch.set_grad_enabled(training), context:
            output = model(batch["depth"], batch["state"], hidden)
            loss, parts = compute_macro_loss(output, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if stateful:
            _remember_hidden(batch, output.hidden, hidden_by_episode)
        mask = batch["loss_mask"] > 0.5
        count = float(mask.sum().item())
        metrics.add("loss", float(loss.detach().item()), count)
        for key, value in parts.items():
            metrics.add(key, float(value.detach().item()), count)
        with torch.no_grad():
            cosine = F.cosine_similarity(
                output.direction[mask],
                batch["macro_direction"][mask], dim=-1).clamp(-1, 1)
            angle = torch.rad2deg(torch.acos(cosine))
            metrics.add("direction_angle_deg", float(angle.mean()), count)
            distance_error = (output.distance_norm[mask] -
                              batch["macro_distance"][mask]).abs()
            metrics.add("distance_mae", float(distance_error.mean()), count)
    return metrics.result()


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path: Path, model: MacroPlannerPolicy, optimizer,
                    scheduler, scaler, epoch: int, best_val: float,
                    train_eps, val_eps, args) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "MacroPlannerPolicy",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val,
        "student_input_fields": ["depth_file"] + list(MACRO_STATE_FIELDS),
        "target_fields": ["macro_direction_flu_x", "macro_direction_flu_y",
                          "macro_direction_flu_z", "macro_distance_norm"],
        "normalization": {
            "depth_max_m": 5.0,
            "macro_state_scale": [float(v) for v in MACRO_STATE_SCALE],
            "macro_type_names": ["PASS_THROUGH", "NORMAL_CORRECTION",
                                  "TURN_LEFT", "TURN_RIGHT"],
        },
        "split": {
            "train_episode_ids": [ep.episode_id for ep in train_eps],
            "val_episode_ids": [ep.episode_id for ep in val_eps],
            "train_scene_ids": sorted(set(ep.scene_id for ep in train_eps)),
            "val_scene_ids": sorted(set(ep.scene_id for ep in val_eps)),
        },
        "train_args": vars(args),
        "saved_at_unix_s": time.time(),
    }
    torch.save(payload, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the schema-v25 causal 5 Hz macro planner")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="checkpoints/macro_v25")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16,
                        help="number of 5 Hz decisions per recurrent chunk")
    parser.add_argument("--burn-in", type=int, default=4)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--mirror-augmentation", action="store_true")
    parser.add_argument(
        "--depth-noise-std-ratio", type=float, default=0.02,
        help="D435i sim-to-real multiplicative Gaussian depth noise applied "
             "at training time (sigma = ratio * normalized depth).  The val "
             "loader always uses 0.")
    parser.add_argument("--stateless-windows", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    episodes = discover_committed_episodes(args.dataset_root)
    macro_decisions = sum(ep.macro_frames for ep in episodes)
    print("dataset: %d committed v25 episodes, %d scenes, %d macro decisions" %
          (len(episodes), len(set(ep.scene_id for ep in episodes)),
           macro_decisions))
    if args.audit_only:
        return
    train_loader, val_loader, train_eps, val_eps = build_macro_dataloaders(
        args.dataset_root, batch_size=args.batch_size,
        sequence_length=args.sequence_length, burn_in=args.burn_in,
        stride=args.stride, val_fraction=args.val_fraction, seed=args.seed,
        workers=args.workers,
        balanced_sampling=not args.no_balanced_sampling,
        stateful=not args.stateless_windows,
        mirror_augmentation=args.mirror_augmentation,
        depth_noise_std_ratio=args.depth_noise_std_ratio)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = MacroPlannerPolicy(MacroPolicyConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(
        optimizer, int(args.warmup_epochs * len(train_loader)), total_steps)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp) \
        if device.type == "cuda" else None
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("schema_version") != SCHEMA_VERSION or \
                checkpoint.get("architecture") != "MacroPlannerPolicy":
            raise ValueError("resume checkpoint is not a schema-v25 macro checkpoint")
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
            args.grad_clip, stateful=not args.stateless_windows)
        val_metrics = run_epoch(
            model, val_loader, device,
            stateful=not args.stateless_windows)
        print("epoch %03d train=%s val=%s" %
              (epoch + 1, json.dumps(train_metrics, sort_keys=True),
               json.dumps(val_metrics, sort_keys=True)))
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler,
                        scaler, epoch, best_val, train_eps, val_eps, args)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, optimizer,
                            scheduler, scaler, epoch, best_val, train_eps,
                            val_eps, args)


if __name__ == "__main__":
    main()
