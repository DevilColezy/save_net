"""Training script for HierarchicalTrendControlPolicy.

Usage (example)::

    python train.py \\
        --dataset-root /path/to/il_data \\
        --model-file /path/to/model.py \\
        --output-dir ./checkpoints \\
        --epochs 50 --batch-size 4 --device cuda

This script does NOT execute on import.  All training happens inside
``main()``, guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW_API = False


def _amp_autocast(enabled: bool = True):
    """Version-compatible autocast context manager."""
    if _AMP_NEW_API:
        return autocast("cuda", enabled=enabled)
    else:
        return autocast(enabled=enabled)


def _amp_scaler(enabled: bool = True) -> Optional[GradScaler]:
    """Version-compatible GradScaler factory."""
    if not enabled:
        return None
    if _AMP_NEW_API:
        return GradScaler("cuda", enabled=enabled)
    else:
        return GradScaler(enabled=enabled)

# ---------------------------------------------------------------------------
# Import dataloader from the same package directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataloader import (  # type: ignore[import-untyped]
    EpisodeInfo,
    build_dataloaders,
)


# ===========================================================================
#  Model loading
# ===========================================================================

def _load_model_from_file(model_file: str) -> Tuple[Any, Any]:
    """Load HierarchicalTrendControlPolicy and TrendControlConfig from a Python file."""
    model_path = Path(model_file).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model_file not found: {model_path}")
    module_name = "model_module_" + str(hash(str(model_path))).replace("-", "_")
    model_parent = str(model_path.parent)
    sys.path.insert(0, model_parent)
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(model_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create module spec for {model_path}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_module
        spec.loader.exec_module(model_module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ImportError(f"Failed to load model from {model_path}: {exc}") from exc
    finally:
        if sys.path and sys.path[0] == model_parent:
            sys.path.pop(0)
    ModelClass = getattr(model_module, "HierarchicalTrendControlPolicy", None)
    ConfigClass = getattr(model_module, "TrendControlConfig", None)
    if ModelClass is None or ConfigClass is None:
        raise ImportError(f"HierarchicalTrendControlPolicy/TrendControlConfig not found in {model_path}")
    return ModelClass, ConfigClass


RECOVERY_YAW_WEIGHT = 4.0


# ===========================================================================
#  Masked reduction
# ===========================================================================

@dataclass
class MaskedReduction:
    numerator: torch.Tensor
    denominator: torch.Tensor
    mean: torch.Tensor


def masked_reduce(values: torch.Tensor, mask: torch.Tensor) -> MaskedReduction:
    m = mask.to(dtype=values.dtype)
    num = (values * m).sum()
    den = m.sum()
    mean = num / den.clamp_min(1)
    return MaskedReduction(numerator=num, denominator=den, mean=mean)


# ===========================================================================
#  Loss computation
# ===========================================================================

def _soft_ce_per_frame(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Per-frame soft CE: -sum(target * log_softmax(logits), dim=-1) -> [B,T,1]."""
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


def compute_losses(
    output: Any, batch: Dict[str, torch.Tensor],
    config: Any, loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    device = output.horizontal_logits.device
    dtype = output.horizontal_logits.dtype
    tgt_mask = batch["target_mask"].to(device=device, dtype=dtype)

    # Horizontal (soft CE)
    h_per = _soft_ce_per_frame(output.horizontal_logits,
                                batch["horizontal_soft_target"].to(device=device, dtype=dtype))
    h_red = masked_reduce(h_per, tgt_mask)

    # Vertical (soft CE, no recovery exclusion)
    v_per = _soft_ce_per_frame(output.vertical_logits,
                                batch["vertical_soft_target"].to(device=device, dtype=dtype))
    v_red = masked_reduce(v_per, tgt_mask)

    # Guide value (use guide_value_raw)
    gv_target = batch["guide_value_target"].to(device=device, dtype=dtype)
    gv_per = F.smooth_l1_loss(output.guide_value_raw, gv_target, reduction="none")
    gv_red = masked_reduce(gv_per, tgt_mask)

    # Control (normalised space)
    cmd_target = batch["command_target"].to(device=device, dtype=dtype)
    cmd_scale = torch.tensor(
        [config.max_vx_flu, config.max_vy_flu, config.max_vz_flu, config.max_yaw_rate],
        device=device, dtype=dtype,
    )
    target_norm = (cmd_target / cmd_scale.unsqueeze(0).unsqueeze(0)).clamp(-1.0, 1.0)
    c_per_dim = F.smooth_l1_loss(output.command_normalized, target_norm, reduction="none")

    # Recovery yaw weighting
    h_tgt = batch["horizontal_target"].to(device=device)
    rec_mask = ((h_tgt == 0) | (h_tgt == 12)).float().unsqueeze(-1).to(device=device, dtype=dtype)
    dim_w = torch.ones_like(c_per_dim)
    dim_w[..., 3] += rec_mask.squeeze(-1) * (RECOVERY_YAW_WEIGHT - 1.0)
    c_per_frame = (c_per_dim * dim_w).sum(dim=-1, keepdim=True) / dim_w.sum(dim=-1, keepdim=True)
    c_red = masked_reduce(c_per_frame, tgt_mask)

    total = (loss_weights["horizontal"] * h_red.mean + loss_weights["vertical"] * v_red.mean
             + loss_weights["guide_value"] * gv_red.mean + loss_weights["control"] * c_red.mean)

    return {"total": total, "horizontal": h_red.mean, "vertical": v_red.mean,
            "guide_value": gv_red.mean, "control": c_red.mean,
            "_horizontal_red": h_red, "_vertical_red": v_red,
            "_guide_value_red": gv_red, "_control_red": c_red}


# ===========================================================================
#  Metric accumulation
# ===========================================================================

class MetricAccumulator:
    def __init__(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, float] = defaultdict(float)

    def add(self, name: str, total: float, count: float) -> None:
        self._sums[name] += total
        self._counts[name] += count

    def add_from_reduction(self, name: str, red: MaskedReduction) -> None:
        self._sums[name] += red.numerator.item()
        self._counts[name] += red.denominator.item()

    def compute(self) -> Dict[str, float]:
        return {n: self._sums[n] / self._counts[n] if self._counts[n] > 0 else 0.0
                for n in self._sums}


def _accum_masked(acc: MetricAccumulator, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
    m = mask.to(dtype=values.dtype)
    total = (values * m).sum().item()
    count = m.sum().item()
    if count > 0:
        acc.add(name, total, count)


def _gather_recurrent_state(
    model: nn.Module,
    state_cache: Dict[str, Any],
    trajectory_ids: List[str],
    num_layers: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Any]:
    """Stack detached per-trajectory states for one chronological batch."""
    if not any(key in state_cache for key in trajectory_ids):
        return None
    zero_state = model.initial_state(
        num_layers, 1, hidden_dim, device, dtype,
    )
    hidden_parts = []
    cell_parts = []
    for key in trajectory_ids:
        state = state_cache.get(key, zero_state)
        hidden_parts.append(state[0])
        cell_parts.append(state[1])
    return (
        torch.cat(hidden_parts, dim=1),
        torch.cat(cell_parts, dim=1),
    )


def _store_recurrent_state(
    model: nn.Module,
    state_cache: Dict[str, Any],
    trajectory_ids: List[str],
    is_last: List[bool],
    state: Any,
) -> None:
    """Detach/split a batch state and release completed trajectory streams."""
    detached = model.detach_state(state)
    for batch_index, (key, finished) in enumerate(
        zip(trajectory_ids, is_last)
    ):
        if finished:
            state_cache.pop(key, None)
            continue
        state_cache[key] = (
            detached[0][:, batch_index:batch_index + 1].contiguous(),
            detached[1][:, batch_index:batch_index + 1].contiguous(),
        )


# ===========================================================================
#  Validation
# ===========================================================================

@torch.no_grad()
def validate(
    model: nn.Module, val_loader: Any, config: Any,
    loss_weights: Dict[str, float], device: torch.device, use_amp: bool,
) -> Dict[str, Any]:
    model.eval()
    loss_acc = MetricAccumulator()
    metric_acc = MetricAccumulator()
    active_episode: Optional[str] = None
    trend_state = None
    control_state = None
    previous_command: Optional[torch.Tensor] = None

    for batch in val_loader:
        if len(batch["episode_id"]) != 1:
            raise ValueError(
                "full-episode validation requires val_loader batch_size=1"
            )
        episode_id = batch["episode_id"][0]
        trajectory_id = batch["trajectory_id"][0]
        target_start = int(batch["target_start"][0])
        if trajectory_id != active_episode:
            if target_start != 0:
                raise RuntimeError(
                    f"first validation chunk for {episode_id} starts at {target_start}"
                )
            active_episode = trajectory_id
            trend_state = None
            control_state = None
            previous_command = None

        depth = batch["depth"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        raw_guide = batch["raw_guide"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        gravity = batch["gravity_flu"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        velocity = batch["velocity_flu"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        yaw_rate = batch["yaw_rate"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        with _amp_autocast(enabled=use_amp):
            output = model(depth=depth, raw_guide=raw_guide, gravity_flu=gravity,
                           velocity_flu=velocity, yaw_rate=yaw_rate,
                           trend_state=trend_state, control_state=control_state)
        trend_state = model.detach_state(output.trend_state)
        control_state = model.detach_state(output.control_state)

        losses = compute_losses(output, batch, config, loss_weights)
        for name in ("horizontal", "vertical", "guide_value", "control"):
            loss_acc.add_from_reduction(f"val_{name}", losses[f"_{name}_red"])

        dtype = output.horizontal_logits.dtype
        tgt_mask = batch["target_mask"].to(device=device, dtype=dtype)
        h_tgt = batch["horizontal_target"].to(device=device)
        rec_mask = ((h_tgt == 0) | (h_tgt == 12)).float().unsqueeze(-1).to(device=device, dtype=dtype)
        normal_mask = 1.0 - rec_mask

        h_correct = (output.horizontal_logits.argmax(dim=-1) == h_tgt).float().unsqueeze(-1)
        _accum_masked(metric_acc, "h_acc", h_correct, tgt_mask)
        _accum_masked(metric_acc, "h_acc_normal", h_correct, tgt_mask * normal_mask)
        _accum_masked(metric_acc, "h_acc_recovery", h_correct, tgt_mask * rec_mask)

        v_correct = (output.vertical_logits.argmax(dim=-1) == batch["vertical_target"].to(device=device)).float().unsqueeze(-1)
        _accum_masked(metric_acc, "v_acc", v_correct, tgt_mask)

        gv_mae = (output.guide_value_raw - batch["guide_value_target"].to(device=device, dtype=dtype)).abs()
        _accum_masked(metric_acc, "gv_mae", gv_mae, tgt_mask)

        cmd_mae = (output.command - batch["command_target"].to(device=device, dtype=dtype)).abs()
        for dim_i, dim_name in enumerate(["vx", "vy", "vz", "yaw"]):
            _accum_masked(metric_acc, f"cmd_{dim_name}_mae", cmd_mae[..., dim_i:dim_i+1], tgt_mask)
        cmd_avg = cmd_mae.mean(dim=-1, keepdim=True)
        _accum_masked(metric_acc, "cmd_avg_mae", cmd_avg, tgt_mask)
        _accum_masked(metric_acc, "cmd_avg_mae_normal", cmd_avg, tgt_mask * normal_mask)
        _accum_masked(metric_acc, "cmd_avg_mae_recovery", cmd_avg, tgt_mask * rec_mask)

        # Command smoothness is evaluated across chunk boundaries as well as
        # inside chunks, which the old reset-per-window validation could not do.
        valid_count = int(tgt_mask.sum().item())
        valid_commands = output.command[0, :valid_count]
        if valid_count > 0:
            if previous_command is not None:
                boundary_delta = (
                    valid_commands[:1] - previous_command.unsqueeze(0)
                ).norm(p=2, dim=-1, keepdim=True)
                metric_acc.add(
                    "cmd_step_l2",
                    boundary_delta.sum().item(),
                    float(boundary_delta.numel()),
                )
            if valid_count > 1:
                inner_delta = (
                    valid_commands[1:] - valid_commands[:-1]
                ).norm(p=2, dim=-1, keepdim=True)
                metric_acc.add(
                    "cmd_step_l2",
                    inner_delta.sum().item(),
                    float(inner_delta.numel()),
                )
            previous_command = valid_commands[-1].detach()

    model.train()
    val_metrics = metric_acc.compute()
    loss_vals = loss_acc.compute()
    global_val_total = (
        loss_weights["horizontal"] * loss_vals.get("val_horizontal", 0.0)
        + loss_weights["vertical"] * loss_vals.get("val_vertical", 0.0)
        + loss_weights["guide_value"] * loss_vals.get("val_guide_value", 0.0)
        + loss_weights["control"] * loss_vals.get("val_control", 0.0)
    )
    return {"total": global_val_total, **loss_vals, **val_metrics}


# ===========================================================================
#  Cosine warmup scheduler
# ===========================================================================

class CosineWarmupScheduler:
    """Linear warmup then cosine decay, per optimizer step.

    ``current_step`` counts successfully completed optimizer updates.
    Before the first update, LR is set for update index 1 (not 0).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self._current_step = 0  # completed updates
        self._base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

        # Set LR for the first update (update_index = 1).
        self._apply_lr_for_next_update()

    def _lr_scale_for_update(self, update_index: int) -> float:
        """Return LR scale for a given update index (1-based)."""
        if update_index < 1:
            update_index = 1
        if self.warmup_steps > 0 and update_index <= self.warmup_steps:
            return update_index / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return self.min_lr_ratio
        progress = (update_index - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def _apply_lr_for_next_update(self) -> None:
        """Set LR for the upcoming (not yet executed) optimizer update."""
        next_update = self._current_step + 1
        scale = self._lr_scale_for_update(next_update)
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "base_lrs": self._base_lrs,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._current_step = state["current_step"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self._base_lrs = state["base_lrs"]
        self.min_lr_ratio = state["min_lr_ratio"]
        self._apply_lr_for_next_update()

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self) -> None:
        """Call after a successful optimizer update."""
        self._current_step += 1
        self._apply_lr_for_next_update()

    def retarget(
        self,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
    ) -> None:
        """Retarget a resumed schedule when the requested epoch count changes."""
        if total_steps < self._current_step:
            raise ValueError(
                f"total_steps={total_steps} is below completed updates "
                f"{self._current_step}"
            )
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self._apply_lr_for_next_update()


# ===========================================================================
#  Checkpoint helpers
# ===========================================================================

def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_recovery_acc: float,
    best_cmd_mae: float,
    model_config: Any,
    args: argparse.Namespace,
    train_ep_ids: List[str],
    val_ep_ids: List[str],
    train_loader: Any = None,
) -> None:
    """Atomically save checkpoint."""
    tmp_path = path + ".tmp"
    state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_recovery_acc": best_recovery_acc,
        "best_cmd_mae": best_cmd_mae,
        "model_config": dataclasses.asdict(model_config) if dataclasses.is_dataclass(model_config) else model_config,
        "args": vars(args),
        "train_episode_ids": train_ep_ids,
        "val_episode_ids": val_ep_ids,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    else:
        state["scaler_state_dict"] = None

    # Save DataLoader generator state for deterministic resume.
    if train_loader is not None:
        loader_gen = getattr(train_loader, "generator", None)
        if loader_gen is not None:
            state["dataloader_generator_state"] = loader_gen.get_state()

    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[CosineWarmupScheduler],
    scaler: Optional[GradScaler],
    device: torch.device,
    strict: bool = True,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load checkpoint, handling ``_orig_mod.`` prefix and PyTorch version compat.

    In strict mode, any model key mismatch raises an error.
    Optimizer/scheduler/scaler loading failures also raise in strict mode.
    """
    if checkpoint is None:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
    else:
        ckpt = checkpoint

    state_dict = ckpt["model_state_dict"]

    # Normalise all keys: strip _orig_mod. prefix.
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k_clean = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        cleaned[k_clean] = v

    if strict:
        # strict=True raises on any mismatch; no return value to capture.
        model.load_state_dict(cleaned, strict=True)
    else:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            warnings.warn(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys when loading checkpoint: {unexpected}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to load optimizer state: {exc}") from exc
            warnings.warn(f"Could not load optimizer state: {exc}")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to load scheduler state: {exc}") from exc
            warnings.warn(f"Could not load scheduler state: {exc}")

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to load scaler state: {exc}") from exc
            warnings.warn(f"Could not load scaler state: {exc}")

    # Restore random states.
    if "python_random_state" in ckpt:
        random.setstate(ckpt["python_random_state"])
    if "numpy_random_state" in ckpt:
        np.random.set_state(ckpt["numpy_random_state"])
    if "torch_random_state" in ckpt:
        torch.random.set_rng_state(ckpt["torch_random_state"])
    if "cuda_random_state_all" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_random_state_all"])

    return ckpt


def _config_from_checkpoint(
    ConfigClass: Any,
    checkpoint: Dict[str, Any],
) -> Any:
    """Rebuild the exact model configuration saved in a checkpoint.

    Tuple-valued dataclass fields may have passed through a JSON-like format,
    so list values are normalised back to tuples before validation. Missing
    newly-added fields retain the current dataclass defaults.
    """
    saved = checkpoint.get("model_config")
    if not saved:
        warnings.warn(
            "Checkpoint has no model_config; falling back to current defaults."
        )
        return ConfigClass()
    if not isinstance(saved, dict):
        raise TypeError("checkpoint model_config must be a dictionary")

    defaults = ConfigClass()
    known_fields = {field.name: field for field in dataclasses.fields(defaults)}
    unknown = sorted(set(saved) - set(known_fields))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown model config fields: {unknown}")

    values = dict(saved)
    for name in known_fields:
        default_value = getattr(defaults, name)
        if isinstance(default_value, tuple) and isinstance(values.get(name), list):
            values[name] = tuple(values[name])
    return ConfigClass(**values)


# ===========================================================================
#  Reproducibility
# ===========================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True


# ===========================================================================
#  Dataset statistics
# ===========================================================================

def _print_stats(train_eps: List[EpisodeInfo], val_eps: List[EpisodeInfo],
                 train_windows: int, val_windows: int) -> None:
    scenes = sorted(set(ep.scene_id for ep in train_eps + val_eps))
    print(f"\n{'='*60}\nDataset\n{'='*60}")
    print(f"  Scenes:                {len(scenes)}")
    print(f"  Train episodes:        {len(train_eps)}")
    print(f"  Val episodes:          {len(val_eps)}")
    print(f"  Train windows:         {train_windows}")
    print(f"  Val windows:           {val_windows}")
    print(f"  Train frames:          {sum(ep.num_frames for ep in train_eps)}")
    print(f"  Val frames:            {sum(ep.num_frames for ep in val_eps)}")
    print(f"{'='*60}\n")


# ===========================================================================
#  Argument parser
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train HierarchicalTrendControlPolicy",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--model-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--burn-in", type=int, default=8)
    p.add_argument("--window-stride", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision")
    p.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    p.add_argument("--teacher-forcing-start", type=float, default=1.0)
    p.add_argument("--teacher-forcing-end", type=float, default=0.0)
    p.add_argument("--teacher-forcing-decay-epochs", type=float, default=None)
    p.add_argument("--horizontal-loss-weight", type=float, default=1.0)
    p.add_argument("--vertical-loss-weight", type=float, default=1.0)
    p.add_argument("--guide-value-loss-weight", type=float, default=1.0)
    p.add_argument("--control-loss-weight", type=float, default=2.0)
    p.add_argument(
        "--stateful-training", action="store_true", default=True,
        help="Carry detached LSTM state across chronological trajectory chunks",
    )
    p.add_argument(
        "--no-stateful-training", action="store_false",
        dest="stateful_training",
    )
    p.add_argument(
        "--mirror-augmentation", action="store_true", default=True,
        help="Train on original and fully mirrored trajectory streams",
    )
    p.add_argument(
        "--no-mirror-augmentation", action="store_false",
        dest="mirror_augmentation",
    )
    p.add_argument("--log-interval", type=int, default=50)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be > 0")
    if not (0 <= args.min_learning_rate <= args.learning_rate):
        raise ValueError("--min-learning-rate must be in [0, learning_rate]")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be > 0")
    for n, v in [("teacher-forcing-start", args.teacher_forcing_start),
                 ("teacher-forcing-end", args.teacher_forcing_end)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"--{n} must be in [0,1]")
    for n, v in [("horizontal-loss-weight", args.horizontal_loss_weight),
                 ("vertical-loss-weight", args.vertical_loss_weight),
                 ("guide-value-loss-weight", args.guide_value_loss_weight),
                 ("control-loss-weight", args.control_loss_weight)]:
        if v < 0:
            raise ValueError(f"--{n} must be >= 0")
    if (args.horizontal_loss_weight + args.vertical_loss_weight
            + args.guide_value_loss_weight + args.control_loss_weight) <= 0:
        raise ValueError("At least one loss weight must be > 0")
    for n, v in [("epochs", args.epochs), ("batch-size", args.batch_size),
                 ("sequence-length", args.sequence_length)]:
        if v <= 0:
            raise ValueError(f"--{n} must be > 0")
    if args.burn_in < 0 or args.window_stride <= 0:
        raise ValueError("--burn-in >= 0, --window-stride > 0 required")
    if (
        args.stateful_training
        and args.window_stride != args.sequence_length
    ):
        raise ValueError(
            "stateful training requires --window-stride to equal "
            "--sequence-length"
        )
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")
    if args.num_workers < 0 or args.log_interval <= 0:
        raise ValueError("--num-workers >= 0, --log-interval > 0 required")
    if args.teacher_forcing_decay_epochs is not None and args.teacher_forcing_decay_epochs <= 0:
        raise ValueError("--teacher-forcing-decay-epochs must be > 0 if provided")


# ===========================================================================
#  Main
# ===========================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("[train] AMP disabled (CUDA not available).")

    set_seed(args.seed)

    # Load model. On resume, the checkpoint is authoritative for every
    # architecture/command-constraint field, not merely a small subset.
    ModelClass, ConfigClass = _load_model_from_file(args.model_file)
    resume_ckpt: Optional[Dict[str, Any]] = None
    if args.resume:
        try:
            resume_ckpt = torch.load(
                args.resume, map_location=device, weights_only=False
            )
        except TypeError:
            resume_ckpt = torch.load(args.resume, map_location=device)
        config = _config_from_checkpoint(ConfigClass, resume_ckpt)
    else:
        config = ConfigClass()
    config.validate()
    for name in ("max_vx_flu", "max_vy_flu", "max_vz_flu", "max_yaw_rate"):
        if getattr(config, name, 0.0) <= 0.0:
            raise ValueError(f"config.{name} must be > 0")
    model = ModelClass(config).to(device)
    print(f"[train] Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[train] Image size: {config.image_size}")

    # Dataloaders
    print(f"[train] Discovering episodes under: {args.dataset_root}")
    train_loader, val_loader, train_eps, val_eps = build_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        burn_in=args.burn_in, window_stride=args.window_stride,
        target_height=config.image_size[0], target_width=config.image_size[1],
        val_ratio=args.val_ratio, seed=args.seed, num_workers=args.num_workers,
        stateful_training=args.stateful_training,
        mirror_augmentation=args.mirror_augmentation,
    )
    _print_stats(train_eps, val_eps, len(train_loader.dataset), len(val_loader.dataset))
    if args.stateful_training:
        print(
            f"[train] Sequence: stateful TBPTT chunks={args.sequence_length}, "
            f"episode-prefix burn-in={args.burn_in}"
        )
    else:
        print(
            f"[train] Sequence: {args.burn_in} burn-in + "
            f"{args.sequence_length} target, stride={args.window_stride}"
        )
    print(
        f"[train] Mirror augmentation: "
        f"{'original + mirrored streams' if args.mirror_augmentation else 'off'}"
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps,
                                       args.min_learning_rate / args.learning_rate)
    scaler = _amp_scaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_recovery_acc = float("-inf")
    best_cmd_mae = float("inf")
    train_ep_ids = sorted(set(ep.episode_id for ep in train_eps))
    val_ep_ids = sorted(set(ep.episode_id for ep in val_eps))

    if args.resume:
        print(f"[train] Resuming from: {args.resume}")
        ckpt = _load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            checkpoint=resume_ckpt,
        )
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_recovery_acc = ckpt.get("best_recovery_acc", float("-inf"))
        best_cmd_mae = ckpt.get("best_cmd_mae", float("inf"))
        scheduler.retarget(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=args.min_learning_rate / args.learning_rate,
        )
        ckpt_args = ckpt.get("args", {})
        for k in (
            "sequence_length", "burn_in", "window_stride",
            "stateful_training", "mirror_augmentation",
        ):
            if k in ckpt_args and ckpt_args[k] != getattr(args, k):
                raise ValueError(f"Argument mismatch: {k}")
        if set(ckpt.get("train_episode_ids", [])) != set(train_ep_ids):
            raise ValueError("Train episode IDs mismatch")
        if set(ckpt.get("val_episode_ids", [])) != set(val_ep_ids):
            raise ValueError("Val episode IDs mismatch")
        gen = getattr(train_loader, "generator", None)
        gs = ckpt.get("dataloader_generator_state")
        if gen is not None and gs is not None:
            gen.set_state(gs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_weights = {"horizontal": args.horizontal_loss_weight, "vertical": args.vertical_loss_weight,
                    "guide_value": args.guide_value_loss_weight, "control": args.control_loss_weight}

    tf_decay_epochs = args.teacher_forcing_decay_epochs
    if tf_decay_epochs is None:
        tf_decay_epochs = int(args.epochs * 0.7)
    tf_decay_epochs = max(1, tf_decay_epochs)

    csv_path = output_dir / "metrics.csv"
    csv_header = ["epoch", "train_loss", "val_loss", "val_h_acc", "val_v_acc", "val_gv_mae",
                  "val_cmd_vx_mae", "val_cmd_vy_mae", "val_cmd_vz_mae", "val_cmd_yaw_mae",
                  "val_cmd_avg_mae", "val_cmd_step_l2",
                  "val_h_acc_normal", "val_h_acc_recovery",
                  "val_cmd_avg_mae_normal", "val_cmd_avg_mae_recovery",
                  "learning_rate", "teacher_forcing_ratio"]
    write_header = not csv_path.exists() or start_epoch == 0
    if csv_path.exists() and not write_header:
        with open(csv_path, "r", newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header != csv_header:
            csv_path = output_dir / "metrics_continuous_validation.csv"
            write_header = not csv_path.exists()
            warnings.warn(
                "Existing metrics.csv uses an older column schema; resumed "
                f"metrics will be written to {csv_path.name}."
            )

    try:
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(csv_header); csv_file.flush()

        print(f"\n[train] Epochs: {args.epochs}, Steps/epoch: {steps_per_epoch}")
        print(f"[train] Device: {device}, AMP: {use_amp}")
        print(f"[train] TF decay: {args.teacher_forcing_start:.2f} -> {args.teacher_forcing_end:.2f} over {tf_decay_epochs} epochs")
        print(f"[train] Output: {output_dir.resolve()}\n")

        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            trend_state_cache: Dict[str, Any] = {}
            control_state_cache: Dict[str, Any] = {}

            tf_prob = args.teacher_forcing_end
            if epoch < tf_decay_epochs:
                progress = epoch / max(1, tf_decay_epochs - 1)
                tf_prob = args.teacher_forcing_start + (args.teacher_forcing_end - args.teacher_forcing_start) * progress
            tf_prob = max(0.0, min(1.0, tf_prob))

            train_acc = MetricAccumulator()
            data_time_total = 0.0; step_start = time.time()
            tf_actual = 0; tf_total = 0

            for batch_idx, batch in enumerate(train_loader):
                data_time_total += time.time() - step_start; step_start = time.time()

                depth = batch["depth"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                raw_guide = batch["raw_guide"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                gravity = batch["gravity_flu"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                velocity = batch["velocity_flu"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                yaw_rate = batch["yaw_rate"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                trajectory_ids = list(batch["trajectory_id"])
                is_last = list(batch["is_last"])
                trend_state = None
                control_state = None
                if args.stateful_training:
                    trend_state = _gather_recurrent_state(
                        model, trend_state_cache, trajectory_ids,
                        config.trend_lstm_layers,
                        config.trend_lstm_hidden_dim,
                        device, depth.dtype,
                    )
                    control_state = _gather_recurrent_state(
                        model, control_state_cache, trajectory_ids,
                        config.control_lstm_layers,
                        config.control_lstm_hidden_dim,
                        device, depth.dtype,
                    )

                # Use one coherent teacher-forcing decision for each complete
                # chunk instead of switching true/predicted guide every frame.
                seq_mask = batch["sequence_mask"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                tf_mask = torch.zeros_like(seq_mask)
                if tf_prob > 0.0:
                    chunk_choice = (
                        torch.rand(
                            (seq_mask.shape[0], 1, 1),
                            device=device,
                        ) < tf_prob
                    ).to(dtype=seq_mask.dtype)
                    tf_mask = chunk_choice * seq_mask
                tf_actual += tf_mask.sum().item(); tf_total += seq_mask.sum().item()

                h_t = batch["horizontal_target"].to(
                    device=device, non_blocking=True
                )
                v_t = batch["vertical_target"].to(
                    device=device, non_blocking=True
                )
                gv_t = batch["guide_value_target"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                # Zero recovery guide values for TF safety.
                rec_m = ((h_t == 0) | (h_t == 12)).float().unsqueeze(-1).to(device=device, dtype=torch.float32)
                gv_t = gv_t * (1.0 - rec_m)

                with _amp_autocast(enabled=use_amp):
                    output = model(depth=depth, raw_guide=raw_guide, gravity_flu=gravity,
                                   velocity_flu=velocity, yaw_rate=yaw_rate,
                                   trend_state=trend_state,
                                   control_state=control_state,
                                   teacher_horizontal=h_t, teacher_vertical=v_t,
                                   teacher_guide_value=gv_t, teacher_forcing_mask=tf_mask)

                losses = compute_losses(output, batch, config, loss_weights)
                total_loss = losses["total"]

                if not torch.isfinite(total_loss):
                    warnings.warn(f"Non-finite loss at epoch {epoch}, step {batch_idx}. Skipping.")
                    for trajectory_id in trajectory_ids:
                        trend_state_cache.pop(trajectory_id, None)
                        control_state_cache.pop(trajectory_id, None)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if args.stateful_training:
                    _store_recurrent_state(
                        model, trend_state_cache, trajectory_ids,
                        is_last, output.trend_state,
                    )
                    _store_recurrent_state(
                        model, control_state_cache, trajectory_ids,
                        is_last, output.control_state,
                    )

                optimizer.zero_grad(set_to_none=True)
                optimizer_updated = True
                if scaler is not None:
                    old_scale = scaler.get_scale()
                    scaler.scale(total_loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                    scaler.step(optimizer); scaler.update()
                    optimizer_updated = scaler.get_scale() >= old_scale
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                    optimizer.step()

                if optimizer_updated:
                    scheduler.step(); global_step += 1

                for name in ("horizontal", "vertical", "guide_value", "control"):
                    train_acc.add_from_reduction(f"train_{name}", losses[f"_{name}_red"])

                if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                    step_time = time.time() - step_start
                    lr = scheduler.get_last_lr()[0]; avg = train_acc.compute()
                    rt = (loss_weights["horizontal"] * avg.get("train_horizontal", 0)
                          + loss_weights["vertical"] * avg.get("train_vertical", 0)
                          + loss_weights["guide_value"] * avg.get("train_guide_value", 0)
                          + loss_weights["control"] * avg.get("train_control", 0))
                    tf_ratio = tf_actual / max(1, tf_total)
                    print(f"  Epoch {epoch:3d} | Step {batch_idx:5d}/{len(train_loader)} | "
                          f"loss={rt:.4f} h={avg.get('train_horizontal',0):.4f} "
                          f"v={avg.get('train_vertical',0):.4f} gv={avg.get('train_guide_value',0):.4f} "
                          f"ctrl={avg.get('train_control',0):.4f} | tf_p={tf_prob:.2f} tf_r={tf_ratio:.2f} | "
                          f"lr={lr:.2e} | data={data_time_total/max(1,batch_idx+1)*1000:.0f}ms "
                          f"step={step_time*1000:.0f}ms")
                    step_start = time.time()

            if args.stateful_training and (
                trend_state_cache or control_state_cache
            ):
                raise RuntimeError(
                    "stateful sampler ended with unfinished trajectory states"
                )

            epoch_time = time.time() - epoch_start
            train_metrics = train_acc.compute()
            train_total = (loss_weights["horizontal"] * train_metrics.get("train_horizontal", 0.0)
                           + loss_weights["vertical"] * train_metrics.get("train_vertical", 0.0)
                           + loss_weights["guide_value"] * train_metrics.get("train_guide_value", 0.0)
                           + loss_weights["control"] * train_metrics.get("train_control", 0.0))

            val_metrics = validate(model, val_loader, config, loss_weights, device, use_amp)
            val_loss = val_metrics["total"]

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            recovery_acc = val_metrics.get("h_acc_recovery", 0.0)
            cmd_mae = val_metrics.get("cmd_avg_mae", float("inf"))
            is_best_recovery = recovery_acc > best_recovery_acc
            is_best_command = cmd_mae < best_cmd_mae
            if is_best_recovery:
                best_recovery_acc = recovery_acc
            if is_best_command:
                best_cmd_mae = cmd_mae

            _save_checkpoint(str(output_dir / "last.pt"), model, optimizer, scheduler, scaler,
                             epoch, global_step, best_val_loss,
                             best_recovery_acc, best_cmd_mae, config, args,
                             train_ep_ids, val_ep_ids, train_loader)
            if is_best:
                _save_checkpoint(str(output_dir / "best.pt"), model, optimizer, scheduler, scaler,
                                 epoch, global_step, best_val_loss,
                                 best_recovery_acc, best_cmd_mae, config, args,
                                 train_ep_ids, val_ep_ids, train_loader)
            if is_best_recovery:
                _save_checkpoint(
                    str(output_dir / "best_recovery.pt"),
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_loss,
                    best_recovery_acc, best_cmd_mae, config, args,
                    train_ep_ids, val_ep_ids, train_loader,
                )
            if is_best_command:
                _save_checkpoint(
                    str(output_dir / "best_command.pt"),
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_loss,
                    best_recovery_acc, best_cmd_mae, config, args,
                    train_ep_ids, val_ep_ids, train_loader,
                )

            lr = scheduler.get_last_lr()[0]
            tf_ratio = tf_actual / max(1, tf_total)
            csv_writer.writerow([epoch, train_total, val_loss] +
                                [val_metrics.get(k, 0) for k in
                                  ("h_acc", "v_acc", "gv_mae", "cmd_vx_mae", "cmd_vy_mae",
                                   "cmd_vz_mae", "cmd_yaw_mae", "cmd_avg_mae", "cmd_step_l2",
                                   "h_acc_normal", "h_acc_recovery",
                                  "cmd_avg_mae_normal", "cmd_avg_mae_recovery")] +
                                [lr, tf_ratio])
            csv_file.flush()

            print(f"\n--- Epoch {epoch:3d} ({epoch_time:.1f}s) ---"
                  f"\n  Train loss: {train_total:.4f}"
                  f"\n  Val loss:   {val_loss:.4f} {'(*BEST*)' if is_best else ''}"
                  f"\n  Val H acc:  {val_metrics.get('h_acc',0):.4f}  V acc: {val_metrics.get('v_acc',0):.4f}"
                  f"\n  Val GV MAE: {val_metrics.get('gv_mae',0):.4f}  Cmd MAE: {val_metrics.get('cmd_avg_mae',0):.4f}"
                  f"\n  Cmd step L2:{val_metrics.get('cmd_step_l2',0):.4f}"
                  f"\n  H acc norm: {val_metrics.get('h_acc_normal',0):.4f}  rec: {val_metrics.get('h_acc_recovery',0):.4f}"
                  f"\n  Cmd norm:   {val_metrics.get('cmd_avg_mae_normal',0):.4f}  rec: {val_metrics.get('cmd_avg_mae_recovery',0):.4f}"
                  f"\n  LR: {lr:.2e}  TF ratio: {tf_ratio:.2f}\n")

    finally:
        csv_file.close()

    print(f"[train] Complete. Best val loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoints: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
