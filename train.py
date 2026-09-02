#!/usr/bin/env python3
"""Train the schema-v25 causal ViT-LSTM imitation policy."""

import argparse
import csv
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

from dataloader import (AVOIDANCE_MODES, COMMAND_SCALE, HIERARCHICAL_MODES,
                        SCHEMA_VERSION, STATE_FIELDS, STATE_SCALE,
                        TARGET_FIELDS, build_dataloaders,
                        discover_committed_episodes)
from model.model import ViTFlyLSTMPolicy, ViTFlyPolicyConfig


@contextmanager
def nullcontext():
    """contextlib.nullcontext for the ROS workspace's Python 3.6."""
    yield


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_avoidance_label_coverage(episodes) -> None:
    """Expose whether an incremental dataset adds decisive avoidance labels
    from the NEW two-level expert (hierarchical_mode + 5 Hz directives)."""
    mode_counts: Dict[str, int] = {}
    avoidance_frames = 0
    macro_frames = 0
    macro_normal = 0
    macro_turn_left = 0
    macro_turn_right = 0
    lateral_02 = 0
    lateral_04 = 0
    yaw_02 = 0
    reactive = 0
    for episode in episodes:
        with episode.csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                mode = row.get("hierarchical_mode", "direct")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                if mode in AVOIDANCE_MODES:
                    avoidance_frames += 1
                    lateral = abs(float(row["target_velocity_flu_y"]))
                    yaw_rate = abs(float(row["target_yaw_rate"]))
                    lateral_02 += int(lateral >= 0.20)
                    lateral_04 += int(lateral >= 0.40)
                    yaw_02 += int(yaw_rate >= 0.20)
                    reactive += int(lateral >= 0.20 or yaw_rate >= 0.20)
                if _truthy(row.get("macro_update_mask", "0")):
                    macro_frames += 1
                    ctype = row.get("macro_correction_type", "PASS_THROUGH")
                    if ctype == "NORMAL_CORRECTION":
                        macro_normal += 1
                    elif ctype == "TURN_LEFT":
                        macro_turn_left += 1
                    elif ctype == "TURN_RIGHT":
                        macro_turn_right += 1
    total_frames = sum(mode_counts.values())
    denominator = max(1, avoidance_frames)
    print("hierarchical modes: %s" % ", ".join(
        "%s=%d(%.1f%%)" % (name, count, 100.0 * count / max(1, total_frames))
        for name, count in sorted(mode_counts.items())))
    print("avoidance labels: avoidance_frames=%d(%.1f%%) "
          "avoid_|vy|>=0.2=%d(%.1f%%) avoid_|vy|>=0.4=%d(%.1f%%) "
          "avoid_|yaw|>=0.2=%d(%.1f%%) reactive=%d(%.1f%%)" % (
              avoidance_frames, 100.0 * avoidance_frames / max(1, total_frames),
              lateral_02, 100.0 * lateral_02 / denominator,
              lateral_04, 100.0 * lateral_04 / denominator,
              yaw_02, 100.0 * yaw_02 / denominator,
              reactive, 100.0 * reactive / denominator))
    print("5 Hz labels: macro_frames=%d normal=%d turn_left=%d turn_right=%d"
          % (macro_frames, macro_normal, macro_turn_left, macro_turn_right))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def compute_loss(normalized_prediction: torch.Tensor,
                 normalized_target: torch.Tensor, loss_mask: torch.Tensor,
                 hierarchical_mode: torch.Tensor, smoothness_weight: float = 0.05,
                 mode_weighting: bool = True,
                 local_avoidance_weight: float = 2.0,
                 depth: Optional[torch.Tensor] = None,
                 clearance_weight: float = 0.0,
                 clearance_margin: float = 0.3,
                 max_depth_m: float = 5.0,
                 max_accel: float = 4.0
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rare decision states deserve more weight without exposing them to the
    # policy: hierarchical_mode is used only by the loss.
    # local_avoidance / macro_* / turn_to_target / blocked -> 2.0x;
    # goal_capture -> 1.5x.  mode_weighting=False is the no-label-weighting
    # ablation (uniform frame weighting).
    mode_weight = torch.ones_like(loss_mask)
    if mode_weighting:
        mode_weight = torch.where(hierarchical_mode == 1,
                                  local_avoidance_weight * mode_weight,
                                  mode_weight)
        mode_weight = torch.where(hierarchical_mode == 2, 2.0 * mode_weight,
                                  mode_weight)
        mode_weight = torch.where(hierarchical_mode == 3, 2.0 * mode_weight,
                                  mode_weight)
        mode_weight = torch.where(hierarchical_mode == 4, 2.0 * mode_weight,
                                  mode_weight)
        mode_weight = torch.where(hierarchical_mode == 5, 2.0 * mode_weight,
                                  mode_weight)
        mode_weight = torch.where(hierarchical_mode == 6, 1.5 * mode_weight,
                                  mode_weight)
    weighted_mask = loss_mask * mode_weight
    element = F.smooth_l1_loss(
        normalized_prediction, normalized_target, reduction="none", beta=0.1)
    velocity = masked_mean(element[..., :3], weighted_mask)
    # ── yaw 收缩补偿（2026-08-27）────────────────────────────────
    # 诊断：yaw_rate 分布极不均衡（大量 0 直飞 + 少数 ±1.5 转向），MSE 下模型
    # yaw 向 0 收缩（右侧目标 [0.2,0.4) 桶学生 yaw 只有专家的 1.5%），导致避障
    # 转向不足 → rollout 7 碰撞 / 7 超时。对策（镜像 velocity 补偿）：
    #   1) |专家 yaw| > 0.3 rad/s 的帧（需要转向）→ yaw 损失 ×3；
    #   2) TURN 帧（mode 3/4，gdn==1 纯旋转）→ yaw 损失 ×2，消除 std 摆动。
    yaw_mask = weighted_mask
    expert_yaw = normalized_target[..., 3] * COMMAND_SCALE[3]
    big_yaw = (torch.abs(expert_yaw) > 0.3).to(normalized_target.dtype)
    yaw_mask = yaw_mask * (1.0 + 2.0 * big_yaw)          # |yaw|>0.3 → ×3
    turn = ((hierarchical_mode == 3) | (hierarchical_mode == 4)).to(
        normalized_target.dtype)
    yaw_mask = yaw_mask * (1.0 + 1.0 * turn)             # TURN → ×2
    yaw = masked_mean(element[..., 3:], yaw_mask)
    if normalized_prediction.shape[1] > 1:
        pred_delta = normalized_prediction[:, 1:] - normalized_prediction[:, :-1]
        target_delta = normalized_target[:, 1:] - normalized_target[:, :-1]
        pair_mask = loss_mask[:, 1:] * loss_mask[:, :-1]
        smooth = masked_mean(F.smooth_l1_loss(
            pred_delta, target_delta, reduction="none", beta=0.05), pair_mask)
    else:
        smooth = velocity.new_zeros(())
    clearance = velocity.new_zeros(())
    if clearance_weight > 0.0 and depth is not None:
        # ── clearance-aware speed loss ──────────────────────────────
        # depth 归一化 [0,1]（0=近, 1=远）。每帧最近障碍距离决定物理可刹停的
        # 安全速度上限 v_safe = sqrt(2*a*(d-margin))。对目标速度超出的部分施加
        # ReLU 惩罚，让模型学会"离障碍越近飞得越慢"，解决 0.28 m 深度盲区下
        # 2 m/s 巡航刹不住导致的碰撞。
        # 只限制前向速度 vx：横向/垂直速度（vy/vz）是避障机动，减速会让绕行
        # 大障碍时失去机动能力（large 尺度碰撞飙升的根源），所以不约束它们。
        # 深度也只取正前方中心一片（避开侧面大障碍表面）：否则绕行大圆柱时
        # 侧面表面占据大片近像素，v_safe 被压得过低，导致绕行犹豫/擦碰。
        #
        # BUGFIX (2026-09-01): 之前 violation 用的是 normalized_target（专家
        # 速度，常量），对模型梯度恒为 0，clearance 项从未参与训练 —— 这正是
        # v35/v36/v37 的 epoch-050 指标逐位相同、clearance_loss 却不同的原因。
        # 改为惩罚模型的「输出速度」：损失是 (输入深度, 模型输出) 的纯函数，
        # 推理时无任何隐藏改动，保持 depth→velocity 一致性；模型从自身看到的
        # 深度序列学「离障碍多近就该飞多慢」，LSTM 时序结构自然学到提前刹车。
        h, w = depth.shape[-2], depth.shape[-1]
        center = depth[..., h // 4:3 * h // 4, w // 3:2 * w // 3]
        d_min = center.reshape(*center.shape[:2], -1).min(dim=-1).values
        d_min_m = d_min * max_depth_m
        v_safe = torch.sqrt(
            2.0 * max_accel * torch.clamp(d_min_m - clearance_margin, min=0.0))
        speed_pred = torch.abs(
            normalized_prediction[..., 0]) * float(COMMAND_SCALE[0])
        violation = torch.relu(speed_pred - v_safe)
        clearance = masked_mean(violation, loss_mask)
    total = velocity + yaw + float(smoothness_weight) * smooth \
        + float(clearance_weight) * clearance
    return total, {"velocity_loss": velocity, "yaw_loss": yaw,
                   "smoothness_loss": smooth, "clearance_loss": clearance}


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


def _move(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device, non_blocking=True)
            if torch.is_tensor(value) else value for key, value in batch.items()}


def _stateful_hidden(model: ViTFlyLSTMPolicy, batch: Dict[str, object],
                     hidden_by_episode: Dict[str, Tuple[torch.Tensor,
                                                        torch.Tensor]],
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble one detached LSTM state column per batch episode."""
    episode_ids = batch["episode_id"]
    first_chunks = batch["is_first_chunk"].detach().cpu().tolist()
    hidden_columns: List[torch.Tensor] = []
    cell_columns: List[torch.Tensor] = []
    for episode_id, is_first in zip(episode_ids, first_chunks):
        if bool(is_first) or episode_id not in hidden_by_episode:
            hidden, cell = model.initial_hidden(1, device=device)
        else:
            hidden, cell = hidden_by_episode[episode_id]
        hidden_columns.append(hidden)
        cell_columns.append(cell)
    return torch.cat(hidden_columns, dim=1), torch.cat(cell_columns, dim=1)


def _remember_hidden(batch: Dict[str, object],
                     hidden: Tuple[torch.Tensor, torch.Tensor],
                     hidden_by_episode: Dict[str, Tuple[torch.Tensor,
                                                        torch.Tensor]]) -> None:
    """Detach at chunk boundaries: standard truncated BPTT."""
    hidden_state, cell_state = hidden
    for column, episode_id in enumerate(batch["episode_id"]):
        hidden_by_episode[episode_id] = (
            hidden_state[:, column:column + 1].detach(),
            cell_state[:, column:column + 1].detach())


def run_epoch(model: ViTFlyLSTMPolicy, loader: Iterable,
              device: torch.device, optimizer=None, scaler=None,
              scheduler=None, amp: bool = False, grad_clip: float = 1.0,
              smoothness_weight: float = 0.05,
              stateful: bool = True,
              mode_weighting: bool = True,
              local_avoidance_weight: float = 2.0,
              clearance_weight: float = 0.0,
              clearance_margin: float = 0.3,
              max_depth_m: float = 5.0,
              max_accel: float = 4.0) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = Metrics()
    scale = torch.as_tensor(COMMAND_SCALE, device=device).view(1, 1, 4)
    hidden_by_episode: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for batch in loader:
        batch = _move(batch, device)
        hidden = _stateful_hidden(
            model, batch, hidden_by_episode, device) if stateful else None
        context = torch.cuda.amp.autocast(enabled=amp) \
            if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(training), context:
            output = model(batch["depth"], batch["state"], hidden)
            loss, parts = compute_loss(
                output.normalized_command, batch["target"],
                batch["loss_mask"], batch["hierarchical_mode"],
                smoothness_weight, mode_weighting,
                local_avoidance_weight=local_avoidance_weight,
                depth=batch.get("depth"),
                clearance_weight=clearance_weight,
                clearance_margin=clearance_margin,
                max_depth_m=max_depth_m,
                max_accel=max_accel)
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
        valid = float(batch["loss_mask"].sum().item())
        metrics.add("loss", float(loss.detach().item()), valid)
        for key, value in parts.items():
            metrics.add(key, float(value.detach().item()), valid)
        error = (output.normalized_command.detach() - batch["target"]).abs() * scale
        mask = batch["loss_mask"].bool()
        if mask.any():
            selected = error[mask]
            names = ("vx_mae", "vy_mae", "vz_mae", "yaw_rate_mae")
            for component, name in enumerate(names):
                metrics.add(name, float(selected[:, component].mean().item()),
                            float(selected.shape[0]))
            for mode_index, mode_name in enumerate(HIERARCHICAL_MODES):
                mode_mask = mask & (batch["hierarchical_mode"] == mode_index)
                if mode_mask.any():
                    mode_error = error[mode_mask]
                    metrics.add(mode_name + "_mae",
                                float(mode_error.mean().item()),
                                float(mode_error.numel()))
    return metrics.result()


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path: Path, model: ViTFlyLSTMPolicy, optimizer,
                    scheduler, scaler, epoch: int, best_val: float,
                    train_eps, val_eps, args) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "ViTFlyLSTMPolicy",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val,
        "student_input_fields": ["depth_file"] + list(STATE_FIELDS),
        "target_fields": list(TARGET_FIELDS),
        "normalization": {
            "depth_max_m": 5.0,
            "state_scale": STATE_SCALE.tolist(),
            "command_scale": COMMAND_SCALE.tolist(),
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
        description="Train causal ViTFly-style policy on il_dataset v25")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="checkpoints/vitfly_v25")
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument(
        "--mirror-augmentation", action="store_true",
        help="opt in to horizontal image/state/command mirroring; disabled "
             "by default because symmetric obstacle scenes can have a "
             "deterministic expert side preference")
    parser.add_argument(
        "--depth-noise-std-ratio", type=float, default=0.02,
        help="D435i sim-to-real multiplicative Gaussian depth noise applied "
             "at training time (sigma = ratio * normalized depth).  0.02 = "
             "D435i <2%% @2m.  The val loader always uses 0.")
    parser.add_argument(
        "--stateless-windows", action="store_true",
        help="use legacy independent windows and burn-in instead of stateful TBPTT")
    parser.add_argument("--resume", default="")
    parser.add_argument("--audit-only", action="store_true",
                        help="validate and summarize the dataset without training")
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
    if args.audit_only:
        return
    train_loader, val_loader, train_eps, val_eps = build_dataloaders(
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
            model, val_loader, device, smoothness_weight=args.smoothness_weight,
            stateful=not args.stateless_windows)
        print("epoch %03d train=%s val=%s" %
              (epoch + 1, json.dumps(train_metrics, sort_keys=True),
               json.dumps(val_metrics, sort_keys=True)))
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler,
                        scaler, epoch, best_val, train_eps, val_eps, args)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler,
                            scaler, epoch, best_val, train_eps, val_eps, args)


if __name__ == "__main__":
    main()
