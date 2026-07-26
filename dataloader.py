"""Hierarchical IL sequence dataset for committed Schema v15 trajectories.

Only reads committed trajectory directories (validated by the collector).
Each traj_xxx/ is one complete, continuous episode.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from PIL import Image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeInfo:
    """Metadata for one committed trajectory."""
    traj_dir: str
    scene_id: str
    task_id: str
    episode_id: str
    num_frames: int
    csv_path: str
    depth_dir: str
    depth_h: int
    depth_w: int


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_committed_episodes(dataset_root: Union[str, Path]) -> List[EpisodeInfo]:
    """Find committed trajectory directories under dataset_root.

    Expected layout: <dataset_root>/<scene>/traj_xxx/
    Each must contain data.csv, metadata.json (schema_version==15, status=="committed"), and depth/.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")

    episodes: List[EpisodeInfo] = []

    for dirpath_str, dirnames, _filenames in os.walk(str(root)):
        dirpath = Path(dirpath_str)
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and not d.endswith(".inprogress")
            and d not in ("_failed", "_debug", "_eval", "scenes", "legacy")
        ]

        csv_path = dirpath / "data.csv"
        meta_path = dirpath / "metadata.json"
        depth_dir = dirpath / "depth"

        if not (csv_path.is_file() and meta_path.is_file() and depth_dir.is_dir()):
            continue

        # Validate metadata.
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("schema_version") != 15:
            raise ValueError(
                f"{meta_path}: schema_version must be 15, got {meta.get('schema_version')}"
            )
        if meta.get("status", "committed") != "committed":
            continue  # skip uncommitted

        depth_h = int(meta.get("depth_h", 0))
        depth_w = int(meta.get("depth_w", 0))
        if depth_h <= 0 or depth_w <= 0:
            raise ValueError(f"{meta_path}: missing or invalid depth_h/depth_w")

        # Read CSV header and count rows.
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
            rows_list = list(reader)
        if len(rows_list) == 0:
            raise ValueError(f"{csv_path}: empty")

        # Verify required columns exist.
        required = [
            "frame_id", "depth_file",
            "global_dir_x_flu", "global_dir_y_flu", "global_dir_z_flu",
            "global_distance_norm",
            "gravity_direction_x_flu", "gravity_direction_y_flu", "gravity_direction_z_flu",
            "state_vx_flu", "state_vy_flu", "state_vz_flu",
            "state_angular_velocity_z_flu",
            "trend_horizontal_class_13",
            "trend_horizontal_soft_00", "trend_horizontal_soft_12",
            "guide_elevation_bin",
            "guide_elevation_soft_0", "guide_elevation_soft_6",
            "guide_distance_norm",
            "expert_vx_flu", "expert_vy_flu", "expert_vz_flu", "expert_yaw_rate",
        ]
        missing = [c for c in required if c not in headers]
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {missing}")

        # Derive identifiers.
        first_row = rows_list[0]
        scene_id = first_row.get("scene_id", "").strip()
        task_id = first_row.get("task_id", "").strip()
        episode_id = first_row.get("episode_id", "").strip()

        if not scene_id:
            rel = dirpath.relative_to(root)
            for p in rel.parts:
                if p.startswith("scene_"):
                    scene_id = p
                    break
            if not scene_id and len(rel.parts) >= 1:
                scene_id = rel.parts[0]
        if not task_id:
            rel = dirpath.relative_to(root)
            task_id = rel.parts[-1] if rel.parts else dirpath.name
        if not episode_id:
            episode_id = f"{scene_id}/{task_id}"

        episodes.append(EpisodeInfo(
            traj_dir=str(dirpath), scene_id=scene_id, task_id=task_id,
            episode_id=episode_id, num_frames=len(rows_list),
            csv_path=str(csv_path), depth_dir=str(depth_dir),
            depth_h=depth_h, depth_w=depth_w,
        ))

    episodes.sort(key=lambda ep: ep.traj_dir)
    if not episodes:
        raise FileNotFoundError(f"No committed trajectory directories found under {root}")
    return episodes


# ---------------------------------------------------------------------------
# Episode splitting
# ---------------------------------------------------------------------------

def split_episodes(
    episodes: List[EpisodeInfo],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[EpisodeInfo], List[EpisodeInfo]]:
    """Split episodes into train/validation by scene, approximating val_ratio by frames."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    scene_to_eps: Dict[str, List[EpisodeInfo]] = OrderedDict()
    for ep in episodes:
        scene_to_eps.setdefault(ep.scene_id, []).append(ep)

    scene_ids = list(scene_to_eps.keys())
    scene_frames = {sid: sum(ep.num_frames for ep in eps) for sid, eps in scene_to_eps.items()}
    total_frames = sum(scene_frames.values())

    if len(scene_ids) >= 2:
        rng = random.Random(seed)
        shuffled = list(scene_ids)
        rng.shuffle(shuffled)
        cum = 0
        val_scenes: Set[str] = set()
        for sid in shuffled:
            cum += scene_frames[sid]
            val_scenes.add(sid)
            if cum / total_frames >= val_ratio:
                break
        train_eps = [ep for ep in episodes if ep.scene_id not in val_scenes]
        val_eps = [ep for ep in episodes if ep.scene_id in val_scenes]
    elif len(scene_ids) == 1 and len(episodes) >= 2:
        warnings.warn(
            "Only 1 scene; splitting by episode. "
            "Cross-scene generalisation cannot be evaluated."
        )
        rng = random.Random(seed)
        ep_list = list(episodes)
        rng.shuffle(ep_list)
        cum = 0
        n_val = 0
        for i, ep in enumerate(ep_list):
            cum += ep.num_frames
            if cum / total_frames >= val_ratio:
                n_val = i + 1
                break
        if n_val == 0:
            n_val = 1
        val_eps = ep_list[:n_val]
        train_eps = ep_list[n_val:]
    else:
        raise ValueError(
            f"Insufficient episodes for train/val split "
            f"({len(episodes)} episodes across {len(scene_ids)} scenes)."
        )

    if not train_eps:
        raise ValueError("Train split is empty. Reduce val_ratio.")
    if not val_eps:
        raise ValueError("Validation split is empty. Increase val_ratio.")
    return train_eps, val_eps


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

_H_SOFT_COUNT = 13
_V_SOFT_COUNT = 7
_H_SOFT_COLS = [f"trend_horizontal_soft_{i:02d}" for i in range(_H_SOFT_COUNT)]
_V_SOFT_COLS = [f"guide_elevation_soft_{i}" for i in range(_V_SOFT_COUNT)]


@dataclass
class _EpisodeData:
    """All scalar data for one episode, stored as numpy arrays."""
    depth_files: np.ndarray
    global_dir_x: np.ndarray
    global_dir_y: np.ndarray
    global_dir_z: np.ndarray
    global_dist_norm: np.ndarray
    grav_x: np.ndarray
    grav_y: np.ndarray
    grav_z: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    vel_z: np.ndarray
    yaw_rate: np.ndarray
    h_class: np.ndarray
    v_class: np.ndarray
    guide_val: np.ndarray
    h_soft: np.ndarray
    v_soft: np.ndarray
    expert_vx: np.ndarray
    expert_vy: np.ndarray
    expert_vz: np.ndarray
    expert_yaw: np.ndarray
    episode_id: str


def load_episode_csv(csv_path: str, episode_id: str) -> _EpisodeData:
    """Parse a committed CSV into numpy arrays."""
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def _fcol(col: str) -> np.ndarray:
        return np.array([float(row[col]) for row in rows], dtype=np.float32)

    return _EpisodeData(
        depth_files=np.array([row["depth_file"].strip() for row in rows], dtype=object),
        global_dir_x=_fcol("global_dir_x_flu"),
        global_dir_y=_fcol("global_dir_y_flu"),
        global_dir_z=_fcol("global_dir_z_flu"),
        global_dist_norm=_fcol("global_distance_norm"),
        grav_x=_fcol("gravity_direction_x_flu"),
        grav_y=_fcol("gravity_direction_y_flu"),
        grav_z=_fcol("gravity_direction_z_flu"),
        vel_x=_fcol("state_vx_flu"),
        vel_y=_fcol("state_vy_flu"),
        vel_z=_fcol("state_vz_flu"),
        yaw_rate=_fcol("state_angular_velocity_z_flu"),
        h_class=np.array([int(row["trend_horizontal_class_13"]) for row in rows], dtype=np.int64),
        v_class=np.array([int(row["guide_elevation_bin"]) for row in rows], dtype=np.int64),
        guide_val=_fcol("guide_distance_norm"),
        h_soft=np.array(
            [[float(row[c]) for c in _H_SOFT_COLS] for row in rows], dtype=np.float32
        ),
        v_soft=np.array(
            [[float(row[c]) for c in _V_SOFT_COLS] for row in rows], dtype=np.float32
        ),
        expert_vx=_fcol("expert_vx_flu"),
        expert_vy=_fcol("expert_vy_flu"),
        expert_vz=_fcol("expert_vz_flu"),
        expert_yaw=_fcol("expert_yaw_rate"),
        episode_id=episode_id,
    )


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def _build_windows(
    episode_len: int, burn_in: int, target_len: int, stride: int,
) -> List[Tuple[int, int, int]]:
    """Build (context_start, target_start, target_end) windows for one episode."""
    windows: List[Tuple[int, int, int]] = []
    target_start = 0
    while target_start < episode_len:
        target_end = min(target_start + target_len, episode_len)
        context_start = max(0, target_start - burn_in)
        windows.append((context_start, target_start, target_end))
        if target_end >= episode_len:
            break
        target_start += stride
    return windows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HierarchicalILSequenceDataset(Dataset):
    """PyTorch Dataset yielding causal trajectory chunks.

    In stateful mode chunks are non-overlapping and every trajectory exposes a
    chronological stream of indices.  Burn-in masks only the beginning of an
    episode because subsequent chunks receive the preceding recurrent state.
    Mirror augmentation duplicates complete streams so a trajectory never
    changes coordinate handedness between adjacent chunks.
    """

    def __init__(
        self,
        episodes: List[EpisodeInfo],
        *,
        sequence_length: int = 16,
        burn_in: int = 8,
        window_stride: int = 16,
        target_height: int = 120,
        target_width: int = 160,
        stateful: bool = False,
        mirror_augmentation: bool = False,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {burn_in}")
        if window_stride <= 0:
            raise ValueError(f"window_stride must be > 0, got {window_stride}")
        if stateful and window_stride != sequence_length:
            raise ValueError(
                "stateful chunks must be contiguous and non-overlapping: "
                "window_stride must equal sequence_length"
            )

        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.window_stride = window_stride
        self.target_height = target_height
        self.target_width = target_width
        self.stateful = stateful
        self.mirror_augmentation = mirror_augmentation
        self._max_window_len = (
            sequence_length if stateful else burn_in + sequence_length
        )

        self._data: List[_EpisodeData] = []
        self._infos: List[EpisodeInfo] = []
        self._windows: List[Tuple[int, int, int, int, bool, bool]] = []
        self.stream_window_indices: List[List[int]] = []

        for ep in episodes:
            data = load_episode_csv(ep.csv_path, ep.episode_id)
            ep_idx = len(self._data)
            self._data.append(data)
            self._infos.append(ep)
            if stateful:
                base_windows = _build_windows(
                    ep.num_frames, 0, sequence_length, sequence_length,
                )
            else:
                base_windows = _build_windows(
                    ep.num_frames, burn_in, sequence_length, window_stride,
                )

            mirror_variants = (False, True) if mirror_augmentation else (False,)
            for mirrored in mirror_variants:
                stream_indices: List[int] = []
                for window_index, (
                    ctx_start, tgt_start, tgt_end,
                ) in enumerate(base_windows):
                    dataset_index = len(self._windows)
                    is_last = window_index == len(base_windows) - 1
                    self._windows.append(
                        (
                            ep_idx, ctx_start, tgt_start, tgt_end,
                            mirrored, is_last,
                        )
                    )
                    stream_indices.append(dataset_index)
                self.stream_window_indices.append(stream_indices)

        if not self._windows:
            raise RuntimeError(
                f"No windows from {len(episodes)} episodes. "
                f"Check sequence_length={sequence_length}, burn_in={burn_in}."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _load_depth(self, depth_dir: str, depth_file: str) -> torch.Tensor:
        img_path = str(Path(depth_dir) / depth_file)
        with Image.open(img_path) as img:
            if img.mode != "I;16":
                img = img.convert("I;16")
            arr = np.array(img, dtype=np.uint16)
        depth = torch.from_numpy(arr.astype(np.float32) / 65535.0).unsqueeze(0)
        if self.target_height and self.target_width:
            h, w = depth.shape[-2], depth.shape[-1]
            if h != self.target_height or w != self.target_width:
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0),
                    size=(self.target_height, self.target_width),
                    mode="area",
                ).squeeze(0).clamp(0.0, 1.0)
        return depth

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        (
            ep_idx, ctx_start, tgt_start, tgt_end, mirrored, is_last,
        ) = self._windows[idx]
        data = self._data[ep_idx]
        info = self._infos[ep_idx]
        indices = list(range(ctx_start, tgt_end))
        actual_len = len(indices)
        T = self._max_window_len
        H, W = self.target_height, self.target_width

        # Allocate.
        depth = torch.zeros(T, 1, H, W)
        raw_guide = torch.zeros(T, 4)
        gravity_flu = torch.zeros(T, 3)
        velocity_flu = torch.zeros(T, 3)
        yaw_rate = torch.zeros(T, 1)
        h_target = torch.zeros(T, dtype=torch.int64)
        v_target = torch.zeros(T, dtype=torch.int64)
        h_soft = torch.zeros(T, _H_SOFT_COUNT)
        v_soft = torch.zeros(T, _V_SOFT_COUNT)
        gv_target = torch.zeros(T, 1)
        cmd_target = torch.zeros(T, 4)
        seq_mask = torch.zeros(T, 1)
        tgt_mask = torch.zeros(T, 1)

        tgt_offset = tgt_start - ctx_start
        if self.stateful and tgt_start == 0:
            # Only the episode prefix is burn-in. Later chunks already receive
            # the exact recurrent state from their chronological predecessor.
            tgt_offset = min(self.burn_in, actual_len)

        for j, fi in enumerate(indices):
            depth[j] = self._load_depth(info.depth_dir, str(data.depth_files[fi]))
            raw_guide[j] = torch.tensor([
                data.global_dir_x[fi], data.global_dir_y[fi],
                data.global_dir_z[fi], data.global_dist_norm[fi],
            ])
            gravity_flu[j] = torch.tensor([data.grav_x[fi], data.grav_y[fi], data.grav_z[fi]])
            velocity_flu[j] = torch.tensor([data.vel_x[fi], data.vel_y[fi], data.vel_z[fi]])
            yaw_rate[j, 0] = float(data.yaw_rate[fi])
            h_target[j] = int(data.h_class[fi])
            v_target[j] = int(data.v_class[fi])
            h_soft[j] = torch.from_numpy(data.h_soft[fi].copy())
            v_soft[j] = torch.from_numpy(data.v_soft[fi].copy())
            gv_target[j, 0] = float(data.guide_val[fi])
            cmd_target[j] = torch.tensor([
                data.expert_vx[fi], data.expert_vy[fi],
                data.expert_vz[fi], data.expert_yaw[fi],
            ])
            seq_mask[j, 0] = 1.0
            if j >= tgt_offset:
                tgt_mask[j, 0] = 1.0

        if mirrored:
            # FLU horizontal reflection: image left/right and every signed
            # lateral/yaw quantity must change together. Horizontal class
            # semantics are symmetric around class 6, including recovery
            # classes 0 <-> 12.
            depth = torch.flip(depth, dims=(-1,))
            raw_guide[:, 1].mul_(-1.0)
            gravity_flu[:, 1].mul_(-1.0)
            velocity_flu[:, 1].mul_(-1.0)
            yaw_rate.mul_(-1.0)
            h_target = (_H_SOFT_COUNT - 1) - h_target
            h_soft = torch.flip(h_soft, dims=(-1,))
            cmd_target[:, 1].mul_(-1.0)
            cmd_target[:, 3].mul_(-1.0)

        trajectory_id = info.traj_dir + ("::mirror" if mirrored else "::original")

        return {
            "depth": depth, "raw_guide": raw_guide,
            "gravity_flu": gravity_flu, "velocity_flu": velocity_flu,
            "yaw_rate": yaw_rate,
            "horizontal_target": h_target, "vertical_target": v_target,
            "horizontal_soft_target": h_soft, "vertical_soft_target": v_soft,
            "guide_value_target": gv_target, "command_target": cmd_target,
            "sequence_mask": seq_mask, "target_mask": tgt_mask,
            "episode_id": data.episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": info.scene_id,
            "target_start": tgt_start,
            "is_last": is_last,
            "mirrored": mirrored,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_sequence_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    tensor_keys = [
        "depth", "raw_guide", "gravity_flu", "velocity_flu", "yaw_rate",
        "horizontal_target", "vertical_target",
        "horizontal_soft_target", "vertical_soft_target",
        "guide_value_target", "command_target",
        "sequence_mask", "target_mask",
    ]
    max_len = max(t["depth"].shape[0] for t in batch)
    result: Dict[str, Any] = {}
    for key in tensor_keys:
        tensors = [item[key] for item in batch]
        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = torch.zeros((max_len - t.shape[0],) + t.shape[1:], dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            else:
                t = t[:max_len]
            padded.append(t)
        result[key] = torch.stack(padded, dim=0)
    result["episode_id"] = [item["episode_id"] for item in batch]
    result["trajectory_id"] = [item["trajectory_id"] for item in batch]
    result["scene_id"] = [item["scene_id"] for item in batch]
    result["target_start"] = [item["target_start"] for item in batch]
    result["is_last"] = [bool(item["is_last"]) for item in batch]
    result["mirrored"] = [bool(item["mirrored"]) for item in batch]
    return result


# ---------------------------------------------------------------------------
# Stateful chronological batch sampling
# ---------------------------------------------------------------------------

class StatefulEpisodeBatchSampler(Sampler[List[int]]):
    """Batch complete trajectory streams while preserving chunk chronology.

    Streams are assigned to stable groups once. Within a group, chunk ``k+1``
    is yielded immediately after chunk ``k``, allowing the trainer to carry
    detached LSTM state without mixing trajectories or relying on shuffled
    window adjacency. Group order changes deterministically each epoch.
    """

    def __init__(
        self,
        dataset: HierarchicalILSequenceDataset,
        batch_size: int,
        seed: int,
    ) -> None:
        if not dataset.stateful:
            raise ValueError("StatefulEpisodeBatchSampler requires stateful dataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        stream_ids = list(range(len(dataset.stream_window_indices)))
        setup_rng = random.Random(self.seed)
        setup_rng.shuffle(stream_ids)
        self._groups = [
            stream_ids[start:start + self.batch_size]
            for start in range(0, len(stream_ids), self.batch_size)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        group_order = list(range(len(self._groups)))
        rng.shuffle(group_order)
        for group_index in group_order:
            group = list(self._groups[group_index])
            rng.shuffle(group)
            max_chunks = max(
                len(self.dataset.stream_window_indices[stream_id])
                for stream_id in group
            )
            for chunk_index in range(max_chunks):
                batch = [
                    self.dataset.stream_window_indices[stream_id][chunk_index]
                    for stream_id in group
                    if chunk_index < len(
                        self.dataset.stream_window_indices[stream_id]
                    )
                ]
                if batch:
                    yield batch

    def __len__(self) -> int:
        return sum(
            max(
                len(self.dataset.stream_window_indices[stream_id])
                for stream_id in group
            )
            for group in self._groups
        )


# ---------------------------------------------------------------------------
# Worker seed
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_root: Union[str, Path],
    *,
    batch_size: int = 4,
    sequence_length: int = 16,
    burn_in: int = 8,
    window_stride: int = 16,
    target_height: int = 120,
    target_width: int = 160,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    stateful_training: bool = True,
    mirror_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, List[EpisodeInfo], List[EpisodeInfo]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    all_eps = discover_committed_episodes(dataset_root)
    train_eps, val_eps = split_episodes(all_eps, val_ratio=val_ratio, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = HierarchicalILSequenceDataset(
        train_eps,
        sequence_length=sequence_length,
        burn_in=burn_in,
        window_stride=window_stride,
        target_height=target_height,
        target_width=target_width,
        stateful=stateful_training,
        mirror_augmentation=mirror_augmentation,
    )
    # Validation is streamed chronologically through each complete episode.
    # It intentionally has no repeated burn-in context: validate() carries the
    # recurrent states from one non-overlapping chunk into the next.
    val_ds = HierarchicalILSequenceDataset(
        val_eps,
        sequence_length=sequence_length,
        burn_in=0,
        window_stride=sequence_length,
        target_height=target_height,
        target_width=target_width,
        stateful=False,
        mirror_augmentation=False,
    )

    if stateful_training:
        stateful_batch_sampler = StatefulEpisodeBatchSampler(
            train_ds, batch_size=batch_size, seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=stateful_batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_sequence_batch,
            pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            generator=g,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_sequence_batch,
            pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            generator=g, drop_last=False,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        # Batch size one preserves episode order and makes recurrent-state
        # ownership unambiguous across chunks.
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_sequence_batch,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_eps, val_eps
