#!/usr/bin/env python3
"""Strict sequence loader for committed il_dataset schema-v25 episodes."""

import csv
import json
import math
import random
import warnings
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image


SCHEMA_VERSION = 25
# 30 Hz student inputs (all sourced from the NEW hierarchical expert: the
# effective target of the 5 Hz corrector + 30 Hz planner).
STATE_FIELDS = (
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "goal_direction_flu_x", "goal_direction_flu_y", "goal_direction_flu_z",
    "goal_distance_norm",
)
TARGET_FIELDS = (
    "target_velocity_flu_x", "target_velocity_flu_y",
    "target_velocity_flu_z", "target_yaw_rate",
)
# 5 Hz student input (the ORIGINAL navigation goal, never the effective
# goal_*); the 5 Hz supervision labels (two_level_expert_labels_v1).
STATE_FIELDS_5HZ = (
    "navigation_goal_direction_flu_x", "navigation_goal_direction_flu_y",
    "navigation_goal_direction_flu_z", "navigation_goal_distance_norm",
)
# R29s: pure-regression macro labels — corrected-target FLU direction +
# distance.  macro_correction_type is kept as a loss-weight column only
# (never a network input/output); token/param_valid stay in the CSV for
# audit/diagnostics but are no longer training labels.
LABEL_FIELDS_5HZ = (
    "macro_update_mask", "macro_label_valid", "macro_correction_type",
    "macro_direction_flu_x", "macro_direction_flu_y",
    "macro_direction_flu_z", "macro_distance_norm",
)
REQUIRED_FIELDS = set(("episode_frame_index", "frame_valid", "depth_file",
                       "hierarchical_mode", "planner_status") +
                      STATE_FIELDS + TARGET_FIELDS + STATE_FIELDS_5HZ +
                      LABEL_FIELDS_5HZ)
STATE_SCALE = np.asarray(
    # gravity_flu is the unit gravity direction written by il_manager,
    # not acceleration in m/s^2.  7-D: velocity/yaw_rate inputs removed so
    # the policy cannot short-circuit on current motion (must read depth).
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32)
# The 5 Hz upper-planner state is self state plus the ORIGINAL navigation
# goal.  It must not use goal_direction_flu_* (the effective 30 Hz target),
# otherwise the upper network would learn the expert's already-corrected
# answer instead of deciding whether correction is needed.
MACRO_STATE_FIELDS = (
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
    "yaw_rate_flu",
    "navigation_goal_direction_flu_x",
    "navigation_goal_direction_flu_y", "navigation_goal_direction_flu_z",
    "navigation_goal_distance_norm",
)
MACRO_STATE_SCALE = np.asarray(
    [1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 1.5, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32)
COMMAND_SCALE = np.asarray([2.5, 2.5, 2.5, 1.5], dtype=np.float32)
# The FULL new-architecture expert state (no lossy legacy projection).
HIERARCHICAL_MODES = (
    "direct", "local_avoidance", "macro_normal", "macro_turn_left",
    "macro_turn_right", "turn_to_target", "goal_capture", "blocked",
)
HIERARCHICAL_MODE_TO_INDEX = {
    name: index for index, name in enumerate(HIERARCHICAL_MODES)
}
# 5 Hz directive classes (macro_correction_type).
MACRO_TYPES = ("PASS_THROUGH", "NORMAL_CORRECTION", "TURN_LEFT", "TURN_RIGHT")
MACRO_TYPE_TO_INDEX = {name: index for index, name in enumerate(MACRO_TYPES)}

# planner_status: the C++ expert status STRING written by il_manager
# (PlannerStatus enum names in types.hpp).  The loader maps the CSV string
# to a stable index and RAISES (with file / line / raw value) on any
# unknown value — never a silent default index (item 一).
PLANNER_STATUSES = (
    "SAFE_PROGRESSING", "SAFE_HOLD", "TERMINAL_SETTLING", "TURNING",
    "EMERGENCY_BRAKE", "BLOCKED_BY_OBSERVED_OBSTACLE", "NO_SAFE_CANDIDATE",
    "STALLED_WITHOUT_PROGRESS", "NO_TARGET",
)
PLANNER_STATUS_TO_INDEX = {
    name: index for index, name in enumerate(PLANNER_STATUSES)
}

# ── 5 Hz direction-token contract (SHARED with the C++ expert) ──────
# The expert's target_encoding.direction_bin_count is 11: token 0 =
# TURN_LEFT, tokens 1..11 = ordinary in-FOV bins left→right (0° sits in
# the middle token 6), token 12 = TURN_RIGHT, PASS token = -1.  These
# constants are the single source for the loader's token validation and
# mirroring; they must stay in sync with the expert contract.
DIRECTION_BIN_COUNT = 11
TOKEN_TURN_LEFT = 0
TOKEN_TURN_RIGHT = 2 * (DIRECTION_BIN_COUNT // 2) + 2  # 12
ORDINARY_TOKEN_MIN = 1
ORDINARY_TOKEN_MAX = TOKEN_TURN_RIGHT - 1              # 11
CENTER_TOKEN = TOKEN_TURN_RIGHT // 2                   # 6


def mirror_direction_token(token: int) -> int:
    """Mirror a direction token over the FULL token range: 0<->12,
    1<->11, ..., 6 stays; PASS (-1) stays -1."""
    if token < 0:
        return token
    return TOKEN_TURN_RIGHT - token


# hierarchical_mode values that count as active avoidance for weighting.
AVOIDANCE_MODES = frozenset(
    ("local_avoidance", "macro_normal", "macro_turn_left", "macro_turn_right",
     "turn_to_target"))


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    csv_path: Path
    episode_id: str
    scene_id: str
    task_id: str
    rows: int
    behavior_class: str
    has_avoidance: bool
    avoidance_frames: int
    macro_frames: int


@dataclass(frozen=True)
class WindowInfo:
    episode_index: int
    target_start: int
    target_length: int
    actual_start: int
    actual_end: int
    left_pad: int


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _read_metadata(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _audit_5hz_row(csv_path: Path, line: int, frame_index: int,
                   row: Dict[str, str]) -> None:
    """Static legality audit of ONE real 5 Hz decision row (item 三).

    Runs only on macro_update_mask==1 rows.  Any violation raises with the
    exact file / line / raw value; a committed episode is never silently
    accepted.  Non-5 Hz rows may carry ZOH diagnostics, but the 5 Hz student
    sampler selects ONLY mask==1 rows.
    """
    def fail(msg: str) -> None:
        raise ValueError("5 Hz audit %s at %s:%d" % (msg, csv_path, line))

    # mask==1 must sit on the 5 Hz grid (item 三).
    if frame_index % 6 != 0:
        fail("macro_update_mask==1 on non-5Hz index %d" % frame_index)
    if str(row.get("macro_label_valid", "0")).strip() not in ("1",):
        fail("macro_label_valid != 1 on a real 5 Hz frame (got %r)"
             % row.get("macro_label_valid"))
    ctype = row.get("macro_correction_type", "")
    if ctype not in MACRO_TYPE_TO_INDEX:
        fail("unknown macro_correction_type %r" % ctype)
    token_raw = row.get("macro_direction_token", -1)
    param_raw = row.get("macro_param_valid", "0")
    dist_raw = row.get("macro_distance_norm", "0.0")
    try:
        token = int(str(token_raw).strip())
        param = int(str(param_raw).strip())
        dist = float(dist_raw)
        dx = float(row.get("macro_direction_flu_x", "0.0"))
        dy = float(row.get("macro_direction_flu_y", "0.0"))
        dz = float(row.get("macro_direction_flu_z", "0.0"))
    except (TypeError, ValueError):
        fail("non-numeric 5 Hz label field (token=%r param=%r dist=%r)"
             % (token_raw, param_raw, dist_raw))
    # 5 Hz inputs (the ORIGINAL navigation goal) and all labels: finite.
    s5 = [float(row.get(name, "0.0")) for name in STATE_FIELDS_5HZ]
    for v in [dist, dx, dy, dz] + s5:
        if not math.isfinite(float(v)):
            fail("non-finite 5 Hz input/label value %r" % v)
    if param not in (0, 1):
        fail("macro_param_valid must be 0 or 1 (got %r)" % param_raw)

    if ctype == "PASS_THROUGH":
        if token != -1:
            fail("PASS token must be -1 (got %d)" % token)
        if param != 0:
            fail("PASS macro_param_valid must be 0 (got %d)" % param)
    elif ctype == "NORMAL_CORRECTION":
        if not (ORDINARY_TOKEN_MIN <= token <= ORDINARY_TOKEN_MAX):
            fail("NORMAL token must be in [%d, %d] (got %d)"
                 % (ORDINARY_TOKEN_MIN, ORDINARY_TOKEN_MAX, token))
        if param != 1:
            fail("NORMAL macro_param_valid must be 1 (got %d)" % param)
        if not (dist < 1.0 - 1e-6):
            fail("NORMAL distance_norm must be < 1 (got %r)" % dist_raw)
        _check_unit_direction(dx, dy, dz, fail)
    elif ctype == "TURN_LEFT":
        if token != TOKEN_TURN_LEFT:
            fail("TURN_LEFT token must be %d (got %d)" % (TOKEN_TURN_LEFT,
                                                          token))
        if param != 1:
            fail("TURN_LEFT macro_param_valid must be 1 (got %d)" % param)
        if abs(dist - 1.0) > 1e-6:
            fail("TURN_LEFT distance_norm must be EXACTLY 1 (got %r)"
                 % dist_raw)
        _check_unit_direction(dx, dy, dz, fail)
    elif ctype == "TURN_RIGHT":
        if token != TOKEN_TURN_RIGHT:
            fail("TURN_RIGHT token must be %d (got %d)" % (TOKEN_TURN_RIGHT,
                                                           token))
        if param != 1:
            fail("TURN_RIGHT macro_param_valid must be 1 (got %d)" % param)
        if abs(dist - 1.0) > 1e-6:
            fail("TURN_RIGHT distance_norm must be EXACTLY 1 (got %r)"
                 % dist_raw)
        _check_unit_direction(dx, dy, dz, fail)


def _check_unit_direction(dx: float, dy: float, dz: float,
                          fail) -> None:
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if abs(norm - 1.0) > 1e-3:
        fail("NORMAL/TURN direction must be a unit vector (norm=%g)" % norm)


def discover_committed_episodes(
        dataset_root: Union[str, Path], verify_depth: bool = True) -> List[EpisodeInfo]:
    """Discover only atomic, successful v25 episodes; reject silent corruption."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("dataset root does not exist: %s" % root)
    episodes: List[EpisodeInfo] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        episode_dir = metadata_path.parent
        # Strict exclusion: the atomic-commit staging roots must NEVER enter
        # the training set.  (The one-level glob already skips the two-level
        # _failed/<ep>/ and _inprogress/<ep>.inprogress/ layout, but the
        # guard stays explicit and defensive.)
        if episode_dir.parent.name in ("_failed", "_inprogress") or \
                episode_dir.name.endswith(".inprogress"):
            continue
        metadata = _read_metadata(metadata_path)
        if metadata.get("status") != "committed" or \
                not bool(metadata.get("quality_committed")) or \
                not bool(metadata.get("reached_goal")):
            continue
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("%s is committed but schema_version != 25" % episode_dir)
        csv_path = episode_dir / "data.csv"
        if not csv_path.is_file():
            raise ValueError("committed episode has no data.csv: %s" % episode_dir)
        row_count = 0
        has_avoidance = False
        avoidance_frames = 0
        macro_frames = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError("%s missing v25 fields: %s" %
                                 (csv_path, sorted(missing)))
            for expected_index, row in enumerate(reader):
                line = expected_index + 2  # 1-based CSV row (header = 1)
                if not _truthy(row["frame_valid"]):
                    raise ValueError("committed episode contains invalid frame: %s:%d" %
                                     (csv_path, line))
                # Strict commit contract: only rows with episode_valid==1
                # belong to a committed episode.
                if not _truthy(row.get("episode_valid", "0")):
                    raise ValueError(
                        "committed episode contains episode_valid!=1 row: "
                        "%s:%d" % (csv_path, line))
                if int(row["episode_frame_index"]) != expected_index:
                    raise ValueError("non-contiguous episode_frame_index: %s" % csv_path)
                values = [float(row[name]) for name in STATE_FIELDS + TARGET_FIELDS]
                if not np.isfinite(values).all():
                    raise ValueError("non-finite student input/target: %s:%d" %
                                     (csv_path, line))
                # ── item 一: planner_status must map to a known C++ status ──
                ps_raw = row.get("planner_status", "")
                if ps_raw not in PLANNER_STATUS_TO_INDEX:
                    raise ValueError(
                        "unknown planner_status %r at %s:%d" %
                        (ps_raw, csv_path, line))
                # ── item 一: hierarchical_mode must be a KNOWN mode (never
                #    silently accepted via get(..., -1)). ──────────────
                mode = row.get("hierarchical_mode", "direct")
                if mode not in HIERARCHICAL_MODE_TO_INDEX:
                    raise ValueError(
                        "unknown hierarchical_mode %r at %s:%d" %
                        (mode, csv_path, line))
                if mode in AVOIDANCE_MODES:
                    has_avoidance = True
                    avoidance_frames += 1
                # ── item 三: 5 Hz field legality audit ────────────────
                mask_raw = row.get("macro_update_mask", "0")
                if str(mask_raw).strip() not in ("0", "1"):
                    raise ValueError(
                        "macro_update_mask must be 0 or 1 at %s:%d (got %r)"
                        % (csv_path, line, mask_raw))
                mask = int(str(mask_raw).strip())
                if mask:
                    _audit_5hz_row(csv_path, line, expected_index, row)
                    macro_frames += 1
                depth_path = episode_dir / row["depth_file"]
                if verify_depth and not depth_path.is_file():
                    raise ValueError("missing depth frame: %s" % depth_path)
                row_count += 1
        declared = int(metadata.get("rows_written", row_count))
        if row_count == 0 or row_count != declared:
            raise ValueError("row-count mismatch in %s: csv=%d metadata=%d" %
                             (episode_dir, row_count, declared))
        episodes.append(EpisodeInfo(
            path=episode_dir, csv_path=csv_path,
            episode_id=str(metadata.get("episode_id", episode_dir.name)),
            scene_id=str(metadata.get("scene_id", "unknown_scene")),
            task_id=str(metadata.get("task_id", "unknown_task")), rows=row_count,
            behavior_class=str(metadata.get(
                "blueprint_behavior_class", "unknown")),
            has_avoidance=has_avoidance, avoidance_frames=avoidance_frames,
            macro_frames=macro_frames))
    if not episodes:
        raise ValueError("no committed schema-v25 episodes found under %s" % root)
    return episodes


def split_episodes(episodes: Sequence[EpisodeInfo], val_fraction: float = 0.2,
                   seed: int = 1337) -> Tuple[List[EpisodeInfo], List[EpisodeInfo]]:
    """Split by scene to prevent geometry leakage; episode fallback is explicit."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    by_scene: Dict[str, List[EpisodeInfo]] = {}
    for episode in episodes:
        by_scene.setdefault(episode.scene_id, []).append(episode)
    if len(by_scene) >= 2:
        scene_ids = sorted(by_scene)
        rng = random.Random(seed)
        rng.shuffle(scene_ids)
        val_count = max(1, min(len(scene_ids) - 1,
                               int(round(len(scene_ids) * val_fraction))))
        val_scenes = set(scene_ids[:val_count])
        train = [ep for ep in episodes if ep.scene_id not in val_scenes]
        val = [ep for ep in episodes if ep.scene_id in val_scenes]
    else:
        warnings.warn(
            "only one scene is available; falling back to episode-level split. "
            "Validation is not a geometry-generalization estimate.", RuntimeWarning)
        shuffled = list(episodes)
        random.Random(seed).shuffle(shuffled)
        val_count = max(1, min(len(shuffled) - 1,
                               int(round(len(shuffled) * val_fraction))))
        val, train = shuffled[:val_count], shuffled[val_count:]
    if not train or not val:
        raise ValueError("at least two committed episodes are required")
    return sorted(train, key=lambda ep: ep.episode_id), \
        sorted(val, key=lambda ep: ep.episode_id)


class V25SequenceDataset(Dataset):
    def __init__(self, episodes: Sequence[EpisodeInfo], sequence_length: int = 32,
                 burn_in: int = 8, stride: Optional[int] = None,
                 max_depth_m: float = 5.0, augment: bool = False,
                 cache_episodes: int = 8, stateful: bool = False):
        self.episodes = list(episodes)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.stateful = bool(stateful)
        # Stateful TBPTT has no overlapping context: the previous chunk's
        # detached LSTM state is its context.  Chunks must consequently be
        # adjacent, fixed-length pieces of an episode.
        self.stride = self.sequence_length if self.stateful else int(
            stride or sequence_length)
        self.context_burn_in = 0 if self.stateful else self.burn_in
        self.total_length = self.context_burn_in + self.sequence_length
        self.max_depth_m = float(max_depth_m)
        self.augment = bool(augment)
        self.cache_episodes = max(1, int(cache_episodes))
        if self.sequence_length <= 0 or self.burn_in < 0 or self.stride <= 0:
            raise ValueError("invalid sequence_length/burn_in/stride")
        self.windows: List[WindowInfo] = []
        for episode_index, episode in enumerate(self.episodes):
            for target_start in range(0, episode.rows, self.stride):
                target_length = min(self.sequence_length,
                                    episode.rows - target_start)
                actual_start = max(0, target_start - self.context_burn_in)
                left_pad = self.context_burn_in - (target_start - actual_start)
                actual_end = target_start + target_length
                self.windows.append(WindowInfo(
                    episode_index, target_start, target_length,
                    actual_start, actual_end, left_pad))
        self._cache: OrderedDict[int, List[Dict[str, str]]] = OrderedDict()
        self._augmentation_epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        """Keep all chunks of one stateful episode in one flip frame."""
        self._augmentation_epoch = int(epoch)

    def _rows(self, episode_index: int) -> List[Dict[str, str]]:
        if episode_index in self._cache:
            rows = self._cache.pop(episode_index)
            self._cache[episode_index] = rows
            return rows
        with self.episodes[episode_index].csv_path.open(
                "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self._cache[episode_index] = rows
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return rows

    def _read_depth(self, path: Path) -> np.ndarray:
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        else:  # pragma: no cover
            image = np.asarray(Image.open(str(path)))
        if image is None or image.ndim != 2:
            raise ValueError("expected one-channel depth PNG: %s" % path)
        # depth_encoding_contract: NORMALIZED uint16 encoding — the valid
        # range [0, max_depth_m] maps onto [0, 65535], so
        #   depth_m = pixel / 65535 * max_depth_m.
        # Pixel 0 is the INVALID marker (non-finite / <=0 depth = no
        # return); it is masked to max range — NEVER treated as a real
        # 0-metre obstacle.  Valid depths are clipped to [0, max_depth_m]
        # and normalized.
        depth_m = image.astype(np.float32) / 65535.0 * float(self.max_depth_m)
        depth_m[image == 0] = float(self.max_depth_m)
        return np.clip(depth_m, 0.0, self.max_depth_m) / self.max_depth_m

    def __getitem__(self, index: int) -> Dict[str, object]:
        window = self.windows[index]
        episode = self.episodes[window.episode_index]
        rows = self._rows(window.episode_index)
        selected = rows[window.actual_start:window.actual_end]
        if not selected:
            raise RuntimeError("empty sequence window")
        first_depth = self._read_depth(episode.path / selected[0]["depth_file"])
        h, w = first_depth.shape
        depth = np.zeros((self.total_length, 1, h, w), dtype=np.float32)
        state = np.zeros((self.total_length, len(STATE_FIELDS)), dtype=np.float32)
        target = np.zeros((self.total_length, len(TARGET_FIELDS)), dtype=np.float32)
        valid_mask = np.zeros(self.total_length, dtype=np.float32)
        loss_mask = np.zeros(self.total_length, dtype=np.float32)
        mode = np.full(self.total_length, -1, dtype=np.int64)
        planner_status = np.full(self.total_length, -1, dtype=np.int64)
        frame_index = np.full(self.total_length, -1, dtype=np.int64)
        # 5 Hz fields (two_level_expert_labels_v1): ZOH values are present
        # on every row; macro_update_mask marks the real 5 Hz decisions.
        state_5hz = np.zeros((self.total_length, len(STATE_FIELDS_5HZ)),
                             dtype=np.float32)
        label_5hz = np.zeros((self.total_length, len(LABEL_FIELDS_5HZ)),
                             dtype=np.float32)
        for local_index, row in enumerate(selected):
            out_index = window.left_pad + local_index
            line = window.actual_start + local_index + 2
            depth[out_index, 0] = self._read_depth(episode.path / row["depth_file"])
            state[out_index] = (
                np.asarray([float(row[name]) for name in STATE_FIELDS],
                           dtype=np.float32) / STATE_SCALE)
            target[out_index] = (
                np.asarray([float(row[name]) for name in TARGET_FIELDS],
                           dtype=np.float32) / COMMAND_SCALE)
            valid_mask[out_index] = 1.0
            # item 一: hierarchical_mode MUST be a known mode (raise with
            # file/line/raw; never a silent get(..., -1)).
            mode_raw = row.get("hierarchical_mode", "direct")
            mode_idx = HIERARCHICAL_MODE_TO_INDEX.get(mode_raw)
            if mode_idx is None:
                raise ValueError(
                    "unknown hierarchical_mode %r at %s:%d" %
                    (mode_raw, self.episodes[window.episode_index].csv_path,
                     line))
            mode[out_index] = mode_idx
            # item 一: planner_status CSV string -> stable index (raise on
            # unknown value; never int(row) on a string or a default -1).
            ps_raw = row.get("planner_status", "")
            ps_idx = PLANNER_STATUS_TO_INDEX.get(ps_raw)
            if ps_idx is None:
                raise ValueError(
                    "unknown planner_status %r at %s:%d" %
                    (ps_raw, self.episodes[window.episode_index].csv_path,
                     line))
            planner_status[out_index] = ps_idx
            frame_index[out_index] = int(row["episode_frame_index"])
            state_5hz[out_index] = np.asarray(
                [float(row[name]) for name in STATE_FIELDS_5HZ],
                dtype=np.float32)
            # macro_correction_type is encoded to an index; the rest are raw.
            # Columns: 0=mask, 1=label_valid, 2=type (loss weight only),
            # 3..5=corrected direction FLU, 6=distance norm (R29s).
            label_5hz[out_index, 0] = float(_truthy(
                row.get("macro_update_mask", "0")))
            label_5hz[out_index, 1] = float(_truthy(
                row.get("macro_label_valid", "0")))
            label_5hz[out_index, 2] = MACRO_TYPE_TO_INDEX.get(
                row.get("macro_correction_type", "PASS_THROUGH"), -1)
            label_5hz[out_index, 3] = float(
                row.get("macro_direction_flu_x", 0.0))
            label_5hz[out_index, 4] = float(
                row.get("macro_direction_flu_y", 0.0))
            label_5hz[out_index, 5] = float(
                row.get("macro_direction_flu_z", 0.0))
            label_5hz[out_index, 6] = float(
                row.get("macro_distance_norm", 0.0))
        loss_begin = self.context_burn_in
        loss_mask[loss_begin:loss_begin + window.target_length] = 1.0
        loss_mask *= valid_mask
        if self.stateful:
            # Randomly flipping each chunk independently would make the hidden
            # state from a mirrored chunk inconsistent with the next chunk.
            flip_key = (episode.episode_id + "|" +
                        str(self._augmentation_epoch)).encode("utf-8")
            should_flip = bool(zlib.crc32(flip_key) & 1)
        else:
            should_flip = random.random() < 0.5
        if self.augment and should_flip:
            depth = depth[:, :, :, ::-1].copy()
            state[:, [1, 4, 6, 8]] *= -1.0
            target[:, [1, 3]] *= -1.0
            # ── 5 Hz mirror (item 二) ────────────────────────────────
            # STATE_FIELDS_5HZ = (nav_goal_dir_x, nav_goal_dir_y,
            # nav_goal_dir_z, nav_goal_dist_norm): ONLY the LEFT axis
            # (index 1) mirrors; x/z are never negated.
            state_5hz[:, 1] *= -1.0
            # LABEL_FIELDS_5HZ = (mask, label_valid, correction_type,
            # dir_x, dir_y, dir_z, distance_norm): ONLY
            # macro_direction_flu_y (index 4) mirrors.
            label_5hz[:, 4] *= -1.0
            # TURN_LEFT <-> TURN_RIGHT categories (index 2) swap.
            turn_left_idx = float(MACRO_TYPE_TO_INDEX["TURN_LEFT"])
            turn_right_idx = float(MACRO_TYPE_TO_INDEX["TURN_RIGHT"])
            is_left = label_5hz[:, 2] == turn_left_idx
            is_right = label_5hz[:, 2] == turn_right_idx
            label_5hz[is_left, 2] = turn_right_idx
            label_5hz[is_right, 2] = turn_left_idx
            # Direction token mirrors over the FULL range (0<->12, 1<->11,
            # ..., 6 stays); PASS (-1) stays -1.  x and z are unchanged.
            tokens = label_5hz[:, 3]
            valid_tokens = tokens >= 0
            label_5hz[valid_tokens, 3] = (
                TOKEN_TURN_RIGHT - tokens[valid_tokens])
            # hierarchical_mode: macro_turn_left <-> macro_turn_right swap.
            mleft = mode == HIERARCHICAL_MODE_TO_INDEX["macro_turn_left"]
            mright = mode == HIERARCHICAL_MODE_TO_INDEX["macro_turn_right"]
            mode[mleft] = HIERARCHICAL_MODE_TO_INDEX["macro_turn_right"]
            mode[mright] = HIERARCHICAL_MODE_TO_INDEX["macro_turn_left"]
        return {
            "depth": torch.from_numpy(depth),
            "state": torch.from_numpy(state),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "hierarchical_mode": torch.from_numpy(mode),
            "planner_status": torch.from_numpy(planner_status),
            "frame_index": torch.from_numpy(frame_index),
            "state_5hz": torch.from_numpy(state_5hz),
            "label_5hz": torch.from_numpy(label_5hz),
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "episode_index": torch.tensor(window.episode_index,
                                          dtype=torch.long),
            "is_first_chunk": torch.tensor(window.target_start == 0,
                                           dtype=torch.bool),
        }

    def sampling_weights(self, rare_weight: float = 2.0) -> torch.Tensor:
        values = []
        for window in self.windows:
            episode = self.episodes[window.episode_index]
            weight = 1.0
            if episode.has_avoidance:
                weight *= rare_weight
            if episode.macro_frames > 0:
                # 5 Hz correction episodes carry the rarer decision labels;
                # cap their influence so they do not erase direct coverage.
                weight *= 1.0 + min(1.0, episode.macro_frames / 60.0)
            values.append(weight)
        return torch.as_tensor(values, dtype=torch.double)


class Macro5HzSequenceDataset(Dataset):
    """Dataset of real 5 Hz decision rows only.

    The CSV stores the macro directive zero-order-held on all six 30 Hz rows.
    This dataset filters by ``macro_update_mask==1`` before making windows,
    so a macro decision is never counted six times.  Windows remain
    episode-local and can be streamed with ``StatefulEpisodeBatchSampler``.
    """

    def __init__(self, episodes: Sequence[EpisodeInfo],
                 sequence_length: int = 16, burn_in: int = 4,
                 stride: Optional[int] = None, max_depth_m: float = 5.0,
                 augment: bool = False, cache_episodes: int = 8,
                 stateful: bool = False):
        self.episodes = list(episodes)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.stateful = bool(stateful)
        self.stride = self.sequence_length if self.stateful else int(
            stride or sequence_length)
        self.context_burn_in = 0 if self.stateful else self.burn_in
        self.total_length = self.context_burn_in + self.sequence_length
        self.max_depth_m = float(max_depth_m)
        self.augment = bool(augment)
        self.cache_episodes = max(1, int(cache_episodes))
        if self.sequence_length <= 0 or self.burn_in < 0 or self.stride <= 0:
            raise ValueError("invalid macro sequence_length/burn_in/stride")
        self.windows: List[WindowInfo] = []
        self._cache: OrderedDict[int, List[Dict[str, str]]] = OrderedDict()
        self._augmentation_epoch = 0
        # Read only the CSV text here.  Depth remains lazy and is decoded only
        # for a sampled window, just like the 30 Hz loader.
        for episode_index, episode in enumerate(self.episodes):
            macro_rows = self._rows(episode_index)
            count = len(macro_rows)
            for target_start in range(0, count, self.stride):
                target_length = min(self.sequence_length, count - target_start)
                actual_start = max(0, target_start - self.context_burn_in)
                left_pad = self.context_burn_in - (target_start - actual_start)
                self.windows.append(WindowInfo(
                    episode_index, target_start, target_length,
                    actual_start, target_start + target_length, left_pad))
        if not self.windows:
            raise ValueError("no macro_update_mask==1 rows found")

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        self._augmentation_epoch = int(epoch)

    def _rows(self, episode_index: int) -> List[Dict[str, str]]:
        if episode_index in self._cache:
            rows = self._cache.pop(episode_index)
            self._cache[episode_index] = rows
            return rows
        with self.episodes[episode_index].csv_path.open(
                "r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if _truthy(row.get("macro_update_mask", "0"))]
        if not rows:
            raise ValueError("episode has no macro decision rows: %s" %
                             self.episodes[episode_index].episode_id)
        self._cache[episode_index] = rows
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return rows

    def _read_depth(self, path: Path) -> np.ndarray:
        # Reuse the exact v25 uint16 decoding and invalid-pixel semantics.
        return V25SequenceDataset._read_depth(self, path)

    def __getitem__(self, index: int) -> Dict[str, object]:
        window = self.windows[index]
        episode = self.episodes[window.episode_index]
        rows = self._rows(window.episode_index)
        selected = rows[window.actual_start:window.actual_end]
        if not selected:
            raise RuntimeError("empty macro sequence window")
        first_depth = self._read_depth(episode.path / selected[0]["depth_file"])
        h, w = first_depth.shape
        depth = np.zeros((self.total_length, 1, h, w), dtype=np.float32)
        state = np.zeros((self.total_length, len(MACRO_STATE_FIELDS)),
                         dtype=np.float32)
        macro_type = np.full(self.total_length, -1, dtype=np.int64)
        macro_direction = np.zeros((self.total_length, 3), dtype=np.float32)
        macro_distance = np.zeros((self.total_length, 1), dtype=np.float32)
        frame_index = np.full(self.total_length, -1, dtype=np.int64)
        valid_mask = np.zeros(self.total_length, dtype=np.float32)
        loss_mask = np.zeros(self.total_length, dtype=np.float32)
        for local_index, row in enumerate(selected):
            out_index = window.left_pad + local_index
            depth[out_index, 0] = self._read_depth(
                episode.path / row["depth_file"])
            state[out_index] = np.asarray(
                [float(row[name]) for name in MACRO_STATE_FIELDS],
                dtype=np.float32) / MACRO_STATE_SCALE
            ctype = row.get("macro_correction_type", "PASS_THROUGH")
            if ctype not in MACRO_TYPE_TO_INDEX:
                raise ValueError("unknown macro type %r in %s" %
                                 (ctype, episode.csv_path))
            macro_type[out_index] = MACRO_TYPE_TO_INDEX[ctype]
            macro_direction[out_index] = np.asarray([
                float(row.get("macro_direction_flu_x", 0.0)),
                float(row.get("macro_direction_flu_y", 0.0)),
                float(row.get("macro_direction_flu_z", 0.0))],
                dtype=np.float32)
            macro_distance[out_index, 0] = float(
                row.get("macro_distance_norm", 0.0))
            frame_index[out_index] = int(row["episode_frame_index"])
            valid_mask[out_index] = 1.0
        loss_begin = self.context_burn_in
        loss_mask[loss_begin:loss_begin + window.target_length] = 1.0
        loss_mask *= valid_mask
        if self.stateful:
            flip_key = (episode.episode_id + "|" +
                        str(self._augmentation_epoch)).encode("utf-8")
            should_flip = bool(zlib.crc32(flip_key) & 1)
        else:
            should_flip = random.random() < 0.5
        if self.augment and should_flip:
            depth = depth[:, :, :, ::-1].copy()
            # gravity-y, velocity-y and original-goal-y are the lateral axes.
            state[:, [1, 4, 8]] *= -1.0
            macro_direction[:, 1] *= -1.0
            left_idx = MACRO_TYPE_TO_INDEX["TURN_LEFT"]
            right_idx = MACRO_TYPE_TO_INDEX["TURN_RIGHT"]
            is_left = macro_type == left_idx
            is_right = macro_type == right_idx
            macro_type[is_left] = right_idx
            macro_type[is_right] = left_idx
        return {
            "depth": torch.from_numpy(depth),
            "state": torch.from_numpy(state),
            "macro_type": torch.from_numpy(macro_type),
            "macro_direction": torch.from_numpy(macro_direction),
            "macro_distance": torch.from_numpy(macro_distance),
            "valid_mask": torch.from_numpy(valid_mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "frame_index": torch.from_numpy(frame_index),
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "episode_index": torch.tensor(window.episode_index,
                                          dtype=torch.long),
            "is_first_chunk": torch.tensor(window.target_start == 0,
                                           dtype=torch.bool),
        }

    def sampling_weights(self, rare_weight: float = 2.0) -> torch.Tensor:
        values = []
        for window in self.windows:
            episode = self.episodes[window.episode_index]
            # Macro decisions are already sparse; retain a modest emphasis for
            # episodes with actual corrections without erasing PASS coverage.
            weight = float(rare_weight) if episode.has_avoidance else 1.0
            values.append(weight)
        return torch.as_tensor(values, dtype=torch.double)


class StatefulEpisodeBatchSampler(Sampler[List[int]]):
    """Batch sequential chunks without ever interleaving one episode stream.

    An emitted batch contains the next chunk for each currently active episode.
    Therefore column ``i`` in consecutive batches belongs to the same episode
    until that episode ends; a replacement episode is inserted only on a later
    batch, where ``is_first_chunk`` clears its state.
    """
    def __init__(self, dataset: V25SequenceDataset, batch_size: int,
                 shuffle: bool, seed: int = 1337):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._by_episode: Dict[int, List[int]] = {}
        for index, window in enumerate(dataset.windows):
            self._by_episode.setdefault(window.episode_index, []).append(index)
        for indices in self._by_episode.values():
            indices.sort(key=lambda index: dataset.windows[index].target_start)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        groups = [list(self._by_episode[key]) for key in sorted(self._by_episode)]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        next_group = 0
        active: List[Tuple[List[int], int]] = []
        while next_group < len(groups) and len(active) < self.batch_size:
            active.append((groups[next_group], 0))
            next_group += 1
        while active:
            yield [group[position] for group, position in active]
            continuing: List[Tuple[List[int], int]] = []
            for group, position in active:
                if position + 1 < len(group):
                    continuing.append((group, position + 1))
            active = continuing
            while next_group < len(groups) and len(active) < self.batch_size:
                active.append((groups[next_group], 0))
                next_group += 1

    def __len__(self) -> int:
        remaining = [len(self._by_episode[key]) for key in sorted(self._by_episode)]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(remaining)
        count = 0
        while remaining:
            active = remaining[:self.batch_size]
            remaining = remaining[self.batch_size:]
            while active:
                count += 1
                active = [length - 1 for length in active if length > 1]
                while remaining and len(active) < self.batch_size:
                    active.append(remaining.pop(0))
        return count


def build_dataloaders(dataset_root: Union[str, Path], batch_size: int = 8,
                      sequence_length: int = 32, burn_in: int = 8,
                      stride: Optional[int] = None, val_fraction: float = 0.2,
                      seed: int = 1337, workers: int = 0,
                      balanced_sampling: bool = True,
                      verify_depth: bool = True, stateful: bool = True,
                      mirror_augmentation: bool = False):
    episodes = discover_committed_episodes(dataset_root, verify_depth=verify_depth)
    train_eps, val_eps = split_episodes(episodes, val_fraction, seed)
    train_ds = V25SequenceDataset(
        train_eps, sequence_length, burn_in, stride,
        augment=mirror_augmentation,
        stateful=stateful)
    val_ds = V25SequenceDataset(
        val_eps, sequence_length, burn_in, stride, augment=False,
        stateful=stateful)
    if stateful:
        if balanced_sampling:
            warnings.warn(
                "stateful TBPTT preserves episode order, so weighted window "
                "sampling is disabled; rare expert modes remain loss-weighted.",
                RuntimeWarning)
        common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(
            train_ds, batch_sampler=StatefulEpisodeBatchSampler(
                train_ds, batch_size, shuffle=True, seed=seed), **common)
        val_loader = DataLoader(
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
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_eps, val_eps


def build_macro_dataloaders(dataset_root: Union[str, Path], batch_size: int = 8,
                            sequence_length: int = 16, burn_in: int = 4,
                            stride: Optional[int] = None,
                            val_fraction: float = 0.2, seed: int = 1337,
                            workers: int = 0,
                            balanced_sampling: bool = True,
                            verify_depth: bool = True, stateful: bool = True,
                            mirror_augmentation: bool = False):
    """Build leakage-free 5 Hz loaders from real macro decision rows.

    ``discover_committed_episodes`` performs the complete v25 legality audit;
    this function then samples only the rows marked by
    ``macro_update_mask==1``.  Scene splitting is shared with the 30 Hz
    trainer, so the two networks can use identical train/validation episodes.
    """
    episodes = discover_committed_episodes(dataset_root,
                                            verify_depth=verify_depth)
    train_eps, val_eps = split_episodes(episodes, val_fraction, seed)
    train_ds = Macro5HzSequenceDataset(
        train_eps, sequence_length, burn_in, stride,
        augment=mirror_augmentation, stateful=stateful)
    val_ds = Macro5HzSequenceDataset(
        val_eps, sequence_length, burn_in, stride,
        augment=False, stateful=stateful)
    if stateful:
        if balanced_sampling:
            warnings.warn(
                "stateful macro TBPTT preserves episode order; weighted "
                "sampling is disabled and macro losses must be balanced.",
                RuntimeWarning)
        common = dict(num_workers=workers,
                      pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(
            train_ds, batch_sampler=StatefulEpisodeBatchSampler(
                train_ds, batch_size, shuffle=True, seed=seed), **common)
        val_loader = DataLoader(
            val_ds, batch_sampler=StatefulEpisodeBatchSampler(
                val_ds, batch_size, shuffle=False, seed=seed), **common)
        return train_loader, val_loader, train_eps, val_eps
    sampler = None
    shuffle = True
    if balanced_sampling:
        sampler = WeightedRandomSampler(
            train_ds.sampling_weights(), len(train_ds.windows),
            replacement=True)
        shuffle = False
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler,
                              **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_eps, val_eps


# Concise compatibility names for callers.
ILSequenceDataset = V25SequenceDataset
