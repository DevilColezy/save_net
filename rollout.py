#!/usr/bin/env python3
"""Closed-loop deterministic rollout for the schema-v25 ViTFly policy.

Examples:
    python3 rollout.py --list-tasks
    python3 rollout.py --checkpoint best.pt
    python3 rollout.py --checkpoint best.pt --model-file model/model.py \
        --tasks clear_straight,forced_left,continuous_5hz

The suite uses reproducible obstacle layouts and includes fixed, abrupt-switch
and continuous 5 Hz goals.  Its preprocessing is intentionally identical to
``V25SequenceDataset``: uint16 centimetre depth divided by 5 m and the exact
11-element state vector/scale stored in the checkpoint.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_IL_SCRIPTS = _THIS_DIR.parent / "il_dataset" / "scripts"
if str(_IL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IL_SCRIPTS))

try:
    import il_common
    import il_dynamics
except ImportError as e:
    il_common = None
    il_dynamics = None
    _IL_IMPORT_ERROR: Optional[ImportError] = e
else:
    _IL_IMPORT_ERROR = None


SCHEMA_VERSION = 25
EXPECTED_ARCHITECTURE = "ViTFlyLSTMPolicy"
EXPECTED_STATE_FIELDS = (
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "goal_direction_flu_x", "goal_direction_flu_y", "goal_direction_flu_z",
    "goal_distance_norm",
)
DEFAULT_STATE_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _load_model_from_file(model_file: str) -> Tuple[Any, Any]:
    model_path = Path(model_file).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    spec = importlib.util.spec_from_file_location(
        f"_rollout_model_{model_path.stem}", str(model_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    MC = getattr(module, "ViTFlyLSTMPolicy", None)
    CC = getattr(module, "ViTFlyPolicyConfig", None)
    if MC is None:
        raise ImportError(f"{model_path} does not define ViTFlyLSTMPolicy")
    if CC is None:
        raise ImportError(f"{model_path} does not define ViTFlyPolicyConfig")
    return MC, CC


def load_policy_checkpoint(checkpoint_file: str, model_file: str,
                           device: torch.device,
                           requested_depth_max_m: float = 5.0):
    """Load and strictly validate one train.py schema-v25 checkpoint."""
    MC, CC = _load_model_from_file(model_file)
    try:
        checkpoint = torch.load(
            checkpoint_file, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only``
        checkpoint = torch.load(checkpoint_file, map_location=device)
    if int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be {SCHEMA_VERSION}, got "
            f"{checkpoint.get('schema_version')!r}")
    if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"checkpoint architecture must be {EXPECTED_ARCHITECTURE}, got "
            f"{checkpoint.get('architecture')!r}")
    checkpoint_fields = tuple(checkpoint.get("student_input_fields", ()))
    expected_fields = ("depth_file",) + EXPECTED_STATE_FIELDS
    if checkpoint_fields and checkpoint_fields != expected_fields:
        raise ValueError(
            "checkpoint student input order does not match schema-v25 rollout")
    normalization = checkpoint.get("normalization") or {}
    checkpoint_depth_max = float(normalization.get("depth_max_m", 5.0))
    if abs(checkpoint_depth_max - requested_depth_max_m) > 1e-6:
        raise ValueError(
            f"requested depth_max_m={requested_depth_max_m} disagrees with "
            f"checkpoint depth_max_m={checkpoint_depth_max}")
    state_scale = tuple(float(value) for value in
                        normalization.get("state_scale", DEFAULT_STATE_SCALE))
    if len(state_scale) != len(EXPECTED_STATE_FIELDS) or \
            min(state_scale) <= 0.0:
        raise ValueError(
            "checkpoint normalization.state_scale must have %d positives"
            % len(EXPECTED_STATE_FIELDS))
    config_dict = checkpoint.get("model_config") or {}
    model_config = CC(**config_dict) if config_dict else CC()
    model_config.validate()
    model = MC(model_config)
    if "model_state" not in checkpoint:
        raise KeyError("schema-v25 checkpoint is missing model_state")
    state_dict = checkpoint["model_state"]
    model.load_state_dict({
        key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
        for key, value in state_dict.items()
    }, strict=True)
    model.to(device=device)
    model.eval()
    return model, model_config, state_scale


def _yaw_from_quat_xyzw(q_xyzw: np.ndarray) -> float:
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_dynamics_config(ctrl_hz: float, max_yaw_rate: float) -> Dict[str, Any]:
    return {
        "global": {
            "dynamics": {
                "simulation_hz": 200.0,
                "control_hz": ctrl_hz,
                "render_hz": 20.0,
                "control_mode": "velocity_yaw_rate",
                "deterministic_time": True,
                "velocity_controller": {
                    "use_existing_flightmare_controller": True,
                    "kp_velocity": [3.0, 3.0, 3.0],
                    "ki_velocity": [0.0, 0.0, 0.0],
                    "kd_velocity": [0.2, 0.2, 0.2],
                    "maximum_acceleration_mps2": [4.0, 4.0, 2.0],
                    "maximum_tilt_deg": 35.0,
                    "maximum_yaw_rate_rps": max_yaw_rate,
                    "attitude_gain": 6.0,
                    "maximum_body_rate_rps": 6.0,
                    "integrator_limit": [1.0, 1.0, 0.5],
                },
                "reset": {
                    "settle_time_s": 0.30,
                    "initial_velocity_noise_std_mps": [0.0, 0.0, 0.0],
                    "initial_angular_velocity_noise_std_rps": [0.0, 0.0, 0.0],
                },
            }
        }
    }


@dataclass
class RolloutConfig:
    checkpoint: str = ""
    model_file: str = ""
    pub_port: str = "10253"
    sub_port: str = "10254"
    scene_id: int = 1
    # Collector defaults: 640x480, fov=90, near=0.01, far=1000.0, max_m=5.0
    depth_width: int = 640
    depth_height: int = 480
    depth_fov: float = 90.0
    depth_near: float = 0.01
    depth_far: float = 1000.0
    depth_max_m: float = 5.0
    # Collector: record_hz=30, control_hz=50
    model_hz: float = 30.0
    ctrl_hz: float = 50.0
    # 0 keeps recurrent state for the complete episode. A positive value is
    # an ablation only; normal schema-v25 inference must leave this at zero.
    lstm_reset_interval: int = 0
    max_yaw_rate: float = 2.0
    max_episode_time: float = 30.0
    goal_tolerance_m: float = 0.30
    goal_speed_tolerance_mps: float = 0.20
    goal_hold_ticks: int = 3
    collision_confirm_frames: int = 1
    drone_radius: float = 0.3
    safety_margin: float = 0.30
    minimum_surface_gap_m: float = 1.20
    state_scale: Tuple[float, ...] = DEFAULT_STATE_SCALE
    device: str = "auto"
    repeats: int = 1
    verbose: bool = True
    log_prefix: str = "rollout_latest"
    render_warmup_frames: int = 5


@dataclass(frozen=True)
class Cylinder:
    """A deterministic vertical cylinder in world coordinates."""

    x: float
    y: float
    radius: float
    height: float = 8.0
    base_z: float = 0.0


@dataclass(frozen=True)
class RolloutTask:
    """One reproducible policy test with fixed geometry and endpoints."""

    name: str
    description: str
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    start_yaw: float
    obstacles: Tuple[Cylinder, ...] = ()
    max_episode_time: Optional[float] = None
    # (simulation time in seconds, new local goal). The first entry may be at
    # t=0.  ``goal`` remains the final endpoint used for success evaluation.
    goal_updates: Tuple[Tuple[float, Tuple[float, float, float]], ...] = ()
    suite: str = "basic"
    # Scene number in an external manifest.  This is intentionally separate
    # from the Unity scene selected by the rollout CLI.
    scene_id: Optional[int] = None


# il_dataset_config.yaml: scene_generation.obstacle_region and the primary
# bottom-to-top start region.  The rollout goal is intentionally closer than
# the collection goal (y=31..32) to expose policies that replay a memorized
# full-length trajectory instead of responding to the current goal.
COLLECTION_OBSTACLE_BOUNDS = (-7.0, 10.0, 0.0, 30.0, 0.0, 8.0)
COLLECTION_START_BOUNDS = (-6.5, 9.5, -1.5, -0.5, 1.8, 2.2)
ROLLOUT_WORKSPACE_BOUNDS = (-6.5, 9.5, -1.5, 29.5, 1.8, 2.2)


def _point_in_bounds(point: np.ndarray, bounds: Tuple[float, ...]) -> bool:
    return (
        bounds[0] <= point[0] <= bounds[1] and
        bounds[2] <= point[1] <= bounds[3] and
        bounds[4] <= point[2] <= bounds[5]
    )


def build_task_registry() -> Dict[str, RolloutTask]:
    """Build basic acceptance tasks plus clearly marked stress tasks.

    Full-height cylinders and >=1.2 m surface gaps match collection.  The
    default CLI runs only the basic suite; compound tasks are opt-in stress
    tests and are not used to decide whether a newly trained policy loads.
    """
    start = (0.0, -1.0, 2.0)
    goal = (0.0, 11.0, 2.0)
    # Match il_manager._get_current_initial_yaw(): navigation/camera forward is
    # Flightlib body +Y, so yaw=0 faces world +Y (not world +X).
    delta_x = goal[0] - start[0]
    delta_y = goal[1] - start[1]
    forward_yaw = math.atan2(delta_y, delta_x) - math.pi / 2.0
    forward_yaw = math.atan2(math.sin(forward_yaw), math.cos(forward_yaw))
    continuous_updates = tuple(
        (0.2 * index, (0.25 * math.sin(index * 0.20),
                       min(11.0, 3.0 + 0.30 * index), 2.0))
        for index in range(28)
    ) + ((5.6, goal),)
    tasks = [
        RolloutTask(
            "clear_straight",
            "Baseline: straight flight with no obstacles.",
            start, goal, forward_yaw,
        ),
        RolloutTask(
            "center_pillar",
            "Symmetric single obstacle; tests stable side selection.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 5.0, 0.60),),
        ),
        RolloutTask(
            "near_pillar",
            "Centered obstacle visible near takeoff; tests early visual response.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 2.5, 0.45),),
        ),
        RolloutTask(
            "far_pillar",
            "Centered obstacle late in the route; tests response timing.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 8.0, 0.80),),
        ),
        RolloutTask(
            "offset_left",
            "Obstacle on body-left (-X); expected detour is body-right (+X).",
            start, goal, forward_yaw,
            (Cylinder(-0.55, 5.0, 0.55),),
        ),
        RolloutTask(
            "offset_right",
            "Obstacle on body-right (+X); expected detour is body-left (-X).",
            start, goal, forward_yaw,
            (Cylinder(0.55, 5.0, 0.55),),
        ),
        RolloutTask(
            "forced_left",
            "Blocked body-right (+X); free detour is body-left (-X).",
            start, goal, forward_yaw,
            (Cylinder(0.0, 5.0, 0.55),
             Cylinder(2.35, 5.0, 0.55)),
        ),
        RolloutTask(
            "forced_right",
            "Blocked body-left (-X); free detour is body-right (+X).",
            start, goal, forward_yaw,
            (Cylinder(0.0, 5.0, 0.55),
             Cylinder(-2.35, 5.0, 0.55)),
        ),
        RolloutTask(
            "wide_gate",
            "Comfortable gate; checks centered passage without oscillation.",
            start, goal, forward_yaw,
            (Cylinder(-1.30, 5.0, 0.65),
             Cylinder(1.30, 5.0, 0.65)),
        ),
        RolloutTask(
            "rear_goal",
            "Goal starts behind the camera; tests turn-before-translation.",
            (0.0, 7.0, 2.0), (0.0, 1.0, 2.0), forward_yaw,
        ),
        RolloutTask(
            "abrupt_goal_switch",
            "One in-episode goal jump without resetting recurrent state.",
            start, (1.5, 10.0, 2.0), forward_yaw, (), 30.0,
            ((0.0, (-1.5, 6.0, 2.0)), (2.5, (1.5, 10.0, 2.0))),
        ),
        RolloutTask(
            "continuous_5hz",
            "Smooth local goal updates at 5 Hz, matching planner integration.",
            start, goal, forward_yaw, (), 30.0, continuous_updates,
        ),
        RolloutTask(
            "two_stage_lr",
            "Body-left obstacle followed by body-right obstacle.",
            start, goal, forward_yaw,
            (Cylinder(-0.75, 4.0, 0.60),
             Cylinder(0.75, 8.0, 0.60)),
            35.0, (), "stress",
        ),
        RolloutTask(
            "two_stage_rl",
            "Body-right obstacle followed by body-left obstacle; mirrored test.",
            start, goal, forward_yaw,
            (Cylinder(0.75, 4.0, 0.60),
             Cylinder(-0.75, 8.0, 0.60)),
            35.0, (), "stress",
        ),
        RolloutTask(
            "double_gate",
            "Two oppositely shifted gates require a mid-route correction.",
            start, goal, forward_yaw,
            (Cylinder(-1.30, 4.0, 0.60),
             Cylinder(1.30, 4.0, 0.60),
             Cylinder(-1.30, 8.0, 0.60),
             Cylinder(1.30, 8.0, 0.60)),
            35.0, (), "stress",
        ),
        RolloutTask(
            "slalom",
            "Alternating obstacles; tests repeated left/right decisions.",
            start, goal, forward_yaw,
            (Cylinder(-0.75, 2.5, 0.55),
             Cylinder(0.75, 5.5, 0.55),
             Cylinder(-0.75, 8.5, 0.55)),
            40.0, (), "stress",
        ),
    ]
    return {task.name: task for task in tasks}


def validate_task_registry(
    tasks: Dict[str, RolloutTask], drone_radius: float, safety_margin: float,
    minimum_surface_gap_m: float = 1.20,
) -> None:
    """Fail fast on malformed tasks or unsafe endpoints."""
    if not tasks:
        raise ValueError("The rollout task registry is empty")
    clearance = drone_radius + safety_margin
    for key, task in tasks.items():
        if key != task.name:
            raise ValueError(f"Task registry key/name mismatch: {key}/{task.name}")
        start = np.asarray(task.start, dtype=np.float64)
        goal = np.asarray(task.goal, dtype=np.float64)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError(f"Task {key}: start/goal must be finite")
        if np.linalg.norm(goal - start) < 1.0:
            raise ValueError(f"Task {key}: start and goal are too close")
        if not _point_in_bounds(start, ROLLOUT_WORKSPACE_BOUNDS):
            raise ValueError(f"Task {key}: start is outside rollout workspace")
        if not _point_in_bounds(goal, ROLLOUT_WORKSPACE_BOUNDS):
            raise ValueError(f"Task {key}: goal is outside rollout workspace")
        endpoints = [("start", start), ("goal", goal)]
        previous_time = -1.0
        for update_index, (update_time, update_goal) in enumerate(task.goal_updates):
            update_point = np.asarray(update_goal, dtype=np.float64)
            if update_time < 0.0 or update_time <= previous_time:
                raise ValueError(f"Task {key}: goal update times must increase")
            if not np.all(np.isfinite(update_point)) or not _point_in_bounds(
                    update_point, ROLLOUT_WORKSPACE_BOUNDS):
                raise ValueError(f"Task {key}: invalid goal update {update_index}")
            endpoints.append((f"goal_update_{update_index}", update_point))
            previous_time = update_time
        for obstacle in task.obstacles:
            if obstacle.radius <= 0.0 or obstacle.height <= 0.0:
                raise ValueError(f"Task {key}: obstacle dimensions must be positive")
            obstacle_bounds = COLLECTION_OBSTACLE_BOUNDS
            if not (
                obstacle.x - obstacle.radius >= obstacle_bounds[0] and
                obstacle.x + obstacle.radius <= obstacle_bounds[1] and
                obstacle.y - obstacle.radius >= obstacle_bounds[2] and
                obstacle.y + obstacle.radius <= obstacle_bounds[3] and
                obstacle.base_z >= obstacle_bounds[4] and
                obstacle.base_z + obstacle.height <= obstacle_bounds[5]
            ):
                raise ValueError(
                    f"Task {key}: obstacle is outside collection bounds")
            for label, endpoint in endpoints:
                within_height = (
                    obstacle.base_z - clearance <= endpoint[2] <=
                    obstacle.base_z + obstacle.height + clearance
                )
                planar_distance = np.linalg.norm(
                    endpoint[:2] - np.array([obstacle.x, obstacle.y]))
                if within_height and planar_distance <= obstacle.radius + clearance:
                    raise ValueError(
                        f"Task {key}: {label} intersects an inflated obstacle")
        for first_index, first in enumerate(task.obstacles):
            for second in task.obstacles[first_index + 1:]:
                vertical_overlap = not (
                    first.base_z + first.height <= second.base_z or
                    second.base_z + second.height <= first.base_z)
                if not vertical_overlap:
                    continue
                center_distance = math.hypot(first.x - second.x,
                                             first.y - second.y)
                surface_gap = center_distance - first.radius - second.radius
                if surface_gap + 1e-9 < minimum_surface_gap_m:
                    raise ValueError(
                        f"Task {key}: obstacle surface gap {surface_gap:.3f} m "
                        f"is below {minimum_surface_gap_m:.3f} m")


def task_to_unity_objects(
    task: RolloutTask, object_slots: int,
) -> List[Dict[str, Any]]:
    """Convert a task to Unity objects and hide unused persistent slots."""
    if len(task.obstacles) > object_slots:
        raise ValueError(
            f"Task {task.name} has {len(task.obstacles)} obstacles, "
            f"but only {object_slots} slots")
    result: List[Dict[str, Any]] = []
    for index in range(object_slots):
        if index < len(task.obstacles):
            obstacle = task.obstacles[index]
            position = [
                float(obstacle.x),
                float(obstacle.base_z + obstacle.height / 2.0),
                float(obstacle.y),
            ]
            size = [
                2.0 * float(obstacle.radius),
                float(obstacle.height),
                2.0 * float(obstacle.radius),
            ]
        else:
            # Unity keeps dynamically created IDs alive. Moving every unused
            # stable slot prevents geometry from a previous task leaking in.
            position = [1000.0 + index, -1000.0, 1000.0]
            size = [0.01, 0.01, 0.01]
        result.append({
            "ID": f"rollout_obstacle_{index:02d}",
            "prefabID": "Object",
            "position": position,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "size": size,
        })
    return result


@dataclass
class EpisodeResult:
    episode: int
    task_name: str
    scene_id: int
    mode: str
    outcome: str
    duration_s: float
    path_length_m: float
    final_goal_distance_m: float
    min_goal_distance_m: float
    num_model_steps: int
    num_collision_frames: int
    first_collision_step: int
    avg_inference_ms: float
    max_inference_ms: float
    num_depth_timeouts: int
    num_frame_mismatches: int
    goal_switch_count: int
    minimum_body_clearance_m: float
    avg_command_delta: float


def _json_safe(value: Any) -> Any:
    """Convert dataclasses/numpy values and non-finite floats for strict JSON."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RolloutDataLogger:
    """Continuously save frame telemetry and atomically update run summaries."""

    _BASE_COLUMNS = [
        "episode", "task", "step", "global_frame", "sim_time_s",
        "state_x", "state_y", "state_z", "state_yaw",
        "camera_heading_yaw",
        "next_x", "next_y", "next_z",
        "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
        "speed_world_mps", "yaw_rate_rps", "goal_distance_m",
        "active_goal_x", "active_goal_y", "active_goal_z", "goal_switch_event",
        "guide_x", "guide_y", "guide_z", "guide_distance_norm",
        "state_gravity_x", "state_gravity_y", "state_gravity_z",
        "minimum_body_clearance_m",
        "depth_min_m", "depth_mean_m",
        "depth_near_1m_frac", "depth_near_2m_frac",
        "depth_near_3m_frac", "depth_near_4m_frac",
        "depth_left_near_4m_frac", "depth_center_near_4m_frac",
        "depth_right_near_4m_frac",
        "normalized_cmd_vx", "normalized_cmd_vy", "normalized_cmd_vz",
        "normalized_cmd_yaw_rate",
        "cmd_vx_flu", "cmd_vy_flu", "cmd_vz_flu", "cmd_yaw_rate",
        "inference_ms",
    ]
    COLUMNS = _BASE_COLUMNS

    def __init__(self, prefix: str, metadata: Dict[str, Any]) -> None:
        prefix_path = Path(prefix).expanduser()
        if not prefix_path.is_absolute():
            prefix_path = Path.cwd() / prefix_path
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        self.steps_path = prefix_path.parent / f"{prefix_path.name}_steps.csv"
        self.summary_path = prefix_path.parent / f"{prefix_path.name}_summary.json"
        self._metadata = metadata
        # Line buffering plus an explicit flush keeps useful data after Ctrl-C.
        self._steps_file = self.steps_path.open(
            "w", newline="", encoding="utf-8", buffering=1,
        )
        self._writer = csv.DictWriter(
            self._steps_file, fieldnames=self.COLUMNS, extrasaction="raise",
        )
        self._writer.writeheader()
        self._steps_file.flush()

    def write_step(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._steps_file.flush()

    def write_summary(
        self, status: str, results: List[EpisodeResult],
    ) -> None:
        payload = {
            **self._metadata,
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_episodes": len(results),
            "results": results,
            "steps_file": str(self.steps_path),
        }
        temp_path = self.summary_path.parent / f"{self.summary_path.name}.tmp"
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload), handle, ensure_ascii=False,
                indent=2, allow_nan=False,
            )
            handle.write("\n")
        temp_path.replace(self.summary_path)

    def close(self) -> None:
        self._steps_file.flush()
        self._steps_file.close()


def compute_global_guide(
    drone_pos: np.ndarray, goal_pos: np.ndarray,
    quat_xyzw: np.ndarray, max_range: float,
) -> Tuple[np.ndarray, float, np.float32]:
    """Global guide in body FLU using full quaternion (same as collector)."""
    dw = goal_pos - drone_pos
    dm = float(np.linalg.norm(dw))
    if dm > 1e-8:
        gf = il_common.world_vector_to_body_flu_quat(dw, quat_xyzw)
        gf = gf / max(float(np.linalg.norm(gf)), 1e-8)
    else:
        gf = np.zeros(3, dtype=np.float64)
    dn = np.float32(min(dm, max_range) / max(max_range, 1e-9))
    return gf, dm, dn


def canonicalize_unity_depth(depth_payload: np.ndarray,
                             max_depth_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the collector/writer/loader depth contract exactly.

    Unity's payload uses hectometres, hence the x100 conversion. Invalid,
    zero and negative samples mean no return and are encoded as max range.
    The writer rounds metres to uint16 centimetres; the loader multiplies by
    0.01 and divides by max range.
    """
    depth_m = np.flipud(np.asarray(depth_payload, dtype=np.float32) * 100.0)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_m = depth_m.copy()
    depth_m[~valid] = float(max_depth_m)
    depth_m = np.clip(depth_m, 0.0, float(max_depth_m))
    depth_cm = np.round(depth_m * 100.0).astype(np.uint16)
    normalized = np.clip(
        depth_cm.astype(np.float32) * 0.01 / float(max_depth_m), 0.0, 1.0)
    return depth_cm, normalized


def preprocess_depth(depth_normalized: np.ndarray,
                     dev: torch.device) -> torch.Tensor:
    """Return [1,1,H,W]; model.step performs the training-size resize."""
    if depth_normalized.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    return torch.from_numpy(depth_normalized.astype(np.float32))[None, None].to(
        device=dev)


def build_normalized_state(gravity_flu: np.ndarray,
                           goal_direction_flu: np.ndarray,
                           goal_distance_norm: float,
                           state_scale: Tuple[float, ...],
                           device: torch.device) -> torch.Tensor:
    """Construct the 7-D student state in field order (gravity 3 + goal
    direction 3 + goal distance 1).  velocity_flu / yaw_rate_flu inputs were
    REMOVED (2026-08-26) so the policy cannot short-circuit on current motion
    and must read the depth image for avoidance."""
    raw = np.concatenate((
        np.asarray(gravity_flu, dtype=np.float32).reshape(3),
        np.asarray(goal_direction_flu, dtype=np.float32).reshape(3),
        np.asarray([goal_distance_norm], dtype=np.float32),
    ))
    scale = np.asarray(state_scale, dtype=np.float32)
    if raw.shape != (7,) or scale.shape != (7,) or np.any(scale <= 0.0):
        raise ValueError("state and state_scale must contain 7 values")
    if not np.isfinite(raw).all():
        raise ValueError("non-finite rollout state")
    return torch.from_numpy(raw / scale)[None].to(device=device)


def active_goal_for_time(task: RolloutTask, sim_time_s: float,
                         previous_index: int) -> Tuple[np.ndarray, int, bool]:
    """Return the scheduled local goal without ever resetting the LSTM."""
    if not task.goal_updates:
        return np.asarray(task.goal, dtype=np.float64), -1, False
    index = previous_index
    for candidate in range(previous_index + 1, len(task.goal_updates)):
        if task.goal_updates[candidate][0] <= sim_time_s + 1e-9:
            index = candidate
        else:
            break
    if index < 0:
        goal = task.goal_updates[0][1]
    else:
        goal = task.goal_updates[index][1]
    return np.asarray(goal, dtype=np.float64), index, index != previous_index


def body_clearance(position: np.ndarray, task: RolloutTask,
                   drone_radius: float) -> float:
    """Analytic clearance from vehicle surface to the nearest cylinder."""
    values = []
    for obstacle in task.obstacles:
        if obstacle.base_z - drone_radius <= position[2] <= \
                obstacle.base_z + obstacle.height + drone_radius:
            values.append(math.hypot(position[0] - obstacle.x,
                                     position[1] - obstacle.y) -
                          obstacle.radius - drone_radius)
    return min(values) if values else float("inf")


def run_rollout(
    model, model_cfg, rollout_cfg: RolloutConfig, device: torch.device,
    ep_idx: int, gid: int, bridge, dyn, task: RolloutTask,
    scene_id: int, obstacles: List[Dict[str, Any]],
    data_logger: RolloutDataLogger,
) -> Tuple[EpisodeResult, int]:
    dts = 1.0 / rollout_cfg.model_hz
    episode_time = (
        task.max_episode_time
        if task.max_episode_time is not None
        else rollout_cfg.max_episode_time
    )
    mx = int(episode_time * rollout_cfg.model_hz)
    dc = {
        "width": rollout_cfg.depth_width, "height": rollout_cfg.depth_height,
        "fov": rollout_cfg.depth_fov, "near": rollout_cfg.depth_near,
        "far": rollout_cfg.depth_far,
    }
    ih, iw = rollout_cfg.depth_height, rollout_cfg.depth_width
    mr = rollout_cfg.depth_max_m
    sp = np.asarray(task.start, dtype=np.float64)
    gp = np.asarray(task.goal, dtype=np.float64)
    syaw = float(task.start_yaw)
    # Dynamics is created once in main() and reused (collector pattern).
    dyn.reset(sp.copy(), syaw, np.zeros(3), np.zeros(3))
    veh = il_common.make_depth_vehicle(
        ros_pos=sp.copy().tolist(), yaw=syaw, depth_cfg=dc,
    )
    # Handshake is done once in main(); here we just send the scene.
    st: Dict[str, Any] = {
        "scene_id": scene_id, "frame_id": gid,
        "vehicles": [veh], "objects": obstacles,
    }
    bridge.send_pose(st)
    # Never reuse the scene-update frame ID for a model input. A delayed
    # response to this message could otherwise be mistaken for the first
    # synchronized rollout frame.
    gid += 1
    time.sleep(0.5)

    # Unity's first render after handshake/scene mutation can contain an
    # uninitialized all-far depth map. Drain that response, then request and
    # discard several uniquely identified renders while the vehicle remains
    # stationary. LSTM state is created only after this renderer warmup.
    try:
        while bridge.try_recv() is not None:
            pass
    except Exception:
        pass
    depth_payload_bytes = iw * ih * 4
    last_warm_depth: Optional[np.ndarray] = None
    for warmup_index in range(rollout_cfg.render_warmup_frames):
        warm_state = dyn.get_state()
        warm_position = warm_state.position_world.copy()
        warm_quaternion = warm_state.quaternion_world_body.copy()
        warm_vehicle = il_common.make_depth_vehicle(
            ros_pos=warm_position.tolist(),
            yaw=_yaw_from_quat_xyzw(warm_quaternion),
            depth_cfg=dc,
            quaternion_xyzw=warm_quaternion.tolist(),
        )
        st["frame_id"] = gid
        st["vehicles"] = [warm_vehicle]
        bridge.send_pose(st)

        warm_depth = None
        warm_start = time.perf_counter()
        while time.perf_counter() - warm_start < 1.0:
            response = bridge.try_recv()
            if response is None:
                time.sleep(0.002)
                continue
            warm_metadata, warm_payloads = response
            warm_frame_id = warm_metadata.get("pub_frame_id")
            if warm_frame_id is None:
                warm_frame_id = warm_metadata.get("frame_id", -1)
            if warm_frame_id != gid:
                continue
            for payload in warm_payloads:
                if len(payload) < depth_payload_bytes:
                    continue
                raw_depth = payload[:depth_payload_bytes]
                depth_float = np.frombuffer(
                    raw_depth, dtype=np.float32,
                ).reshape((ih, iw))
                warm_depth, _ = canonicalize_unity_depth(depth_float, mr)
                break
            if warm_depth is not None:
                break
        if warm_depth is None:
            raise RuntimeError(
                f"Task {task.name}: Unity render warmup frame "
                f"{warmup_index + 1}/{rollout_cfg.render_warmup_frames} timed out"
            )
        last_warm_depth = warm_depth
        gid += 1

    if rollout_cfg.verbose and last_warm_depth is not None:
        warm_depth_m = last_warm_depth.astype(np.float32) * 0.01
        print(
            f"  [rollout] render warmup complete: "
            f"frames={rollout_cfg.render_warmup_frames} "
            f"depth_min={np.min(warm_depth_m):.2f}m "
            f"depth_mean={np.mean(warm_depth_m):.2f}m "
            f"near4={np.mean(warm_depth_m < 4.0):.1%}",
            flush=True,
        )

    dt = torch.float32
    hidden = model.initial_hidden(1, device=device, dtype=dt)
    rollout_start_position = dyn.get_state().position_world.copy()
    pp: List[np.ndarray] = [rollout_start_position]
    cf = 0; fcs = -1; mts: List[float] = []; ndt = 0; nfm = 0
    pc: Optional[np.ndarray] = None; cds: List[float] = []
    md = float(np.linalg.norm(rollout_start_position - gp))
    fd = md; ot = "timeout"; gh = 0
    goal_schedule_index = -1
    goal_switch_count = 0
    minimum_clearance = body_clearance(
        rollout_start_position, task, rollout_cfg.drone_radius)
    for step in range(mx):
        if (
            rollout_cfg.lstm_reset_interval > 0
            and step > 0
            and step % rollout_cfg.lstm_reset_interval == 0
        ):
            hidden = model.initial_hidden(1, device=device, dtype=dt)

        stt = dyn.get_state()
        pw = stt.position_world.copy()
        qq = stt.quaternion_world_body.copy()
        veh = il_common.make_depth_vehicle(
            ros_pos=pw.tolist(), yaw=_yaw_from_quat_xyzw(qq),
            depth_cfg=dc, quaternion_xyzw=qq.tolist(),
        )
        st["frame_id"] = gid; st["vehicles"] = [veh]
        bridge.send_pose(st)
        du = None; col = False
        t0 = time.perf_counter(); to = 1.0; dfl = iw * ih * 4
        while time.perf_counter() - t0 < to:
            r = bridge.try_recv()
            if r is None:
                time.sleep(0.002); continue
            md_, rp = r
            fid = md_.get("pub_frame_id")
            if fid is None:
                fid = md_.get("frame_id", -1)
            if fid != gid:
                nfm += 1
                if nfm <= 3 or nfm % 100 == 0:
                    print(f"  [rollout] frame ID mismatch: expected={gid} got={fid} "
                          f"(nfm={nfm})", flush=True)
                continue
            for pt in rp:
                if len(pt) >= dfl:
                    raw = pt[:dfl]
                    df = np.frombuffer(raw, dtype=np.float32).reshape((ih, iw))
                    du, depth_normalized = canonicalize_unity_depth(
                        df, rollout_cfg.depth_max_m)
                    break
            vs = md_.get("pub_vehicles", [])
            if vs and vs[0].get("collision", False):
                col = True
            break
        if col:
            cf += 1
            if fcs < 0:
                fcs = step
            if cf >= rollout_cfg.collision_confirm_frames:
                ot = "collision"; break
        if du is None:
            ndt += 1; ot = "error"
            warnings.warn(f"[rollout] Ep {ep_idx} step {step}: depth timeout")
            break
        sim_time_s = step * dts
        active_goal, new_goal_index, goal_switch_event = active_goal_for_time(
            task, sim_time_s, goal_schedule_index)
        if goal_switch_event:
            # Index zero establishes the initial local goal; later changes
            # are actual planner updates. Hidden state intentionally persists.
            if goal_schedule_index >= 0:
                goal_switch_count += 1
            goal_schedule_index = new_goal_index
        gf, gdm, gdn = compute_global_guide(pw, active_goal, qq, mr)
        state_pw = pw.copy()
        state_yaw = _yaw_from_quat_xyzw(qq)
        state_velocity_flu = stt.velocity_flu.copy()
        state_speed = float(np.linalg.norm(stt.velocity_world))
        depth_m = du.astype(np.float32) * 0.01
        first_split = iw // 3
        second_split = 2 * iw // 3
        left_depth = depth_m[:, :first_split]
        center_depth = depth_m[:, first_split:second_split]
        right_depth = depth_m[:, second_split:]
        dt_ = preprocess_depth(depth_normalized, device)
        grav_flu = il_common.world_vector_to_body_flu_quat(
            np.array([0.0, 0.0, -1.0], dtype=np.float64), qq,
        )
        omega_body = np.asarray(stt.angular_velocity_body, dtype=np.float32)
        omega_flu = np.array(
            [omega_body[1], -omega_body[0], omega_body[2]],
            dtype=np.float32)
        yr = float(omega_flu[2])
        state_tensor = build_normalized_state(
            grav_flu, gf, float(gdn),
            rollout_cfg.state_scale, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ti = time.perf_counter()
        with torch.no_grad():
            out = model.step(dt_, state_tensor, hidden)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ims = (time.perf_counter() - ti) * 1000.0; mts.append(ims)
        hidden = tuple(value.detach() for value in out.hidden)
        cmf = out.command[0].cpu().numpy().copy()
        normalized_command = out.normalized_command[0].cpu().numpy().copy()
        if pc is not None:
            cds.append(float(np.linalg.norm(cmf - pc)))
        pc = cmf.copy()
        vc = cmf[:3].copy(); yc = float(cmf[3])
        dyn.step_velocity_command(vc, yc, dts)
        stt = dyn.get_state()
        pw = stt.position_world.copy()
        pp.append(pw.copy())
        dst = float(np.linalg.norm(pw - gp))
        fd = dst
        if dst < md:
            md = dst
        clearance = body_clearance(pw, task, rollout_cfg.drone_radius)
        minimum_clearance = min(minimum_clearance, clearance)
        spd = float(np.linalg.norm(stt.velocity_world))
        step_row: Dict[str, Any] = {
            "episode": ep_idx,
            "task": task.name,
            "step": step,
            "global_frame": gid,
            "sim_time_s": (step + 1) * dts,
            "state_x": float(state_pw[0]),
            "state_y": float(state_pw[1]),
            "state_z": float(state_pw[2]),
            "state_yaw": float(state_yaw),
            "camera_heading_yaw": math.atan2(
                math.sin(state_yaw + math.pi / 2.0),
                math.cos(state_yaw + math.pi / 2.0),
            ),
            "next_x": float(pw[0]),
            "next_y": float(pw[1]),
            "next_z": float(pw[2]),
            "velocity_flu_x": float(state_velocity_flu[0]),
            "velocity_flu_y": float(state_velocity_flu[1]),
            "velocity_flu_z": float(state_velocity_flu[2]),
            "speed_world_mps": state_speed,
            "yaw_rate_rps": yr,
            "goal_distance_m": dst,
            "active_goal_x": float(active_goal[0]),
            "active_goal_y": float(active_goal[1]),
            "active_goal_z": float(active_goal[2]),
            "goal_switch_event": int(goal_switch_event and step > 0),
            "guide_x": float(gf[0]),
            "guide_y": float(gf[1]),
            "guide_z": float(gf[2]),
            "guide_distance_norm": float(gdn),
            "state_gravity_x": float(grav_flu[0]),
            "state_gravity_y": float(grav_flu[1]),
            "state_gravity_z": float(grav_flu[2]),
            "minimum_body_clearance_m": clearance,
            "depth_min_m": float(np.min(depth_m)),
            "depth_mean_m": float(np.mean(depth_m)),
            "depth_near_1m_frac": float(np.mean(depth_m < 1.0)),
            "depth_near_2m_frac": float(np.mean(depth_m < 2.0)),
            "depth_near_3m_frac": float(np.mean(depth_m < 3.0)),
            "depth_near_4m_frac": float(np.mean(depth_m < 4.0)),
            "depth_left_near_4m_frac": float(np.mean(left_depth < 4.0)),
            "depth_center_near_4m_frac": float(np.mean(center_depth < 4.0)),
            "depth_right_near_4m_frac": float(np.mean(right_depth < 4.0)),
            "normalized_cmd_vx": float(normalized_command[0]),
            "normalized_cmd_vy": float(normalized_command[1]),
            "normalized_cmd_vz": float(normalized_command[2]),
            "normalized_cmd_yaw_rate": float(normalized_command[3]),
            "cmd_vx_flu": float(vc[0]),
            "cmd_vy_flu": float(vc[1]),
            "cmd_vz_flu": float(vc[2]),
            "cmd_yaw_rate": yc,
            "inference_ms": ims,
        }
        data_logger.write_step(step_row)
        schedule_complete = (
            not task.goal_updates or
            goal_schedule_index == len(task.goal_updates) - 1)
        if schedule_complete and dst <= rollout_cfg.goal_tolerance_m and \
                spd <= rollout_cfg.goal_speed_tolerance_mps:
            gh += 1
            if gh >= rollout_cfg.goal_hold_ticks:
                ot = "success"; break
        else:
            gh = 0
        if rollout_cfg.verbose and step % 30 == 0:
            print(f"  [{ep_idx}:{step:04d}] dist={dst:.2f}m spd={spd:.2f}m/s | "
                  f"cmd=[{vc[0]:+.2f},{vc[1]:+.2f},{vc[2]:+.2f},{yc:+.2f}] | "
                  f"infer={ims:.1f}ms | guide={gdm:.1f}m | "
                  f"clear={clearance:.2f}m", flush=True)
        gid += 1
    dur = (step + 1) * dts
    plen = float(sum(np.linalg.norm(pp[i]-pp[i-1]) for i in range(1, len(pp))))
    ai = float(np.mean(mts)) if mts else 0.0
    xi = float(np.max(mts)) if mts else 0.0
    ac = float(np.mean(cds)) if cds else 0.0
    res = EpisodeResult(
        episode=ep_idx, task_name=task.name, scene_id=scene_id, mode="fixed",
        outcome=ot, duration_s=dur, path_length_m=plen,
        final_goal_distance_m=fd, min_goal_distance_m=md,
        num_model_steps=step+1, num_collision_frames=cf, first_collision_step=fcs,
        avg_inference_ms=ai, max_inference_ms=xi,
        num_depth_timeouts=ndt, num_frame_mismatches=nfm,
        goal_switch_count=goal_switch_count,
        minimum_body_clearance_m=minimum_clearance,
        avg_command_delta=ac,
    )
    return res, gid + 1


def main() -> None:
    import os as _os  # noqa: F811  (used at function exit)
    p = argparse.ArgumentParser(
        description="Closed-loop schema-v25 ViTFlyLSTMPolicy rollout")
    p.add_argument("--checkpoint")
    p.add_argument(
        "--model-file", default=str(_THIS_DIR / "model" / "model.py"),
        help="Policy implementation (default: save_net/model/model.py).")
    p.add_argument(
        "--tasks", default="basic",
        help="Comma-separated task names, or basic/stress/all (default: basic).")
    p.add_argument(
        "--list-tasks", action="store_true",
        help="List deterministic tasks and exit without loading the model.")
    p.add_argument(
        "--repeats", type=int, default=1,
        help="Repeat each selected deterministic task this many times.")
    p.add_argument("--pub-port", default="10253")
    p.add_argument("--sub-port", default="10254")
    p.add_argument("--scene-id", type=int, default=1)
    p.add_argument("--max-episode-time", type=float, default=30.0)
    p.add_argument("--goal-tolerance", type=float, default=0.30)
    p.add_argument("--goal-speed-tolerance", type=float, default=0.20)
    p.add_argument("--goal-hold-ticks", type=int, default=3)
    p.add_argument("--model-hz", type=float, default=30.0)
    p.add_argument("--ctrl-hz", type=float, default=50.0)
    p.add_argument(
        "--lstm-reset-interval", type=int, default=0,
        help=(
            "Reset the policy LSTM every N model steps (ablation only); "
            "0 keeps state for the complete episode."
        ),
    )
    p.add_argument("--max-yaw-rate", type=float, default=2.0)
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--depth-fov", type=float, default=90.0)
    p.add_argument("--depth-max-m", type=float, default=5.0)
    p.add_argument(
        "--render-warmup-frames", type=int, default=5,
        help=(
            "Discard this many synchronized Unity depth frames after each "
            "scene reset before starting the LSTM (default: 5)."
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--log-prefix", default="rollout_latest",
        help=(
            "Output prefix for continuously updated CSV/JSON diagnostics "
            "(default: ./rollout_latest)."
        ),
    )
    a = p.parse_args()

    task_registry = build_task_registry()
    validate_task_registry(
        task_registry, drone_radius=0.30, safety_margin=0.30,
        minimum_surface_gap_m=1.20)
    if a.list_tasks:
        print("Deterministic rollout tasks:")
        for task in task_registry.values():
            print(
                f"  {task.name:<20} suite={task.suite:<6} "
                f"obstacles={len(task.obstacles):>2}  "
                f"{task.description}")
        return
    if not a.checkpoint:
        p.error("--checkpoint is required unless --list-tasks is used")
    if a.repeats <= 0:
        p.error("--repeats must be > 0")
    if a.render_warmup_frames < 1:
        p.error("--render-warmup-frames must be >= 1")
    if a.lstm_reset_interval < 0:
        p.error("--lstm-reset-interval must be >= 0")
    task_selector = a.tasks.strip().lower()
    if task_selector == "all":
        selected_tasks = list(task_registry.values())
    elif task_selector in ("basic", "stress"):
        selected_tasks = [task for task in task_registry.values()
                          if task.suite == task_selector]
    else:
        requested = [name.strip() for name in a.tasks.split(",") if name.strip()]
        unknown = [name for name in requested if name not in task_registry]
        if unknown:
            p.error(
                "unknown task(s): {}. Use --list-tasks.".format(
                    ", ".join(unknown)))
        if not requested:
            p.error("--tasks must select at least one task")
        selected_tasks = [task_registry[name] for name in requested]
    if _IL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import IL dataset modules: {}. Expected path: {}. "
            "Source devel/setup.bash before running rollout."
            .format(_IL_IMPORT_ERROR, _IL_SCRIPTS))

    cf = RolloutConfig(
        checkpoint=a.checkpoint, model_file=a.model_file,
        pub_port=a.pub_port, sub_port=a.sub_port, scene_id=a.scene_id,
        depth_width=a.depth_width, depth_height=a.depth_height,
        depth_fov=a.depth_fov, depth_max_m=a.depth_max_m,
        model_hz=a.model_hz, ctrl_hz=a.ctrl_hz,
        lstm_reset_interval=a.lstm_reset_interval,
        max_yaw_rate=a.max_yaw_rate,
        max_episode_time=a.max_episode_time,
        goal_tolerance_m=a.goal_tolerance,
        goal_speed_tolerance_mps=a.goal_speed_tolerance,
        goal_hold_ticks=a.goal_hold_ticks,
        device=a.device, repeats=a.repeats, verbose=not a.quiet,
        log_prefix=a.log_prefix,
        render_warmup_frames=a.render_warmup_frames,
    )
    if a.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(a.device)
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda but CUDA unavailable")
    model, mc, checkpoint_scale = load_policy_checkpoint(
        cf.checkpoint, cf.model_file, dev, cf.depth_max_m)
    cf.state_scale = checkpoint_scale
    # CUDA warmup: first inference is slow (kernel compilation).
    with torch.no_grad():
        _ = model.step(
            torch.ones(1, 1, mc.image_height, mc.image_width,
                       device=dev, dtype=torch.float32),
            torch.zeros(1, mc.state_dim, device=dev, dtype=torch.float32),
            model.initial_hidden(1, device=dev, dtype=torch.float32))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print("[rollout] CUDA warmup complete.")
    total_episodes = len(selected_tasks) * cf.repeats
    print("="*60)
    print("Policy Rollout - ViTFlyLSTMPolicy (schema v25)")
    print("="*60)
    print(f"  Checkpoint:  {cf.checkpoint}")
    print(f"  Model file:  {cf.model_file}")
    print(f"  Device:      {dev}")
    print("  Mode:        fixed task suite")
    print(f"  Tasks:       {', '.join(task.name for task in selected_tasks)}")
    print(f"  Repeats:     {cf.repeats}")
    print(f"  Episodes:    {total_episodes}")
    print(f"  Model Hz:    {cf.model_hz}  (collector record rate)")
    print(f"  Ctrl Hz:     {cf.ctrl_hz}   (collector control rate)")
    if cf.lstm_reset_interval > 0:
        print(
            f"  LSTM state:  reset every {cf.lstm_reset_interval} model steps"
        )
    else:
        print("  LSTM state:  continuous for complete episode")
    print(f"  Depth:       {cf.depth_width}x{cf.depth_height} max={cf.depth_max_m}m")
    print(f"  Render warm: {cf.render_warmup_frames} discarded frames/episode")
    print(f"  Goal:        {cf.goal_tolerance_m}m <={cf.goal_speed_tolerance_mps}m/s x{cf.goal_hold_ticks}")
    print(f"  Ports:       PUB={cf.pub_port} SUB={cf.sub_port}")
    print(f"  Params:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model image: {(mc.image_height, mc.image_width)}")
    print(f"  State:       {len(cf.state_scale)}-D (gravity+goal), scale={cf.state_scale}")
    log_metadata = {
        "format_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_config": cf,
        "model_config": mc,
        "tasks": selected_tasks,
    }
    data_logger = RolloutDataLogger(cf.log_prefix, log_metadata)
    data_logger.write_summary("running", [])
    print(f"  Step log:    {data_logger.steps_path}")
    print(f"  Summary:     {data_logger.summary_path}")
    print("="*60)
    print("[rollout] Connecting to Unity...")
    bridge = il_common.UnityBridge(pub_port=cf.pub_port, sub_port=cf.sub_port)
    bridge.bind()
    print("[rollout] Unity bridge bound.")
    # Handshake once (collector does it once at startup, not per-episode).
    dc_main = {
        "width": cf.depth_width, "height": cf.depth_height,
        "fov": cf.depth_fov, "near": cf.depth_near, "far": cf.depth_far,
    }
    ok = bridge.connect_handshake(cf.scene_id, dc_main, timeout=60.0)
    if not ok:
        raise RuntimeError("Unity handshake failed")
    print("[rollout] Unity handshake OK.")
    # Create dynamics once (collector creates one backend and reuses it).
    dycfg = _build_dynamics_config(cf.ctrl_hz, cf.max_yaw_rate)
    dyn = il_dynamics.FlightmareDynamicsBackend(dycfg)
    results: List[EpisodeResult] = []
    gid = 0
    object_slots = max(len(task.obstacles) for task in task_registry.values())
    episode_plan = [
        (task, repeat_index)
        for task in selected_tasks
        for repeat_index in range(cf.repeats)
    ]
    for ep, (task, repeat_index) in enumerate(episode_plan):
        # Drain stale messages; protect against ZMQ socket errors.
        try:
            while bridge.try_recv() is not None:
                pass
        except Exception:
            pass
        time.sleep(0.1)  # brief pause between episodes
        sp = np.asarray(task.start, dtype=np.float64)
        gp = np.asarray(task.goal, dtype=np.float64)
        obs = task_to_unity_objects(task, object_slots)
        print(f"\n{'-'*60}")
        print(
            f"Episode {ep+1}/{len(episode_plan)}  "
            f"(task={task.name}, repeat={repeat_index+1}/{cf.repeats})")
        print(f"  Purpose: {task.description}")
        print(f"  Start: [{sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f}]")
        print(f"  Goal:  [{gp[0]:.1f},{gp[1]:.1f},{gp[2]:.1f}]")
        print(f"  Dist:  {np.linalg.norm(gp-sp):.1f}m")
        print(f"  Scene: {len(task.obstacles)} fixed obstacles")
        print(f"{'-'*60}")
        try:
            result, gid = run_rollout(
                model=model, model_cfg=mc, rollout_cfg=cf, device=dev,
                ep_idx=ep, gid=gid, bridge=bridge, dyn=dyn, task=task,
                scene_id=cf.scene_id, obstacles=obs,
                data_logger=data_logger,
            )
        except Exception as exc:
            print(f"[rollout] Episode {ep} ERROR: {exc}")
            import traceback; traceback.print_exc()
            result = EpisodeResult(
                episode=ep, task_name=task.name, scene_id=cf.scene_id,
                mode="fixed", outcome="error",
                duration_s=0, path_length_m=0, final_goal_distance_m=float("inf"),
                min_goal_distance_m=float("inf"), num_model_steps=0,
                num_collision_frames=0, first_collision_step=-1,
                avg_inference_ms=0, max_inference_ms=0,
                num_depth_timeouts=0, num_frame_mismatches=0,
                goal_switch_count=0,
                minimum_body_clearance_m=float("inf"),
                avg_command_delta=0,
            )
            gid += 10
        results.append(result)
        data_logger.write_summary("running", results)
        cm = ""
        if result.first_collision_step >= 0:
            cm = f" | 1st_coll={result.first_collision_step} total_coll={result.num_collision_frames}"
        print(f"  -> {result.outcome.upper()} | dur={result.duration_s:.1f}s | "
              f"path={result.path_length_m:.1f}m | "
              f"final_dist={result.final_goal_distance_m:.2f}m "
              f"(min={result.min_goal_distance_m:.2f}m) | "
              f"steps={result.num_model_steps} | "
              f"infer={result.avg_inference_ms:.1f}/{result.max_inference_ms:.1f}ms | "
              f"cmdD={result.avg_command_delta:.3f} | "
              f"switches={result.goal_switch_count} "
              f"min_clear={result.minimum_body_clearance_m:.2f}m | "
              f"dto={result.num_depth_timeouts} fmm={result.num_frame_mismatches}"
              + cm, flush=True)
    # Safe cleanup: avoid segfault from ZMQ / C++ dynamics teardown.
    try:
        bridge.close()
    except Exception:
        pass
    print(f"\n{'='*60}")
    print("SUMMARY"); print(f"{'='*60}")
    nt = len(results)
    ns = sum(1 for r in results if r.outcome == "success")
    nc = sum(1 for r in results if r.outcome == "collision")
    nto = sum(1 for r in results if r.outcome == "timeout")
    ne = sum(1 for r in results if r.outcome == "error")
    print("  Mode:        fixed task suite")
    print(f"  Total:       {nt}")
    print(f"  Success:     {ns} ({100*ns/max(nt,1):.1f}%)")
    print(f"  Collision:   {nc} ({100*nc/max(nt,1):.1f}%)")
    print(f"  Timeout:     {nto} ({100*nto/max(nt,1):.1f}%)")
    print(f"  Error:       {ne}")
    print("\n  Per-task outcomes:")
    for task in selected_tasks:
        task_results = [r for r in results if r.task_name == task.name]
        successes = sum(r.outcome == "success" for r in task_results)
        collisions = sum(r.outcome == "collision" for r in task_results)
        timeouts = sum(r.outcome == "timeout" for r in task_results)
        errors = sum(r.outcome == "error" for r in task_results)
        print(
            f"    {task.name:<16} success={successes}/{len(task_results)} "
            f"collision={collisions} timeout={timeouts} error={errors}")
    fin = [r for r in results if r.outcome in ("success","collision","timeout")]
    if fin:
        print(f"\n  Avg dur:        {np.mean([r.duration_s for r in fin]):.1f}s")
        print(f"  Avg path:       {np.mean([r.path_length_m for r in fin]):.1f}m")
        print(f"  Avg final dist: {np.mean([r.final_goal_distance_m for r in fin]):.2f}m")
        print(f"  Avg infer:      {np.mean([r.avg_inference_ms for r in fin]):.1f}ms")
        print(f"  Avg cmd delta:  {np.mean([r.avg_command_delta for r in fin]):.4f}")
    switches = sum(r.goal_switch_count for r in results)
    finite_clearances = [r.minimum_body_clearance_m for r in results
                         if math.isfinite(r.minimum_body_clearance_m)]
    if finite_clearances:
        print(f"\n  Safety:  minimum body clearance={min(finite_clearances):.2f}m")
    print(f"  Goals:   applied in-episode updates={switches}")
    tto = sum(r.num_depth_timeouts for r in results)
    tmm = sum(r.num_frame_mismatches for r in results)
    print(f"  Infra:   depth_timeouts={tto}  frame_mismatches={tmm}")
    data_logger.write_summary("completed", results)
    data_logger.close()
    print(f"  Saved:   {data_logger.steps_path}")
    print(f"           {data_logger.summary_path}")
    print(f"{'='*60}")
    # Flush stdout before forced exit (avoids segfault in C++ pybind destructors).
    sys.stdout.flush()
    sys.stderr.flush()
    # Use os._exit to skip C++ global/static destructor chain that crashes.
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
