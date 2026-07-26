#!/usr/bin/env python3
"""Deterministic policy rollout for HierarchicalTrendControlPolicy.

Examples:
    python3 rollout.py --list-tasks
    python3 rollout.py --checkpoint best.pt --model-file model/model.py
    python3 rollout.py --checkpoint best.pt --model-file model/model.py \
        --tasks clear_straight,forced_left,narrow_gate

The suite uses fixed obstacle layouts and fixed start/goal states so results
are reproducible and timeouts/collisions can be attributed to a known task.
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
    MC = getattr(module, "HierarchicalTrendControlPolicy", None)
    CC = getattr(module, "TrendControlConfig", None)
    if MC is None:
        raise ImportError(f"{model_path} does not define HierarchicalTrendControlPolicy")
    if CC is None:
        raise ImportError(f"{model_path} does not define TrendControlConfig")
    return MC, CC


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
    max_yaw_rate: float = 2.0
    max_episode_time: float = 30.0
    goal_tolerance_m: float = 0.30
    goal_speed_tolerance_mps: float = 0.20
    goal_hold_ticks: int = 3
    collision_confirm_frames: int = 1
    drone_radius: float = 0.3
    safety_margin: float = 0.10
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


# il_dataset_config.yaml: scene_generation.obstacle_region and the primary
# bottom-to-top start region.  The rollout goal is intentionally closer than
# the collection goal (y=31..32) to expose policies that replay a memorized
# full-length trajectory instead of responding to the current goal.
COLLECTION_OBSTACLE_BOUNDS = (-7.0, 10.0, 0.0, 30.0, 0.0, 8.0)
COLLECTION_START_BOUNDS = (-6.5, 9.5, -1.5, -0.5, 1.8, 2.2)
ROLLOUT_GOAL_BOUNDS = (-6.5, 9.5, 18.5, 19.5, 1.8, 2.2)


def _point_in_bounds(point: np.ndarray, bounds: Tuple[float, ...]) -> bool:
    return (
        bounds[0] <= point[0] <= bounds[1] and
        bounds[2] <= point[1] <= bounds[3] and
        bounds[4] <= point[2] <= bounds[5]
    )


def build_task_registry() -> Dict[str, RolloutTask]:
    """Build deterministic 20 m tasks inside the collector's +Y workspace."""
    start = (0.0, -1.0, 2.0)
    goal = (0.0, 19.0, 2.0)
    # Match il_manager._get_current_initial_yaw(): navigation/camera forward is
    # Flightlib body +Y, so yaw=0 faces world +Y (not world +X).
    delta_x = goal[0] - start[0]
    delta_y = goal[1] - start[1]
    forward_yaw = math.atan2(delta_y, delta_x) - math.pi / 2.0
    forward_yaw = math.atan2(math.sin(forward_yaw), math.cos(forward_yaw))
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
            (Cylinder(0.0, 9.0, 1.0),),
        ),
        RolloutTask(
            "near_pillar",
            "Centered obstacle visible near takeoff; tests early visual response.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 4.0, 1.0),),
        ),
        RolloutTask(
            "far_pillar",
            "Centered obstacle late in the route; tests response timing.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 14.0, 1.0),),
        ),
        RolloutTask(
            "offset_left",
            "Obstacle on body-left (-X); expected detour is body-right (+X).",
            start, goal, forward_yaw,
            (Cylinder(-0.70, 9.0, 1.0),),
        ),
        RolloutTask(
            "offset_right",
            "Obstacle on body-right (+X); expected detour is body-left (-X).",
            start, goal, forward_yaw,
            (Cylinder(0.70, 9.0, 1.0),),
        ),
        RolloutTask(
            "forced_left",
            "Blocked body-right (+X); free detour is body-left (-X).",
            start, goal, forward_yaw,
            (Cylinder(0.0, 9.0, 0.95),
             Cylinder(1.65, 9.0, 0.75)),
        ),
        RolloutTask(
            "forced_right",
            "Blocked body-left (-X); free detour is body-right (+X).",
            start, goal, forward_yaw,
            (Cylinder(0.0, 9.0, 0.95),
             Cylinder(-1.65, 9.0, 0.75)),
        ),
        RolloutTask(
            "wide_gate",
            "Comfortable gate; checks centered passage without oscillation.",
            start, goal, forward_yaw,
            (Cylinder(-1.25, 9.0, 0.65),
             Cylinder(1.25, 9.0, 0.65)),
        ),
        RolloutTask(
            "narrow_gate",
            "0.9 m raw opening for a 0.6 m vehicle; precision stress test.",
            start, goal, forward_yaw,
            (Cylinder(-1.0, 9.0, 0.55),
             Cylinder(1.0, 9.0, 0.55)),
        ),
        RolloutTask(
            "gate_left",
            "Wide gate centered on body-left; tests lateral goal correction.",
            start, goal, forward_yaw,
            (Cylinder(-2.25, 9.0, 0.65),
             Cylinder(0.25, 9.0, 0.65)),
        ),
        RolloutTask(
            "gate_right",
            "Wide gate centered on body-right; mirror of gate_left.",
            start, goal, forward_yaw,
            (Cylinder(-0.25, 9.0, 0.65),
             Cylinder(2.25, 9.0, 0.65)),
        ),
        RolloutTask(
            "two_stage_lr",
            "Body-left obstacle followed by body-right obstacle.",
            start, goal, forward_yaw,
            (Cylinder(-0.75, 6.0, 0.90),
             Cylinder(0.75, 12.0, 0.90)),
        ),
        RolloutTask(
            "two_stage_rl",
            "Body-right obstacle followed by body-left obstacle; mirrored test.",
            start, goal, forward_yaw,
            (Cylinder(0.75, 6.0, 0.90),
             Cylinder(-0.75, 12.0, 0.90)),
        ),
        RolloutTask(
            "double_gate",
            "Two oppositely shifted gates require a mid-route correction.",
            start, goal, forward_yaw,
            (Cylinder(-2.05, 6.0, 0.60),
             Cylinder(0.45, 6.0, 0.60),
             Cylinder(-0.45, 12.0, 0.60),
             Cylinder(2.05, 12.0, 0.60)),
        ),
        RolloutTask(
            "funnel",
            "A wide entrance narrows into a centered precision gate.",
            start, goal, forward_yaw,
            (Cylinder(-2.0, 6.0, 0.65),
             Cylinder(2.0, 6.0, 0.65),
             Cylinder(-1.0, 11.0, 0.55),
             Cylinder(1.0, 11.0, 0.55)),
        ),
        RolloutTask(
            "reverse_funnel",
            "A narrow gate opens into a wide exit; mirror in route order.",
            start, goal, forward_yaw,
            (Cylinder(-1.0, 6.0, 0.55),
             Cylinder(1.0, 6.0, 0.55),
             Cylinder(-2.0, 11.0, 0.65),
             Cylinder(2.0, 11.0, 0.65)),
        ),
        RolloutTask(
            "slalom",
            "Alternating obstacles; tests repeated left/right decisions.",
            start, goal, forward_yaw,
            (Cylinder(-0.75, 3.0, 0.80),
             Cylinder(0.75, 7.0, 0.80),
             Cylinder(-0.75, 11.0, 0.80),
             Cylinder(0.75, 15.0, 0.80)),
            40.0,
        ),
        RolloutTask(
            "long_corridor",
            "1.5 m raw corridor; tests drift and command smoothness.",
            start, goal, forward_yaw,
            tuple(
                Cylinder(x, y, 0.55)
                for y in (2.5, 6.0, 9.5, 13.0, 16.0)
                for x in (-1.30, 1.30)
            ),
            40.0,
        ),
        RolloutTask(
            "climb_over",
            "Low wide obstacle forces a climb and return to collection height.",
            start, goal, forward_yaw,
            (Cylinder(0.0, 9.0, 2.0, height=1.6),),
            40.0,
        ),
    ]
    return {task.name: task for task in tasks}


def validate_task_registry(
    tasks: Dict[str, RolloutTask], drone_radius: float, safety_margin: float,
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
        if not _point_in_bounds(start, COLLECTION_START_BOUNDS):
            raise ValueError(
                f"Task {key}: start is outside the collection start region")
        if not _point_in_bounds(goal, ROLLOUT_GOAL_BOUNDS):
            raise ValueError(
                f"Task {key}: goal is outside the shortened rollout goal region")
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
            for label, endpoint in (("start", start), ("goal", goal)):
                within_height = (
                    obstacle.base_z - clearance <= endpoint[2] <=
                    obstacle.base_z + obstacle.height + clearance
                )
                planar_distance = np.linalg.norm(
                    endpoint[:2] - np.array([obstacle.x, obstacle.y]))
                if within_height and planar_distance <= obstacle.radius + clearance:
                    raise ValueError(
                        f"Task {key}: {label} intersects an inflated obstacle")


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
    normal_fraction: float
    recover_left_fraction: float
    recover_right_fraction: float
    recovery_entry_count: int
    max_consecutive_recovery: int
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
        "guide_x", "guide_y", "guide_z", "guide_distance_norm",
        "depth_min_m", "depth_mean_m",
        "depth_near_1m_frac", "depth_near_2m_frac",
        "depth_near_3m_frac", "depth_near_4m_frac",
        "depth_left_near_4m_frac", "depth_center_near_4m_frac",
        "depth_right_near_4m_frac",
        "horizontal_index", "vertical_index",
        "guide_value_raw", "guide_value",
        "cmd_vx_flu", "cmd_vy_flu", "cmd_vz_flu", "cmd_yaw_rate",
        "inference_ms",
    ]
    _HORIZONTAL_COLUMNS = [f"horizontal_prob_{index}" for index in range(13)]
    _VERTICAL_COLUMNS = [f"vertical_prob_{index}" for index in range(7)]
    COLUMNS = _BASE_COLUMNS + _HORIZONTAL_COLUMNS + _VERTICAL_COLUMNS

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


def preprocess_depth(
    du16: np.ndarray, th: int, tw: int, dev: torch.device,
) -> torch.Tensor:
    d = torch.from_numpy(du16.astype(np.float32) / 65535.0)
    if d.shape[0] != th or d.shape[1] != tw:
        d = torch.nn.functional.interpolate(
            d[None, None, ...], size=(th, tw), mode="area",
        ).squeeze(0)
    else:
        d = d.unsqueeze(0)
    return d.unsqueeze(0).to(device=dev)


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
    mh, mw = model_cfg.image_size
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
                depth_m = np.flipud(depth_float * 100.0)
                depth_m = np.nan_to_num(
                    depth_m, nan=mr, posinf=mr, neginf=0.0,
                )
                warm_depth = np.clip(
                    depth_m / mr * 65535.0, 0, 65535,
                ).astype(np.uint16)
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
        warm_depth_m = (
            last_warm_depth.astype(np.float32) * (mr / 65535.0)
        )
        print(
            f"  [rollout] render warmup complete: "
            f"frames={rollout_cfg.render_warmup_frames} "
            f"depth_min={np.min(warm_depth_m):.2f}m "
            f"depth_mean={np.mean(warm_depth_m):.2f}m "
            f"near4={np.mean(warm_depth_m < 4.0):.1%}",
            flush=True,
        )

    dt = torch.float32
    ts = model.initial_state(model_cfg.trend_lstm_layers, 1, model_cfg.trend_lstm_hidden_dim, device, dt)
    cs = model.initial_state(model_cfg.control_lstm_layers, 1, model_cfg.control_lstm_hidden_dim, device, dt)
    rollout_start_position = dyn.get_state().position_world.copy()
    pp: List[np.ndarray] = [rollout_start_position]
    cf = 0; fcs = -1; mts: List[float] = []; ndt = 0; nfm = 0
    tn, tl, tr = 0, 0, 0; rec = 0; wr = False; cr = 0; mcr = 0
    pc: Optional[np.ndarray] = None; cds: List[float] = []
    md = float(np.linalg.norm(rollout_start_position - gp))
    fd = md; ot = "timeout"; gh = 0
    for step in range(mx):
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
                    dm_ = np.flipud(df * 100.0)
                    dm_ = np.nan_to_num(dm_, nan=rollout_cfg.depth_max_m,
                                        posinf=rollout_cfg.depth_max_m, neginf=0.0)
                    du = np.clip(dm_ / rollout_cfg.depth_max_m * 65535.0, 0, 65535).astype(np.uint16)
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
        gf, gdm, gdn = compute_global_guide(pw, gp, qq, mr)
        state_pw = pw.copy()
        state_yaw = _yaw_from_quat_xyzw(qq)
        state_velocity_flu = stt.velocity_flu.copy()
        state_speed = float(np.linalg.norm(stt.velocity_world))
        depth_m = du.astype(np.float32) * (mr / 65535.0)
        first_split = iw // 3
        second_split = 2 * iw // 3
        left_depth = depth_m[:, :first_split]
        center_depth = depth_m[:, first_split:second_split]
        right_depth = depth_m[:, second_split:]
        dt_ = preprocess_depth(du, mh, mw, device)
        rgt = torch.as_tensor(np.array(
            [[float(gf[0]), float(gf[1]), float(gf[2]), float(gdn)]], dtype=np.float32,
        ), device=device, dtype=dt)
        grav_flu = il_common.world_vector_to_body_flu_quat(
            np.array([0.0, 0.0, -1.0], dtype=np.float64), qq,
        )
        gft = torch.as_tensor(np.asarray([grav_flu], dtype=np.float32), device=device, dtype=dt)
        vft = torch.as_tensor(np.asarray([stt.velocity_flu], dtype=np.float32).reshape(1, 3), device=device, dtype=dt)
        yr = float(stt.angular_velocity_body[2])
        yt = torch.as_tensor(np.array([[yr]], dtype=np.float32), device=device, dtype=dt)
        ti = time.perf_counter()
        with torch.no_grad():
            out = model.forward_step(
                depth=dt_, raw_guide=rgt, gravity_flu=gft,
                velocity_flu=vft, yaw_rate=yt,
                trend_state=ts, control_state=cs,
            )
        ims = (time.perf_counter() - ti) * 1000.0; mts.append(ims)
        ts = model.detach_state(out.trend_state)
        cs = model.detach_state(out.control_state)
        cmf = out.command[0, 0].cpu().numpy().copy()
        horizontal_prob = torch.softmax(
            out.horizontal_logits[0, 0], dim=-1,
        ).cpu().numpy()
        vertical_prob = torch.softmax(
            out.vertical_logits[0, 0], dim=-1,
        ).cpu().numpy()
        if pc is not None:
            cds.append(float(np.linalg.norm(cmf - pc)))
        pc = cmf.copy()
        hi = int(out.horizontal_index[0, 0].item())
        if hi == 0:
            tl += 1
            if not wr:
                rec += 1; wr = True
            cr += 1
        elif hi == 12:
            tr += 1
            if not wr:
                rec += 1; wr = True
            cr += 1
        else:
            tn += 1
            if wr:
                mcr = max(mcr, cr); cr = 0; wr = False
        vc = cmf[:3].copy(); yc = float(cmf[3])
        dyn.step_velocity_command(vc, yc, dts)
        stt = dyn.get_state()
        pw = stt.position_world.copy()
        pp.append(pw.copy())
        dst = float(np.linalg.norm(pw - gp))
        fd = dst
        if dst < md:
            md = dst
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
            "guide_x": float(gf[0]),
            "guide_y": float(gf[1]),
            "guide_z": float(gf[2]),
            "guide_distance_norm": float(gdn),
            "depth_min_m": float(np.min(depth_m)),
            "depth_mean_m": float(np.mean(depth_m)),
            "depth_near_1m_frac": float(np.mean(depth_m < 1.0)),
            "depth_near_2m_frac": float(np.mean(depth_m < 2.0)),
            "depth_near_3m_frac": float(np.mean(depth_m < 3.0)),
            "depth_near_4m_frac": float(np.mean(depth_m < 4.0)),
            "depth_left_near_4m_frac": float(np.mean(left_depth < 4.0)),
            "depth_center_near_4m_frac": float(np.mean(center_depth < 4.0)),
            "depth_right_near_4m_frac": float(np.mean(right_depth < 4.0)),
            "horizontal_index": hi,
            "vertical_index": int(out.vertical_index[0, 0].item()),
            "guide_value_raw": float(out.guide_value_raw[0, 0, 0].item()),
            "guide_value": float(out.guide_value[0, 0, 0].item()),
            "cmd_vx_flu": float(vc[0]),
            "cmd_vy_flu": float(vc[1]),
            "cmd_vz_flu": float(vc[2]),
            "cmd_yaw_rate": yc,
            "inference_ms": ims,
        }
        step_row.update({
            f"horizontal_prob_{index}": float(probability)
            for index, probability in enumerate(horizontal_prob)
        })
        step_row.update({
            f"vertical_prob_{index}": float(probability)
            for index, probability in enumerate(vertical_prob)
        })
        data_logger.write_step(step_row)
        if dst <= rollout_cfg.goal_tolerance_m and spd <= rollout_cfg.goal_speed_tolerance_mps:
            gh += 1
            if gh >= rollout_cfg.goal_hold_ticks:
                ot = "success"; break
        else:
            gh = 0
        if rollout_cfg.verbose and step % 30 == 0:
            hl = "REC_L" if hi == 0 else ("REC_R" if hi == 12 else "NORM")
            print(f"  [{ep_idx}:{step:04d}] dist={dst:.2f}m spd={spd:.2f}m/s | "
                  f"cmd=[{vc[0]:+.2f},{vc[1]:+.2f},{vc[2]:+.2f},{yc:+.2f}] | "
                  f"trend={hl} | infer={ims:.1f}ms | guide={gdm:.1f}m", flush=True)
        gid += 1
    if wr:
        mcr = max(mcr, cr)
    dur = (step + 1) * dts
    plen = float(sum(np.linalg.norm(pp[i]-pp[i-1]) for i in range(1, len(pp))))
    ai = float(np.mean(mts)) if mts else 0.0
    xi = float(np.max(mts)) if mts else 0.0
    ac = float(np.mean(cds)) if cds else 0.0
    tot = tn + tl + tr or 1
    res = EpisodeResult(
        episode=ep_idx, task_name=task.name, scene_id=scene_id, mode="fixed",
        outcome=ot, duration_s=dur, path_length_m=plen,
        final_goal_distance_m=fd, min_goal_distance_m=md,
        num_model_steps=step+1, num_collision_frames=cf, first_collision_step=fcs,
        avg_inference_ms=ai, max_inference_ms=xi,
        num_depth_timeouts=ndt, num_frame_mismatches=nfm,
        normal_fraction=tn/tot, recover_left_fraction=tl/tot,
        recover_right_fraction=tr/tot,
        recovery_entry_count=rec, max_consecutive_recovery=mcr,
        avg_command_delta=ac,
    )
    return res, gid + 1


def main() -> None:
    import os as _os  # noqa: F811  (used at function exit)
    p = argparse.ArgumentParser(
        description="Deterministic task-suite rollout for HierarchicalTrendControlPolicy")
    p.add_argument("--checkpoint")
    p.add_argument("--model-file")
    p.add_argument(
        "--tasks", default="all",
        help="Comma-separated task names, or 'all' (default).")
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
    validate_task_registry(task_registry, drone_radius=0.30, safety_margin=0.10)
    if a.list_tasks:
        print("Deterministic rollout tasks:")
        for task in task_registry.values():
            print(
                f"  {task.name:<16} obstacles={len(task.obstacles):>2}  "
                f"{task.description}")
        return
    if not a.checkpoint or not a.model_file:
        p.error("--checkpoint and --model-file are required unless --list-tasks is used")
    if a.repeats <= 0:
        p.error("--repeats must be > 0")
    if a.render_warmup_frames < 1:
        p.error("--render-warmup-frames must be >= 1")
    if a.tasks.strip().lower() == "all":
        selected_tasks = list(task_registry.values())
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
        model_hz=a.model_hz, ctrl_hz=a.ctrl_hz, max_yaw_rate=a.max_yaw_rate,
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
    MC, CC = _load_model_from_file(cf.model_file)
    try:
        ckpt = torch.load(cf.checkpoint, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(cf.checkpoint, map_location=dev)
    cd = ckpt.get("model_config") or ckpt.get("config") or {}
    mc = CC(**cd) if cd else CC()
    mc.validate()
    model = MC(mc)
    sd = ckpt["model_state_dict"]
    model.load_state_dict({
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in sd.items()
    }, strict=True)
    model.to(device=dev); model.eval()
    # CUDA warmup: first inference is slow (kernel compilation).
    with torch.no_grad():
        _ = model.forward_step(
            depth=torch.zeros(1, 1, mc.image_size[0], mc.image_size[1], device=dev, dtype=torch.float32),
            raw_guide=torch.zeros(1, 4, device=dev, dtype=torch.float32),
            gravity_flu=torch.zeros(1, 3, device=dev, dtype=torch.float32),
            velocity_flu=torch.zeros(1, 3, device=dev, dtype=torch.float32),
            yaw_rate=torch.zeros(1, 1, device=dev, dtype=torch.float32),
        )
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print("[rollout] CUDA warmup complete.")
    total_episodes = len(selected_tasks) * cf.repeats
    print("="*60)
    print("Policy Rollout - HierarchicalTrendControlPolicy")
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
    print(f"  Depth:       {cf.depth_width}x{cf.depth_height} max={cf.depth_max_m}m")
    print(f"  Render warm: {cf.render_warmup_frames} discarded frames/episode")
    print(f"  Goal:        {cf.goal_tolerance_m}m <={cf.goal_speed_tolerance_mps}m/s x{cf.goal_hold_ticks}")
    print(f"  Ports:       PUB={cf.pub_port} SUB={cf.sub_port}")
    print(f"  Params:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model image: {mc.image_size}")
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
                normal_fraction=0, recover_left_fraction=0, recover_right_fraction=0,
                recovery_entry_count=0, max_consecutive_recovery=0,
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
              f"rec_entries={result.recovery_entry_count} "
              f"max_cons={result.max_consecutive_recovery} | "
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
    an = np.mean([r.normal_fraction for r in results]) if results else 0
    al = np.mean([r.recover_left_fraction for r in results]) if results else 0
    ar = np.mean([r.recover_right_fraction for r in results]) if results else 0
    ae = np.mean([r.recovery_entry_count for r in results]) if results else 0
    print(f"\n  Trend:   normal={an:.1%} rec_left={al:.1%} rec_right={ar:.1%} entries={ae:.1f}")
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
