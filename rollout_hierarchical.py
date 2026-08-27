#!/usr/bin/env python3
"""Closed-loop hierarchical rollout: C++ 5 Hz expert (upper) + 30 Hz student (lower).

This script mirrors the ``rollout.py`` interface (task registry, Unity bridge,
dynamics, telemetry logger, CLI) but runs the TWO-RATE hierarchical stack
exactly as the schema-v25 collector does:

  * UPPER (5 Hz): the real C++ ``HierarchicalExpert`` (pybind
    ``_il_hierarchical_expert``).  Its macro corrector decides every
    ``tick % 6 == 0`` whether to PASS, NORMAL-correct or TURN, world-latches
    the chosen corrected target, and its effective-target adapter re-expresses
    the goal in the live 30 Hz body frame every tick.  The expert is fully
    stateful (FSM / history map / corridor diagnostics) and consumes the same
    metre-valued depth frames the collector feeds it.
  * LOWER (30 Hz): the trained schema-v25 ``ViTFlyLSTMPolicy`` student.  It
    receives the depth frame plus the 7-D state whose goal part is the
    expert's *effective* target (``goal_direction_flu_*`` /
    ``goal_distance_norm`` from ``ExpertStepOutput``) and regresses the
    body-FLU velocity + yaw-rate command that the dynamics executes.

So this is an end-to-end closed-loop test of the deployed hierarchy:
"upper expert decides where to fly, lower student flies there".

Task handling mirrors ``rollout.py``: deterministic obstacle layouts with
fixed endpoints.  Tasks that carry ``goal_updates`` drive the expert's
``accept_new_goal`` at 5 Hz boundaries (exactly what the joint_v2 runtime
adapter does when the mission goal changes), while fixed-goal tasks keep the
original navigation goal constant.

Examples:
    python3 rollout_hierarchical.py --list-tasks
    python3 rollout_hierarchical.py \
        --checkpoint checkpoints/vitfly_v25_joint_v2_1/best.pt \
        --expert-config ../il_dataset/config/il_dataset_joint_v2_config.yaml \
        --tasks basic
    python3 rollout_hierarchical.py \
        --checkpoint best.pt --model-file model/model.py \
        --expert-config ../il_dataset/config/il_dataset_joint_v2_config.yaml \
        --tasks clear_straight,forced_left,continuous_5hz
"""

import argparse
import importlib.util
import json
import math
import random
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

# Reuse the single-policy rollout machinery (task registry, bridge handling,
# depth/state preprocessing, telemetry, checkpoint loader).
from rollout import (  # noqa: E402
    DEFAULT_STATE_SCALE,
    Cylinder,
    EpisodeResult,
    RolloutDataLogger,
    RolloutTask,
    body_clearance,
    build_normalized_state,
    canonicalize_unity_depth,
    load_policy_checkpoint,
    preprocess_depth,
    task_to_unity_objects,
    validate_task_registry,
    _build_dynamics_config,
    _json_safe,
    _yaw_from_quat_xyzw,
)

try:
    import il_common
    import il_dynamics
    import il_config
    import il_expert_config
    import _il_hierarchical_expert as expert_mod
except ImportError as e:  # pragma: no cover
    il_common = None
    il_dynamics = None
    il_config = None
    il_expert_config = None
    expert_mod = None
    _EXPERT_IMPORT_ERROR: Optional[ImportError] = e
else:
    _EXPERT_IMPORT_ERROR = None


class HierarchicalDataLogger(RolloutDataLogger):
    """Step telemetry writer with the hierarchical expert columns appended.

    The base ``RolloutDataLogger`` writes the single-policy column set; this
    subclass extends the header with the C++ expert's per-tick diagnostics
    (effective-target source, hierarchical mode, planner status, macro label
    and the exact effective goal fed to the 30 Hz student).
    """

    _HI_COLUMNS = [
        "episode", "task", "step", "global_frame", "sim_time_s",
        "state_x", "state_y", "state_z", "state_yaw",
        "speed_world_mps", "yaw_rate_rps", "goal_distance_m",
        "effective_target_source", "hierarchical_mode", "planner_status",
        "macro_update_mask", "macro_correction_type",
        "expert_goal_dir_x", "expert_goal_dir_y", "expert_goal_dir_z",
        "expert_goal_dist_norm",
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
    COLUMNS = _HI_COLUMNS


# ============================================================================
#  Schema-v25 episode writer (interactive_trajectory_debug.py compatible)
# ============================================================================

# Column set consumed by il_dataset/test/interactive_trajectory_debug.py
# (plus the roll-back schema-v25 fields).  One ``data.csv`` per episode is
# written into a per-episode subdirectory under ``episode_out_root``.
SCHEMA25_ROLLOUT_COLUMNS = [
    "episode_frame_index", "trajectory_time_s", "control_dt_s",
    "x", "y", "z", "yaw", "yaw_rate",
    "state_vx_flu", "state_vy_flu", "state_vz_flu",
    "speed_world_mps", "inference_ms",
    "effective_target_world_x", "effective_target_world_y",
    "effective_target_world_z",
    "goal_direction_flu_x", "goal_direction_flu_y",
    "goal_direction_flu_z", "goal_distance_norm",
    # Expert 30 Hz executable command (the training label / ground truth).
    "target_velocity_flu_x", "target_velocity_flu_y",
    "target_velocity_flu_z", "target_yaw_rate",
    # Student (30 Hz model) command actually executed by the dynamics.
    "student_velocity_flu_x", "student_velocity_flu_y",
    "student_velocity_flu_z", "student_yaw_rate",
    "hierarchical_mode", "planner_status", "planner_failure_reason",
    "plan_valid", "plan_terminal", "plan_points_xy",
    "macro_update_mask", "macro_label_valid", "macro_correction_type",
    "macro_direction_token",
    "macro_direction_flu_x", "macro_direction_flu_y",
    "macro_direction_flu_z", "macro_distance_norm",
    "min_observed_clearance_m",
    "truth_minimum_clearance_m", "truth_brake_risk",
    "truth_brake_would_trigger",
    "emergency_brake", "local_corridor_blocked",
    "fsm_state", "effective_target_source", "target_correction_active",
    "observability_reason",
    "scene_id", "task_id",
    "navigation_goal_world_x", "navigation_goal_world_y",
    "navigation_goal_world_z",
    "original_navigation_goal_world_x", "original_navigation_goal_world_y",
    "original_navigation_goal_world_z",
    "episode_valid", "failure_taxonomy",
]


class SchemaV25EpisodeWriter:
    """Write one episode's roll-out as a schema-v25 ``data.csv`` so the
    interactive top-down debugger can step through every tick.

    The roll-out replays the deployed hierarchy: the 30 Hz student executes
    the C++ expert's effective target.  ``data.csv`` uses the collector's
    schema-v25 column names so ``interactive_trajectory_debug.py`` renders
    trajectory / plan / macro / safety fields without modification.
    """

    def __init__(self, episode_dir, scene_id, task_id,
                 navigation_goal_world, initial_yaw):
        import csv as _csv
        self._csv = _csv
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.episode_dir / "data.csv"
        self._handle = self.path.open(
            "w", newline="", encoding="utf-8", buffering=1)
        self._writer = _csv.DictWriter(
            self._handle, fieldnames=SCHEMA25_ROLLOUT_COLUMNS,
            extrasaction="raise")
        self._writer.writeheader()
        self._handle.flush()
        self.scene_id = int(scene_id)
        self.task_id = int(task_id)
        self._nav = [float(v) for v in navigation_goal_world]
        self.initial_yaw = float(initial_yaw)
        self.episode_valid = 1
        self.failure_taxonomy = ""
        self._row_count = 0

    def configure_truth(self, obstacles, vehicle_radius,
                        min_bounds, max_bounds):
        """Create the exact-cylinder truth audit (same as the collector)."""
        if expert_mod is None:
            self._truth = None
            return
        self._truth = expert_mod.TruthCylinderAudit()
        obs = [[float(o.x), float(o.y), float(o.radius), float(o.height)]
               for o in obstacles]
        self._truth.configure(obs, float(vehicle_radius),
                              [float(v) for v in min_bounds],
                              [float(v) for v in max_bounds])

    def write_row(self, step, dts, pw, qq, stt, yaw, yr, spd, inference_ms,
                  eout, vc, yc, gdn, nav_dir, obs_clearance,
                  truth_cfg, task_goal):
        """Write one schema-v25 row for the current tick.

        ``truth_cfg`` is a dict with the brake/audit parameters
        (eff_accel, stop_margin) or None to skip truth computation.
        """
        row = {
            "episode_frame_index": int(step),
            "trajectory_time_s": round((step + 1) * dts, 6),
            "control_dt_s": round(dts, 6),
            "x": float(pw[0]), "y": float(pw[1]), "z": float(pw[2]),
            "yaw": float(yaw), "yaw_rate": float(yr),
            "state_vx_flu": float(stt.velocity_flu[0]),
            "state_vy_flu": float(stt.velocity_flu[1]),
            "state_vz_flu": float(stt.velocity_flu[2]),
            "speed_world_mps": float(spd),
            "inference_ms": float(inference_ms),
            "effective_target_world_x": float(eout.effective_target_world_x),
            "effective_target_world_y": float(eout.effective_target_world_y),
            "effective_target_world_z": float(eout.effective_target_world_z),
            "goal_direction_flu_x": float(eout.goal_direction_flu_x),
            "goal_direction_flu_y": float(eout.goal_direction_flu_y),
            "goal_direction_flu_z": float(eout.goal_direction_flu_z),
            "goal_distance_norm": float(gdn),
            # Expert 30 Hz executable command (the ground-truth label).
            "target_velocity_flu_x": float(eout.target_velocity_flu_x),
            "target_velocity_flu_y": float(eout.target_velocity_flu_y),
            "target_velocity_flu_z": float(eout.target_velocity_flu_z),
            "target_yaw_rate": float(eout.target_yaw_rate),
            # Student (30 Hz model) command actually executed.
            "student_velocity_flu_x": float(vc[0]),
            "student_velocity_flu_y": float(vc[1]),
            "student_velocity_flu_z": float(vc[2]),
            "student_yaw_rate": float(yc),
            "hierarchical_mode": str(eout.hierarchical_mode),
            "planner_status": str(eout.planner_status),
            "planner_failure_reason": str(eout.failure_reason),
            "plan_valid": int(eout.plan_valid),
            "plan_terminal": int(eout.plan_terminal),
            "plan_points_xy": _encode_plan_points(eout.plan_points_x,
                                                  eout.plan_points_y),
            "macro_update_mask": int(eout.macro_update_mask),
            "macro_label_valid": int(eout.macro_label_valid),
            "macro_correction_type": str(eout.macro_correction_type),
            "macro_direction_token": int(eout.macro_direction_token),
            "macro_direction_flu_x": float(eout.macro_direction_flu_x),
            "macro_direction_flu_y": float(eout.macro_direction_flu_y),
            "macro_direction_flu_z": float(eout.macro_direction_flu_z),
            "macro_distance_norm": float(eout.macro_distance_norm),
            "min_observed_clearance_m": float(eout.min_observed_clearance_m),
            "emergency_brake": int(eout.emergency_brake),
            "local_corridor_blocked": int(eout.local_corridor_blocked),
            "fsm_state": str(eout.fsm_state),
            "effective_target_source": str(eout.effective_target_source),
            "target_correction_active": int(eout.target_correction_active),
            "observability_reason": str(eout.observability_reason),
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "navigation_goal_world_x": self._nav[0],
            "navigation_goal_world_y": self._nav[1],
            "navigation_goal_world_z": self._nav[2],
            "original_navigation_goal_world_x": self._nav[0],
            "original_navigation_goal_world_y": self._nav[1],
            "original_navigation_goal_world_z": self._nav[2],
            "episode_valid": self.episode_valid,
            "failure_taxonomy": self.failure_taxonomy,
        }
        # ── Truth audit (judge-only; exact cylinders) ─────────────
        truth_clear = float("nan")
        truth_risk = float("nan")
        truth_would = ""
        if getattr(self, "_truth", None) is not None and truth_cfg:
            truth_clear = self._truth.segment_min_clearance(
                float(pw[0]), float(pw[1]),
                float(pw[0]), float(pw[1]))
            speed2 = float(np.linalg.norm(stt.velocity_world[0:2]))
            risk, would = self._truth.brake_risk(
                float(pw[0]), float(pw[1]),
                float(stt.velocity_world[0]), float(stt.velocity_world[1]),
                float(truth_cfg["eff_accel"]),
                float(truth_cfg["stop_margin"]))
            truth_risk = float(risk)
            truth_would = int(bool(would))
        row["truth_minimum_clearance_m"] = truth_clear
        row["truth_brake_risk"] = truth_risk
        row["truth_brake_would_trigger"] = truth_would
        self._writer.writerow(row)
        self._handle.flush()
        self._row_count += 1

    def finalize(self, episode_valid, failure_taxonomy, outcome):
        """Set the per-episode fields, close, and write metadata."""
        self.episode_valid = int(episode_valid)
        self.failure_taxonomy = str(failure_taxonomy)
        # Rewrite rows is costly; instead we record the summary in a sidecar
        # (the debugger only reads data.csv rows; per-episode fields are set
        # on the LAST row by reopening is not needed for the viewer).
        self._handle.flush()
        self._handle.close()
        metadata = {
            "episode_valid": self.episode_valid,
            "failure_taxonomy": self.failure_taxonomy,
            "outcome": str(outcome),
            "rows": self._row_count,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "navigation_goal_world": self._nav,
            "initial_yaw": self.initial_yaw,
        }
        meta_path = self.episode_dir / "rollout_metadata.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=1)


def _encode_plan_points(px, py):
    """Encode world-XY plan points as 'x1,y1;x2,y2;...' (collector format)."""
    try:
        xs = list(px or [])
        ys = list(py or [])
    except TypeError:
        return ""
    if not xs or not ys or len(xs) != len(ys):
        return ""
    return ";".join("%.3f,%.3f" % (xs[i], ys[i]) for i in range(len(xs)))


# ============================================================================
#  Avoidance test scenes — EXACT mirror of gen_avoid_scenes.py
#  Every task's straight start->goal line provably crosses an obstacle CORE
#  (or threads the S_gap narrow passage), so a successful flight MUST detour.
#  The scene/task data and the deterministic initial-yaw sampling are reused
#  verbatim from il_dataset/gen_avoid_scenes.py (single source of truth).
#  The generator is currently kept under il_dataset/test, but the legacy
#  package-root location remains supported for older workspaces, so the
#  rollout exercises exactly the handcrafted collection layout.
#
#  A second generator, gen_avoid_scenes_4level.py, mirrors the scene_parallel
#  COLLECTION recipe (small / medium / large / mixed — 2 scenes x 2 tasks per
#  level = 8 scenes / 16 tasks) and is exposed via ``--tasks 4level``.
# ============================================================================

_IL_DATASET_DIR = _THIS_DIR.parent / "il_dataset"
_AVOID_SCENE_PATHS = (
    _IL_DATASET_DIR / "gen_avoid_scenes.py",
    _IL_DATASET_DIR / "test" / "gen_avoid_scenes.py",
)
_FOURLEVEL_SCENE_PATHS = (
    _IL_DATASET_DIR / "gen_avoid_scenes_4level.py",
    _IL_DATASET_DIR / "test" / "gen_avoid_scenes_4level.py",
)


def _load_scene_module(module_name, paths):
    """Import a handcrafted scene generator by path (single source)."""
    scene_path = next((path for path in paths if path.is_file()), None)
    if scene_path is None:
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(scene_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_avoid = _load_scene_module(
    "_il_dataset_gen_avoid_scenes", _AVOID_SCENE_PATHS)
_4level = _load_scene_module(
    "_il_dataset_gen_avoid_scenes_4level", _FOURLEVEL_SCENE_PATHS)


def _scene_initial_yaw(module, task, scene_index, rng):
    """Deterministic per-scene initial yaw identical to the generator.

    ``sample_initial_yaw`` returns FM convention B (expert_yaw - pi/2), the
    same convention rollout.py uses for ``forward_yaw``.
    """
    sx, sy, gx, gy, _label = task
    goal_bearing_expert = math.atan2(gy - sy, gx - sx)
    return module.sample_initial_yaw(goal_bearing_expert, rng)


def _build_scene_task_registry(module, suite: str) -> Dict[str, RolloutTask]:
    """Build one ``RolloutTask`` per start/goal pair from a handcrafted
    scene generator module (``.SCENES`` list of dicts).  The initial yaw uses
    the same deterministic per-scene RNG seed as the collector blueprint so
    the rollout replicates the collection initial conditions."""
    if module is None:  # pragma: no cover
        raise RuntimeError(
            "Cannot import scene generator module for suite '%s'" % suite)
    tasks: Dict[str, RolloutTask] = {}
    for scene_index, sc in enumerate(module.SCENES):
        rng = random.Random(20260824 + scene_index * 7919)
        obstacles = tuple(
            Cylinder(float(o[0]), float(o[1]), float(o[2]))
            for o in sc["obstacles"])
        for task in sc["tasks"]:
            sx, sy, gx, gy, label = task
            start_yaw = _scene_initial_yaw(module, task, scene_index, rng)
            name = "{}".format(label)
            description = (
                "{}: start->goal line crosses obstacle core -> must detour"
                .format(sc["name"]))
            tasks[name] = RolloutTask(
                name=name,
                description=description,
                start=(float(sx), float(sy), 2.0),
                goal=(float(gx), float(gy), 2.0),
                start_yaw=start_yaw,
                obstacles=obstacles,
                suite=suite,
                scene_id=scene_index,
            )
    return tasks


def build_avoid_task_registry() -> Dict[str, RolloutTask]:
    """Handcrafted avoidance tasks (gen_avoid_scenes layout)."""
    return _build_scene_task_registry(_avoid, "avoid")


def build_4level_task_registry() -> Dict[str, RolloutTask]:
    """4-level scene_parallel-mirror tasks (gen_avoid_scenes_4level layout):
    small / medium / large / mixed, 2 scenes x 2 tasks per level."""
    return _build_scene_task_registry(_4level, "4level")


_TASK_REGISTRY_BUILDERS = {
    "avoid": build_avoid_task_registry,
    "4level": build_4level_task_registry,
}


@dataclass
class HierarchicalRolloutConfig:
    """Upper/lower stack knobs on top of the single-policy RolloutConfig."""

    checkpoint: str = ""
    model_file: str = ""
    expert_config: str = ""
    pub_port: str = "10253"
    sub_port: str = "10254"
    scene_id: int = 1
    depth_width: int = 640
    depth_height: int = 480
    depth_fov: float = 90.0
    depth_near: float = 0.01
    depth_far: float = 1000.0
    depth_max_m: float = 5.0
    # The authoritative camera->body matrix (16 floats row-major) from the
    # expert YAML's global.depth.t_bc; used by every make_depth_vehicle call.
    depth_t_bc: Tuple[float, ...] = ()
    model_hz: float = 30.0
    ctrl_hz: float = 50.0
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
    log_prefix: str = "rollout_hi_latest"
    render_warmup_frames: int = 5
    flight_z: float = 2.0
    # Frame-sync retry policy (mirrors il_manager: frame_match_timeout_s +
    # max_frame_retries).  A single render can occasionally exceed the per-
    # attempt window; the collector retries with fresh frame ids instead of
    # aborting the episode, so the rollout must too (else transient render
    # jitter is misread as a policy failure — measured 14/14 errors were
    # dto=1 depth timeouts while the flight was progressing normally).
    frame_match_timeout_s: float = 0.15
    max_frame_retries: int = 5
    # Directory where per-episode schema-v25 ``data.csv`` files are written
    # (one subdirectory per episode, loadable by
    # il_dataset/test/interactive_trajectory_debug.py).  Empty disables.
    episode_out_root: str = ""


def load_expert_stack(expert_config: str):
    """Load the C++ HierarchicalExpert configured from the YAML file.

    Returns (expert, params, min_bounds, max_bounds, depth_cfg).  The expert
    is the ONE stateful instance; reset it with ``reset_task`` before every
    episode.  ``depth_cfg`` is the authoritative ``global.depth`` dict from
    the same YAML (the loader injects the unique ``t_bc`` camera->body
    matrix that BOTH the Unity wire and the C++ camera rig read), so every
    ``make_depth_vehicle`` call uses the single validated matrix.
    """
    if _EXPERT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import the C++ expert modules: {}. Expected path: {} "
            "(source devel/setup.bash; the .so must be built with "
            "`catkin build il_dataset`).".format(
                _EXPERT_IMPORT_ERROR, _IL_SCRIPTS))
    cfg_path = Path(expert_config).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError("Expert config not found: {}".format(cfg_path))
    config = il_config.load_config(str(cfg_path))
    global_cfg = config["global"]
    errors: List[str] = []
    params = il_expert_config.build_params(global_cfg, errors)
    if errors:
        raise ValueError("hierarchical_expert config errors:\n  - " +
                         "\n  - ".join(errors))
    min_b, max_b = il_expert_config.build_scene_bounds(global_cfg)
    depth_cfg = dict(global_cfg.get("depth", {}) or {})
    if not depth_cfg.get("t_bc"):
        raise ValueError(
            "global.depth.t_bc missing from {}: the config loader must "
            "inject the single camera->body matrix".format(cfg_path))
    expert = expert_mod.HierarchicalExpert()
    revision = getattr(expert_mod, "EXPERT_REVISION",
                       "<no EXPERT_REVISION — STALE .so>")
    expert.configure(params, list(min_b), list(max_b))
    return expert, params, min_b, max_b, depth_cfg


def run_hierarchical_rollout(
    student, student_cfg, expert, params, hr_cfg: HierarchicalRolloutConfig,
    device: torch.device, ep_idx: int, gid: int, bridge, dyn, task: RolloutTask,
    scene_id: int, obstacles: List[Dict[str, Any]],
    data_logger: RolloutDataLogger,
    schema_writer: Optional[SchemaV25EpisodeWriter] = None,
    truth_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[EpisodeResult, int]:
    """One closed-loop episode: C++ expert decides the target, the 30 Hz
    student executes.  Returns (EpisodeResult, next_global_frame_id).

    ``schema_writer`` (optional) receives each tick as a schema-v25 row for
    the interactive trajectory debugger; ``truth_cfg`` enables the exact-
    cylinder truth audit on those rows.
    """
    dts = 1.0 / hr_cfg.model_hz
    episode_time = (
        task.max_episode_time
        if task.max_episode_time is not None
        else hr_cfg.max_episode_time
    )
    mx = int(episode_time * hr_cfg.model_hz)
    dc = {
        "width": hr_cfg.depth_width, "height": hr_cfg.depth_height,
        "fov": hr_cfg.depth_fov, "near": hr_cfg.depth_near,
        "far": hr_cfg.depth_far,
        "t_bc": list(hr_cfg.depth_t_bc),
    }
    ih, iw = hr_cfg.depth_height, hr_cfg.depth_width
    mr = hr_cfg.depth_max_m
    sp = np.asarray(task.start, dtype=np.float64)
    gp = np.asarray(task.goal, dtype=np.float64)
    syaw = float(task.start_yaw)
    flight_z = float(hr_cfg.flight_z)
    tick_base = 0  # 5 Hz macro decisions land on tick % 6 == 0 (tick 0,6,12,...)

    # Reset the C++ expert for this episode (fresh FSM / history map).
    expert.reset_task([float(sp[0]), float(sp[1])],
                      [float(gp[0]), float(gp[1])], syaw, tick_base, flight_z)

    dyn.reset(sp.copy(), syaw, np.zeros(3), np.zeros(3))
    veh = il_common.make_depth_vehicle(
        ros_pos=sp.copy().tolist(), yaw=syaw, depth_cfg=dc,
    )
    st: Dict[str, Any] = {
        "scene_id": scene_id, "frame_id": gid,
        "vehicles": [veh], "objects": obstacles,
    }
    bridge.send_pose(st)
    gid += 1
    time.sleep(0.5)

    # Renderer warm-up (same as rollout.py): drain stale responses, then
    # discard several uniquely identified renders while the vehicle rests.
    try:
        while bridge.try_recv() is not None:
            pass
    except Exception:
        pass
    depth_payload_bytes = iw * ih * 4
    for warmup_index in range(hr_cfg.render_warmup_frames):
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
                    raw_depth, dtype=np.float32).reshape((ih, iw))
                warm_depth, _ = canonicalize_unity_depth(depth_float, mr)
                break
            if warm_depth is not None:
                break
        if warm_depth is None:
            raise RuntimeError(
                f"Task {task.name}: Unity render warmup frame "
                f"{warmup_index + 1}/{hr_cfg.render_warmup_frames} timed out")
        gid += 1

    if hr_cfg.verbose:
        warm_depth_m = warm_depth.astype(np.float32) * 0.01
        print(
            f"  [rollout-hi] render warmup complete: frames="
            f"{hr_cfg.render_warmup_frames} "
            f"depth_min={np.min(warm_depth_m):.2f}m "
            f"depth_mean={np.mean(warm_depth_m):.2f}m",
            flush=True,
        )

    dt = torch.float32
    hidden = student.initial_hidden(1, device=device, dtype=dt)
    rollout_start_position = dyn.get_state().position_world.copy()
    pp: List[np.ndarray] = [rollout_start_position]
    cf = 0
    fcs = -1
    mts: List[float] = []
    ndt = 0
    nfm = 0
    pc: Optional[np.ndarray] = None
    cds: List[float] = []
    md = float(np.linalg.norm(rollout_start_position - gp))
    fd = md
    ot = "timeout"
    goal_switch_count = 0
    goal_schedule_index = -1
    minimum_clearance = body_clearance(
        rollout_start_position, task, hr_cfg.drone_radius)
    macro_types: List[str] = []
    last_directive_type = ""
    gh = 0
    applied_updates = -1  # index of the last applied goal_update (task goal)

    for step in range(mx):
        if (
            hr_cfg.lstm_reset_interval > 0
            and step > 0
            and step % hr_cfg.lstm_reset_interval == 0
        ):
            hidden = student.initial_hidden(1, device=device, dtype=dt)

        stt = dyn.get_state()
        pw = stt.position_world.copy()
        qq = stt.quaternion_world_body.copy()
        yaw = _yaw_from_quat_xyzw(qq)
        vel = stt.velocity_world.copy()
        yaw_rate_body = float(stt.angular_velocity_body[2])
        veh = il_common.make_depth_vehicle(
            ros_pos=pw.tolist(), yaw=yaw, depth_cfg=dc,
            quaternion_xyzw=qq.tolist(),
        )
        st["frame_id"] = gid
        st["vehicles"] = [veh]
        bridge.send_pose(st)
        du = None
        col = False
        dfl = iw * ih * 4
        # ── Strict frame synchronisation WITH retries (collector contract) ──
        # On a mismatch / timeout we re-send with a NEW frame id (never reuse
        # the timed-out id) and retry up to max_frame_retries.  Only after all
        # attempts fail do we abort the episode as a depth timeout.
        for attempt in range(hr_cfg.max_frame_retries + 1):
            if attempt > 0:
                # Fresh frame id for the retry (a timed-out id is dead).
                gid += 1
                st["frame_id"] = gid
                st["vehicles"] = [veh]
                bridge.send_pose(st)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < hr_cfg.frame_match_timeout_s:
                r = bridge.try_recv()
                if r is None:
                    time.sleep(0.002)
                    continue
                md_, rp = r
                fid = md_.get("pub_frame_id")
                if fid is None:
                    fid = md_.get("frame_id", -1)
                if fid != gid:
                    nfm += 1
                    if nfm <= 3 or nfm % 100 == 0:
                        print(
                            f"  [rollout-hi] frame ID mismatch: expected={gid} "
                            f"got={fid} (nfm={nfm})", flush=True)
                    continue
                for pt in rp:
                    if len(pt) >= dfl:
                        raw = pt[:dfl]
                        df = np.frombuffer(raw, dtype=np.float32).reshape((ih, iw))
                        du, depth_normalized = canonicalize_unity_depth(df, mr)
                        break
                vs = md_.get("pub_vehicles", [])
                if vs and vs[0].get("collision", False):
                    col = True
                break
            if du is not None:
                break  # matched depth on this attempt
            if hr_cfg.verbose and attempt < hr_cfg.max_frame_retries:
                print(
                    f"  [rollout-hi] Ep {ep_idx} step {step}: frame match "
                    f"timeout on attempt {attempt + 1}/"
                    f"{hr_cfg.max_frame_retries + 1}; retrying", flush=True)
        if col:
            cf += 1
            if fcs < 0:
                fcs = step
            if cf >= hr_cfg.collision_confirm_frames:
                ot = "collision"
                break
        if du is None:
            ndt += 1
            ot = "error"
            warnings.warn(
                f"[rollout-hi] Ep {ep_idx} step {step}: depth timeout after "
                f"{hr_cfg.max_frame_retries + 1} render attempts")
            break

        sim_time_s = step * dts
        # ── UPPER: C++ expert step (30 Hz tick; macro decides on %6==0) ──
        # CRITICAL: feed the expert the RAW flipud depth (AvoidBench payload
        # * 100.0) with invalid samples LEFT as 0 / NaN / huge values — the
        # C++ observation builder treats non-finite / ~0 as UNKNOWN and
        # > range as valid no-hit.  canonicalize_unity_depth() instead fills
        # invalid samples with max_depth_m (5.0 m), which the expert reads as
        # a REAL obstacle at the range boundary → phantom walls → spurious
        # macro NORMAL/TURN takeovers (measured: takeover rate 13.4% vs 5.2%
        # in real collection data).  The student still gets the canonicalized
        # depth (matches training normalisation).
        expert_depth = np.flipud(df.astype(np.float64) * 100.0)
        try:
            eout = expert.step(
                [float(pw[0]), float(pw[1]), float(pw[2])], yaw,
                [float(vel[0]), float(vel[1]), float(vel[2])], yaw_rate_body,
                np.ascontiguousarray(expert_depth, dtype=np.float32).ravel(),
                int(hr_cfg.depth_width), int(hr_cfg.depth_height),
                [float(pw[0]), float(pw[1]), float(pw[2])],
                [float(qq[0]), float(qq[1]), float(qq[2]), float(qq[3])],
                flight_z, int(tick_base + step), col)
        except Exception as exc:  # noqa: BLE001
            ot = "error"
            warnings.warn(
                f"[rollout-hi] Ep {ep_idx} step {step}: expert.step error: "
                f"{exc}")
            break

        # ── Task goal updates (only meaningful at a 5 Hz boundary) ──
        # ``accept_new_goal`` is a mission-level goal revision; joint_v2 keeps
        # it fixed per episode, and the rollout goal_updates tasks exercise
        # the expert's re-planning on a boundary.  Mirror rollout.py's
        # active_goal_for_time() so only the CURRENT schedule entry is applied.
        if eout.macro_update_mask and task.goal_updates:
            sim_time_now = (step + 1) * dts
            schedule_index = applied_updates
            for candidate in range(applied_updates + 1, len(task.goal_updates)):
                if task.goal_updates[candidate][0] <= sim_time_now + 1e-9:
                    schedule_index = candidate
                else:
                    break
            if schedule_index > applied_updates:
                update_time, update_goal = task.goal_updates[schedule_index]
                expert.accept_new_goal(
                    [float(update_goal[0]), float(update_goal[1])],
                    int(tick_base + step))
                goal_switch_count += 1
                applied_updates = schedule_index

        # ── LOWER: 30 Hz student consumes the expert's effective target ──
        gf = np.array(
            [float(eout.goal_direction_flu_x),
             float(eout.goal_direction_flu_y),
             float(eout.goal_direction_flu_z)], dtype=np.float64)
        gdn = float(eout.goal_distance_norm)
        if np.linalg.norm(gf) < 1e-8:
            gf = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gf = gf / float(np.linalg.norm(gf))

        depth_m_frame = du.astype(np.float32) * 0.01
        left = depth_m_frame[:, : iw // 3]
        center = depth_m_frame[:, iw // 3: 2 * iw // 3]
        right = depth_m_frame[:, 2 * iw // 3:]

        dt_ = preprocess_depth(depth_normalized, device)
        grav_flu = il_common.world_vector_to_body_flu_quat(
            np.array([0.0, 0.0, -1.0], dtype=np.float64), qq)
        omega_body = np.asarray(stt.angular_velocity_body, dtype=np.float32)
        omega_flu = np.array(
            [omega_body[1], -omega_body[0], omega_body[2]], dtype=np.float32)
        yr = float(omega_flu[2])
        state_tensor = build_normalized_state(
            grav_flu, gf, gdn,
            hr_cfg.state_scale, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ti = time.perf_counter()
        with torch.no_grad():
            out = student.step(dt_, state_tensor, hidden)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ims = (time.perf_counter() - ti) * 1000.0
        mts.append(ims)
        hidden = tuple(value.detach() for value in out.hidden)
        cmf = out.command[0].cpu().numpy().copy()
        normalized_command = out.normalized_command[0].cpu().numpy().copy()
        if pc is not None:
            cds.append(float(np.linalg.norm(cmf - pc)))
        pc = cmf.copy()
        vc = cmf[:3].copy()
        yc = float(cmf[3])
        dyn.step_velocity_command(vc, yc, dts)

        # Record the expert's macro directive type on decision frames.
        if eout.macro_update_mask:
            macro_types.append(eout.macro_correction_type)
            last_directive_type = eout.macro_correction_type

        stt = dyn.get_state()
        pw = stt.position_world.copy()
        pp.append(pw.copy())
        dst = float(np.linalg.norm(pw - gp))
        fd = dst
        if dst < md:
            md = dst
        clearance = body_clearance(pw, task, hr_cfg.drone_radius)
        minimum_clearance = min(minimum_clearance, clearance)
        spd = float(np.linalg.norm(stt.velocity_world))

        step_row: Dict[str, Any] = {
            "episode": ep_idx,
            "task": task.name,
            "step": step,
            "global_frame": gid,
            "sim_time_s": (step + 1) * dts,
            "state_x": float(pw[0]),
            "state_y": float(pw[1]),
            "state_z": float(pw[2]),
            "state_yaw": float(yaw),
            "speed_world_mps": spd,
            "yaw_rate_rps": yr,
            "goal_distance_m": dst,
            "effective_target_source": eout.effective_target_source,
            "hierarchical_mode": eout.hierarchical_mode,
            "planner_status": eout.planner_status,
            "macro_update_mask": int(eout.macro_update_mask),
            "macro_correction_type": eout.macro_correction_type,
            "expert_goal_dir_x": float(eout.goal_direction_flu_x),
            "expert_goal_dir_y": float(eout.goal_direction_flu_y),
            "expert_goal_dir_z": float(eout.goal_direction_flu_z),
            "expert_goal_dist_norm": gdn,
            "guide_x": float(gf[0]),
            "guide_y": float(gf[1]),
            "guide_z": float(gf[2]),
            "guide_distance_norm": gdn,
            "state_gravity_x": float(grav_flu[0]),
            "state_gravity_y": float(grav_flu[1]),
            "state_gravity_z": float(grav_flu[2]),
            "minimum_body_clearance_m": clearance,
            "depth_min_m": float(np.min(depth_m_frame)),
            "depth_mean_m": float(np.mean(depth_m_frame)),
            "depth_near_1m_frac": float(np.mean(depth_m_frame < 1.0)),
            "depth_near_2m_frac": float(np.mean(depth_m_frame < 2.0)),
            "depth_near_3m_frac": float(np.mean(depth_m_frame < 3.0)),
            "depth_near_4m_frac": float(np.mean(depth_m_frame < 4.0)),
            "depth_left_near_4m_frac": float(np.mean(left < 4.0)),
            "depth_center_near_4m_frac": float(np.mean(center < 4.0)),
            "depth_right_near_4m_frac": float(np.mean(right < 4.0)),
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
        if schema_writer is not None:
            schema_writer.write_row(
                step=step, dts=dts, pw=pw, qq=qq, stt=stt, yaw=yaw, yr=yr,
                spd=spd, inference_ms=ims, eout=eout, vc=vc, yc=yc,
                gdn=gdn, nav_dir=None,
                obs_clearance=clearance, truth_cfg=truth_cfg,
                task_goal=gp)

        # Success: reach the ORIGINAL navigation goal (the final task goal).
        if dst <= hr_cfg.goal_tolerance_m and \
                spd <= hr_cfg.goal_speed_tolerance_mps:
            gh += 1
            if gh >= hr_cfg.goal_hold_ticks:
                ot = "success"
                break
        else:
            gh = 0

        if hr_cfg.verbose and step % 30 == 0:
            print(
                f"  [{ep_idx}:{step:04d}] dist={dst:.2f}m spd={spd:.2f}m/s | "
                f"src={eout.effective_target_source} mode={eout.hierarchical_mode} "
                f"| cmd=[{vc[0]:+.2f},{vc[1]:+.2f},{vc[2]:+.2f},{yc:+.2f}] | "
                f"infer={ims:.1f}ms | clear={clearance:.2f}m", flush=True)
        gid += 1

    dur = (step + 1) * dts
    plen = float(sum(np.linalg.norm(pp[i] - pp[i - 1])
                     for i in range(1, len(pp))))
    ai = float(np.mean(mts)) if mts else 0.0
    xi = float(np.max(mts)) if mts else 0.0
    ac = float(np.mean(cds)) if cds else 0.0
    res = EpisodeResult(
        episode=ep_idx, task_name=task.name, scene_id=scene_id, mode="hier",
        outcome=ot, duration_s=dur, path_length_m=plen,
        final_goal_distance_m=fd, min_goal_distance_m=md,
        num_model_steps=step + 1, num_collision_frames=cf,
        first_collision_step=fcs, avg_inference_ms=ai, max_inference_ms=xi,
        num_depth_timeouts=ndt, num_frame_mismatches=nfm,
        goal_switch_count=goal_switch_count,
        minimum_body_clearance_m=minimum_clearance, avg_command_delta=ac,
    )
    return res, gid + 1


def main() -> None:
    import os as _os  # noqa: F811
    p = argparse.ArgumentParser(
        description="Closed-loop hierarchical rollout: C++ 5 Hz expert "
                    "(upper) + 30 Hz ViTFlyLSTMPolicy student (lower).")
    p.add_argument("--checkpoint",
                   help="30 Hz student checkpoint (schema-v25 ViTFlyLSTMPolicy).")
    p.add_argument(
        "--model-file", default=str(_THIS_DIR / "model" / "model.py"),
        help="Policy implementation (default: save_net/model/model.py).")
    p.add_argument(
        "--expert-config",
        help="il_dataset YAML used to build the C++ Params2D (e.g. "
             "../il_dataset/config/il_dataset_joint_v2_config.yaml).")
    p.add_argument(
        "--tasks", default="avoid",
        help="Comma-separated task names, or a suite selector: "
             "avoid (gen_avoid_scenes), 4level (gen_avoid_scenes_4level), "
             "all (default: avoid).")
    p.add_argument(
        "--list-tasks", action="store_true",
        help="List avoidance tasks and exit without loading the model.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--pub-port", default="10253")
    p.add_argument("--sub-port", default="10254")
    p.add_argument("--scene-id", type=int, default=1)
    p.add_argument("--max-episode-time", type=float, default=30.0)
    p.add_argument("--goal-tolerance", type=float, default=0.30)
    p.add_argument("--goal-speed-tolerance", type=float, default=0.20)
    p.add_argument("--goal-hold-ticks", type=int, default=3)
    p.add_argument("--model-hz", type=float, default=30.0)
    p.add_argument("--ctrl-hz", type=float, default=50.0)
    p.add_argument("--lstm-reset-interval", type=int, default=0)
    p.add_argument("--max-yaw-rate", type=float, default=2.0)
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--depth-fov", type=float, default=90.0)
    p.add_argument("--depth-near", type=float, default=0.01)
    p.add_argument("--depth-far", type=float, default=1000.0)
    p.add_argument("--depth-max-m", type=float, default=5.0)
    p.add_argument("--render-warmup-frames", type=int, default=5)
    p.add_argument("--flight-z", type=float, default=2.0)
    p.add_argument("--frame-match-timeout", type=float, default=0.15,
                   help="Per-attempt depth frame match timeout (s).")
    p.add_argument("--max-frame-retries", type=int, default=5,
                   help="Depth frame match retries before aborting (0 = "
                        "abort on the first timeout).")
    p.add_argument(
        "--episode-out-root", default="",
        help="Directory for per-episode schema-v25 data.csv (one subdir per "
             "episode, loadable by interactive_trajectory_debug.py).  "
             "Defaults to <checkpoint_dir>/rollout_episodes when empty.")
    p.add_argument(
        "--no-truth-audit", action="store_true",
        help="Disable the exact-cylinder truth audit in the episode output.")
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--log-prefix", default="rollout_hi_latest",
        help="Output prefix for CSV/JSON diagnostics (default: ./rollout_hi_latest).")
    a = p.parse_args()

    task_selector = a.tasks.strip().lower()
    # Suite selectors (avoid / 4level) pick a single scene generator; the
    # named-task and all selectors merge every registry.
    if task_selector in _TASK_REGISTRY_BUILDERS:
        task_registry = _TASK_REGISTRY_BUILDERS[task_selector]()
        registry_label = task_selector
    else:
        task_registry = {}
        for builder in _TASK_REGISTRY_BUILDERS.values():
            task_registry.update(builder())
        registry_label = "all" if task_selector == "all" else "named"
    # The handcrafted avoidance layout intentionally places goals close to
    # obstacle edges (surface gap ~0.5 m on large_short) to force a precise
    # detour; the strict single-policy 0.6 m clearance pad is inappropriate
    # here, so validation keeps only the drone-radius (0.3 m) pad.
    validate_task_registry(
        task_registry, drone_radius=0.30, safety_margin=0.0,
        minimum_surface_gap_m=1.20)
    if a.list_tasks:
        print("Rollout tasks (registry='%s'):" % registry_label)
        for task in task_registry.values():
            print(
                f"  {task.name:<20} suite={task.suite:<6} "
                f"obstacles={len(task.obstacles):>2}  {task.description}")
        return
    if not a.checkpoint:
        p.error("--checkpoint is required unless --list-tasks is used")
    if not a.expert_config:
        p.error("--expert-config is required (il_dataset YAML for the C++ "
                "expert Params2D)")
    if a.repeats <= 0:
        p.error("--repeats must be > 0")
    if a.render_warmup_frames < 1:
        p.error("--render-warmup-frames must be >= 1")
    if a.lstm_reset_interval < 0:
        p.error("--lstm-reset-interval must be >= 0")
    if task_selector == "all":
        selected_tasks = list(task_registry.values())
    elif task_selector in _TASK_REGISTRY_BUILDERS:
        selected_tasks = [task for task in task_registry.values()
                          if task.suite == task_selector]
    else:
        requested = [name.strip() for name in a.tasks.split(",") if name.strip()]
        unknown = [name for name in requested if name not in task_registry]
        if unknown:
            p.error("unknown task(s): {}. Use --list-tasks.".format(
                ", ".join(unknown)))
        if not requested:
            p.error("--tasks must select at least one task")
        selected_tasks = [task_registry[name] for name in requested]

    if a.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(a.device)
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda but CUDA unavailable")

    # ── Upper stack: C++ expert ────────────────────────────────────
    expert, params, min_b, max_b, depth_cfg = load_expert_stack(a.expert_config)
    print("=" * 60)
    print("Hierarchical Rollout - C++ 5 Hz expert + 30 Hz student")
    print("=" * 60)
    print(f"  Expert .so revision: {getattr(expert_mod, 'EXPERT_REVISION', '<n/a>')}")
    print(f"  Expert config:       {a.expert_config}")
    print(f"  Scene bounds:        {min_b} .. {max_b}")
    print(f"  Macro cadence:       every 6 control ticks (5 Hz @ 30 Hz)")

    # ── Lower stack: 30 Hz student ─────────────────────────────────
    student, mc, checkpoint_scale = load_policy_checkpoint(
        a.checkpoint, a.model_file, dev, a.depth_max_m)
    hr_cfg = HierarchicalRolloutConfig(
        checkpoint=a.checkpoint, model_file=a.model_file,
        expert_config=a.expert_config,
        pub_port=a.pub_port, sub_port=a.sub_port, scene_id=a.scene_id,
        depth_width=a.depth_width, depth_height=a.depth_height,
        depth_fov=a.depth_fov, depth_near=a.depth_near, depth_far=a.depth_far,
        depth_max_m=a.depth_max_m,
        depth_t_bc=tuple(float(v) for v in depth_cfg["t_bc"]),
        model_hz=a.model_hz, ctrl_hz=a.ctrl_hz,
        lstm_reset_interval=a.lstm_reset_interval,
        max_yaw_rate=a.max_yaw_rate, max_episode_time=a.max_episode_time,
        goal_tolerance_m=a.goal_tolerance,
        goal_speed_tolerance_mps=a.goal_speed_tolerance,
        goal_hold_ticks=a.goal_hold_ticks,
        device=a.device, repeats=a.repeats, verbose=not a.quiet,
        log_prefix=a.log_prefix,
        render_warmup_frames=a.render_warmup_frames,
        flight_z=a.flight_z,
        frame_match_timeout_s=a.frame_match_timeout,
        max_frame_retries=a.max_frame_retries,
        episode_out_root=a.episode_out_root or
            str(Path(a.checkpoint).resolve().parent / "rollout_episodes"),
    )
    hr_cfg.state_scale = checkpoint_scale

    # CUDA warmup (kernel compilation).
    with torch.no_grad():
        _ = student.step(
            torch.ones(1, 1, mc.image_height, mc.image_width,
                       device=dev, dtype=torch.float32),
            torch.zeros(1, mc.state_dim, device=dev, dtype=torch.float32),
            student.initial_hidden(1, device=dev, dtype=torch.float32))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print("[rollout-hi] CUDA warmup complete.")

    total_episodes = len(selected_tasks) * a.repeats
    print(f"  Checkpoint:  {a.checkpoint}")
    print(f"  Model file:  {a.model_file}")
    print(f"  Device:      {dev}")
    print(f"  Mode:        hierarchical (expert upper / student lower)")
    print(f"  Tasks:       {', '.join(t.name for t in selected_tasks)}")
    print(f"  Repeats:     {a.repeats}   Episodes: {total_episodes}")
    print(f"  Model Hz:    {a.model_hz}   Ctrl Hz: {a.ctrl_hz}")
    print(f"  Depth:       {a.depth_width}x{a.depth_height} max={a.depth_max_m}m")
    print(f"  Ports:       PUB={a.pub_port} SUB={a.sub_port}")
    print(f"  Params:      {sum(p.numel() for p in student.parameters()):,}")
    print(f"  State:       7-D (gravity+goal), scale={hr_cfg.state_scale}")

    log_metadata = {
        "format_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_config": hr_cfg,
        "model_config": mc,
        "tasks": selected_tasks,
        "expert_revision": getattr(expert_mod, "EXPERT_REVISION", "<n/a>"),
        "expert_config": a.expert_config,
    }
    data_logger = HierarchicalDataLogger(a.log_prefix, log_metadata)
    data_logger.write_summary("running", [])
    print(f"  Step log:    {data_logger.steps_path}")
    print(f"  Summary:     {data_logger.summary_path}")
    print("=" * 60)

    print("[rollout-hi] Connecting to Unity...")
    bridge = il_common.UnityBridge(pub_port=a.pub_port, sub_port=a.sub_port)
    bridge.bind()
    print("[rollout-hi] Unity bridge bound.")
    dc_main = {
        "width": a.depth_width, "height": a.depth_height,
        "fov": a.depth_fov, "near": a.depth_near, "far": a.depth_far,
        "t_bc": list(depth_cfg["t_bc"]),
    }
    ok = bridge.connect_handshake(a.scene_id, dc_main, timeout=60.0)
    if not ok:
        raise RuntimeError("Unity handshake failed")
    print("[rollout-hi] Unity handshake OK.")
    dycfg = _build_dynamics_config(a.ctrl_hz, a.max_yaw_rate)
    dyn = il_dynamics.FlightmareDynamicsBackend(dycfg)

    results: List[EpisodeResult] = []
    gid = 0
    object_slots = max(len(task.obstacles) for task in task_registry.values())
    episode_plan = [
        (task, repeat_index)
        for task in selected_tasks
        for repeat_index in range(a.repeats)
    ]
    for ep, (task, repeat_index) in enumerate(episode_plan):
        try:
            while bridge.try_recv() is not None:
                pass
        except Exception:
            pass
        time.sleep(0.1)
        sp = np.asarray(task.start, dtype=np.float64)
        gp = np.asarray(task.goal, dtype=np.float64)
        obs = task_to_unity_objects(task, object_slots)

        # ── Schema-v25 episode output (interactive_trajectory_debug) ──
        schema_writer: Optional[SchemaV25EpisodeWriter] = None
        truth_cfg: Optional[Dict[str, float]] = None
        if hr_cfg.episode_out_root:
            ep_dir = Path(hr_cfg.episode_out_root) / (
                "ep%04d_%s" % (ep, task.name))
            schema_writer = SchemaV25EpisodeWriter(
                str(ep_dir),
                # The row scene_id is used by interactive_trajectory_debug
                # to index the handcrafted manifest.  It must be the avoid
                # scene id, not the Unity scene selected with --scene-id.
                scene_id=(task.scene_id if task.scene_id is not None
                          else a.scene_id),
                task_id=ep,
                navigation_goal_world=[float(gp[0]), float(gp[1]),
                                       float(gp[2])],
                initial_yaw=float(task.start_yaw))
            if not a.no_truth_audit and task.obstacles:
                schema_writer.configure_truth(
                    task.obstacles, vehicle_radius=hr_cfg.drone_radius,
                    min_bounds=min_b, max_bounds=max_b)
                truth_cfg = {
                    "eff_accel": float(getattr(params, "lp_eff_accel_mps2",
                                               getattr(params, "lp_max_accel",
                                                       2.0))),
                    "stop_margin": float(
                        getattr(params, "lp_brake_stop_margin_m", 0.3)),
                }
            print(f"  Schema out: {schema_writer.path}")

        print(f"\n{'-' * 60}")
        print(f"Episode {ep + 1}/{len(episode_plan)}  "
              f"(task={task.name}, repeat={repeat_index + 1}/{a.repeats})")
        print(f"  Purpose: {task.description}")
        print(f"  Start: [{sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f}]")
        print(f"  Goal:  [{gp[0]:.1f},{gp[1]:.1f},{gp[2]:.1f}]")
        print(f"  Dist:  {np.linalg.norm(gp - sp):.1f}m")
        print(f"  Scene: {len(task.obstacles)} fixed obstacles")
        print(f"{'-' * 60}")
        try:
            result, gid = run_hierarchical_rollout(
                student=student, student_cfg=mc, expert=expert, params=params,
                hr_cfg=hr_cfg, device=dev, ep_idx=ep, gid=gid, bridge=bridge,
                dyn=dyn, task=task, scene_id=a.scene_id, obstacles=obs,
                data_logger=data_logger,
                schema_writer=schema_writer, truth_cfg=truth_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[rollout-hi] Episode {ep} ERROR: {exc}")
            import traceback
            traceback.print_exc()
            result = EpisodeResult(
                episode=ep, task_name=task.name, scene_id=a.scene_id,
                mode="hier", outcome="error", duration_s=0, path_length_m=0,
                final_goal_distance_m=float("inf"),
                min_goal_distance_m=float("inf"), num_model_steps=0,
                num_collision_frames=0, first_collision_step=-1,
                avg_inference_ms=0, max_inference_ms=0,
                num_depth_timeouts=0, num_frame_mismatches=0,
                goal_switch_count=0,
                minimum_body_clearance_m=float("inf"),
                avg_command_delta=0,
            )
            gid += 10
        if schema_writer is not None:
            ep_valid = 1 if result.outcome == "success" else 0
            schema_writer.finalize(
                episode_valid=ep_valid,
                failure_taxonomy="",
                outcome=result.outcome)
        results.append(result)
        data_logger.write_summary("running", results)
        cm = ""
        if result.first_collision_step >= 0:
            cm = (f" | 1st_coll={result.first_collision_step} "
                  f"total_coll={result.num_collision_frames}")
        print(
            f"  -> {result.outcome.upper()} | dur={result.duration_s:.1f}s | "
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

    try:
        bridge.close()
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    nt = len(results)
    ns = sum(1 for r in results if r.outcome == "success")
    nc = sum(1 for r in results if r.outcome == "collision")
    nto = sum(1 for r in results if r.outcome == "timeout")
    ne = sum(1 for r in results if r.outcome == "error")
    print(f"  Mode:        hierarchical (C++ 5 Hz expert + 30 Hz student)")
    print(f"  Total:       {nt}")
    print(f"  Success:     {ns} ({100 * ns / max(nt, 1):.1f}%)")
    print(f"  Collision:   {nc} ({100 * nc / max(nt, 1):.1f}%)")
    print(f"  Timeout:     {nto} ({100 * nto / max(nt, 1):.1f}%)")
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
    fin = [r for r in results if r.outcome in ("success", "collision", "timeout")]
    if fin:
        print(f"\n  Avg dur:        {np.mean([r.duration_s for r in fin]):.1f}s")
        print(f"  Avg path:       {np.mean([r.path_length_m for r in fin]):.1f}m")
        print(f"  Avg final dist: {np.mean([r.final_goal_distance_m for r in fin]):.2f}m")
        print(f"  Avg infer:      {np.mean([r.avg_inference_ms for r in fin]):.1f}ms")
    finite_clearances = [r.minimum_body_clearance_m for r in results
                         if math.isfinite(r.minimum_body_clearance_m)]
    if finite_clearances:
        print(f"\n  Safety:  minimum body clearance={min(finite_clearances):.2f}m")
    print(f"  Goals:   applied in-episode updates={sum(r.goal_switch_count for r in results)}")
    tto = sum(r.num_depth_timeouts for r in results)
    tmm = sum(r.num_frame_mismatches for r in results)
    print(f"  Infra:   depth_timeouts={tto}  frame_mismatches={tmm}")
    data_logger.write_summary("completed", results)
    data_logger.close()
    print(f"  Saved:   {data_logger.steps_path}")
    print(f"           {data_logger.summary_path}")
    print(f"{'=' * 60}")
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
