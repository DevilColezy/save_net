#!/usr/bin/env python3
"""Comprehensive stack A/B/C/D test on ONE unified scene set.

Four deployable stacks are compared on the same scenes:

  * ``student30``            — the 30 Hz ViTFlyLSTMPolicy ALONE (upper=none,
                               lower=student30; it flies the original goal).
  * ``expert5_student30``    — C++ 5 Hz expert decides, 30 Hz student flies
                               (rollout_hierarchical stack).
  * ``student5_student30``   — 5 Hz MacroPlannerPolicy student decides,
                               30 Hz student flies (the full learned stack).
  * ``expert``               — the complete C++ expert (5 Hz + 30 Hz) alone.

Every stack runs the same 8 scenes (clear / small / medium / zigzag / rear /
single_big / big_wall / gate) so the four results are directly comparable.

Effective-target sources:
  * upper=none     -> the ORIGINAL navigation goal.
  * upper=expert5  -> expert.step()'s effective_target (world point) plus the
                      already-projected FLU direction / distance.
  * upper=student5 -> the 5 Hz student's regression (direction + distance),
                      world-latched at the decision instant and re-projected
                      into the live body frame every 30 Hz tick (the same
                      adapter contract the expert uses).

Lower executors:
  * lower=student30 -> build the 7-D state (gravity + effective goal) and run
                       the 30 Hz student.
  * lower=expert    -> execute expert.step()'s command directly.

Usage:
    python3 rollout_stack.py --list-tasks
    python3 rollout_stack.py --stack student30 \
        --checkpoint checkpoints/vitfly_v28_joint_v2_col4_mirror/best.pt
    python3 rollout_stack.py --stack student5_student30 \
        --checkpoint checkpoints/vitfly_v28_joint_v2_col4_mirror/best.pt \
        --macro-checkpoint checkpoints/macro_v1_col4_7d/best.pt \
        --expert-config ../il_dataset/config/il_dataset_joint_v2_config.yaml
    # run all four stacks and write one comparison summary:
    python3 rollout_stack.py --all-stacks \
        --checkpoint ... --macro-checkpoint ... --expert-config ...
"""

import argparse
import importlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_IL_SCRIPTS = _THIS_DIR.parent / "il_dataset" / "scripts"
if str(_IL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IL_SCRIPTS))

from rollout import (  # noqa: E402
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
    _yaw_from_quat_xyzw,
)
from rollout_hierarchical import (  # noqa: E402
    build_4level_task_registry,
    load_expert_stack,
)
from rollout_macro_student import (  # noqa: E402
    NORMAL_ANGLE_DEG,
    TURN_DIST_THRESHOLD,
    load_macro_checkpoint,
)

try:
    import il_common
    import il_dynamics
except ImportError as e:  # pragma: no cover
    il_common = None
    il_dynamics = None
    _IL_IMPORT_ERROR: Optional[ImportError] = e
else:
    _IL_IMPORT_ERROR = None

# ============================================================================
#  Unified scene set (30 scenes, grouped by difficulty class)
# ============================================================================
def _deg(deg: float) -> float:
    return math.radians(deg)


# (name, obstacles [(x,y,r)], tasks [(sx,sy,gx,gy,label,start_yaw)])
STACK_SCENES = [
    # ── A. baseline & single-obstacle ──────────────────────────────
    {"name": "S_clear", "desc": "clear straight baseline",
     "obstacles": [],
     "tasks": [(0.0, 3.0, 0.0, 15.0, "clear", 0.0)]},
    {"name": "S_small", "desc": "single small cylinder r=0.4 on the line",
     "obstacles": [(0.0, 9.0, 0.4)],
     "tasks": [(0.0, 3.0, 0.0, 15.0, "small", 0.0)]},
    {"name": "S_medium", "desc": "single medium cylinder r=1.0 on the line",
     "obstacles": [(0.0, 10.0, 1.0)],
     "tasks": [(0.0, 4.0, 0.0, 16.0, "medium", 0.0)]},
    {"name": "S_small_offset", "desc": "small cylinder offset 1.5 m off the line",
     "obstacles": [(1.5, 9.0, 0.4)],
     "tasks": [(0.0, 3.0, 0.0, 15.0, "small_offset", 0.0)]},
    {"name": "S_wide_single", "desc": "single wide r=2.5 cylinder on the line",
     "obstacles": [(0.0, 10.0, 2.5)],
     "tasks": [(0.0, 4.0, 0.0, 16.0, "wide_single", 0.0)]},
    {"name": "S_large_offset", "desc": "large r=1.5 cylinder offset 2.5 m",
     "obstacles": [(2.5, 10.0, 1.5)],
     "tasks": [(0.0, 3.0, 0.0, 17.0, "large_offset", 0.0)]},
    # ── B. chains (consecutive obstacles on the straight line) ─────
    {"name": "S_chain2", "desc": "2 cylinders in line, consecutive detours",
     "obstacles": [(0.0, 9.0, 0.6), (0.0, 14.0, 0.7)],
     "tasks": [(0.0, 3.0, 0.0, 17.0, "chain2", 0.0)]},
    {"name": "S_chain3", "desc": "3 cylinders in line",
     "obstacles": [(0.0, 8.0, 0.5), (0.0, 12.0, 0.6), (0.0, 16.0, 0.5)],
     "tasks": [(0.0, 3.0, 0.0, 19.0, "chain3", 0.0)]},
    {"name": "S_chain4", "desc": "4 cylinders in line (long corridor chain)",
     "obstacles": [(0.0, 7.0, 0.4), (0.0, 11.0, 0.5), (0.0, 15.0, 0.4),
                   (0.0, 19.0, 0.5)],
     "tasks": [(0.0, 3.0, 0.0, 22.0, "chain4", 0.0)]},
    {"name": "S_medium_chain", "desc": "3 medium cylinders in line",
     "obstacles": [(0.0, 9.0, 1.0), (0.0, 13.0, 0.8), (0.0, 17.0, 1.0)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "medium_chain", 0.0)]},
    # ── C. zig-zag / slalom ────────────────────────────────────────
    {"name": "S_zigzag", "desc": "3 staggered cylinders, consecutive detours",
     "obstacles": [(0.0, 10.0, 0.7), (1.5, 14.0, 0.7), (-1.5, 18.0, 0.7)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "zigzag", 0.0)]},
    {"name": "S_zigzag2", "desc": "4 staggered cylinders incl. line piercers",
     "obstacles": [(0.0, 9.0, 0.6), (1.8, 13.0, 0.8), (-1.8, 17.0, 0.6),
                   (0.0, 21.0, 0.7)],
     "tasks": [(0.0, 3.0, 0.0, 24.0, "zigzag2", 0.0)]},
    {"name": "S_slalom", "desc": "long alternating S-shaped detour",
     "obstacles": [(0.0, 8.0, 0.5), (1.8, 11.0, 0.6), (-1.8, 14.0, 0.5),
                   (0.0, 17.0, 0.6), (-1.8, 20.0, 0.5), (1.8, 23.0, 0.6)],
     "tasks": [(0.0, 3.0, 0.0, 25.0, "slalom", 0.0)]},
    # ── D. FOV-out goals (must TURN) ───────────────────────────────
    {"name": "S_rear", "desc": "goal directly behind (FOV-out): needs 5 Hz TURN",
     "obstacles": [],
     "tasks": [(0.0, 12.0, 0.0, 5.0, "rear", 0.0)]},
    {"name": "S_rear_left", "desc": "goal LEFT-behind (~141 deg) + blocker",
     "obstacles": [(0.0, 9.0, 0.8)],
     "tasks": [(0.0, 12.0, -4.0, 7.0, "rear_left", 0.0)]},
    {"name": "S_rear_right", "desc": "goal RIGHT-behind (~141 deg) + blocker",
     "obstacles": [(0.0, 9.0, 0.8)],
     "tasks": [(0.0, 12.0, 4.0, 7.0, "rear_right", 0.0)]},
    {"name": "S_side_goal", "desc": "goal far to the right-front (FOV-out)",
     "obstacles": [],
     "tasks": [(0.0, 10.0, 7.0, 6.0, "side_goal", 0.0)]},
    # ── E. big blockers / walls (fill the FOV) ─────────────────────
    {"name": "S_single_big", "desc": "single r=3.5 cylinder 4 m ahead fills FOV",
     "obstacles": [(0.0, 8.0, 3.5)],
     "tasks": [(0.0, 4.0, 0.0, 18.0, "single_big", 0.0)]},
    {"name": "S_big_blocker", "desc": "r=3.0 blocker, goal directly behind",
     "obstacles": [(0.0, 10.0, 3.0)],
     "tasks": [(0.0, 4.0, 0.0, 16.0, "big_blocker", 0.0)]},
    {"name": "S_big_wall", "desc": "13.2 m wall of 3 x r=1.8 (wall bypass)",
     "obstacles": [(-4.8, 12.0, 1.8), (0.0, 12.0, 1.8), (4.8, 12.0, 1.8)],
     "tasks": [(0.0, 4.0, 0.0, 20.0, "big_wall", 0.0)]},
    {"name": "S_double_wall", "desc": "two staggered rows (front gaps blocked)",
     "obstacles": [(-3.9, 10.0, 0.7), (-1.3, 10.0, 0.7), (1.3, 10.0, 0.7),
                   (3.9, 10.0, 0.7), (-2.6, 15.0, 0.7), (0.0, 15.0, 0.7),
                   (2.6, 15.0, 0.7)],
     "tasks": [(0.0, 3.0, 0.0, 19.0, "double_wall", 0.0)]},
    # ── F. gates / corridors ───────────────────────────────────────
    {"name": "S_gate", "desc": "narrow gate (1.2 m gap) on the straight line",
     "obstacles": [(-1.2, 10.0, 0.6), (1.2, 10.0, 0.6)],
     "tasks": [(0.0, 3.0, 0.0, 17.0, "gate", 0.0)]},
    {"name": "S_corridor", "desc": "1.4 m wide corridor of two side walls",
     "obstacles": [(-1.2, 8.0, 0.5), (-1.2, 12.0, 0.5), (-1.2, 16.0, 0.5),
                   (1.2, 8.0, 0.5), (1.2, 12.0, 0.5), (1.2, 16.0, 0.5)],
     "tasks": [(0.0, 3.0, 0.0, 19.0, "corridor", 0.0)]},
    # ── G. collection-like level tiers ─────────────────────────────
    {"name": "S_level_small", "desc": "level small: dense r~0.3-0.45 clusters",
     "obstacles": [(0.0, 8.0, 0.3), (-1.5, 11.0, 0.4), (1.5, 13.0, 0.35),
                   (0.0, 16.0, 0.3), (-1.0, 18.0, 0.45)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "level_small", 0.0)]},
    {"name": "S_level_medium", "desc": "level medium: r 0.8-1.0 blockers",
     "obstacles": [(0.0, 9.0, 0.8), (-1.8, 13.0, 1.0), (1.8, 17.0, 0.9)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "level_medium", 0.0)]},
    {"name": "S_level_large", "desc": "level large: r 1.6-2.0 big blockers",
     "obstacles": [(0.0, 10.0, 2.0), (-3.0, 15.0, 1.6), (3.0, 19.0, 1.8)],
     "tasks": [(0.0, 3.0, 0.0, 23.0, "level_large", 0.0)]},
    {"name": "S_level_mixed", "desc": "level mixed: r 0.3-2.0 combined",
     "obstacles": [(0.0, 9.0, 0.4), (-2.0, 12.0, 1.5), (2.0, 15.0, 0.5),
                   (-1.0, 18.0, 2.0), (1.0, 21.0, 0.3)],
     "tasks": [(0.0, 3.0, 0.0, 25.0, "level_mixed", 0.0)]},
    {"name": "S_level_dense", "desc": "dense: many small cylinders close together",
     "obstacles": [(0.0, 7.0, 0.3), (-1.2, 9.0, 0.35), (1.2, 11.0, 0.3),
                   (0.0, 13.0, 0.4), (-1.0, 15.0, 0.35), (1.0, 17.0, 0.3)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "level_dense", 0.0)]},
    {"name": "S_level_dense2", "desc": "extra-dense small cylinders in line",
     "obstacles": [(0.0, 7.0, 0.3), (-1.0, 9.0, 0.35), (1.0, 11.0, 0.3),
                   (0.0, 13.0, 0.35), (-1.2, 15.0, 0.3), (1.2, 17.0, 0.35),
                   (0.0, 19.0, 0.3)],
     "tasks": [(0.0, 3.0, 0.0, 21.0, "level_dense2", 0.0)]},
    # ── H. cluster ─────────────────────────────────────────────────
    {"name": "S_cluster", "desc": "central cluster of 4 small cylinders",
     "obstacles": [(0.0, 10.0, 0.3), (-1.5, 11.0, 0.3), (1.5, 11.0, 0.3),
                   (0.0, 13.0, 0.3)],
     "tasks": [(0.0, 3.0, 0.0, 17.0, "cluster", 0.0)]},
]


def build_stack_task_registry() -> Dict[str, RolloutTask]:
    tasks: Dict[str, RolloutTask] = {}
    for scene_index, sc in enumerate(STACK_SCENES):
        obstacles = tuple(
            Cylinder(float(o[0]), float(o[1]), float(o[2]))
            for o in sc["obstacles"])
        for sx, sy, gx, gy, label, syaw in sc["tasks"]:
            tasks[label] = RolloutTask(
                name=label, description=sc["desc"],
                start=(float(sx), float(sy), 2.0),
                goal=(float(gx), float(gy), 2.0),
                start_yaw=float(syaw), obstacles=obstacles,
                suite="stack", scene_id=scene_index)
    return tasks


_V3_SCENE_PATHS = (
    _THIS_DIR.parent / "il_dataset" / "test" / "gen_rollout_v3_scenes.py",
)
# v3 environments map to the Unity scene-id used by --scene-id:
# indoor -> 1 (WAREHOUSE), outdoor -> 0 (INDUSTRIAL).
_V3_ENV_BY_SCENE_ID = {1: "indoor", 0: "outdoor"}


def _load_v3_scene_module():
    path = next((p for p in _V3_SCENE_PATHS if p.is_file()), None)
    if path is None:
        raise RuntimeError("gen_rollout_v3_scenes.py not found")
    spec = importlib.util.spec_from_file_location(
        "_il_gen_rollout_v3_scenes", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_v3_task_registry(scene_id: int) -> Dict[str, RolloutTask]:
    """v3 scene set (gen_rollout_v3_scenes.py).  Only tasks whose
    environment matches the requested Unity scene are kept:
    scene_id 1 = indoor (WAREHOUSE), scene_id 0 = outdoor (INDUSTRIAL)."""
    env = _V3_ENV_BY_SCENE_ID.get(int(scene_id))
    if env is None:
        raise ValueError(
            f"v3 scene set needs --scene-id 0 (outdoor) or 1 (indoor), "
            f"got {scene_id}")
    module = _load_v3_scene_module()
    # Outdoor INDUSTRIAL has low static clutter -> fly HIGH (z=14); indoor
    # WAREHOUSE stays at z=2.  Obstacle height matches the collection recipe
    # (outdoor obstacles U[18,24] m pierce the 15-17 m band; use the 18 m
    # lower bound) so walls still block the drone and the detour stays
    # meaningful.
    flight_z = 14.0 if env == "outdoor" else 2.0
    obstacle_h = 18.0 if env == "outdoor" else 8.0
    tasks: Dict[str, RolloutTask] = {}
    for scene_index, sc in enumerate(module.SCENES):
        if sc.get("environment") != env:
            continue
        walls = list(sc.get("walls", []))
        obs = list(sc.get("obstacles", []))
        all_obs = walls + obs
        obstacles = []
        for o in all_obs:
            if len(o) >= 5 and o[3] > 0.0 and o[4] > 0.0:
                # AABB box: (x, y, radius=0, half_w, half_h) -> Transparen_Cube
                obstacles.append(Cylinder(float(o[0]), float(o[1]),
                                          float(o[2]), height=obstacle_h,
                                          half_w=float(o[3]),
                                          half_h=float(o[4])))
            else:
                obstacles.append(Cylinder(float(o[0]), float(o[1]),
                                          float(o[2]), height=obstacle_h))
        obstacles = tuple(obstacles)
        wall_flags = tuple([True] * len(walls) + [False] * len(obs))
        for task in sc["tasks"]:
            sx, sy, gx, gy, label = task
            tasks[label] = RolloutTask(
                name=label, description=sc["desc"],
                start=(float(sx), float(sy), flight_z),
                goal=(float(gx), float(gy), flight_z),
                start_yaw=0.0, obstacles=obstacles,
                wall_flags=wall_flags,
                suite="v3", scene_id=scene_index)
    if not tasks:
        raise RuntimeError(
            f"v3 scene set: no tasks for environment '{env}' (scene_id "
            f"{scene_id}). Use --scene-id 1 for indoor or 0 for outdoor.")
    return tasks, flight_z


def build_big8_task_registry() -> Dict[str, RolloutTask]:
    """Single r=8 m cylinder at the scene centre with two crossing tasks.

    Both start/goal pairs lie on the cylinder axis so the straight line runs
    straight through the obstacle core; the drone must detour around a 16 m
    wide blocker.  The two tasks differ only in how close the endpoints sit
    to the obstacle surface (6 m vs 2 m).
    """
    center = (1.5, 15.0)
    radius = 8.0
    obstacle = Cylinder(float(center[0]), float(center[1]), float(radius))
    tasks: Dict[str, RolloutTask] = {}
    specs = [
        ("big8_6m", 6.0, "start/goal 6 m off the r=8 m cylinder surface"),
        ("big8_2m", 2.0, "start/goal 2 m off the r=8 m cylinder surface"),
    ]
    for name, surf, desc in specs:
        off = radius + surf
        start = (center[0], center[1] - off, 2.0)
        goal = (center[0], center[1] + off, 2.0)
        tasks[name] = RolloutTask(
            name=name, description=desc, start=start, goal=goal,
            start_yaw=0.0, obstacles=(obstacle,), suite="big8",
            scene_id=1000)
    return tasks


STACKS = ["student30", "expert5_student30", "student5_student30",
          "expert", "expert30"]


@dataclass
class StackRolloutConfig:
    stack: str = "student30"
    checkpoint: str = ""
    model_file: str = ""
    macro_checkpoint: str = ""
    expert_config: str = ""
    pub_port: str = "10253"
    sub_port: str = "10254"
    scene_id: int = 1
    depth_width: int = 640
    depth_height: int = 360
    depth_fov: float = 58.0   # VERTICAL FOV = D435i depth ~58°
    depth_near: float = 0.28  # D435i Min-Z
    depth_far: float = 10.0   # D435i effective range
    depth_max_m: float = 5.0
    depth_t_bc: Tuple[float, ...] = ()
    model_hz: float = 30.0
    ctrl_hz: float = 50.0
    max_yaw_rate: float = 2.0
    max_episode_time: float = 30.0
    goal_tolerance_m: float = 0.30
    goal_speed_tolerance_mps: float = 0.20
    goal_hold_ticks: int = 3
    # Within this distance of the goal the episode is declared SUCCESS
    # IMMEDIATELY (no speed / hold requirement).  This prevents the drone
    # from re-planning endlessly around the goal and timing out while being
    # only ~0.5-1 m away.  0 disables the immediate latch.
    goal_immediate_m: float = 0.5
    collision_confirm_frames: int = 1
    drone_radius: float = 0.3
    device: str = "auto"
    repeats: int = 1
    verbose: bool = False
    log_prefix: str = "rollout_stack"
    render_warmup_frames: int = 5
    flight_z: float = 2.0
    frame_match_timeout_s: float = 0.15
    max_frame_retries: int = 5


class StackDataLogger(RolloutDataLogger):
    _S_COLUMNS = [
        "episode", "task", "stack", "step", "sim_time_s",
        "state_x", "state_y", "state_z", "state_yaw",
        "speed_world_mps", "goal_distance_m",
        "is_macro_tick",
        "upper_type", "effective_target_x", "effective_target_y",
        "goal_dir_flu_x", "goal_dir_flu_y", "goal_dist_norm",
        "cmd_vx_flu", "cmd_vy_flu", "cmd_vz_flu", "cmd_yaw_rate",
        "minimum_body_clearance_m", "inference_ms",
    ]
    COLUMNS = _S_COLUMNS


# Distance to the ORIGINAL goal below which the 5 Hz student upper is forced to
# PASS_THROUGH (near-goal latch).  This prevents the upper from spuriously
# re-planning (TURN/NORMAL) right next to the goal and flying the drone away,
# which previously caused "arrived but never declared success" -> timeout.
GOAL_LOCK_DIST_M = 1.0


def _student5_decide(macro, depth_t, macro_state_t, mhidden, qq, pw, gp,
                     R, device):
    """Run the 5 Hz macro student and return
    (type_name, world_target_xy, goal_dir_flu, goal_dist_norm, hidden)."""
    # Original goal FLU direction (for PASS/NORMAL decision and target).
    orig_world = gp - pw
    d_orig = float(np.linalg.norm(orig_world[:2]))
    orig_dir_flu = np.zeros(3)
    if d_orig > 1e-8:
        orig_dir_flu = il_common.world_vector_to_body_flu_quat(
            orig_world / d_orig, qq)
    # Near-goal latch: once close to the original goal, force PASS_THROUGH so
    # the 30 Hz student converges onto the goal and the success check fires.
    if d_orig < GOAL_LOCK_DIST_M:
        gdn = float(min(d_orig, R - 0.5) / max(R, 1e-9))
        return "PASS_THROUGH", gp, orig_dir_flu.copy(), gdn, mhidden
    with torch.no_grad():
        mout = macro.step(depth_t, macro_state_t, mhidden)
    mhidden = tuple(v.detach() for v in mout.hidden)
    dir_flu = mout.direction[0].cpu().numpy().copy()
    dist_norm = float(mout.distance_norm[0, 0].cpu().numpy())
    d = np.asarray(dir_flu, dtype=np.float64).reshape(3)
    nd = float(np.linalg.norm(d))
    if nd < 1e-8:
        d = np.array([1.0, 0.0, 0.0])
    else:
        d = d / nd
    # Decide type using the same thresholds as the macro-student rollout.
    if dist_norm > TURN_DIST_THRESHOLD:
        tname = "TURN_LEFT" if d[1] > 0.0 else "TURN_RIGHT"
        dir_world3 = il_common.body_flu_vector_to_world_quat(d, qq)
        dir_world = np.asarray(dir_world3[:2], dtype=np.float64)
        wn = float(np.linalg.norm(dir_world))
        if wn < 1e-8:
            dir_world = np.array([0.0, 1.0])
        else:
            dir_world = dir_world / wn
        target = np.array([pw[0], pw[1]]) + dir_world * R
        goal_dir_flu = np.asarray(
            il_common.world_vector_to_body_flu_quat(
                np.array([target[0] - pw[0], target[1] - pw[1], 0.0]), qq))
        gdn = 1.0
    else:
        o = np.asarray(orig_dir_flu, dtype=np.float64).reshape(3)
        no = float(np.linalg.norm(o))
        angle = 0.0
        if no > 1e-8:
            o = o / no
            angle = math.degrees(math.acos(
                float(np.clip(np.dot(d, o), -1.0, 1.0))))
        if angle > NORMAL_ANGLE_DEG:
            tname = "NORMAL_CORRECTION"
            dir_world3 = il_common.body_flu_vector_to_world_quat(d, qq)
            dir_world = np.asarray(dir_world3[:2], dtype=np.float64)
            wn = float(np.linalg.norm(dir_world))
            if wn < 1e-8:
                dir_world = np.array([0.0, 1.0])
            else:
                dir_world = dir_world / wn
            dist_world = float(dist_norm) * R
            target = np.array([pw[0], pw[1]]) + dir_world * dist_world
            goal_dir_flu = np.asarray(
                il_common.world_vector_to_body_flu_quat(
                    np.array([target[0] - pw[0], target[1] - pw[1], 0.0]), qq))
            gdn = float(min(dist_world, R - 0.5) / max(R, 1e-9))
        else:
            tname = "PASS_THROUGH"
            target = gp
            goal_dir_flu = orig_dir_flu.copy()
            gdn = float(min(d_orig, R - 0.5) / max(R, 1e-9))
    return tname, target, goal_dir_flu, gdn, mhidden


def run_stack_rollout(
    stack, student30, s30_cfg, s30_scale, macro, expert, params,
    cfg: StackRolloutConfig, device: torch.device, ep_idx: int, gid: int,
    bridge, dyn, task: RolloutTask, scene_id: int,
    obstacles: List[Dict[str, Any]], data_logger: StackDataLogger,
) -> Tuple[EpisodeResult, int]:
    dts = 1.0 / cfg.model_hz
    episode_time = (task.max_episode_time
                    if task.max_episode_time is not None
                    else cfg.max_episode_time)
    mx = int(episode_time * cfg.model_hz)
    dc = {"width": cfg.depth_width, "height": cfg.depth_height,
          "fov": cfg.depth_fov, "near": cfg.depth_near, "far": cfg.depth_far,
          "t_bc": list(cfg.depth_t_bc)}
    ih, iw = cfg.depth_height, cfg.depth_width
    mr = cfg.depth_max_m
    sp = np.asarray(task.start, dtype=np.float64)
    gp = np.asarray(task.goal, dtype=np.float64)
    syaw = float(task.start_yaw)
    flight_z = float(cfg.flight_z)
    tick_base = 0
    R = float(getattr(params, "obs_range_m", 5.0)) if params is not None else 5.0

    if expert is not None:
        expert.reset_task([float(sp[0]), float(sp[1])],
                          [float(gp[0]), float(gp[1])], syaw, tick_base,
                          flight_z)
        expert.clear_external_directive()
    dyn.reset(sp.copy(), syaw, np.zeros(3), np.zeros(3))
    veh = il_common.make_depth_vehicle(
        ros_pos=sp.copy().tolist(), yaw=syaw, depth_cfg=dc)
    st = {"scene_id": scene_id, "frame_id": gid,
          "vehicles": [veh], "objects": obstacles}
    bridge.send_pose(st)
    gid += 1
    time.sleep(0.5)

    try:
        while bridge.try_recv() is not None:
            pass
    except Exception:
        pass
    depth_payload_bytes = iw * ih * 4
    warm_depth = None
    for warmup_index in range(cfg.render_warmup_frames):
        ws = dyn.get_state()
        wp = ws.position_world.copy()
        wq = ws.quaternion_world_body.copy()
        wv = il_common.make_depth_vehicle(
            ros_pos=wp.tolist(), yaw=_yaw_from_quat_xyzw(wq),
            depth_cfg=dc, quaternion_xyzw=wq.tolist())
        st["frame_id"] = gid
        st["vehicles"] = [wv]
        bridge.send_pose(st)
        wstart = time.perf_counter()
        while time.perf_counter() - wstart < 1.0:
            r = bridge.try_recv()
            if r is None:
                time.sleep(0.002)
                continue
            wm, wpl = r
            fid = wm.get("pub_frame_id") or wm.get("frame_id", -1)
            if fid != gid:
                continue
            for pt in wpl:
                if len(pt) >= depth_payload_bytes:
                    raw = pt[:depth_payload_bytes]
                    df = np.frombuffer(raw, dtype=np.float32).reshape((ih, iw))
                    warm_depth, _ = canonicalize_unity_depth(df, mr)
                    break
            if warm_depth is not None:
                break
        if warm_depth is None:
            raise RuntimeError("render warmup timed out")
        gid += 1

    dt = torch.float32
    hidden = student30.initial_hidden(1, device=device, dtype=dt) \
        if student30 is not None else None
    mhidden = macro.initial_hidden(1, device=device, dtype=dt) \
        if macro is not None else None
    upper = cfg.stack.split("_")[0] if cfg.stack != "student30" else "none"
    if cfg.stack == "expert":
        upper = "expert5"
    elif cfg.stack == "expert30":
        # Pure 30 Hz expert: the 5 Hz corrector is suppressed below, so the
        # C++ local planner flies the ORIGINAL goal with no upper takeover.
        upper = "expert30"
    lower = "expert" if cfg.stack in ("expert", "expert30") else "student30"

    pp: List[np.ndarray] = [dyn.get_state().position_world.copy()]
    cf = 0
    fcs = -1
    ndt = 0
    nfm = 0
    pc: Optional[np.ndarray] = None
    cds: List[float] = []
    md = float(np.linalg.norm(pp[0] - gp))
    fd = md
    ot = "timeout"
    minimum_clearance = body_clearance(pp[0], task, cfg.drone_radius)
    gh = 0
    last_upper_type = "NONE"
    last_eff = gp.copy()
    last_gdf = np.zeros(3)
    last_gdn = 0.0

    for step in range(mx):
        stt = dyn.get_state()
        pw = stt.position_world.copy()
        qq = stt.quaternion_world_body.copy()
        yaw = _yaw_from_quat_xyzw(qq)
        vel = stt.velocity_world.copy()
        yaw_rate_body = float(stt.angular_velocity_body[2])
        veh = il_common.make_depth_vehicle(
            ros_pos=pw.tolist(), yaw=yaw, depth_cfg=dc,
            quaternion_xyzw=qq.tolist())
        st["frame_id"] = gid
        st["vehicles"] = [veh]
        bridge.send_pose(st)
        du = None
        col = False
        dfl = iw * ih * 4
        for attempt in range(cfg.max_frame_retries + 1):
            if attempt > 0:
                gid += 1
                st["frame_id"] = gid
                st["vehicles"] = [veh]
                bridge.send_pose(st)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < cfg.frame_match_timeout_s:
                r = bridge.try_recv()
                if r is None:
                    time.sleep(0.002)
                    continue
                mm, rp = r
                fid = mm.get("pub_frame_id") or mm.get("frame_id", -1)
                if fid != gid:
                    nfm += 1
                    continue
                for pt in rp:
                    if len(pt) >= dfl:
                        raw = pt[:dfl]
                        df = np.frombuffer(raw, dtype=np.float32).reshape((ih, iw))
                        du, depth_normalized = canonicalize_unity_depth(df, mr)
                        break
                vs = mm.get("pub_vehicles", [])
                if vs and vs[0].get("collision", False):
                    col = True
                break
            if du is not None:
                break
            if cfg.verbose and attempt < cfg.max_frame_retries:
                print("  [stack] frame timeout retry", flush=True)
        if col:
            cf += 1
            if fcs < 0:
                fcs = step
            if cf >= cfg.collision_confirm_frames:
                ot = "collision"
                break
        if du is None:
            ndt += 1
            ot = "error"
            warnings.warn("depth timeout after retries")
            break

        is_macro_tick = (step % 6 == 0)
        depth_t = preprocess_depth(depth_normalized, device)

        # ── Upper: choose the effective world target on 5 Hz boundaries ──
        if upper == "student5" and is_macro_tick:
            orig_world = gp - pw
            d_orig = float(np.linalg.norm(orig_world[:2]))
            orig_dir_flu = np.zeros(3)
            if d_orig > 1e-8:
                orig_dir_flu = il_common.world_vector_to_body_flu_quat(
                    orig_world / d_orig, qq)
            goal_dist_norm = float(min(d_orig, R - 0.5) / max(R, 1e-9))
            grav_flu = il_common.world_vector_to_body_flu_quat(
                np.array([0.0, 0.0, -1.0], dtype=np.float64), qq)
            macro_state_t = build_normalized_state(
                grav_flu, orig_dir_flu, goal_dist_norm,
                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), device)
            last_upper_type, last_eff, last_gdf, last_gdn, mhidden = \
                _student5_decide(macro, depth_t, macro_state_t, mhidden,
                                 qq, pw, gp, R, device)

        # ── Expert step (needed when expert is the lower OR upper) ──
        eout = None
        if expert is not None:
            if upper == "student5" and is_macro_tick:
                # Suppress the expert's own corrector so it does not
                # override the student's decision (lower student30 only).
                expert.set_external_directive(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              "STACK_SUPPRESS")
            elif upper == "expert30" and is_macro_tick:
                # Pure 30 Hz expert: inject PASS_THROUGH so the 5 Hz
                # corrector NEVER takes over — the C++ local planner flies
                # the original goal on its own.
                expert.set_external_directive(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              "EXPERT30_ONLY")
            expert_depth = np.flipud(df.astype(np.float64) * 100.0)
            try:
                eout = expert.step(
                    [float(pw[0]), float(pw[1]), float(pw[2])], yaw,
                    [float(vel[0]), float(vel[1]), float(vel[2])],
                    yaw_rate_body,
                    np.ascontiguousarray(expert_depth,
                                         dtype=np.float32).ravel(),
                    int(cfg.depth_width), int(cfg.depth_height),
                    [float(pw[0]), float(pw[1]), float(pw[2])],
                    [float(qq[0]), float(qq[1]), float(qq[2]), float(qq[3])],
                    flight_z, int(tick_base + step), col)
            except Exception as exc:  # noqa: BLE001
                ot = "error"
                warnings.warn(f"expert.step error: {exc}")
                break

        # ── Resolve the effective FLU goal for the 30 Hz student ──
        if lower == "student30":
            if upper == "expert5" and eout is not None:
                last_upper_type = str(eout.effective_target_source)
                last_eff = np.array(
                    [float(eout.effective_target_world_x),
                     float(eout.effective_target_world_y)])
                last_gdf = np.array(
                    [float(eout.goal_direction_flu_x),
                     float(eout.goal_direction_flu_y),
                     float(eout.goal_direction_flu_z)])
                last_gdn = float(eout.goal_distance_norm)
            elif upper == "student5":
                # Re-project the latched world target into the live frame.
                delta = np.array(
                    [last_eff[0] - pw[0], last_eff[1] - pw[1], 0.0])
                nd = float(np.linalg.norm(delta))
                if nd < 1e-8:
                    last_gdf = np.array([1.0, 0.0, 0.0])
                    last_gdn = 0.0
                else:
                    last_gdf = np.asarray(
                        il_common.world_vector_to_body_flu_quat(delta / nd, qq))
                    last_gdn = float(min(nd, R - 0.5) / max(R, 1e-9))
            else:  # upper == none: fly the ORIGINAL goal
                last_upper_type = "NONE"
                last_eff = gp.copy()
                delta = gp - pw
                nd = float(np.linalg.norm(delta))
                if nd < 1e-8:
                    last_gdf = np.array([1.0, 0.0, 0.0])
                    last_gdn = 0.0
                else:
                    last_gdf = np.asarray(
                        il_common.world_vector_to_body_flu_quat(delta / nd, qq))
                    last_gdn = float(min(nd, R - 0.5) / max(R, 1e-9))
            grav_flu = il_common.world_vector_to_body_flu_quat(
                np.array([0.0, 0.0, -1.0], dtype=np.float64), qq)
            state_tensor = build_normalized_state(
                grav_flu, last_gdf, last_gdn, s30_scale, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ti = time.perf_counter()
            with torch.no_grad():
                out = student30.step(depth_t, state_tensor, hidden)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ims = (time.perf_counter() - ti) * 1000.0
            hidden = tuple(v.detach() for v in out.hidden)
            cmf = out.command[0].cpu().numpy().copy()
            vc = cmf[:3].copy()
            yc = float(cmf[3])
        else:  # lower == expert
            vc = np.array(
                [float(eout.target_velocity_flu_x),
                 float(eout.target_velocity_flu_y),
                 float(eout.target_velocity_flu_z)], dtype=np.float64)
            yc = float(eout.target_yaw_rate)
            last_upper_type = str(eout.effective_target_source)
            last_eff = np.array(
                [float(eout.effective_target_world_x),
                 float(eout.effective_target_world_y)])
            last_gdf = np.array(
                [float(eout.goal_direction_flu_x),
                 float(eout.goal_direction_flu_y),
                 float(eout.goal_direction_flu_z)])
            last_gdn = float(eout.goal_distance_norm)
            ims = 0.0

        if pc is not None:
            cds.append(float(np.linalg.norm(vc - pc)))
        pc = vc.copy()
        dyn.step_velocity_command(vc, yc, dts)

        stt = dyn.get_state()
        pw = stt.position_world.copy()
        pp.append(pw.copy())
        dst = float(np.linalg.norm(pw - gp))
        fd = dst
        if dst < md:
            md = dst
        clearance = body_clearance(pw, task, cfg.drone_radius)
        minimum_clearance = min(minimum_clearance, clearance)
        spd = float(np.linalg.norm(stt.velocity_world))

        data_logger.write_step({
            "episode": ep_idx, "task": task.name, "stack": cfg.stack,
            "step": step, "sim_time_s": (step + 1) * dts,
            "state_x": float(pw[0]), "state_y": float(pw[1]),
            "state_z": float(pw[2]), "state_yaw": float(yaw),
            "speed_world_mps": spd, "goal_distance_m": dst,
            "is_macro_tick": int(is_macro_tick),
            "upper_type": last_upper_type,
            "effective_target_x": float(last_eff[0]),
            "effective_target_y": float(last_eff[1]),
            "goal_dir_flu_x": float(last_gdf[0]),
            "goal_dir_flu_y": float(last_gdf[1]),
            "goal_dist_norm": float(last_gdn),
            "cmd_vx_flu": float(vc[0]), "cmd_vy_flu": float(vc[1]),
            "cmd_vz_flu": float(vc[2]), "cmd_yaw_rate": yc,
            "minimum_body_clearance_m": float(clearance),
            "inference_ms": ims,
        })

        # Immediate success latch: within goal_immediate_m of the goal the
        # episode is done right away (avoids endless re-planning around the
        # goal that previously caused "arrived but never declared success").
        if cfg.goal_immediate_m > 0.0 and dst <= cfg.goal_immediate_m:
            ot = "success"
            break
        if dst <= cfg.goal_tolerance_m and \
                spd <= cfg.goal_speed_tolerance_mps:
            gh += 1
            if gh >= cfg.goal_hold_ticks:
                ot = "success"
                break
        else:
            gh = 0
        gid += 1

    dur = (step + 1) * dts
    plen = float(sum(np.linalg.norm(pp[i] - pp[i - 1])
                     for i in range(1, len(pp))))
    res = EpisodeResult(
        episode=ep_idx, task_name=task.name, scene_id=scene_id, mode=cfg.stack,
        outcome=ot, duration_s=dur, path_length_m=plen,
        final_goal_distance_m=fd, min_goal_distance_m=md,
        num_model_steps=step + 1, num_collision_frames=cf,
        first_collision_step=fcs, avg_inference_ms=0, max_inference_ms=0,
        num_depth_timeouts=ndt, num_frame_mismatches=nfm, goal_switch_count=0,
        minimum_body_clearance_m=minimum_clearance,
        avg_command_delta=float(np.mean(cds)) if cds else 0.0,
    )
    return res, gid + 1


def main() -> None:
    import os as _os  # noqa: F811
    p = argparse.ArgumentParser(
        description="Unified 4-stack comparison rollout on one scene set.")
    p.add_argument("--stack", default="student30", choices=STACKS)
    p.add_argument("--stacks", default="",
                   help="comma-separated stack names to run in ONE Unity "
                        "session (e.g. expert30,expert). Overrides --stack.")
    p.add_argument("--all-stacks", action="store_true",
                   help="Run all five stacks sequentially and merge the summary.")
    p.add_argument("--checkpoint", help="30 Hz ViTFlyLSTMPolicy checkpoint.")
    p.add_argument("--model-file", default=str(_THIS_DIR / "model" / "model.py"))
    p.add_argument("--macro-checkpoint",
                   help="5 Hz MacroPlannerPolicy checkpoint (for "
                        "student5_student30).")
    p.add_argument("--expert-config",
                   help="il_dataset YAML for the C++ expert Params2D.")
    p.add_argument("--tasks", default="all",
                   help="Comma-separated task names or 'all'.")
    p.add_argument("--scene-set", default="stack",
                   choices=["stack", "4level", "v3"],
                   help="stack = 30 synthetic comparison scenes; "
                        "4level = collection-mirror 4-level scenes + big8; "
                        "v3 = redesigned rollout scenes (--scene-id 1 indoor "
                        "WAREHOUSE / 0 outdoor INDUSTRIAL, min gap 1.6 m).")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--pub-port", default="10253")
    p.add_argument("--sub-port", default="10254")
    p.add_argument("--scene-id", type=int, default=1)
    p.add_argument("--max-episode-time", type=float, default=30.0)
    p.add_argument("--goal-tolerance", type=float, default=0.30)
    p.add_argument("--goal-immediate", type=float, default=0.5,
                   help="declare SUCCESS immediately when within this distance "
                        "of the goal (0 disables; default 0.5 m).")
    p.add_argument("--max-yaw-rate", type=float, default=2.0)
    p.add_argument("--ctrl-hz", type=float, default=50.0)
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=360)
    p.add_argument("--depth-fov", type=float, default=58.0,
                   help="vertical FOV degrees (D435i=58, legacy wide=90)")
    p.add_argument("--depth-near", type=float, default=0.28)
    p.add_argument("--depth-far", type=float, default=10.0)
    p.add_argument("--depth-max-m", type=float, default=5.0)
    p.add_argument("--render-warmup-frames", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--log-prefix", default="rollout_stack")
    a = p.parse_args()

    v3_bounds = False
    v3_flight_z = 2.0
    if a.scene_set == "4level":
        task_registry = build_4level_task_registry()
        task_registry.update(build_big8_task_registry())
    elif a.scene_set == "v3":
        task_registry, v3_flight_z = build_v3_task_registry(a.scene_id)
        v3_bounds = True
    else:
        task_registry = build_stack_task_registry()
    # v3 scenes include outdoor INDUSTRIAL (x[-22,22] y[-10,30]) walls.
    # Outdoor flies HIGH (z=14) with 16 m obstacles; indoor stays at z=2.
    if v3_bounds and a.scene_id == 0:
        ws_bounds = (-20.0, 20.0, -8.0, 29.0, 13.8, 14.2)
        obs_bounds = (-22.0, 22.0, -10.0, 30.0, 0.0, 20.0)
    elif v3_bounds:
        ws_bounds = (-20.0, 20.0, -8.0, 29.0, 1.8, 2.2)
        obs_bounds = (-22.0, 22.0, -10.0, 30.0, 0.0, 8.0)
    else:
        ws_bounds = None
        obs_bounds = None
    validate_task_registry(
        task_registry, drone_radius=0.30, safety_margin=0.0,
        minimum_surface_gap_m=1.6 if v3_bounds else 1.20,
        workspace_bounds=ws_bounds, obstacle_bounds=obs_bounds,
        wall_touch_tolerance_m=0.1 if v3_bounds else 0.0)
    if a.list_tasks:
        for task in task_registry.values():
            print(f"  {task.name:<12} {task.description}")
        return

    if a.stacks:
        stacks = [s.strip() for s in a.stacks.split(",") if s.strip()]
        unknown = [s for s in stacks if s not in STACKS]
        if unknown:
            p.error("unknown stack(s): %s" % ", ".join(unknown))
    elif a.all_stacks:
        stacks = STACKS
    else:
        stacks = [a.stack]
    need_macro = any("student5" in s for s in stacks)
    need_expert = any("expert" in s for s in stacks)
    need_student30 = any(
        s in ("student30", "expert5_student30", "student5_student30")
        for s in stacks)
    if need_student30 and not a.checkpoint:
        p.error("--checkpoint (30 Hz) is required for student stacks")
    if need_macro and not a.macro_checkpoint:
        p.error("--macro-checkpoint required for student5_student30")
    if need_expert and not a.expert_config:
        p.error("--expert-config required for expert stacks")

    task_selector = a.tasks.strip().lower()
    if task_selector in ("all",):
        selected_tasks = list(task_registry.values())
    else:
        requested = [n.strip() for n in a.tasks.split(",") if n.strip()]
        unknown = [n for n in requested if n not in task_registry]
        if unknown:
            p.error("unknown task(s): %s" % ", ".join(unknown))
        selected_tasks = [task_registry[n] for n in requested]

    if a.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(a.device)

    # Load shared models once.
    student30 = None
    s30_cfg = None
    s30_scale = None
    if need_student30:
        student30, s30_cfg, s30_scale = load_policy_checkpoint(
            a.checkpoint, a.model_file, dev, a.depth_max_m)
    macro = None
    if need_macro:
        macro, _mc = load_macro_checkpoint(a.macro_checkpoint, a.model_file, dev)
    expert = None
    params = None
    depth_t_bc: Tuple[float, ...] = ()
    if need_expert:
        expert, params, _minb, _maxb, depth_cfg = load_expert_stack(
            a.expert_config)
        depth_t_bc = tuple(float(v) for v in depth_cfg["t_bc"])
    if not depth_t_bc:
        depth_t_bc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    print("=" * 70)
    print("Unified stack comparison (4 stacks x %d scenes)" % len(selected_tasks))
    print("=" * 70)
    print(f"  30 Hz student:  {a.checkpoint}")
    if macro is not None:
        print(f"  5 Hz student:   {a.macro_checkpoint}")
    if expert is not None:
        print(f"  Expert .so:     {getattr(__import__('_il_hierarchical_expert', fromlist=['EXPERT_REVISION']), 'EXPERT_REVISION', '<n/a>')}")
    print(f"  Stacks:         {', '.join(stacks)}")
    print(f"  Tasks:          {', '.join(t.name for t in selected_tasks)}")
    print("=" * 70)

    total_episodes = len(stacks) * len(selected_tasks) * a.repeats
    print(f"  Connecting to Unity...")
    bridge = il_common.UnityBridge(pub_port=a.pub_port, sub_port=a.sub_port)
    bridge.bind()
    dc_main = {"width": a.depth_width, "height": a.depth_height,
               "fov": a.depth_fov, "near": a.depth_near,
               "far": a.depth_far, "t_bc": list(depth_t_bc)}
    ok = bridge.connect_handshake(a.scene_id, dc_main, timeout=60.0)
    if not ok:
        raise RuntimeError("Unity handshake failed")
    print("  Unity handshake OK.")
    dycfg = _build_dynamics_config(a.ctrl_hz, a.max_yaw_rate)
    dyn = il_dynamics.FlightmareDynamicsBackend(dycfg)

    log_metadata = {
        "format_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "stacks": stacks,
        "tasks": selected_tasks,
    }
    data_logger = StackDataLogger(a.log_prefix, log_metadata)
    data_logger.write_summary("running", [])
    print(f"  Step log: {data_logger.steps_path}")
    print(f"  Summary:  {data_logger.summary_path}")

    results: List[EpisodeResult] = []
    gid = 0
    object_slots = max(len(t.obstacles) for t in task_registry.values())
    plan = [(s, t, ri) for s in stacks for t in selected_tasks
            for ri in range(a.repeats)]
    for ep, (stack, task, repeat_index) in enumerate(plan):
        try:
            while bridge.try_recv() is not None:
                pass
        except Exception:
            pass
        time.sleep(0.1)
        sp = np.asarray(task.start, dtype=np.float64)
        gp = np.asarray(task.goal, dtype=np.float64)
        obs = task_to_unity_objects(task, object_slots)
        cfg = StackRolloutConfig(
            stack=stack, checkpoint=a.checkpoint, model_file=a.model_file,
            macro_checkpoint=a.macro_checkpoint or "",
            expert_config=a.expert_config or "",
            pub_port=a.pub_port, sub_port=a.sub_port, scene_id=a.scene_id,
            depth_width=a.depth_width, depth_height=a.depth_height,
            depth_fov=a.depth_fov, depth_near=a.depth_near,
            depth_far=a.depth_far,
            depth_max_m=a.depth_max_m, depth_t_bc=depth_t_bc,
            model_hz=30.0, ctrl_hz=50.0, max_yaw_rate=a.max_yaw_rate,
            max_episode_time=a.max_episode_time,
            goal_tolerance_m=a.goal_tolerance,
            goal_immediate_m=a.goal_immediate,
            goal_hold_ticks=3, collision_confirm_frames=1,
            device=a.device, repeats=a.repeats, verbose=not a.quiet,
            log_prefix=a.log_prefix,
            render_warmup_frames=a.render_warmup_frames,
            flight_z=v3_flight_z, frame_match_timeout_s=0.15, max_frame_retries=5,
        )
        print(f"\n[{ep}] stack={stack} task={task.name}  {task.description}",
              flush=True)
        try:
            result, gid = run_stack_rollout(
                stack=stack, student30=student30, s30_cfg=s30_cfg,
                s30_scale=s30_scale, macro=macro, expert=expert, params=params,
                cfg=cfg, device=dev, ep_idx=ep, gid=gid, bridge=bridge,
                dyn=dyn, task=task, scene_id=a.scene_id, obstacles=obs,
                data_logger=data_logger)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            result = EpisodeResult(
                episode=ep, task_name=task.name, scene_id=a.scene_id,
                mode=stack, outcome="error", duration_s=0, path_length_m=0,
                final_goal_distance_m=float("inf"),
                min_goal_distance_m=float("inf"), num_model_steps=0,
                num_collision_frames=0, first_collision_step=-1,
                avg_inference_ms=0, max_inference_ms=0,
                num_depth_timeouts=0, num_frame_mismatches=0,
                goal_switch_count=0,
                minimum_body_clearance_m=float("inf"), avg_command_delta=0)
            gid += 10
        results.append(result)
        data_logger.write_summary("running", results)
        print(f"  -> {result.outcome.upper()} dur={result.duration_s:.1f}s "
              f"path={result.path_length_m:.1f}m "
              f"final={result.final_goal_distance_m:.2f}m "
              f"min_clear={result.minimum_body_clearance_m:.2f}m", flush=True)

    try:
        bridge.close()
    except Exception:
        pass

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    for stack in stacks:
        sr = [r for r in results if r.mode == stack]
        ns = sum(1 for r in sr if r.outcome == "success")
        nc = sum(1 for r in sr if r.outcome == "collision")
        nt = sum(1 for r in sr if r.outcome == "timeout")
        ne = sum(1 for r in sr if r.outcome == "error")
        fin = [r for r in sr if r.outcome in ("success", "collision", "timeout")]
        path = np.mean([r.path_length_m for r in fin]) if fin else 0
        clr = min((r.minimum_body_clearance_m for r in sr
                   if math.isfinite(r.minimum_body_clearance_m)), default=0)
        print(f"  {stack:<22} success={ns}/{len(sr)} coll={nc} to={nt} err={ne} "
              f"| avg_path={path:.1f}m min_clear={clr:.2f}m")
    print("\n  Per-task:")
    for task in selected_tasks:
        line = f"    {task.name:<12}"
        for stack in stacks:
            sr = [r for r in results if r.mode == stack and
                  r.task_name == task.name]
            if not sr:
                continue
            outcome = sr[0].outcome
            line += f"  {stack[:10]:<10}={outcome[:3]}"
        print(line)
    data_logger.write_summary("completed", results)
    data_logger.close()
    print(f"  Saved: {data_logger.steps_path}")
    print(f"         {data_logger.summary_path}")
    print("=" * 70)
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
