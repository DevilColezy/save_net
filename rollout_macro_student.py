#!/usr/bin/env python3
"""Closed-loop 5 Hz MACRO-STUDENT rollout: 5 Hz student decides the target,
the C++ 30 Hz expert executes it.

This is the INVERSE of ``rollout_hierarchical.py``:

  * rollout_hierarchical.py  = C++ 5 Hz expert (upper) + 30 Hz student (lower)
  * rollout_macro_student.py = 5 Hz MacroPlannerPolicy STUDENT (upper)
                               + C++ 30 Hz LocalPlanner30Hz EXPERT (lower)

The C++ expert runs completely unchanged at 30 Hz (FSM, local planner,
effective-target adapter, brake/terminal logic).  Its 5 Hz corrector is
BYPASSED via the new ``set_external_directive`` pybind interface: on every
5 Hz boundary the student's decision (PASS / NORMAL / TURN, world-latched
target) is injected, and the 30 Hz local planner then flies toward it
exactly as it would fly toward the expert's own correction.  This isolates
the 5 Hz planner's quality — if the student picks a wrong/colliding target,
the expert WILL faithfully fly into it.

Student decision rule (derived from the col_4 label distribution, 22911
macro rows):

  * ``distance_norm > 0.95``  -> TURN (pure rotation; FLU-y sign picks side)
  * else angle(student dir, original goal dir) > 12 deg -> NORMAL_CORRECTION
  * else PASS_THROUGH (copy the original goal)

col_4 evidence: PASS distance <= 0.900 (median 0.900), TURN == 1.000 exactly;
PASS direction angle <= 14.3 deg (median 0), NORMAL median 15.2 deg.

Three upper modes are supported:

  * ``--upper student`` — the 5 Hz MacroPlannerPolicy decides and is
    injected into the expert (the macro-student test).
  * ``--upper expert``  — the pure C++ 5 Hz expert decides (baseline).
  * ``--upper pass``    — the upper layer is FORCED to PASS on every 5 Hz
    boundary (the expert's own corrector is suppressed via an injected
    PASS directive).  This is the "upper layer does nothing" control: the
    30 Hz LocalPlanner30Hz flies alone.  Scenes where ``--upper pass``
    FAILS (collision / timeout / stuck) but ``--upper expert`` SUCCEEDS are
    precisely the scenes where a 5 Hz upper planner is INDISPENSABLE — use
    this to validate that a macro test scene actually requires a takeover.

The scenes in ``MACRO_SCENES`` are deliberately MACRO-targeted (unlike the
general avoid scenes): rear/FOV-out goals (force TURN), corridor blocked by
a big cylinder (force NORMAL side choice), consecutive zig-zag detours
(force multiple 5 Hz decisions), a late blocker (mid-flight decision), a
big obstacle behind which the goal lies, and a clear baseline (the student
must stay PASS).  Run ``--upper pass`` first on any scene: it must FAIL
for the scene to be a genuine upper-layer takeover test.  Run ``--upper
expert`` on the same scenes to get the pure C++ expert baseline.
    python3 rollout_macro_student.py \
        --checkpoint checkpoints/macro_v1_col4_7d/best.pt \
        --expert-config ../il_dataset/config/il_dataset_joint_v2_config.yaml \
        --tasks macro_suite --upper expert --log-prefix macro_expert_baseline
"""

import argparse
import importlib.util
import json
import math
import random
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
    DEFAULT_STATE_SCALE,
    Cylinder,
    EpisodeResult,
    RolloutDataLogger,
    RolloutTask,
    body_clearance,
    build_normalized_state,
    canonicalize_unity_depth,
    preprocess_depth,
    task_to_unity_objects,
    validate_task_registry,
    _build_dynamics_config,
    _yaw_from_quat_xyzw,
)
from rollout_hierarchical import (  # noqa: E402
    SchemaV25EpisodeWriter,
    load_expert_stack,
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

# ============================================================================
#  Decision thresholds (from the col_4 macro label distribution)
# ============================================================================
TURN_DIST_THRESHOLD = 0.95      # TURN rows carry distance_norm == 1.0 exactly
NORMAL_ANGLE_DEG = 12.0         # PASS <= 14.3 deg (median 0), NORMAL median 15.2

# ============================================================================
#  Macro-targeted test scenes
# ============================================================================
# (name, obstacles [(x, y, r)], tasks [(sx, sy, gx, gy, label, start_yaw_rad)])
# start_yaw is the FM convention (yaw=0 -> nose toward world +Y).
def _deg(yaw_deg: float) -> float:
    return math.radians(yaw_deg)


MACRO_SCENES = [
    # ── Baseline: straight clear flight; the student MUST stay PASS ──
    {
        "name": "M_clear",
        "description": "clear straight: student must stay PASS, no takeover",
        "obstacles": [],
        "tasks": [
            (0.0, 3.0, 0.0, 15.0, "macro_clear", 0.0),
        ],
    },
    # ── Rear / FOV-out goal: forces a TURN decision (student upper) ──
    {
        "name": "M_rear_left",
        "description": "goal LEFT-BEHIND (out of FOV) + blocker ahead: must TURN_LEFT",
        "obstacles": [(0.0, 9.0, 0.8)],
        "tasks": [
            (0.0, 3.0, -4.0, 1.0, "macro_rear_left", 0.0),
        ],
    },
    # ── Goal DIRECTLY BEHIND (180 deg, out of FOV): local cannot rotate ──
    #    R29 single-mode: a target outside the physical FOV is an immediate
    #    NO_SAFE_CANDIDATE handoff.  Verified: --upper pass times out with
    #    path=0.0 m (the drone parks for 30 s).  The upper MUST TURN.
    {
        "name": "M_rear",
        "description": "goal DIRECTLY behind (180 deg, FOV-out): upper MUST TURN",
        "obstacles": [],
        "tasks": [
            (0.0, 12.0, 0.0, 5.0, "macro_rear", 0.0),
        ],
    },
    # ── Goal to the RIGHT-BEHIND (144 deg, FOV-out) + blocker ahead ──
    {
        "name": "M_rear_right",
        "description": "goal RIGHT-BEHIND (144 deg, FOV-out): upper MUST TURN_RIGHT",
        "obstacles": [(0.0, 9.0, 0.8)],
        "tasks": [
            (0.0, 12.0, 5.0, 5.0, "macro_rear_right", 0.0),
        ],
    },
    # ── Max-size blocker (r=3.5) with the goal DIRECTLY BEHIND it: the
    #    goal bearing is permanently blocked by a 7 m wide wall; the local
    #    planner sees only an arc of the cylinder and cannot decide which
    #    side to bypass.  The upper must latch a lateral correction target.
    {
        "name": "M_big_blocker",
        "description": "r=3.5 wall hides the goal behind it: upper must latch "
                       "a lateral target",
        "obstacles": [(0.0, 14.0, 3.5)],
        "tasks": [
            (0.0, 5.0, 0.0, 23.0, "macro_big_blocker", 0.0),
        ],
    },
    # ── Wide picket wall (5 x r=0.8 cylinders, 13.6 m wide) across the
    #    straight line.  The gaps are 1.4 m (drone radius 0.3 -> flyable)
    #    but the wall spans far beyond the 5 m local observation patch, so
    #    the local planner sees a wall that fills the FOV and cannot reason
    #    about the bypass side; the upper's historical map can.
    {
        "name": "M_wide_wall",
        "description": "13.6 m picket wall: local FOV cannot see the ends, "
                       "upper must choose the bypass side",
        "obstacles": [(-6.0, 14.0, 0.8), (-3.0, 14.0, 0.8), (0.0, 14.0, 0.8),
                      (3.0, 14.0, 0.8), (6.0, 14.0, 0.8)],
        "tasks": [
            (0.0, 4.0, 0.0, 24.0, "macro_wide_wall", 0.0),
        ],
    },
    # ── Big-cylinder wall: 3 x r=1.8 cylinders (13.2 m wide) across the
    #    straight line; the goal is directly behind the central cylinder.
    #    All three cylinders are inside the 5 m FOV patch when the drone
    #    reaches planning range, so the local planner sees a continuous
    #    wall filling the FOV.  Forces the WAYPOINT (NORMAL) mode.
    #    Verified: --upper pass times out; --upper expert succeeds with
    #    132 NORMAL_CORRECTION waypoint steps (plans a lateral bypass).
    #    The 5 Hz student currently FAILS here (keeps feeding a 32 deg
    #    NORMAL target that lands inside the wall and parks the drone) —
    #    a genuine student-vs-expert capability gap on wall bypassing.
    {
        "name": "M_big_wall",
        "description": "13.2 m wall of 3 x r=1.8, goal behind: upper must "
                       "plan a waypoint (NORMAL) bypass",
        "obstacles": [(-4.8, 12.0, 1.8), (0.0, 12.0, 1.8), (4.8, 12.0, 1.8)],
        "tasks": [
            (0.0, 4.0, 0.0, 20.0, "macro_big_wall", 0.0),
        ],
    },
    # ── SINGLE max-radius cylinder parked CLOSE ahead: r=3.5 at 4.0 m from
    #    the start fills the whole 90 deg FOV (half-angle 61 deg > 45 deg),
    #    so the local planner's rays are ALL truncated at the cylinder and
    #    it cannot see a bypass side at all — the drone must plan around it.
    #    This is the single-cylinder case that DOES exist in the col_4
    #    training distribution (r_max 3.49), unlike the multi-cylinder wall.
    #    Goal is directly ahead (in FOV) to force the WAYPOINT (NORMAL) mode.
    {
        "name": "M_single_wall",
        "description": "single r=3.5 cylinder 4 m ahead fills the FOV: upper "
                       "must plan a waypoint (NORMAL) bypass",
        "obstacles": [(0.0, 8.0, 3.5)],
        "tasks": [
            (0.0, 4.0, 0.0, 18.0, "macro_single_wall", 0.0),
        ],
    },
]


def build_macro_task_registry() -> Dict[str, RolloutTask]:
    """Build one ``RolloutTask`` per macro test task (explicit start yaw)."""
    tasks: Dict[str, RolloutTask] = {}
    for scene_index, sc in enumerate(MACRO_SCENES):
        obstacles = tuple(
            Cylinder(float(o[0]), float(o[1]), float(o[2]))
            for o in sc["obstacles"])
        for sx, sy, gx, gy, label, syaw in sc["tasks"]:
            tasks[label] = RolloutTask(
                name=label,
                description=sc["description"],
                start=(float(sx), float(sy), 2.0),
                goal=(float(gx), float(gy), 2.0),
                start_yaw=float(syaw),
                obstacles=obstacles,
                suite="macro",
                scene_id=scene_index,
            )
    return tasks


# ============================================================================
#  Checkpoint loader (MacroPlannerPolicy, not the 30 Hz ViTFlyLSTMPolicy)
# ============================================================================
def load_macro_checkpoint(checkpoint_file: str, model_file: str,
                          device: torch.device):
    """Load a train_macro.py checkpoint and return (model, config)."""
    spec = importlib.util.spec_from_file_location(
        "macro_model_runtime", str(model_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load model file %s" % model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if checkpoint.get("architecture") != "MacroPlannerPolicy":
        raise ValueError(
            "%s is not a MacroPlannerPolicy checkpoint (architecture=%r)" %
            (checkpoint_file, checkpoint.get("architecture")))
    cfg = model_module.MacroPolicyConfig(**checkpoint["model_config"])
    model = model_module.MacroPlannerPolicy(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


# ============================================================================
#  Student decision
# ============================================================================
def student_decide(direction_flu: np.ndarray, distance_norm: float,
                   original_dir_flu: np.ndarray):
    """Map the 5 Hz student's regression output onto the four directive
    classes using the col_4-derived thresholds.  Returns
    (type_int, type_name, angle_deg) where type_int 0=PASS 1=NORMAL
    2=TURN_LEFT 3=TURN_RIGHT."""
    d = np.asarray(direction_flu, dtype=np.float64).reshape(3)
    nd = float(np.linalg.norm(d))
    if nd < 1e-8:
        return 0, "PASS_THROUGH", 0.0
    d = d / nd
    if distance_norm > TURN_DIST_THRESHOLD:
        if d[1] > 0.0:   # FLU +y = left
            return 2, "TURN_LEFT", 0.0
        return 3, "TURN_RIGHT", 0.0
    o = np.asarray(original_dir_flu, dtype=np.float64).reshape(3)
    no = float(np.linalg.norm(o))
    if no < 1e-8:
        return 0, "PASS_THROUGH", 0.0
    o = o / no
    cosang = float(np.clip(np.dot(d, o), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosang))
    if angle_deg > NORMAL_ANGLE_DEG:
        return 1, "NORMAL_CORRECTION", angle_deg
    return 0, "PASS_THROUGH", angle_deg


# ============================================================================
#  Per-tick logger with the 5 Hz student decision columns
# ============================================================================
class MacroStudentDataLogger(RolloutDataLogger):
    """Step telemetry with the 5 Hz student's decision columns appended."""

    _MS_COLUMNS = [
        "episode", "task", "step", "sim_time_s",
        "state_x", "state_y", "state_z", "state_yaw",
        "speed_world_mps", "goal_distance_m",
        "is_macro_tick",
        # Student upper decision (on macro ticks; ZOH otherwise).
        "student_directive_type", "student_angle_deg", "student_dist_norm",
        "student_dir_flu_x", "student_dir_flu_y", "student_dir_flu_z",
        "student_dir_world_x", "student_dir_world_y",
        # Expert effective target actually executed by the 30 Hz planner.
        "effective_target_source", "effective_target_x", "effective_target_y",
        "expert_goal_dir_flu_x", "expert_goal_dir_flu_y",
        "expert_goal_dist_norm",
        "hierarchical_mode", "planner_status", "emergency_brake",
        "local_corridor_blocked",
        "minimum_body_clearance_m",
        "cmd_vx_flu", "cmd_vy_flu", "cmd_vz_flu", "cmd_yaw_rate",
        "inference_ms",
    ]
    COLUMNS = _MS_COLUMNS


@dataclass
class MacroStudentRolloutConfig:
    checkpoint: str = ""
    model_file: str = ""
    expert_config: str = ""
    upper: str = "student"      # student | expert
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
    collision_confirm_frames: int = 1
    drone_radius: float = 0.3
    state_scale: Tuple[float, ...] = DEFAULT_STATE_SCALE
    device: str = "auto"
    repeats: int = 1
    verbose: bool = True
    log_prefix: str = "rollout_macro_latest"
    render_warmup_frames: int = 5
    flight_z: float = 2.0
    frame_match_timeout_s: float = 0.15
    max_frame_retries: int = 5
    episode_out_root: str = ""


def run_macro_student_rollout(
    student, student_cfg, expert, params, ms_cfg: MacroStudentRolloutConfig,
    device: torch.device, ep_idx: int, gid: int, bridge, dyn, task: RolloutTask,
    scene_id: int, obstacles: List[Dict[str, Any]],
    data_logger: RolloutDataLogger,
    schema_writer: Optional[SchemaV25EpisodeWriter] = None,
) -> Tuple[EpisodeResult, int]:
    """One closed-loop episode: the 5 Hz student (or C++ expert, when
    ``--upper expert``) decides the target every 6 ticks; the C++ 30 Hz
    expert executes it."""
    dts = 1.0 / ms_cfg.model_hz
    episode_time = (
        task.max_episode_time
        if task.max_episode_time is not None
        else ms_cfg.max_episode_time
    )
    mx = int(episode_time * ms_cfg.model_hz)
    dc = {
        "width": ms_cfg.depth_width, "height": ms_cfg.depth_height,
        "fov": ms_cfg.depth_fov, "near": ms_cfg.depth_near,
        "far": ms_cfg.depth_far,
        "t_bc": list(ms_cfg.depth_t_bc),
    }
    ih, iw = ms_cfg.depth_height, ms_cfg.depth_width
    mr = ms_cfg.depth_max_m
    sp = np.asarray(task.start, dtype=np.float64)
    gp = np.asarray(task.goal, dtype=np.float64)
    syaw = float(task.start_yaw)
    flight_z = float(ms_cfg.flight_z)
    tick_base = 0

    expert.reset_task([float(sp[0]), float(sp[1])],
                      [float(gp[0]), float(gp[1])], syaw, tick_base, flight_z)
    expert.clear_external_directive()

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

    # Renderer warm-up (same contract as the collectors / other rollouts).
    try:
        while bridge.try_recv() is not None:
            pass
    except Exception:
        pass
    depth_payload_bytes = iw * ih * 4
    warm_depth = None
    for warmup_index in range(ms_cfg.render_warmup_frames):
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
                f"{warmup_index + 1}/{ms_cfg.render_warmup_frames} timed out")
        gid += 1

    dt = torch.float32
    # 5 Hz student recurrent state (advanced only on macro ticks).
    mhidden = None
    if student is not None:
        mhidden = student.initial_hidden(1, device=device, dtype=dt)
    student_upper = ms_cfg.upper == "student" and student is not None

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
    minimum_clearance = body_clearance(
        rollout_start_position, task, ms_cfg.drone_radius)
    gh = 0
    # Student decision statistics for the summary.
    decision_counts: Dict[str, int] = {
        "PASS_THROUGH": 0, "NORMAL_CORRECTION": 0,
        "TURN_LEFT": 0, "TURN_RIGHT": 0}
    last_student_type = "PASS_THROUGH"
    last_student_angle = 0.0
    last_student_dist = 0.0
    last_student_dir_flu = np.zeros(3)
    last_student_dir_world = np.zeros(2)
    R = float(getattr(params, "obs_range_m", 5.0))

    for step in range(mx):
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
        for attempt in range(ms_cfg.max_frame_retries + 1):
            if attempt > 0:
                gid += 1
                st["frame_id"] = gid
                st["vehicles"] = [veh]
                bridge.send_pose(st)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < ms_cfg.frame_match_timeout_s:
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
                break
            if ms_cfg.verbose and attempt < ms_cfg.max_frame_retries:
                print(
                    f"  [macro-st] Ep {ep_idx} step {step}: frame match "
                    f"timeout on attempt {attempt + 1}/"
                    f"{ms_cfg.max_frame_retries + 1}; retrying", flush=True)
        if col:
            cf += 1
            if fcs < 0:
                fcs = step
            if cf >= ms_cfg.collision_confirm_frames:
                ot = "collision"
                break
        if du is None:
            ndt += 1
            ot = "error"
            warnings.warn(
                f"[macro-st] Ep {ep_idx} step {step}: depth timeout after "
                f"{ms_cfg.max_frame_retries + 1} render attempts")
            break

        is_macro_tick = (step % 6 == 0)

        # ── 5 Hz boundary: the student (or the pure expert) decides ──
        if is_macro_tick:
            if student_upper:
                # Original navigation goal in the live FLU frame (the
                # student's 7-D input goal part — NOT the effective target).
                orig_world = gp - pw
                d_orig = float(np.linalg.norm(orig_world[:2]))
                orig_dir_flu = np.zeros(3)
                if d_orig > 1e-8:
                    orig_dir_flu = il_common.world_vector_to_body_flu_quat(
                        orig_world / d_orig, qq)
                goal_dist_norm = float(
                    min(d_orig, R - 0.5) / max(R, 1e-9))
                grav_flu = il_common.world_vector_to_body_flu_quat(
                    np.array([0.0, 0.0, -1.0], dtype=np.float64), qq)
                state_tensor = build_normalized_state(
                    grav_flu, orig_dir_flu, goal_dist_norm,
                    ms_cfg.state_scale, device)
                depth_t = preprocess_depth(depth_normalized, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ti = time.perf_counter()
                with torch.no_grad():
                    mout = student.step(depth_t, state_tensor, mhidden)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                mts.append((time.perf_counter() - ti) * 1000.0)
                mhidden = tuple(v.detach() for v in mout.hidden)
                dir_flu = mout.direction[0].cpu().numpy().copy()
                dist_norm = float(mout.distance_norm[0, 0].cpu().numpy())
                mtype, mtype_name, mang = student_decide(
                    dir_flu, dist_norm, orig_dir_flu)
                # World-latch the student's direction at decision time.
                dir_world3 = il_common.body_flu_vector_to_world_quat(
                    dir_flu, qq)
                dir_world = np.asarray(dir_world3[:2], dtype=np.float64)
                wn = float(np.linalg.norm(dir_world))
                if wn < 1e-8:
                    dir_world = np.array([0.0, 1.0], dtype=np.float64)
                else:
                    dir_world = dir_world / wn
                last_student_type = mtype_name
                last_student_angle = mang
                last_student_dist = dist_norm
                last_student_dir_flu = dir_flu.copy()
                last_student_dir_world = dir_world.copy()
                decision_counts[mtype_name] = \
                    decision_counts.get(mtype_name, 0) + 1
                # Inject: PASS clears (expert tracks the original goal);
                # NORMAL latches a world target point; TURN latches a world
                # unit direction.
                if mtype == 0:
                    expert.clear_external_directive()
                elif mtype == 1:
                    dist_world = float(dist_norm) * R
                    tx = float(pw[0]) + dir_world[0] * dist_world
                    ty = float(pw[1]) + dir_world[1] * dist_world
                    expert.set_external_directive(
                        1, tx, ty, 0.0, 0.0, float(dist_norm),
                        "STUDENT_NORMAL")
                else:
                    expert.set_external_directive(
                        mtype, 0.0, 0.0,
                        float(dir_world[0]), float(dir_world[1]), 1.0,
                        "STUDENT_TURN")
                if ms_cfg.verbose:
                    print(
                        f"  [macro-st] Ep {ep_idx} 5Hz tick {step}: "
                        f"{mtype_name} angle={mang:.1f}deg "
                        f"dist={dist_norm:.3f} "
                        f"dir_flu=[{dir_flu[0]:+.2f},{dir_flu[1]:+.2f},"
                        f"{dir_flu[2]:+.2f}] dir_world="
                        f"[{dir_world[0]:+.2f},{dir_world[1]:+.2f}]",
                        flush=True)
            elif ms_cfg.upper == "pass":
                # Pass control: the upper layer is FORCED to PASS on every
                # 5 Hz boundary, so the expert's own corrector can never
                # take over.  The 30 Hz LocalPlanner30Hz flies completely
                # alone.  Scenes that FAIL in this mode but SUCCEED with
                # --upper expert/student are genuine takeover tests.
                expert.set_external_directive(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              "FORCE_PASS")
                last_student_type = "PASS_FORCED"
                last_student_angle = 0.0
                last_student_dist = 0.0
                last_student_dir_flu = np.zeros(3)
                last_student_dir_world = np.zeros(2)
            else:
                # Pure expert upper (baseline): nothing to inject.
                last_student_type = "EXPERT"
                last_student_angle = 0.0
                last_student_dist = 0.0
                last_student_dir_flu = np.zeros(3)
                last_student_dir_world = np.zeros(2)

        # ── Expert step: 30 Hz local planner executes the directive ──
        expert_depth = np.flipud(df.astype(np.float64) * 100.0)
        try:
            eout = expert.step(
                [float(pw[0]), float(pw[1]), float(pw[2])], yaw,
                [float(vel[0]), float(vel[1]), float(vel[2])], yaw_rate_body,
                np.ascontiguousarray(expert_depth, dtype=np.float32).ravel(),
                int(ms_cfg.depth_width), int(ms_cfg.depth_height),
                [float(pw[0]), float(pw[1]), float(pw[2])],
                [float(qq[0]), float(qq[1]), float(qq[2]), float(qq[3])],
                flight_z, int(tick_base + step), col)
        except Exception as exc:  # noqa: BLE001
            ot = "error"
            warnings.warn(
                f"[macro-st] Ep {ep_idx} step {step}: expert.step error: {exc}")
            break

        # ── Execute the expert's 30 Hz command ──
        vc = np.array(
            [float(eout.target_velocity_flu_x),
             float(eout.target_velocity_flu_y),
             float(eout.target_velocity_flu_z)], dtype=np.float64)
        yc = float(eout.target_yaw_rate)
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
        clearance = body_clearance(pw, task, ms_cfg.drone_radius)
        minimum_clearance = min(minimum_clearance, clearance)
        spd = float(np.linalg.norm(stt.velocity_world))

        depth_m_frame = du.astype(np.float32) * 0.01
        step_row: Dict[str, Any] = {
            "episode": ep_idx,
            "task": task.name,
            "step": step,
            "sim_time_s": (step + 1) * dts,
            "state_x": float(pw[0]),
            "state_y": float(pw[1]),
            "state_z": float(pw[2]),
            "state_yaw": float(yaw),
            "speed_world_mps": spd,
            "goal_distance_m": dst,
            "is_macro_tick": int(is_macro_tick),
            "student_directive_type": last_student_type,
            "student_angle_deg": float(last_student_angle),
            "student_dist_norm": float(last_student_dist),
            "student_dir_flu_x": float(last_student_dir_flu[0]),
            "student_dir_flu_y": float(last_student_dir_flu[1]),
            "student_dir_flu_z": float(last_student_dir_flu[2]),
            "student_dir_world_x": float(last_student_dir_world[0]),
            "student_dir_world_y": float(last_student_dir_world[1]),
            "effective_target_source": eout.effective_target_source,
            "effective_target_x": float(eout.effective_target_world_x),
            "effective_target_y": float(eout.effective_target_world_y),
            "expert_goal_dir_flu_x": float(eout.goal_direction_flu_x),
            "expert_goal_dir_flu_y": float(eout.goal_direction_flu_y),
            "expert_goal_dist_norm": float(eout.goal_distance_norm),
            "hierarchical_mode": str(eout.hierarchical_mode),
            "planner_status": str(eout.planner_status),
            "emergency_brake": int(eout.emergency_brake),
            "local_corridor_blocked": int(eout.local_corridor_blocked),
            "minimum_body_clearance_m": float(clearance),
            "cmd_vx_flu": float(vc[0]),
            "cmd_vy_flu": float(vc[1]),
            "cmd_vz_flu": float(vc[2]),
            "cmd_yaw_rate": yc,
            "inference_ms": 0.0,
        }
        data_logger.write_step(step_row)

        # Success: reach the ORIGINAL navigation goal.
        if dst <= ms_cfg.goal_tolerance_m and \
                spd <= ms_cfg.goal_speed_tolerance_mps:
            gh += 1
            if gh >= ms_cfg.goal_hold_ticks:
                ot = "success"
                break
        else:
            gh = 0

        if ms_cfg.verbose and step % 30 == 0:
            print(
                f"  [{ep_idx}:{step:04d}] dist={dst:.2f}m spd={spd:.2f}m/s | "
                f"src={eout.effective_target_source} "
                f"mode={eout.hierarchical_mode} | "
                f"cmd=[{vc[0]:+.2f},{vc[1]:+.2f},{vc[2]:+.2f},{yc:+.2f}] | "
                f"clear={clearance:.2f}m", flush=True)
        gid += 1

    dur = (step + 1) * dts
    plen = float(sum(np.linalg.norm(pp[i] - pp[i - 1])
                     for i in range(1, len(pp))))
    ai = float(np.mean(mts)) if mts else 0.0
    xi = float(np.max(mts)) if mts else 0.0
    ac = float(np.mean(cds)) if cds else 0.0
    res = EpisodeResult(
        episode=ep_idx, task_name=task.name, scene_id=scene_id, mode="macro-st",
        outcome=ot, duration_s=dur, path_length_m=plen,
        final_goal_distance_m=fd, min_goal_distance_m=md,
        num_model_steps=step + 1, num_collision_frames=cf,
        first_collision_step=fcs, avg_inference_ms=ai, max_inference_ms=xi,
        num_depth_timeouts=ndt, num_frame_mismatches=nfm,
        goal_switch_count=0,
        minimum_body_clearance_m=minimum_clearance, avg_command_delta=ac,
    )
    if student_upper and ms_cfg.verbose:
        print(
            f"  [macro-st] Ep {ep_idx} student decisions: "
            + ", ".join("%s=%d" % (k, v) for k, v in decision_counts.items()),
            flush=True)
    return res, gid + 1


def main() -> None:
    import os as _os  # noqa: F811
    p = argparse.ArgumentParser(
        description="Closed-loop 5 Hz macro-student rollout: the 5 Hz "
                    "MacroPlannerPolicy decides the target, the C++ 30 Hz "
                    "expert executes it.")
    p.add_argument("--checkpoint",
                   help="5 Hz macro checkpoint (MacroPlannerPolicy, from "
                        "train_macro.py).")
    p.add_argument(
        "--model-file", default=str(_THIS_DIR / "model" / "model.py"),
        help="Policy implementation (default: save_net/model/model.py).")
    p.add_argument(
        "--expert-config",
        help="il_dataset YAML used to build the C++ Params2D (e.g. "
             "../il_dataset/config/il_dataset_joint_v2_config.yaml).")
    p.add_argument(
        "--upper", default="student",
        choices=["student", "expert", "pass"],
        help="Who decides at 5 Hz: 'student' (5 Hz MacroPlannerPolicy, "
             "injected into the expert), 'expert' (pure C++ expert "
             "baseline) or 'pass' (upper layer forced PASS — pure 30 Hz "
             "local planner; scenes that FAIL here are genuine takeover "
             "tests).")
    p.add_argument(
        "--tasks", default="macro",
        help="Comma-separated task names, or 'macro' for the full macro "
             "suite.")
    p.add_argument("--list-tasks", action="store_true")
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
    p.add_argument("--max-yaw-rate", type=float, default=2.0)
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=360)
    p.add_argument("--depth-fov", type=float, default=58.0)
    p.add_argument("--depth-near", type=float, default=0.28)
    p.add_argument("--depth-far", type=float, default=10.0)
    p.add_argument("--depth-max-m", type=float, default=5.0)
    p.add_argument("--render-warmup-frames", type=int, default=5)
    p.add_argument("--flight-z", type=float, default=2.0)
    p.add_argument("--frame-match-timeout", type=float, default=0.15)
    p.add_argument("--max-frame-retries", type=int, default=5)
    p.add_argument("--episode-out-root", default="",
                   help="Directory for per-episode schema-v25 data.csv "
                        "(optional, loadable by interactive_trajectory_debug).")
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--log-prefix", default="rollout_macro_latest")
    a = p.parse_args()

    task_registry = build_macro_task_registry()
    validate_task_registry(
        task_registry, drone_radius=0.30, safety_margin=0.0,
        minimum_surface_gap_m=1.20)
    if a.list_tasks:
        print("Macro rollout tasks:")
        for task in task_registry.values():
            print(
                f"  {task.name:<22} suite={task.suite:<6} "
                f"obstacles={len(task.obstacles):>2}  {task.description}")
        return
    if a.upper == "student" and not a.checkpoint:
        p.error("--checkpoint is required with --upper student")
    if not a.expert_config:
        p.error("--expert-config is required (il_dataset YAML for the C++ "
                "expert Params2D)")
    if a.repeats <= 0:
        p.error("--repeats must be > 0")

    task_selector = a.tasks.strip().lower()
    if task_selector in ("macro", "all"):
        selected_tasks = list(task_registry.values())
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

    expert, params, min_b, max_b, depth_cfg = load_expert_stack(a.expert_config)
    print("=" * 64)
    print("Macro-student Rollout - 5 Hz student (upper) + 30 Hz expert (lower)")
    print("=" * 64)
    print(f"  Expert .so revision: {getattr(expert_mod, 'EXPERT_REVISION', '<n/a>')}")
    print(f"  Upper 5 Hz:          {a.upper}")
    if a.upper == "student":
        student, mc = load_macro_checkpoint(a.checkpoint, a.model_file, dev)
        print(f"  Macro checkpoint:    {a.checkpoint}")
        print(f"  Macro state dim:     {mc.state_dim}  "
              f"recurrent={mc.recurrent_hidden_dim}/{mc.recurrent_layers}")
    else:
        student = None
        mc = None
    print(f"  Scene bounds:        {min_b} .. {max_b}")

    ms_cfg = MacroStudentRolloutConfig(
        checkpoint=a.checkpoint, model_file=a.model_file,
        expert_config=a.expert_config, upper=a.upper,
        pub_port=a.pub_port, sub_port=a.sub_port, scene_id=a.scene_id,
        depth_width=a.depth_width, depth_height=a.depth_height,
        depth_fov=a.depth_fov, depth_near=a.depth_near, depth_far=a.depth_far,
        depth_max_m=a.depth_max_m,
        depth_t_bc=tuple(float(v) for v in depth_cfg["t_bc"]),
        model_hz=a.model_hz, ctrl_hz=a.ctrl_hz,
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
        episode_out_root=a.episode_out_root,
    )

    log_metadata = {
        "format_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_config": ms_cfg,
        "tasks": selected_tasks,
        "expert_revision": getattr(expert_mod, "EXPERT_REVISION", "<n/a>"),
        "expert_config": a.expert_config,
    }
    data_logger = MacroStudentDataLogger(a.log_prefix, log_metadata)
    data_logger.write_summary("running", [])

    print(f"  Step log:    {data_logger.steps_path}")
    print(f"  Summary:     {data_logger.summary_path}")
    print("=" * 64)

    print("[macro-st] Connecting to Unity...")
    bridge = il_common.UnityBridge(pub_port=a.pub_port, sub_port=a.sub_port)
    bridge.bind()
    dc_main = {
        "width": a.depth_width, "height": a.depth_height,
        "fov": a.depth_fov, "near": a.depth_near, "far": a.depth_far,
        "t_bc": list(depth_cfg["t_bc"]),
    }
    ok = bridge.connect_handshake(a.scene_id, dc_main, timeout=60.0)
    if not ok:
        raise RuntimeError("Unity handshake failed")
    print("[macro-st] Unity handshake OK.")
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

        schema_writer: Optional[SchemaV25EpisodeWriter] = None
        if ms_cfg.episode_out_root:
            ep_dir = Path(ms_cfg.episode_out_root) / (
                "ep%04d_%s" % (ep, task.name))
            schema_writer = SchemaV25EpisodeWriter(
                str(ep_dir),
                scene_id=(task.scene_id if task.scene_id is not None
                          else a.scene_id),
                task_id=ep,
                navigation_goal_world=[float(gp[0]), float(gp[1]),
                                       float(gp[2])],
                initial_yaw=float(task.start_yaw))
            print(f"  Schema out: {schema_writer.path}")

        print(f"\n{'-' * 64}")
        print(f"Episode {ep + 1}/{len(episode_plan)}  "
              f"(task={task.name}, repeat={repeat_index + 1}/{a.repeats})")
        print(f"  Purpose: {task.description}")
        print(f"  Start: [{sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f}]  yaw={math.degrees(task.start_yaw):.1f}deg")
        print(f"  Goal:  [{gp[0]:.1f},{gp[1]:.1f},{gp[2]:.1f}]  "
              f"dist={np.linalg.norm(gp - sp):.1f}m")
        print(f"{'-' * 64}")
        try:
            result, gid = run_macro_student_rollout(
                student=student, student_cfg=mc, expert=expert, params=params,
                ms_cfg=ms_cfg, device=dev, ep_idx=ep, gid=gid, bridge=bridge,
                dyn=dyn, task=task, scene_id=a.scene_id, obstacles=obs,
                data_logger=data_logger, schema_writer=schema_writer,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[macro-st] Episode {ep} ERROR: {exc}")
            import traceback
            traceback.print_exc()
            result = EpisodeResult(
                episode=ep, task_name=task.name, scene_id=a.scene_id,
                mode="macro-st", outcome="error", duration_s=0, path_length_m=0,
                final_goal_distance_m=float("inf"),
                min_goal_distance_m=float("inf"), num_model_steps=0,
                num_collision_frames=0, first_collision_step=-1,
                avg_inference_ms=0, max_inference_ms=0,
                num_depth_timeouts=0, num_frame_mismatches=0,
                goal_switch_count=0,
                minimum_body_clearance_m=float("inf"), avg_command_delta=0,
            )
            gid += 10
        if schema_writer is not None:
            schema_writer.finalize(
                episode_valid=1 if result.outcome == "success" else 0,
                failure_taxonomy="",
                outcome=result.outcome)
        results.append(result)
        data_logger.write_summary("running", results)
        print(
            f"  -> {result.outcome.upper()} | dur={result.duration_s:.1f}s | "
            f"path={result.path_length_m:.1f}m | "
            f"final_dist={result.final_goal_distance_m:.2f}m "
            f"(min={result.min_goal_distance_m:.2f}m) | "
            f"steps={result.num_model_steps} | "
            f"min_clear={result.minimum_body_clearance_m:.2f}m | "
            f"dto={result.num_depth_timeouts} fmm={result.num_frame_mismatches}",
            flush=True)

    try:
        bridge.close()
    except Exception:
        pass

    print(f"\n{'=' * 64}")
    print("SUMMARY")
    print(f"{'=' * 64}")
    nt = len(results)
    ns = sum(1 for r in results if r.outcome == "success")
    nc = sum(1 for r in results if r.outcome == "collision")
    nto = sum(1 for r in results if r.outcome == "timeout")
    ne = sum(1 for r in results if r.outcome == "error")
    print(f"  Upper:       {a.upper}   (5 Hz student + 30 Hz expert)")
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
            f"    {task.name:<22} success={successes}/{len(task_results)} "
            f"collision={collisions} timeout={timeouts} error={errors}")
    fin = [r for r in results if r.outcome in ("success", "collision", "timeout")]
    if fin:
        print(f"\n  Avg dur:        {np.mean([r.duration_s for r in fin]):.1f}s")
        print(f"  Avg path:       {np.mean([r.path_length_m for r in fin]):.1f}m")
        print(f"  Avg final dist: {np.mean([r.final_goal_distance_m for r in fin]):.2f}m")
    finite_clearances = [r.minimum_body_clearance_m for r in results
                         if math.isfinite(r.minimum_body_clearance_m)]
    if finite_clearances:
        print(f"\n  Safety:  minimum body clearance={min(finite_clearances):.2f}m")
    print(f"  Infra:   depth_timeouts={sum(r.num_depth_timeouts for r in results)}"
          f"  frame_mismatches={sum(r.num_frame_mismatches for r in results)}")
    data_logger.write_summary("completed", results)
    data_logger.close()
    print(f"  Saved:   {data_logger.steps_path}")
    print(f"           {data_logger.summary_path}")
    print(f"{'=' * 64}")
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
