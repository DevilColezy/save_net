#!/usr/bin/env python3
"""Convert rollout_stack.py's flat steps.csv + summary.json into per-episode
data.csv directories loadable by il_dataset/test/interactive_trajectory_debug.py.

Usage:
    python3 export_stack_episodes.py \
        --steps rollout_d435i_nonoise_steps.csv \
        --summary rollout_d435i_nonoise_summary.json \
        --out-dir rollout_episodes_d435i \
        [--stack student5_student30]   # optional: only export one stack

Output layout (one data.csv per (stack, task) episode):
    <out-dir>/
        stack_manifest.json          # scene_id -> obstacle cylinders (overlay)
        <stack>/
            <task>/data.csv
            stack_manifest.json      # copy so the debugger auto-discovers it
"""

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

# Column set consumed by interactive_trajectory_debug.py (schema-v25 names).
COLUMNS = [
    "episode_frame_index", "trajectory_time_s", "control_dt_s",
    "x", "y", "z", "yaw", "yaw_rate",
    "state_vx_flu", "state_vy_flu", "state_vz_flu",
    "speed_world_mps", "inference_ms",
    "effective_target_world_x", "effective_target_world_y",
    "effective_target_world_z",
    "goal_direction_flu_x", "goal_direction_flu_y",
    "goal_direction_flu_z", "goal_distance_norm",
    "target_velocity_flu_x", "target_velocity_flu_y",
    "target_velocity_flu_z", "target_yaw_rate",
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

CTRL_DT = 1.0 / 30.0  # model_hz=30


def _f(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else ""
    except (TypeError, ValueError):
        return ""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stack", default="", help="only export this stack")
    a = p.parse_args()

    with open(a.summary, encoding="utf-8") as fh:
        summary = json.load(fh)
    task_by_name = {t["name"]: t for t in summary["tasks"]}
    outcome_by_key = {
        (r["task_name"], r["mode"]): r.get("outcome", "?")
        for r in summary.get("results", [])
    }

    # group rows by (stack, task)
    rows = defaultdict(list)
    with open(a.steps, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[(row["stack"], row["task"])].append(row)

    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # manifest: scene_id -> obstacles
    manifest_scenes = []
    for t in summary["tasks"]:
        scene_id = int(t.get("scene_id", 0))
        obs = [
            {"x": float(o["x"]), "y": float(o["y"]),
             "radius": float(o["radius"])}
            for o in t.get("obstacles", [])
            if all(k in o for k in ("x", "y", "radius"))
        ]
        manifest_scenes.append({"scene_id": scene_id, "obstacles": obs})
    manifest_payload = {"scenes": manifest_scenes}
    manifest_path = out_root / "stack_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest_payload, fh, indent=1)

    written = 0
    for (stack, task), group in sorted(rows.items()):
        if a.stack and stack != a.stack:
            continue
        t = task_by_name.get(task)
        if t is None:
            continue
        goal = t["goal"]
        scene_id = int(t.get("scene_id", 0))
        outcome = outcome_by_key.get((task, stack), "?")
        ep_dir = out_root / stack / task
        ep_dir.mkdir(parents=True, exist_ok=True)

        with (ep_dir / "data.csv").open("w", newline="",
                                        encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            for r in group:
                step = r.get("step", "")
                sim_t = _f(r.get("sim_time_s"))
                x, y, z = (r.get("state_x", ""), r.get("state_y", ""),
                           r.get("state_z", ""))
                out = {
                    "episode_frame_index": step,
                    "trajectory_time_s": sim_t,
                    "control_dt_s": round(CTRL_DT, 6),
                    "x": x, "y": y, "z": z,
                    "yaw": r.get("state_yaw", ""),
                    "yaw_rate": "",
                    "state_vx_flu": "", "state_vy_flu": "",
                    "state_vz_flu": "",
                    "speed_world_mps": r.get("speed_world_mps", ""),
                    "inference_ms": r.get("inference_ms", ""),
                    "effective_target_world_x": r.get(
                        "effective_target_x", ""),
                    "effective_target_world_y": r.get(
                        "effective_target_y", ""),
                    "effective_target_world_z": "",
                    "goal_direction_flu_x": r.get("goal_dir_flu_x", ""),
                    "goal_direction_flu_y": r.get("goal_dir_flu_y", ""),
                    "goal_direction_flu_z": "",
                    "goal_distance_norm": r.get("goal_dist_norm", ""),
                    "target_velocity_flu_x": r.get("cmd_vx_flu", ""),
                    "target_velocity_flu_y": r.get("cmd_vy_flu", ""),
                    "target_velocity_flu_z": r.get("cmd_vz_flu", ""),
                    "target_yaw_rate": r.get("cmd_yaw_rate", ""),
                    "student_velocity_flu_x": r.get("cmd_vx_flu", ""),
                    "student_velocity_flu_y": r.get("cmd_vy_flu", ""),
                    "student_velocity_flu_z": r.get("cmd_vz_flu", ""),
                    "student_yaw_rate": r.get("cmd_yaw_rate", ""),
                    "hierarchical_mode": r.get("stack", ""),
                    "planner_status": "", "planner_failure_reason": "",
                    "plan_valid": "", "plan_terminal": "",
                    "plan_points_xy": "",
                    "macro_update_mask": r.get("is_macro_tick", "0"),
                    "macro_label_valid": "", "macro_correction_type": "",
                    "macro_direction_token": "",
                    "macro_direction_flu_x": "",
                    "macro_direction_flu_y": "",
                    "macro_direction_flu_z": "",
                    "macro_distance_norm": "",
                    "min_observed_clearance_m": r.get(
                        "minimum_body_clearance_m", ""),
                    "truth_minimum_clearance_m": "", "truth_brake_risk": "",
                    "truth_brake_would_trigger": "",
                    "emergency_brake": "", "local_corridor_blocked": "",
                    "fsm_state": "",
                    "effective_target_source": r.get("upper_type", ""),
                    "target_correction_active": "",
                    "observability_reason": "",
                    "scene_id": scene_id,
                    "task_id": task,
                    "navigation_goal_world_x": goal[0],
                    "navigation_goal_world_y": goal[1],
                    "navigation_goal_world_z": goal[2],
                    "original_navigation_goal_world_x": goal[0],
                    "original_navigation_goal_world_y": goal[1],
                    "original_navigation_goal_world_z": goal[2],
                    "episode_valid": 1,
                    "failure_taxonomy": outcome,
                }
                w.writerow(out)
        # copy manifest into the stack dir for auto-discovery
        shutil.copy(manifest_path, out_root / stack / "stack_manifest.json")
        written += 1

    print("exported %d episodes -> %s" % (written, out_root))
    print("debugger command example:")
    print("  python3 il_dataset/test/interactive_trajectory_debug.py \\")
    print("      %s/student5_student30" % out_root)


if __name__ == "__main__":
    main()
