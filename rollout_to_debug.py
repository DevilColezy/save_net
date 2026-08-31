#!/usr/bin/env python3
"""Convert rollout_stack_steps.csv into per-episode directories loadable by
interactive_trajectory_debug.py.

rollout_stack writes ONE flat CSV (episode,task,stack,step,...) while the
debugger expects one directory per episode containing a collector-style
data.csv.  This script:

  * groups rows by (stack, task, episode);
  * writes  <out>/<stack>/<task>_ep<ep>/data.csv  with the collector column
    names the debugger reads (mapping rollout fields onto them);
  * copies the rollout-v3 scene manifest next to the episodes so the debugger
    overlays the real obstacles (r15 cylinder + bypass belt).

Usage:
    python3 rollout_to_debug.py [--steps path/to/rollout_stack_steps.csv] \\
        [--manifest path/to/rollout_v3_scenes_manifest.json] [--out OUT]
"""
import argparse
import csv
import json
import math
import os
import shutil

# rollout_stack column -> collector column used by the debugger.
ROLLOUT_COL = {
    "x": "state_x", "y": "state_y", "z": "state_z",
    "yaw": "state_yaw",
    "episode_frame_index": "step",
    "trajectory_time_s": "sim_time_s",
    "goal_direction_flu_x": "goal_dir_flu_x",
    "goal_direction_flu_y": "goal_dir_flu_y",
    "goal_distance_norm": "goal_dist_norm",
    "effective_target_world_x": "effective_target_x",
    "effective_target_world_y": "effective_target_y",
    "target_velocity_flu_x": "cmd_vx_flu",
    "target_velocity_flu_y": "cmd_vy_flu",
    "target_velocity_flu_z": "cmd_vz_flu",
    "target_yaw_rate": "cmd_yaw_rate",
    "macro_update_mask": "is_macro_tick",
    "min_observed_clearance_m": "minimum_body_clearance_m",
    "inference_ms": "inference_ms",
}
# Collector columns the debugger prints; ones not present stay empty.
DEBUG_COLUMNS = [
    "episode_frame_index", "trajectory_time_s", "control_dt_s",
    "x", "y", "z", "yaw", "yaw_rate",
    "state_vx_flu", "state_vy_flu",
    "effective_target_world_x", "effective_target_world_y",
    "goal_direction_flu_x", "goal_direction_flu_y", "goal_distance_norm",
    "navigation_goal_world_x", "navigation_goal_world_y",
    "original_navigation_goal_world_x", "original_navigation_goal_world_y",
    "target_velocity_flu_x", "target_velocity_flu_y", "target_velocity_flu_z",
    "target_yaw_rate",
    "hierarchical_mode", "planner_status", "planner_failure_reason",
    "plan_valid", "plan_terminal", "local_corridor_blocked",
    "fsm_state", "effective_target_source", "target_correction_active",
    "observability_reason", "plan_points_xy",
    "macro_update_mask", "macro_label_valid", "macro_correction_type",
    "macro_direction_token", "macro_direction_flu_x", "macro_direction_flu_y",
    "macro_distance_norm",
    "min_observed_clearance_m", "truth_minimum_clearance_m",
    "truth_brake_risk", "truth_brake_would_trigger", "emergency_brake",
    "scene_id", "task_id", "inference_ms",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="rollout_stack_steps.csv")
    ap.add_argument("--manifest",
                    default="/home/rgzn/flightmare_ws/il_data_joint_v2/"
                            "rollout_v3_scenes_manifest.json")
    ap.add_argument("--out", default="debug_episodes")
    a = ap.parse_args()

    # scene profile -> scene_id (from the rollout-v3 manifest)
    scene_id_by_task = {}
    if os.path.isfile(a.manifest):
        with open(a.manifest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for scene in payload.get("scenes", []):
            scene_id_by_task[scene.get("profile")] = int(scene["scene_id"])

    with open(a.steps, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("empty steps file: %s" % a.steps)

    groups = {}
    for row in rows:
        key = (row["stack"], row["task"], int(row["episode"]))
        groups.setdefault(key, []).append(row)

    for key, group in groups.items():
        stack, task, ep = key
        episode_dir = os.path.join(a.out, stack, "%s_ep%03d" % (task, ep))
        os.makedirs(episode_dir, exist_ok=True)
        data_path = os.path.join(episode_dir, "data.csv")
        scene_id = scene_id_by_task.get(task, 0)
        with open(data_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DEBUG_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in sorted(group, key=lambda r: int(r["step"])):
                out = {}
                for col in DEBUG_COLUMNS:
                    if col in ("scene_id",):
                        out[col] = scene_id
                    elif col == "task_id":
                        out[col] = ep
                    elif col == "control_dt_s":
                        out[col] = "0.033333"
                    elif col == "macro_correction_type":
                        out[col] = row.get("upper_type", "")
                    elif col == "hierarchical_mode":
                        out[col] = row.get("upper_type", "")
                    elif col == "macro_direction_flu_x":
                        out[col] = row.get("goal_dir_flu_x", "")
                    elif col == "macro_direction_flu_y":
                        out[col] = row.get("goal_dir_flu_y", "")
                    elif col == "macro_distance_norm":
                        out[col] = row.get("goal_dist_norm", "")
                    elif col in ROLLOUT_COL:
                        out[col] = row.get(ROLLOUT_COL[col], "")
                    else:
                        out[col] = ""
                writer.writerow(out)

    # Copy the manifest so the debugger auto-loads the obstacle overlay.
    if os.path.isfile(a.manifest):
        shutil.copy2(a.manifest, os.path.join(a.out, os.path.basename(a.manifest)))
        print("manifest copied: %s" % os.path.join(a.out, os.path.basename(a.manifest)))
    print("wrote %d episodes -> %s" % (len(groups), a.out))


if __name__ == "__main__":
    main()
