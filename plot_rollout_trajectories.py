#!/usr/bin/env python3
"""Plot rollout trajectories as top-down views, overlaying start/goal and
obstacles, to expose the global trajectory shape (e.g. systematic right-turn
detours vs reactive avoidance).

Also prints a per-task side-bias metric: mean signed cross-track offset of the
trajectory relative to the start->goal segment (positive = left of the
segment, negative = right of it), and the fraction of the path spent on the
right side.
"""
import csv
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rollout_hierarchical import (  # noqa: E402
    build_4level_task_registry,
    build_avoid_task_registry,
)


def _task_name_from_dir(d: Path) -> str:
    return d.name.split("_", 1)[1]


def _cross_track(xy: np.ndarray, start, goal) -> np.ndarray:
    """Signed distance of each point from the start->goal line.

    Uses the standard 2D cross-product sign: positive = left of travel
    direction (start->goal), negative = right.
    """
    d = np.array(goal[:2], float) - np.array(start[:2], float)
    d = d / (np.linalg.norm(d) + 1e-9)
    p = xy[:, :2] - np.array(start[:2], float)
    # left-normal n = (-dy, dx)
    n = np.array([-d[1], d[0]], float)
    return p @ n


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        THIS_DIR / "checkpoints" / "vitfly_v26_joint_v2_col3"
        / "rollout_episodes")
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rollout_traj"

    reg = {}
    reg.update(build_avoid_task_registry())
    reg.update(build_4level_task_registry())

    dirs = sorted(root.glob("ep*/"))
    print(f"episodes: {len(dirs)}")

    rows = []  # (suite, name, traj_xy, task)
    for d in dirs:
        name = _task_name_from_dir(d)
        task = reg.get(name)
        csv_path = d / "data.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            pts = [(float(r["x"]), float(r["y"])) for r in reader]
        if not pts:
            continue
        rows.append((task.suite if task else "?", name, np.array(pts), task))

    # --- bias metric ---
    print("\n=== cross-track bias (negative = right of start->goal) ===")
    bias_by_suite = {}
    for suite, name, xy, task in rows:
        if task is None:
            continue
        c = _cross_track(xy, task.start, task.goal)
        frac_right = float((c < -0.05).mean())
        bias_by_suite.setdefault(suite, []).append(
            (name, float(c.mean()), frac_right))
    for suite, vals in bias_by_suite.items():
        means = [v[1] for v in vals]
        rights = [v[2] for v in vals]
        print(f"\n[{suite}] mean_cross={np.mean(means):+.3f}  "
              f"frac_right={np.mean(rights):.3f}")
        for name, m, fr in vals:
            print(f"  {name:18s} cross={m:+.3f}  right_frac={fr:.2f}")

    # --- plot: split by suite ---
    for suite in ("avoid", "4level"):
        subset = [r for r in rows if r[0] == suite]
        if not subset:
            continue
        ncols = 5 if suite == "avoid" else 4
        nrows = int(np.ceil(len(subset) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 2.6, nrows * 2.8))
        axes = np.atleast_1d(axes).ravel()
        for i, (s, name, xy, task) in enumerate(subset):
            ax = axes[i]
            ax.set_aspect("equal")
            # obstacles
            for o in task.obstacles:
                c = plt.Circle((o.x, o.y), o.radius, color="0.75",
                               alpha=0.9, zorder=1)
                ax.add_patch(c)
            # start / goal
            ax.plot(task.start[0], task.start[1], "go", ms=6, zorder=5)
            ax.plot(task.goal[0], task.goal[1], "r*", ms=12, zorder=5)
            # trajectory colored by time
            t = np.linspace(0, 1, len(xy))
            ax.scatter(xy[:, 0], xy[:, 1], c=t, cmap="viridis", s=4,
                       zorder=3, linewidths=0)
            # start->goal reference line
            ax.plot([task.start[0], task.goal[0]],
                    [task.start[1], task.goal[1]], "r--", lw=0.7, zorder=2)
            ax.set_title(name, fontsize=7)
            ax.tick_params(labelsize=5)
            ax.grid(alpha=0.2)
        for j in range(len(subset), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"rollout trajectories — {suite} (green=start, red=goal, "
                     "gray=obstacles, viridis=time)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"{out_prefix}_{suite}.png"
        fig.savefig(out, dpi=110)
        print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
