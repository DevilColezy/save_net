#!/usr/bin/env python3
"""Unified test benchmark — 6 scales x 4 scenes x 2 tasks = 48 tasks.

Per scale: 3 cylinder scenes (density 2.0 / 1.6 / 1.2) + 1 wall scene.

Cylinder scenes:
  * exactly ONE upper-bound-radius cylinder (the key structure)
  * all other cylinders use the LOWER-bound radius, uniform-random layout,
    min surface gap = density.

Wall scene:
  * one wall fixed at the upper-bound half-width (faces the start, along X)
  * lower-bound cylinders in the two side bands (sparse gap 2.0).

Scale (lower / upper radius):
  small   0.1 / 0.5
  medium  0.5 / 3.0
  large   3.0 / 7.0
  mixed   0.1 / 7.0
  x-large 7.0 / 15.0        (scene0 high-altitude)
  x-mixed 1.0 / 10.0        (scene0 high-altitude)

Tasks: 2 per scene, no yaw bias — centre take-off and side take-off, both
flying straight along +Y through the band.

Writes a JSON manifest and renders one top-down PNG per scene.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser(os.environ.get(
    "UNIFIED_BENCH_DIR", "~/flightmare_ws/unified_benchmark"))
PIC_DIR = os.path.join(OUT_DIR, "pics")


# ---------------------------------------------------------------------------
# Geometry helpers: obstacle = (x, y, r, hw, hh); hw/hh > 0 -> wall box.
# ---------------------------------------------------------------------------
def _aabb_dist(px, py, x, y, hw, hh):
    dx = max(abs(px - x) - hw, 0.0)
    dy = max(abs(py - y) - hh, 0.0)
    return math.hypot(dx, dy)


def surface_gap(o1, o2):
    x1, y1, r1, hw1, hh1 = o1
    x2, y2, r2, hw2, hh2 = o2
    b1 = hw1 > 0 and hh1 > 0
    b2 = hw2 > 0 and hh2 > 0
    if b1 and b2:
        gx = abs(x1 - x2) - (hw1 + hw2)
        gy = abs(y1 - y2) - (hh1 + hh2)
        if gx <= 0 and gy <= 0:
            return -math.hypot(gx, gy)
        if gx <= 0:
            return gy
        if gy <= 0:
            return gx
        return math.hypot(gx, gy)
    if b1:
        return max(_aabb_dist(x2, y2, x1, y1, hw1, hh1) - r2, 0.0)
    if b2:
        return max(_aabb_dist(x1, y1, x2, y2, hw2, hh2) - r1, 0.0)
    return max(math.hypot(x1 - x2, y1 - y2) - r1 - r2, 0.0)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
SCALES = {
    # name: dict(r=(lo,hi), lows=[sparse,mid,dense] lower bounds, n_big,
    #            scene); lows=None -> lower-bound radius uniform in [1,3]
    "small":   {"r": (0.1, 0.5), "lows": [0.30, 0.20, 0.10],
                "n_big": 10, "scene": 1},
    "medium":  {"r": (0.5, 3.0), "lows": [1.50, 1.00, 0.50],
                "n_big": 2, "scene": 1},
    "large":   {"r": (3.0, 7.0), "lows": [5.00, 4.00, 3.00],
                "n_big": 1, "scene": 1},
    "mixed":   {"r": (0.1, 7.0), "lows": [3.00, 1.50, 0.50],
                "n_big": 1, "scene": 1},
    "x-large": {"r": (7.0, 15.0), "lows": [11.00, 9.00, 7.00],
                "n_big": 1, "scene": 0},
    "x-mixed": {"r": (1.0, 10.0), "lows": [3.00, 2.00, 1.00],
                "n_big": 1, "scene": 0},
}

DENSITIES = [("sparse", 2.0), ("mid", 1.8), ("dense", 1.6)]

CHANNELS = {
    1: {"xmin": -8.0, "xmax": 11.0, "ymin": 0.0, "ymax": 34.0},
    0: {"xmin": -17.0, "xmax": 17.0, "ymin": 0.0, "ymax": 75.0},
}

# Obstacle layout uses the ORIGINAL straight-run endpoints (unchanged from
# the pre-2026-09-02 revision): the ±4 m endpoint shift is applied only to
# the task start/goal in prepare_unified_manifest.py, so the obstacle field
# keeps its exact layout and only the spawn/target move.
Y_LAYOUT = {
    "small":   {"start": 1.0, "end": 21.0, "wall_len": 3.0},
    "medium":  {"start": 1.0, "end": 25.0, "wall_len": 4.0},
    "large":   {"start": 1.0, "end": 33.0, "wall_len": 5.0},
    "mixed":   {"start": 1.0, "end": 29.0, "wall_len": 5.0},
    "x-large": {"start": 1.0, "end": 71.0, "wall_len": 6.0},
    "x-mixed": {"start": 1.0, "end": 37.0, "wall_len": 6.0},
}

WALL_THICKNESS = 0.3

# Side-wall keep-out for obstacle placement (m).  scene 1 lost its boundary
# walls in the 2026-09-01 revision; scene 0 keeps them and requires 1.8 m
# between a wall face and any spawned obstacle.
WALL_MARGIN = {1: 0.2, 0: 1.8}

# Wall-scene wall layout per scale. half_w=None -> default single wall whose
# half-width is the scale upper bound; medium uses three 3 m-wide walls on
# one horizontal line with gaps equal to the scene density (1.8 m).
WALL_CONFIG = {
    "medium": {"n_walls": 3, "half_w": 1.5},
}


def _rect_blocked(x, y, r, rect):
    """True if a cylinder centred at (x, y) with radius r would enter rect."""
    x0, x1, y0, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy) < r + 0.05


def _midline_corridor(scale, density, y, ch, bigs=None, walls=None):
    """Obstacle-free midline corridor (half-width = density / 2) from the
    mid-task start up to the primary obstacle, for the last four scales."""
    if scale not in ("large", "mixed", "x-large", "x-mixed"):
        return []
    cx = 0.5 * (ch["xmin"] + ch["xmax"])
    if walls:
        front_y = min(w[1] - w[4] for w in walls)
    elif bigs:
        front_y = min(b[1] - b[2] for b in bigs)
    else:
        return []
    y_start = y["start"]
    if front_y <= y_start:
        return []
    half = 0.5 * density
    return [(cx - half, cx + half, y_start, front_y)]


def scatter_fixed_r(r, density, ch, y0, y1, rng, avoid=(),
                    exclude_rects=(), wall_margin=0.2):
    """Uniform-random cylinders of FIXED radius r, min surface gap >= density.

    ``exclude_rects`` are (x0, x1, y0, y1) corridors kept obstacle-free: a
    cylinder is rejected if its surface would enter a corridor.
    ``wall_margin`` is the keep-out from the channel walls (scene 0: 1.8 m)."""
    x_lo = ch["xmin"] + r + wall_margin
    x_hi = ch["xmax"] - r - wall_margin
    if x_hi <= x_lo or y1 <= y0 + 2 * r:
        return []
    cell = density + 2.0 * r
    area = (x_hi - x_lo) * (y1 - y0)
    target = max(1, int(area / (cell * cell) * 0.95))
    placed = list(avoid)
    out = []
    for _ in range(target * 60):
        if len(out) >= target:
            break
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y0 + r, y1 - r)
        if any(_rect_blocked(x, y, r, rect) for rect in exclude_rects):
            continue
        o = (x, y, r, 0.0, 0.0)
        if all(surface_gap(o, p) >= density for p in placed):
            placed.append(o)
            out.append(o)
    return out


def boundary_walls(scene, ch):
    """Two channel boundary walls along Y (long side along Y, thickness
    0.3 m along X).  Only scene 0 keeps them, so the high-altitude drone
    collides with the side walls instead of exiting the region.  scene 1's
    walls were removed in the 2026-09-01 revision."""
    if scene != 0:
        return []
    y_mid = 0.5 * (ch["ymin"] + ch["ymax"])
    hh = 0.5 * (ch["ymax"] - ch["ymin"])
    return [(ch["xmin"], y_mid, hh, 0.15, hh),
            (ch["xmax"], y_mid, hh, 0.15, hh)]


def add_boundary_walls(scene, ch, obstacles):
    return boundary_walls(scene, ch) + list(obstacles)


def lower_radius(scale, low_idx, rng):
    """Lower-bound radius for a density level (dense -> small, sparse -> big)."""
    lows = SCALES[scale]["lows"]
    if lows is None:  # x-mixed: uniform in [1, 3]
        return rng.uniform(1.0, 3.0)
    return lows[low_idx]


def gen_cylinder_scene(scale, dkey, density, low_idx, seed):
    ch = CHANNELS[SCALES[scale]["scene"]]
    y = Y_LAYOUT[scale]
    scene = SCALES[scale]["scene"]
    wall_margin = WALL_MARGIN[scene]
    rng = random.Random(seed)
    r_lo, r_hi = SCALES[scale]["r"]
    n_big = SCALES[scale]["n_big"]
    y0, y1 = y["start"] + 0.8, y["end"] - 0.8
    cx = 0.5 * (ch["xmin"] + ch["xmax"])
    # n_big upper-bound key cylinders with random jittered positions
    bigs = []
    bx_lo = ch["xmin"] + r_hi + wall_margin
    bx_hi = ch["xmax"] - r_hi - wall_margin
    for _ in range(n_big):
        for _try in range(300):
            bx = rng.uniform(bx_lo, bx_hi)
            by = rng.uniform(y0 + r_hi, y1 - r_hi)
            o = (bx, by, r_hi, 0.0, 0.0)
            if all(surface_gap(o, b) >= density for b in bigs):
                bigs.append(o)
                break
    low_r = lower_radius(scale, low_idx, rng)
    avoid = bigs + boundary_walls(SCALES[scale]["scene"], ch)
    corridor = _midline_corridor(scale, density, y, ch, bigs=bigs)
    smalls = scatter_fixed_r(low_r, density, ch, y0, y1, rng, avoid=avoid,
                             exclude_rects=corridor,
                             wall_margin=wall_margin)
    side_x = ch["xmin"] + 1.5
    tasks = [(cx, y["start"], cx, y["end"], "%s_%s_mid" % (scale, dkey)),
             (side_x, y["start"], side_x, y["end"],
              "%s_%s_side" % (scale, dkey))]
    return {
        "name": "U_%s_%s" % (scale, dkey), "scale": scale,
        "density_key": dkey, "density": density,
        "scene": SCALES[scale]["scene"], "kind": "cylinders",
        "obstacles": add_boundary_walls(SCALES[scale]["scene"], ch,
                                        bigs + smalls),
        "tasks": tasks, "seed": seed,
    }


def gen_wall_scene(scale, seed):
    ch = CHANNELS[SCALES[scale]["scene"]]
    y = Y_LAYOUT[scale]
    scene = SCALES[scale]["scene"]
    wall_margin = WALL_MARGIN[scene]
    rng = random.Random(seed)
    r_lo, r_hi = SCALES[scale]["r"]
    wall_y0 = 0.5 * (y["start"] + y["end"]) - y["wall_len"] / 2.0
    wall_y1 = wall_y0 + y["wall_len"]
    cx = 0.5 * (ch["xmin"] + ch["xmax"])
    density = 1.8  # wall scene side cylinders use MID density
    wcfg = WALL_CONFIG.get(scale, {})
    n_walls = int(wcfg.get("n_walls", 1))
    half_w = wcfg.get("half_w")
    if half_w is None:
        half_w = min(r_hi,
                     (ch["xmax"] - ch["xmin"] - 2.0 * density) / 2.0)
    wall_yc = 0.5 * (wall_y0 + wall_y1)
    wall_gap = density  # wall-to-wall gap = scene density
    if n_walls <= 1:
        wall_xs = [cx]
    else:
        pitch = 2.0 * half_w + wall_gap
        wall_xs = [cx + (i - 0.5 * (n_walls - 1)) * pitch
                   for i in range(n_walls)]
    walls = [(wx, wall_yc, half_w, half_w, WALL_THICKNESS / 2.0)
             for wx in wall_xs]
    low_r = lower_radius(scale, 2, rng)  # smallest (dense) lower bound
    avoid = walls + boundary_walls(SCALES[scale]["scene"], ch)
    corridor = _midline_corridor(scale, density, y, ch, walls=walls)
    c1 = scatter_fixed_r(low_r, density, ch, y["start"] + 0.8, wall_y0, rng,
                         avoid=avoid, exclude_rects=corridor,
                         wall_margin=wall_margin)
    c2 = scatter_fixed_r(low_r, density, ch, wall_y1, y["end"] - 0.8, rng,
                         avoid=avoid, wall_margin=wall_margin)
    side_x = ch["xmin"] + 1.5
    tasks = [(cx, y["start"], cx, y["end"], "%s_wall_mid" % scale),
             (side_x, y["start"], side_x, y["end"], "%s_wall_side" % scale)]
    return {
        "name": "U_%s_wall" % scale, "scale": scale,
        "density_key": "wall", "density": density,
        "scene": SCALES[scale]["scene"], "kind": "wall",
        "obstacles": add_boundary_walls(SCALES[scale]["scene"], ch,
                                        walls + c1 + c2),
        "tasks": tasks, "seed": seed,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_scene(sc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = CHANNELS[sc["scene"]]
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    ax.set_xlim(ch["xmin"] - 2, ch["xmax"] + 2)
    ax.set_ylim(ch["ymin"] - 2, ch["ymax"] + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(ch["xmin"], color="0.2", lw=2)
    ax.axvline(ch["xmax"], color="0.2", lw=2)
    for o in sc["obstacles"]:
        x, y, r, hw, hh = o
        if hw > 0 and hh > 0:
            ax.add_patch(plt.Rectangle((x - hw, y - hh), 2 * hw, 2 * hh,
                                       fc="0.25", ec="0.1", lw=0.6))
        else:
            ax.add_patch(plt.Circle((x, y), r, fc="0.75", ec="0.1", lw=0.5))
    for (sx, sy, gx, gy, label) in sc["tasks"]:
        ax.plot([sx, gx], [sy, gy], color="tab:red", lw=0.6, alpha=0.55)
        ax.scatter(sx, sy, s=34, c="tab:green", marker="o",
                   edgecolors="k", linewidths=0.4, zorder=3)
        ax.scatter(gx, gy, s=48, c="tab:red", marker="*", zorder=3)
    ax.set_title("%s  %s gap=%.1f" % (sc["name"], sc["scale"], sc["density"]),
                 fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def generate_scenes():
    """Generate all 24 scenes (6 scales x 4 scenes)."""
    scenes = []
    seed = 20260831
    for scale in SCALES:
        for li, (dkey, dval) in enumerate(DENSITIES):
            scenes.append(gen_cylinder_scene(scale, dkey, dval, li, seed))
            seed += 7919
        scenes.append(gen_wall_scene(scale, seed))
        seed += 7919
    return scenes


def main():
    export_pics = "--pics" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = generate_scenes()

    manifest = {
        "kind": "UNIFIED_BENCHMARK_V2",
        "scales": list(SCALES.keys()),
        "densities": [d[0] for d in DENSITIES] + ["wall"],
        "scenes": [{
            "name": s["name"], "scale": s["scale"], "density": s["density"],
            "scene": s["scene"], "kind": s["kind"], "seed": s["seed"],
            "obstacles": s["obstacles"],
            "tasks": [list(t) for t in s["tasks"]],
        } for s in scenes],
    }
    with open(os.path.join(OUT_DIR, "unified_benchmark.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(">>> %d scenes -> %s" % (len(scenes), OUT_DIR))
    for scale in SCALES:
        for s in [x for x in scenes if x["scale"] == scale]:
            nw = sum(1 for o in s["obstacles"] if o[3] > 0 and o[4] > 0)
            nc = len(s["obstacles"]) - nw
            nbig = SCALES[scale]["n_big"]
            print("  %-8s %-6s gap=%.1f  cyl=%2d (big=%d) wall=%d"
                  % (scale, s["density_key"], s["density"], nc, nbig, nw))

    if export_pics:
        os.makedirs(PIC_DIR, exist_ok=True)
        for sc in scenes:
            render_scene(sc, os.path.join(PIC_DIR, "%s.png" % sc["name"]))
        print(">>> top-down PNGs -> %s" % PIC_DIR)


if __name__ == "__main__":
    main()
