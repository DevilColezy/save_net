#!/usr/bin/env python3
"""Probe the INDUSTRIAL (scene 0) Unity scene for collision-free spawn cells.

Connects once, then parks the vehicle at every grid cell (x, y, z=2.0) and
reports whether Unity flags a collision.  Used to find a safe region to
relocate the outdoor v3 wall scenes.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "il_dataset" / "scripts"))
import il_common

PUB, SUB = "10253", "10254"
DEPTH = {"width": 640, "height": 360, "fov": 58.0, "near": 0.28, "far": 10.0,
         "t_bc": [0.0]*12 + [1.0, 0.0, 0.0, 1.0]}


def main():
    bridge = il_common.UnityBridge(pub_port=PUB, sub_port=SUB)
    bridge.bind()
    ok = bridge.connect_handshake(0, DEPTH, timeout=60.0)
    if not ok:
        print("handshake FAILED")
        return
    print("handshake OK")
    free, blocked = [], []
    for x in range(-22, 23, 2):
        for y in range(-8, 28, 2):
            veh = il_common.make_depth_vehicle([x, y, 2.0], 0.0, DEPTH)
            st = {"scene_id": 0, "frame_id": 0, "vehicles": [veh], "objects": []}
            bridge.send_pose(st)
            coll = False
            deadline = time.time() + 3.0
            while time.time() < deadline:
                r = bridge.try_recv()
                if r is None:
                    time.sleep(0.01)
                    continue
                mm, _rp = r
                vs = mm.get("pub_vehicles", [])
                if vs and vs[0].get("collision", False):
                    coll = True
                break
            (blocked if coll else free).append((x, y))
        print(f"  row x={x}: free so far {len(free)} blocked {len(blocked)}",
              flush=True)
    print("\nFREE cells (x,y):", sorted(free))
    print("\nBLOCKED cells (x,y):", sorted(blocked))
    bridge.close()


if __name__ == "__main__":
    main()
