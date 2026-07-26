#!/usr/bin/env python3

import unittest
import math

try:
    from .rollout import (
        build_task_registry,
        task_to_unity_objects,
        validate_task_registry,
    )
except ImportError:
    from rollout import (
        build_task_registry,
        task_to_unity_objects,
        validate_task_registry,
    )


class RolloutTaskSuiteTest(unittest.TestCase):

    def test_registry_is_valid_and_covers_expected_behaviours(self):
        tasks = build_task_registry()
        validate_task_registry(tasks, drone_radius=0.30, safety_margin=0.10)
        self.assertEqual(len(tasks), 9)
        self.assertIn("clear_straight", tasks)
        self.assertIn("forced_left", tasks)
        self.assertIn("forced_right", tasks)
        self.assertIn("narrow_gate", tasks)
        self.assertIn("slalom", tasks)
        self.assertIn("climb_over", tasks)
        for task in tasks.values():
            self.assertEqual(task.start, (0.0, -1.0, 2.0))
            self.assertEqual(task.goal, (0.0, 31.0, 2.0))
            self.assertAlmostEqual(task.start_yaw, math.pi / 2.0)

    def test_task_objects_use_stable_slots_and_hide_unused_geometry(self):
        tasks = build_task_registry()
        slots = max(len(task.obstacles) for task in tasks.values())
        clear_objects = task_to_unity_objects(tasks["clear_straight"], slots)
        corridor_objects = task_to_unity_objects(tasks["long_corridor"], slots)

        self.assertEqual(len(clear_objects), slots)
        self.assertEqual(len(corridor_objects), slots)
        self.assertEqual(
            [obj["ID"] for obj in clear_objects],
            [obj["ID"] for obj in corridor_objects],
        )
        self.assertTrue(all(obj["position"][1] == -1000.0
                            for obj in clear_objects))
        self.assertTrue(all(obj["position"][1] >= 0.0
                            for obj in corridor_objects))

    def test_unity_axis_and_cylinder_dimensions(self):
        task = build_task_registry()["center_pillar"]
        obj = task_to_unity_objects(task, 1)[0]
        # Unity order is [world X, world Z, world Y].
        self.assertEqual(obj["position"], [0.0, 4.0, 15.0])
        self.assertEqual(obj["size"], [2.0, 8.0, 2.0])


if __name__ == "__main__":
    unittest.main()
