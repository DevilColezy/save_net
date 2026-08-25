#!/usr/bin/env python3

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    from .model.model import ViTFlyLSTMPolicy, ViTFlyPolicyConfig
    from .rollout import (
        DEFAULT_STATE_SCALE,
        EXPECTED_STATE_FIELDS,
        active_goal_for_time,
        build_normalized_state,
        build_task_registry,
        canonicalize_unity_depth,
        load_policy_checkpoint,
        task_to_unity_objects,
        validate_task_registry,
    )
except ImportError:
    from model.model import ViTFlyLSTMPolicy, ViTFlyPolicyConfig
    from rollout import (
        DEFAULT_STATE_SCALE,
        EXPECTED_STATE_FIELDS,
        active_goal_for_time,
        build_normalized_state,
        build_task_registry,
        canonicalize_unity_depth,
        load_policy_checkpoint,
        task_to_unity_objects,
        validate_task_registry,
    )


class RolloutTaskSuiteTest(unittest.TestCase):

    def test_registry_matches_training_geometry_and_behaviours(self):
        tasks = build_task_registry()
        validate_task_registry(
            tasks, drone_radius=0.30, safety_margin=0.30,
            minimum_surface_gap_m=1.20)
        self.assertGreaterEqual(len(tasks), 12)
        for name in (
                "clear_straight", "center_pillar", "forced_left",
                "forced_right", "rear_goal", "abrupt_goal_switch",
                "continuous_5hz"):
            self.assertIn(name, tasks)
            self.assertEqual(tasks[name].suite, "basic")
        self.assertNotIn("climb_over", tasks)
        self.assertGreaterEqual(len(tasks["continuous_5hz"].goal_updates), 20)
        intervals = np.diff(
            [update[0] for update in tasks["continuous_5hz"].goal_updates])
        self.assertTrue(np.allclose(intervals, 0.2))
        for task in tasks.values():
            for obstacle in task.obstacles:
                self.assertEqual(obstacle.base_z, 0.0)
                self.assertEqual(obstacle.height, 8.0)
            for first_index, first in enumerate(task.obstacles):
                for second in task.obstacles[first_index + 1:]:
                    gap = math.hypot(first.x - second.x,
                                     first.y - second.y) - \
                        first.radius - second.radius
                    self.assertGreaterEqual(gap + 1e-9, 1.20)

    def test_task_objects_use_stable_slots_and_hide_unused_geometry(self):
        tasks = build_task_registry()
        slots = max(len(task.obstacles) for task in tasks.values())
        clear_objects = task_to_unity_objects(tasks["clear_straight"], slots)
        compound_objects = task_to_unity_objects(tasks["double_gate"], slots)

        self.assertEqual(len(clear_objects), slots)
        self.assertEqual(len(compound_objects), slots)
        self.assertEqual(
            [obj["ID"] for obj in clear_objects],
            [obj["ID"] for obj in compound_objects],
        )
        self.assertTrue(all(obj["position"][1] == -1000.0
                            for obj in clear_objects))
        self.assertTrue(all(obj["position"][1] >= 0.0
                            for obj in compound_objects))

    def test_unity_axis_and_cylinder_dimensions(self):
        task = build_task_registry()["center_pillar"]
        obj = task_to_unity_objects(task, 1)[0]
        # Unity order is [world X, world Z, world Y].
        self.assertEqual(obj["position"], [0.0, 4.0, 5.0])
        self.assertEqual(obj["size"], [1.2, 8.0, 1.2])

    def test_goal_updates_are_piecewise_continuous_without_implicit_reset(self):
        task = build_task_registry()["abrupt_goal_switch"]
        goal0, index0, changed0 = active_goal_for_time(task, 0.0, -1)
        goal1, index1, changed1 = active_goal_for_time(task, 2.0, index0)
        goal2, index2, changed2 = active_goal_for_time(task, 2.5, index1)
        np.testing.assert_allclose(goal0, (-1.5, 6.0, 2.0))
        np.testing.assert_allclose(goal1, goal0)
        np.testing.assert_allclose(goal2, (1.5, 10.0, 2.0))
        self.assertTrue(changed0)
        self.assertFalse(changed1)
        self.assertTrue(changed2)
        self.assertEqual((index0, index1, index2), (0, 0, 1))

    def test_depth_and_state_preprocessing_match_v25_contract(self):
        # Payload is flipped vertically; values are Unity hectometres.
        payload = np.asarray([
            [np.nan, 0.0, -1.0, np.inf],
            [0.01234, 0.05000, 0.10000, 1.00000],
        ], dtype=np.float32)
        depth_cm, normalized = canonicalize_unity_depth(payload, 5.0)
        np.testing.assert_array_equal(
            depth_cm[0], np.asarray([123, 500, 500, 500], dtype=np.uint16))
        np.testing.assert_array_equal(
            depth_cm[1], np.asarray([500, 500, 500, 500], dtype=np.uint16))
        np.testing.assert_allclose(normalized, depth_cm * 0.01 / 5.0)

        state = build_normalized_state(
            np.asarray([0.0, 0.0, -1.0]),
            np.asarray([2.5, -1.25, 0.0]), 0.75,
            np.asarray([0.0, 1.0, 0.0]), 0.5,
            DEFAULT_STATE_SCALE, torch.device("cpu"))
        self.assertEqual(tuple(state.shape), (1, 11))
        np.testing.assert_allclose(
            state.numpy()[0],
            [0.0, 0.0, -1.0, 1.0, -0.5, 0.0, 0.5,
             0.0, 1.0, 0.0, 0.5])

    def test_current_train_checkpoint_loads_and_steps(self):
        config = ViTFlyPolicyConfig(
            image_height=32, image_width=48,
            stage_dims=(8, 16), stage_depths=(1, 1),
            stage_heads=(1, 1), sr_ratios=(4, 2), visual_dim=24,
            state_hidden_dim=16, lstm_hidden_dim=20, lstm_layers=1,
            dropout=0.0)
        source = ViTFlyLSTMPolicy(config)
        payload = {
            "schema_version": 25,
            "architecture": "ViTFlyLSTMPolicy",
            "model_config": config.to_dict(),
            "model_state": source.state_dict(),
            "student_input_fields": ["depth_file"] + list(EXPECTED_STATE_FIELDS),
            "normalization": {
                "depth_max_m": 5.0,
                "state_scale": list(DEFAULT_STATE_SCALE),
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            torch.save(payload, str(checkpoint))
            model_file = Path(__file__).resolve().parent / "model" / "model.py"
            loaded, loaded_config, state_scale = load_policy_checkpoint(
                str(checkpoint), str(model_file), torch.device("cpu"), 5.0)
        self.assertEqual(loaded_config.image_height, 32)
        self.assertEqual(state_scale, DEFAULT_STATE_SCALE)
        hidden = loaded.initial_hidden(1)
        with torch.no_grad():
            output = loaded.step(
                torch.ones(1, 1, 32, 48), torch.zeros(1, 11), hidden)
        self.assertEqual(tuple(output.command.shape), (1, 4))
        self.assertEqual(tuple(output.hidden[0].shape), (1, 1, 20))


if __name__ == "__main__":
    unittest.main()
