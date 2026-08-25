"""Regression tests for strict whole-trajectory Schema-v17 validity."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from dataloader import (
    EpisodeInfo,
    HierarchicalILSequenceDataset,
    STRICT_EPISODE_VALIDITY_POLICY,
    discover_committed_episodes,
    split_episodes,
)
from train import compute_losses


H_SOFT_COLUMNS = [f"trend_horizontal_soft_{i:02d}" for i in range(13)]
V_SOFT_COLUMNS = [f"guide_elevation_soft_{i}" for i in range(7)]


def _row(frame: int, **validity_overrides: int) -> dict[str, object]:
    validity = {
        "frame_valid": 1,
        "expert_label_valid": 1,
        "macro_label_valid": 1,
        "global_direction_valid": 1,
    }
    validity.update(validity_overrides)
    row: dict[str, object] = {
        "frame_id": frame,
        "depth_file": f"{frame:06d}.png",
        "scene_id": "scene_a",
        "task_id": "task_a",
        "episode_id": "episode_a",
        "global_dir_x_flu": 1.0,
        "global_dir_y_flu": 0.25,
        "global_dir_z_flu": 0.0,
        "global_distance_norm": 0.5,
        "gravity_direction_x_flu": 0.0,
        "gravity_direction_y_flu": 0.0,
        "gravity_direction_z_flu": -1.0,
        "state_vx_flu": 0.1,
        "state_vy_flu": 0.2,
        "state_vz_flu": 0.3,
        "state_angular_velocity_x_flu": 0.2,
        "state_angular_velocity_y_flu": 0.3,
        "state_angular_velocity_z_flu": 0.4,
        "macro_update": 1,
        "macro_mode": "BYPASS_LEFT",
        "macro_committed_side": 1,
        "macro_move_dir_x_flu": 0.8,
        "macro_move_dir_y_flu": 0.6,
        "macro_move_dir_z_flu": 0.0,
        "macro_move_distance_m": 2.0,
        "macro_move_distance_norm": 0.5,
        "macro_yaw_dir_x_flu": 0.8,
        "macro_yaw_dir_y_flu": 0.6,
        "trend_horizontal_class_13": 5,
        "guide_elevation_bin": 3,
        "guide_distance_norm": 0.6,
        "expert_vx_flu": 0.7,
        "expert_vy_flu": 0.8,
        "expert_vz_flu": 0.9,
        "expert_yaw_rate": 1.0,
        **validity,
    }
    row.update({column: float(i == 5) for i, column in enumerate(H_SOFT_COLUMNS)})
    row.update({column: float(i == 3) for i, column in enumerate(V_SOFT_COLUMNS)})
    return row


def _write_episode(
    root: Path,
    rows: list[dict[str, object]],
    *,
    schema_version: int = 17,
) -> Path:
    traj_dir = root / "scene_a" / "traj_001"
    depth_dir = traj_dir / "depth"
    depth_dir.mkdir(parents=True)
    with (traj_dir / "data.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for frame in range(len(rows)):
        pixels = np.full((2, 3), frame + 1, dtype=np.uint16)
        Image.fromarray(pixels).save(depth_dir / f"{frame:06d}.png")
    with (traj_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": schema_version,
            "terminal_label_semantics": "goal_hold_v1",
            "episode_validity_policy": STRICT_EPISODE_VALIDITY_POLICY,
            "status": "committed",
            "depth_h": 2,
            "depth_w": 3,
            "scene_density_tier": "sparse",
            "scene_profile_name": "test_sparse_profile",
        }, handle)
    return traj_dir


class StrictEpisodeDiscoveryTest(unittest.TestCase):
    def test_schema_v15_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_episode(Path(tmp), [_row(0)], schema_version=15)
            with self.assertRaisesRegex(ValueError, "must be 17"):
                discover_committed_episodes(tmp)

    def test_any_invalid_label_rejects_complete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_episode(
                Path(tmp),
                [_row(0), _row(1, macro_label_valid=0)],
            )
            with self.assertRaisesRegex(
                    ValueError, "complete trajectory must be rejected"):
                discover_committed_episodes(tmp)

    def test_valid_episode_has_no_per_head_loss_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_episode(Path(tmp), [_row(0), _row(1)])
            episodes = discover_committed_episodes(tmp)
            self.assertEqual(episodes[0].scene_density_tier, "sparse")
            self.assertEqual(
                episodes[0].scene_profile_name, "test_sparse_profile"
            )
            dataset = HierarchicalILSequenceDataset(
                episodes,
                sequence_length=2,
                burn_in=0,
                window_stride=2,
                target_height=2,
                target_width=3,
                mirror_augmentation=True,
            )
            original = dataset[0]
            mirrored = dataset[1]
            self.assertFalse(any("loss_valid" in key for key in original))
            self.assertEqual(original["target_mask"].squeeze(-1).tolist(),
                             [1.0, 1.0])
            self.assertEqual(original["macro_mode_target"].tolist(), [1, 1])
            self.assertEqual(mirrored["macro_mode_target"].tolist(), [2, 2])
            self.assertTrue(torch.allclose(
                mirrored["command_target"][:, 1],
                -original["command_target"][:, 1],
            ))

    def test_scene_split_never_leaks_a_scene(self) -> None:
        episodes = [
            EpisodeInfo(
                "", "scene_a", "a", "a", 10, "", "", 1, 1,
                scene_density_tier="sparse",
                scene_profile_name="test_sparse_profile",
            ),
            EpisodeInfo(
                "", "scene_b", "b", "b", 10, "", "", 1, 1,
                scene_density_tier="sparse",
                scene_profile_name="test_sparse_profile",
            ),
        ]
        train, val = split_episodes(episodes, val_ratio=0.5, seed=7)
        self.assertTrue({ep.scene_id for ep in train}.isdisjoint(
            {ep.scene_id for ep in val}
        ))


class StructuralMaskTest(unittest.TestCase):
    def test_all_losses_use_target_mask_only(self) -> None:
        batch = {
            "target_mask": torch.tensor([[[1.0], [0.0], [1.0]]]),
            "macro_update_mask": torch.ones(1, 3, 1),
            "macro_valid_mask": torch.ones(1, 3, 1),
            "macro_move_direction_target": torch.tensor(
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]]),
            "macro_move_distance_target": torch.zeros(1, 3, 1),
            "macro_yaw_direction_target": torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
            "macro_mode_target": torch.zeros(1, 3, dtype=torch.long),
            "command_target": torch.zeros(1, 3, 4),
        }
        output = SimpleNamespace(
            move_direction_flu=torch.tensor(
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]]),
            move_distance_norm=torch.zeros(1, 3, 1),
            yaw_direction_flu_xy=torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
            macro_mode_logits=torch.zeros(1, 3, 8),
            command_normalized=torch.zeros(1, 3, 4),
        )
        config = SimpleNamespace(
            max_vx_flu=1.0,
            max_vy_flu=1.0,
            max_vz_flu=1.0,
            max_yaw_rate=1.0,
        )
        weights = {
            "macro_direction": 1.0,
            "macro_distance": 1.0,
            "macro_yaw": 1.0,
            "macro_mode": 1.0,
            "control": 1.0,
        }
        baseline = compute_losses(output, batch, config, weights)
        for name in (
                "macro_direction", "macro_distance", "macro_yaw",
                "macro_mode", "control"):
            self.assertEqual(
                baseline[f"_{name}_red"].denominator.item(), 2.0
            )

        # The middle frame is padding/burn-in and is the only frame excluded
        # by training. There is no label-validity mask path.
        output.move_direction_flu[:, 1] = torch.tensor([0.0, 1.0, 0.0])
        output.move_distance_norm[:, 1] = 100.0
        output.yaw_direction_flu_xy[:, 1] = torch.tensor([0.0, 1.0])
        output.macro_mode_logits[:, 1, 7] = 100.0
        output.command_normalized[:, 1] = 100.0
        changed = compute_losses(output, batch, config, weights)
        for name in (
                "macro_direction", "macro_distance", "macro_yaw",
                "macro_mode", "control"):
            self.assertTrue(torch.allclose(baseline[name], changed[name]))


if __name__ == "__main__":
    unittest.main()
