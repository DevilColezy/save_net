"""Regression tests for leakage-free density-stratified evaluation splits."""

from __future__ import annotations

import contextlib
import io
import unittest

from dataloader import DENSITY_TIERS, EpisodeInfo, split_episodes
from train import _print_stats


def _episode(
    scene_id: str,
    tier: str,
    frames: int,
    episode_index: int = 0,
) -> EpisodeInfo:
    return EpisodeInfo(
        traj_dir=f"/{scene_id}/traj_{episode_index:03d}",
        scene_id=scene_id,
        task_id=f"task_{episode_index:03d}",
        episode_id=f"{scene_id}/episode_{episode_index:03d}",
        num_frames=frames,
        csv_path="",
        depth_dir="",
        depth_h=2,
        depth_w=3,
        scene_density_tier=tier,
        scene_profile_name=f"profile_{tier}",
    )


class DensityTierSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = []
        for tier_index, tier in enumerate(DENSITY_TIERS):
            for scene_index in range(4):
                scene_id = f"{tier}_scene_{scene_index}"
                for episode_index in range(2):
                    self.episodes.append(_episode(
                        scene_id,
                        tier,
                        frames=10 + tier_index * 3 + scene_index,
                        episode_index=episode_index,
                    ))

    def test_both_splits_cover_every_tier_without_scene_leakage(self) -> None:
        train, val = split_episodes(self.episodes, val_ratio=0.25, seed=17)

        train_scenes = {ep.scene_id for ep in train}
        val_scenes = {ep.scene_id for ep in val}
        self.assertTrue(train_scenes.isdisjoint(val_scenes))
        self.assertEqual(
            {ep.scene_density_tier for ep in train}, set(DENSITY_TIERS)
        )
        self.assertEqual(
            {ep.scene_density_tier for ep in val}, set(DENSITY_TIERS)
        )

        # All episodes belonging to one scene must land on the same side.
        for scene_id in train_scenes | val_scenes:
            sides = {
                "train" if ep in train else "val"
                for ep in self.episodes if ep.scene_id == scene_id
            }
            self.assertEqual(len(sides), 1)

    def test_same_seed_is_reproducible_independent_of_episode_order(self) -> None:
        train_a, val_a = split_episodes(
            self.episodes, val_ratio=0.25, seed=123
        )
        train_b, val_b = split_episodes(
            list(reversed(self.episodes)), val_ratio=0.25, seed=123
        )
        self.assertEqual(
            {ep.scene_id for ep in train_a}, {ep.scene_id for ep in train_b}
        )
        self.assertEqual(
            {ep.scene_id for ep in val_a}, {ep.scene_id for ep in val_b}
        )

    def test_conflicting_tier_within_scene_is_rejected(self) -> None:
        episodes = [
            _episode("scene_a", "sparse", 10, 0),
            _episode("scene_a", "dense", 10, 1),
            _episode("scene_b", "medium", 10, 0),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent density tiers"):
            split_episodes(episodes, val_ratio=0.2, seed=1)

    def test_one_scene_cannot_form_leakage_free_split(self) -> None:
        episodes = [
            _episode("scene_a", "sparse", 10, 0),
            _episode("scene_a", "sparse", 10, 1),
        ]
        with self.assertRaisesRegex(ValueError, "At least 2 distinct scenes"):
            split_episodes(episodes, val_ratio=0.2, seed=1)

    def test_training_stats_report_each_density_tier(self) -> None:
        train, val = split_episodes(self.episodes, val_ratio=0.25, seed=17)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_stats(train, val, train_windows=20, val_windows=5)
        report = output.getvalue()
        self.assertIn("Density tiers (scenes / episodes / frames):", report)
        for tier in DENSITY_TIERS:
            self.assertIn(tier, report)


if __name__ == "__main__":
    unittest.main()
