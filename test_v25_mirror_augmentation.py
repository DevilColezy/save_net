"""Regression tests for schema-v25 horizontal mirroring configuration."""

import csv
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from dataloader import (EpisodeInfo, LABEL_FIELDS_5HZ, STATE_FIELDS,
                        STATE_FIELDS_5HZ, TARGET_FIELDS, V25SequenceDataset)


class V25MirrorAugmentationTest(unittest.TestCase):
    def _episode(self, root):
        image = np.asarray([[100, 200, 300], [400, 500, 600]], dtype=np.uint16)
        path = root / "depth_000000.png"
        if cv2 is not None:
            self.assertTrue(cv2.imwrite(str(path), image))
        else:  # pragma: no cover - OpenCV is a project requirement.
            Image.fromarray(image.astype(np.int32), mode="I").save(str(path))
        fields = (list(STATE_FIELDS + TARGET_FIELDS) + STATE_FIELDS_5HZ +
                  LABEL_FIELDS_5HZ + ["depth_file", "hierarchical_mode",
                                      "planner_status", "episode_frame_index"])
        row = {name: "0" for name in fields}
        row.update({
            "depth_file": "depth_000000.png", "hierarchical_mode": "direct",
            # planner_status is the C++ PlannerStatus STRING (item 一); the
            # loader maps it to a stable index and rejects unknown values.
            "planner_status": "SAFE_PROGRESSING", "episode_frame_index": "0",
            "gravity_flu_y": "0.2", "velocity_flu_y": "0.4",
            "yaw_rate_flu": "0.6", "goal_direction_flu_y": "0.8",
            "target_velocity_flu_y": "0.5", "target_yaw_rate": "0.7",
        })
        with (root / "data.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerow(row)
        return EpisodeInfo(root, root / "data.csv", "episode", "scene",
                           "task", 1, "direct", False, 0, 0)

    def test_mirror_is_opt_in_and_flips_all_lateral_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = self._episode(Path(tmp))
            plain = V25SequenceDataset([episode], sequence_length=1,
                                        burn_in=0, augment=False,
                                        stateful=True)[0]
            mirrored_ds = V25SequenceDataset(
                [episode], sequence_length=1, burn_in=0, augment=True,
                stateful=True)
            epoch = next(e for e in range(100) if zlib.crc32(
                ("episode|%d" % e).encode("utf-8")) & 1)
            mirrored_ds.set_epoch(epoch)
            mirrored = mirrored_ds[0]
            self.assertTrue(np.array_equal(
                plain["depth"].numpy()[:, :, :, ::-1],
                mirrored["depth"].numpy()))
            for index in (1, 4, 6, 8):
                self.assertAlmostEqual(
                    mirrored["state"][0, index].item(),
                    -plain["state"][0, index].item())
            for index in (1, 3):
                self.assertAlmostEqual(
                    mirrored["target"][0, index].item(),
                    -plain["target"][0, index].item())


if __name__ == "__main__":
    unittest.main()
