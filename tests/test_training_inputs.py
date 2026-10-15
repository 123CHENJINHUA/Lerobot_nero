"""Check selected model inputs and normalization without touching hardware."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pyarrow as pa

from nero_common import FEATURES, LIMITS, OBSERVATION_FEATURES
from training_inputs import (
    InputSelection, ReaderMetadata, SelectedDataset, SelectedMetadata, apply_saved_selection,
    install_training_input_selection,
)


class FakeMetadata:
    def __init__(self):
        self.info = {"features": {
            "action": {"dtype": "float32", "shape": [8], "names": list(FEATURES)},
            "observation.state": {
                "dtype": "float32", "shape": [27], "names": list(OBSERVATION_FEATURES)
            },
            **{
                f"observation.images.{name}": {"dtype": "video", "shape": [2, 2, 3]}
                for name in ("front", "camera2", "front_depth", "camera2_depth")
            },
        }}
        self.stats = {
            "action": {"mean": torch.zeros(8)},
            "observation.state": {
                "mean": torch.arange(27, dtype=torch.float32),
                "count": torch.tensor([100]),
            },
        }

    @property
    def features(self):
        return self.info["features"]

    @property
    def video_keys(self):
        return [key for key in self.features if key.startswith("observation.images.")]

    @property
    def camera_keys(self):
        return self.video_keys


class FakeDataset:
    def __init__(self, actions=None):
        self.meta = FakeMetadata()
        self.actions = torch.zeros(1, 8) if actions is None else torch.tensor(actions, dtype=torch.float32)
        self.reader = SimpleNamespace(
            _meta=self.meta,
            hf_dataset=SimpleNamespace(data=pa.table({"action": self.actions.tolist()})),
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return {
            "action": self.actions[index].clone(),
            "observation.state": torch.arange(27, dtype=torch.float32),
            **{key: torch.zeros(3, 2, 2) for key in self.meta.video_keys},
        }


class TrainingInputsTest(unittest.TestCase):
    def test_pretrained_inputs_are_rebuilt_for_selected_dataset(self):
        from lerobot.datasets import factory

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(factory, "make_dataset", return_value=FakeDataset()), \
                patch.object(factory, "resolve_delta_timestamps"):
            cfg = SimpleNamespace(output_dir=temp, policy=SimpleNamespace(
                input_features={"observation.images.camera1": object()}))
            install_training_input_selection("front,camera2", "joints,pose")
            dataset = factory.make_dataset(cfg)
            self.assertEqual(cfg.policy.input_features, {})
            self.assertEqual(dataset.meta.features["observation.state"]["shape"], [14])
            self.assertEqual(dataset.meta.video_keys, [
                "observation.images.front", "observation.images.camera2"])

    def test_action_clipping_recomputes_stats_without_changing_base(self):
        actions = [[0.0] * 8, [0.0, 0.0, 0.0, 2.199359, 0.0, 0.996548, 1.688065, 0.081]]
        base = FakeDataset(actions)
        selection = InputSelection(base.meta, "front", "joints")
        dataset = SelectedDataset(base, selection)
        clipped = torch.stack([dataset[i]["action"] for i in range(len(dataset))])
        self.assertAlmostEqual(clipped[1, 3].item(), LIMITS[3][1], places=6)
        self.assertAlmostEqual(clipped[1, 5].item(), LIMITS[5][1], places=6)
        self.assertAlmostEqual(clipped[1, 6].item(), LIMITS[6][1], places=6)
        self.assertAlmostEqual(clipped[1, 7].item(), 0.08, places=6)
        stats = dataset.meta.stats["action"]
        self.assertTrue(torch.allclose(stats["min"], clipped.min(dim=0).values))
        self.assertTrue(torch.allclose(stats["max"], clipped.max(dim=0).values))
        self.assertTrue(torch.allclose(stats["mean"], clipped.mean(dim=0)))
        self.assertTrue(torch.allclose(stats["std"], clipped.std(dim=0, unbiased=False)))
        self.assertEqual(stats["count"].item(), 2)
        self.assertEqual(dataset.clipped_action_frames, 1)
        self.assertTrue(torch.equal(base.actions, torch.tensor(actions)))
        self.assertTrue(torch.equal(base.meta.stats["action"]["mean"], torch.zeros(8)))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input_selection.json"
            path.write_text(json.dumps(selection.to_dict()))
            replay = apply_saved_selection(base[1], base.meta, path)
            self.assertTrue(torch.equal(replay["action"], clipped[1]))

    def test_invalid_action_order_or_values_rejected(self):
        meta = FakeMetadata()
        meta.info["features"]["action"]["names"] = list(reversed(FEATURES))
        with self.assertRaisesRegex(ValueError, "action order"):
            InputSelection(meta, "front", "joints")
        base = FakeDataset([[float("nan")] * 8])
        with self.assertRaisesRegex(ValueError, "finite Nero"):
            SelectedDataset(base, InputSelection(base.meta, "front", "joints"))

    def test_uninitialized_wrappers_report_missing_attributes(self):
        for cls in (ReaderMetadata, SelectedMetadata, SelectedDataset):
            with self.subTest(wrapper=cls.__name__):
                wrapper = cls.__new__(cls)
                with self.assertRaises(AttributeError):
                    getattr(wrapper, "__setstate__")

    def test_pickle_roundtrip_preserves_selection(self):
        base = FakeDataset()
        dataset = SelectedDataset(base, InputSelection(base.meta, "front,camera2", "joints,pose"))
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.meta.features["observation.state"]["shape"], [14])
        self.assertEqual(set(restored.base.reader._meta.video_keys), set(restored.meta.video_keys))
        self.assertTrue(torch.equal(restored[0]["observation.state"], dataset[0]["observation.state"]))

    def test_spawn_dataloader_preserves_selection(self):
        base = FakeDataset()
        dataset = SelectedDataset(base, InputSelection(base.meta, "front,camera2", "joints,pose"))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=2, multiprocessing_context="spawn", timeout=20,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["observation.state"].tolist(), [[*range(8), *range(15, 21)]])
        self.assertNotIn("observation.images.front_depth", batches[0])

    def test_rgb_joints_pose_filters_images_state_and_stats(self):
        base = FakeDataset()
        selection = InputSelection(base.meta, "front,camera2", "joints,pose")
        dataset = SelectedDataset(base, selection)
        frame = dataset[0]

        self.assertEqual(dataset.meta.features["observation.state"]["shape"], [14])
        self.assertEqual(frame["observation.state"].shape, (14,))
        self.assertEqual(dataset.meta.stats["observation.state"]["mean"].tolist(),
                         [*range(8), *range(15, 21)])
        self.assertEqual(dataset.meta.stats["observation.state"]["count"].tolist(), [100])
        self.assertEqual(set(dataset.meta.video_keys),
                         {"observation.images.front", "observation.images.camera2"})
        self.assertEqual(set(base.reader._meta.video_keys), set(dataset.meta.video_keys))
        self.assertNotIn("observation.images.front_depth", frame)
        self.assertEqual(base.meta.features["observation.state"]["shape"], [27])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input_selection.json"
            path.write_text(json.dumps(selection.to_dict()))
            replay = apply_saved_selection(base[0], base.meta, path)
            self.assertEqual(set(replay), set(frame))
            self.assertTrue(torch.equal(replay["observation.state"], frame["observation.state"]))

    def test_all_inputs_and_image_only(self):
        base = FakeDataset()
        all_inputs = InputSelection(base.meta, "front,camera2,front_depth,camera2_depth",
                                    "joints,pose,torque,velocity")
        self.assertEqual(SelectedDataset(base, all_inputs)[0]["observation.state"].shape, (27,))

        image_only = InputSelection(FakeMetadata(), "front", "")
        dataset = SelectedDataset(FakeDataset(), image_only)
        self.assertNotIn("observation.state", dataset.meta.features)
        self.assertNotIn("observation.state", dataset[0])

        pose_only = InputSelection(FakeMetadata(), "front,camera2", "pose")
        self.assertEqual(SelectedDataset(FakeDataset(), pose_only)[0]["observation.state"].shape, (6,))

    def test_unknown_or_duplicate_inputs_fail(self):
        meta = FakeMetadata()
        with self.assertRaises(ValueError):
            InputSelection(meta, "front,front", "joints")
        with self.assertRaises(ValueError):
            InputSelection(meta, "missing", "pose")
        with self.assertRaises(ValueError):
            InputSelection(meta, "front", "unknown")


if __name__ == "__main__":
    unittest.main()
