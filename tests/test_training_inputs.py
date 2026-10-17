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
    install_training_input_selection, selected_dataset_from_saved,
)
from action_space import TCP_FEATURES, TCP_6D_FEATURES, decode_tcp_action, joint_action_to_tcp, nero_mdh, validate_checkpoint_action


class FakeMetadata:
    def __init__(self):
        self.info = {"features": {
            "action": {"dtype": "float32", "shape": [8], "names": list(FEATURES)},
            "observation.state": {
                "dtype": "float32", "shape": [27], "names": list(OBSERVATION_FEATURES)
            },
            **{
                f"observation.images.{name}": {"dtype": "video", "shape": [2, 2, 3],
                                             "names": ["height", "width", "channels"]}
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
    def test_tcp_6d_labels_stats_metadata_and_replay(self):
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh
        actions = [[0.0] * 8, [0.1, -0.2, 0.1, 2.1, -0.3, 0.2, 0.0, 0.04]]
        base = FakeDataset(actions)
        selection = InputSelection(base.meta, "front", "joints", "tcp", "6d")
        dataset = SelectedDataset(base, selection)
        converted = torch.stack([dataset[i]["action"] for i in range(2)])
        self.assertEqual(converted.shape, (2, 10))
        self.assertEqual(dataset.meta.features["action"]["names"], list(TCP_6D_FEATURES))
        self.assertTrue(torch.allclose(dataset.meta.stats["action"]["mean"], converted.mean(0), atol=1e-6))
        self.assertEqual(joint_action_to_tcp(base.actions[None], nero_mdh(), "6d").shape, (1, 2, 10))
        decoded = decode_tcp_action(converted[1].numpy(), "6d")
        from action_space import pose_distance
        self.assertLess(max(pose_distance(decoded[:6], fk_from_mdh(nero_mdh(), actions[1][:7]))), 1e-6)
        saved = selection.to_dict()
        self.assertEqual(saved["tcp_rotation"], "6d")
        config = SimpleNamespace(output_features={"action": SimpleNamespace(shape=(10,))})
        self.assertEqual(validate_checkpoint_action(config, saved, "tcp", "6d"), "tcp")
        with self.assertRaisesRegex(ValueError, "TCP_ROTATION"):
            validate_checkpoint_action(config, saved, "tcp", "rpy")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input_selection.json"
            path.write_text(json.dumps(saved))
            self.assertTrue(torch.equal(apply_saved_selection(base[1], base.meta, path)["action"], converted[1]))
        restored = selected_dataset_from_saved(FakeDataset(actions), saved)
        self.assertTrue(torch.equal(restored[1]["action"], converted[1]))
        self.assertEqual(pickle.loads(pickle.dumps(restored))[1]["action"].shape, (10,))
        self.assertEqual(base.meta.features["action"]["shape"], [8])

    def test_tcp_targets_chunks_stats_and_saved_replay(self):
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh

        actions = [[0.0] * 8, [0.1, -0.2, 0.1, 2.199, -0.3, 0.2, 0.0, 0.09]]
        base = FakeDataset(actions)
        selection = InputSelection(base.meta, "front", "joints,pose", "tcp")
        dataset = SelectedDataset(base, selection)
        converted = torch.stack([dataset[i]["action"] for i in range(2)])
        source = [0.1, -0.2, 0.1, LIMITS[3][1], -0.3, 0.2, 0.0]
        expected = torch.tensor(fk_from_mdh(nero_mdh(), source) + [0.08])
        self.assertTrue(torch.allclose(converted[1], expected, atol=1e-6))
        self.assertEqual(dataset.meta.features["action"]["names"], list(TCP_FEATURES))
        self.assertEqual(dataset.meta.features["action"]["shape"], [7])
        self.assertEqual(base.meta.features["action"]["shape"], [8])
        for key, expected_stat in (("min", converted.min(dim=0).values),
                                   ("max", converted.max(dim=0).values),
                                   ("mean", converted.mean(dim=0)),
                                   ("std", converted.std(dim=0, unbiased=False))):
            self.assertTrue(torch.allclose(dataset.meta.stats["action"][key], expected_stat, atol=1e-6))
        self.assertEqual(joint_action_to_tcp(base.actions[None], nero_mdh()).shape, (1, 2, 7))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input_selection.json"
            path.write_text(json.dumps(selection.to_dict()))
            replay = apply_saved_selection(base[1], base.meta, path)
            self.assertTrue(torch.equal(replay["action"], converted[1]))
        restored = selected_dataset_from_saved(FakeDataset(actions), selection.to_dict())
        self.assertTrue(torch.equal(restored[1]["action"], converted[1]))
        self.assertTrue(torch.equal(base.actions, torch.tensor(actions)))

    def test_checkpoint_action_mismatch_and_missing_metadata_rejected(self):
        dataset = SelectedDataset(FakeDataset(), InputSelection(FakeMetadata(), "front", "joints", "tcp"))
        saved = dataset.selection.to_dict()
        config = SimpleNamespace(output_features={"action": SimpleNamespace(shape=(7,))})
        self.assertEqual(validate_checkpoint_action(config, saved), "tcp")
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            validate_checkpoint_action(config, saved, "joints")
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_checkpoint_action(config, {})
        with self.assertRaisesRegex(ValueError, "kinematics/frame"):
            validate_checkpoint_action(config, {**saved, "mdh": []})
        with self.assertRaisesRegex(ValueError, "workspace"):
            validate_checkpoint_action(config, {**saved, "tcp_workspace": None})
        config.output_features["action"].shape = (8,)
        self.assertEqual(validate_checkpoint_action(config, {}), "joints")

    def test_tcp_live_input_matches_training_state_and_images(self):
        from lerobot.utils.feature_utils import dataset_to_policy_features
        from real_test import build_live_input

        base = FakeDataset()
        dataset = SelectedDataset(base, InputSelection(base.meta, "front", "joints,pose", "tcp"))
        features = dataset_to_policy_features(dataset.meta.features)
        features.pop("action")
        raw = {key: float(i) for i, key in enumerate(OBSERVATION_FEATURES)}
        raw["front"] = torch.zeros(2, 2, 3, dtype=torch.uint8).numpy()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input_selection.json"
            path.write_text(json.dumps(dataset.selection.to_dict()))
            frame = build_live_input(base.meta, features, raw, path, "pick")
        self.assertEqual(frame["observation.state"].tolist(), [*range(8), *range(15, 21)])
        self.assertEqual(frame["observation.images.front"].shape, (3, 2, 2))
        self.assertNotIn("action", frame)

    def test_tcp_spawn_dataloader_converts_action_chunks(self):
        base = FakeDataset([[0.0] * 8, [0.1] * 7 + [0.04]])
        dataset = SelectedDataset(base, InputSelection(base.meta, "front", "joints", "tcp"))
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored[0]["action"].shape, (7,))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn", timeout=20,
        )
        self.assertEqual(next(iter(loader))["action"].shape, (2, 7))

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
