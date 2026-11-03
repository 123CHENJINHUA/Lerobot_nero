"""Dual routing, training layout and validation without real hardware."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config

from dual_nero import NeroDualFollower, NeroDualFollowerConfig, NeroDualLeaderConfig
from nero_common import DUAL_FEATURES, FEATURES, OBSERVATION_FEATURES, SIDES
from action_space import validate_checkpoint_action
from real_test import build_live_input, required_cameras, check_execution_ready
from training_inputs import InputSelection, SelectedDataset
from test_safety import FakeArm, FakeGripper
from test_training_inputs import FakeDataset


class DualTest(unittest.TestCase):
    def test_four_camera_scripts_share_serials_and_reject_duplicates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name in ("record_dual.sh", "tele_dual.sh", "real_test_dual.sh"):
                shutil.copy2(root / name, temp / name)
            config = (root / "dual_config.sh").read_text()
            config = re.sub(r'^SIDE_CAMERA_SERIAL="[^"]*"', 'SIDE_CAMERA_SERIAL="444"', config, flags=re.M)
            (temp / "dual_config.sh").write_text(config)
            launcher = temp / "nero.sh"
            launcher.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
if sys.argv[1] == "camera-path":
    print("/dev/camera_" + sys.argv[2])
else:
    print(json.dumps({"args": sys.argv[1:], "depth": os.environ.get("NERO_DEPTH_CAMERAS"),
                      "images": os.environ["TRAIN_IMAGES"]}))
''')
            launcher.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if key not in {"TRAIN_IMAGES", "DATASET_ROOT", "DATASET_REPO_ID", "OUTPUT_DIR"}}
            for name in ("record_dual.sh", "tele_dual.sh", "real_test_dual.sh"):
                result = subprocess.run(["bash", str(temp / name)], env=env, capture_output=True, text=True, check=True)
                saved = json.loads(result.stdout)
                self.assertEqual(saved["images"], "front,left_wrist,right_wrist,side")
                args = saved["args"]
                if name == "real_test_dual.sh":
                    self.assertIn("--side-serial=444", args)
                else:
                    import yaml
                    cameras = yaml.safe_load(next(arg.split("=", 1)[1] for arg in args
                                                  if arg.startswith("--robot.cameras=")))
                    self.assertEqual(set(cameras), {"front", "left_wrist", "right_wrist", "side"})
                    self.assertEqual(cameras["side"]["index_or_path"], "/dev/camera_444")
                if name == "record_dual.sh":
                    self.assertIn("side:444", saved["depth"])
                    self.assertIn("--dataset.root=datasets/nero_dual_02", args)
            original = (root / "dual_config.sh").read_text()
            serials = re.findall(r'^(?:FRONT|LEFT_WRIST|RIGHT_WRIST)_CAMERA_SERIAL="(\d+)"', original, re.M)
            for serial in ["", *serials]:
                (temp / "dual_config.sh").write_text(config.replace('SIDE_CAMERA_SERIAL="444"',
                                                                    f'SIDE_CAMERA_SERIAL="{serial}"'))
                result = subprocess.run(["bash", str(temp / "record_dual.sh")], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def setUp(self):
        self.robot = make_robot_from_config(NeroDualFollowerConfig(
            allow_motion=True, control_mode="host"))
        for arm in self.robot.arms.values():
            arm.arm, arm.gripper = FakeArm(), FakeGripper()
        self.action = {f"{side}.{key}": (0.05 if key == "gripper.pos" else 0.1)
                       for side in SIDES for key in FEATURES}

    def test_factory_and_leader_routing(self):
        self.assertIsInstance(self.robot, NeroDualFollower)
        leader = make_teleoperator_from_config(NeroDualLeaderConfig())
        self.assertEqual([arm.config.channel for arm in leader.arms.values()], ["can0", "can1"])
        with patch.object(leader.arms["left"], "get_action", return_value=dict.fromkeys(FEATURES, 1)), \
             patch.object(leader.arms["right"], "get_action", return_value=dict.fromkeys(FEATURES, 2)):
            result = leader.get_action()
        self.assertEqual(tuple(result), DUAL_FEATURES)
        self.assertEqual(result["right.joint1.pos"], 2)

    def test_two_arm_step_limits_and_routing(self):
        self.action["right.joint1.pos"] = -0.1
        result = self.robot.send_action(self.action)
        self.assertAlmostEqual(result["left.joint1.pos"], 0.06)
        self.assertAlmostEqual(result["right.joint1.pos"], -0.06)
        for arm in self.robot.arms.values():
            self.assertEqual(len(arm.arm.sent), 1)
            self.assertAlmostEqual(arm.gripper.sent[0][0], 0.026)

    def test_invalid_right_target_never_moves_left(self):
        for key, value in (("right.joint1.pos", float("nan")), ("right.joint1.pos", 0.3),
                           ("right.gripper.pos", 0.09)):
            with self.subTest(key=key, value=value):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.robot.send_action({**self.action, key: value})
                self.assertTrue(all(not arm.arm.sent and not arm.gripper.sent
                                    for arm in self.robot.arms.values()))
        self.robot.arms["right"].arm._parser.joint_12.timestamp = 0
        with self.assertRaises(RuntimeError):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arms["left"].arm.sent)

    def test_preview_never_sends(self):
        self.robot.config.control_mode = "external"
        for arm in self.robot.arms.values():
            arm.config.control_mode = "external"
        self.assertEqual(self.robot.send_action(self.action), self.action)
        self.assertTrue(all(not arm.arm.sent for arm in self.robot.arms.values()))

    def test_shared_bus_host_motion_is_rejected_by_launcher(self):
        from nero import run_lerobot
        args = ["--robot.type=nero_dual_follower", "--teleop.type=nero_dual_leader"]
        with self.assertRaisesRegex(ValueError, "Host motion on a shared CAN bus"):
            run_lerobot("record", args + ["--robot.control_mode=host", "--robot.allow_motion=true"])
        with self.assertRaisesRegex(ValueError, "pair the same left/right"):
            run_lerobot("record", args + ["--teleop.left_channel=can1", "--teleop.right_channel=can0"])

    def test_depth_hook_is_owned_once_for_all_four_views(self):
        hook = SimpleNamespace(image_features={f"{name}_depth": (480, 640, 3)
                                              for name in ("front", "left_wrist", "right_wrist", "side")})
        with patch("depth_record.ACTIVE_HOOK", hook):
            robot = NeroDualFollower(NeroDualFollowerConfig())
        self.assertIs(robot.depth_hook, hook)
        self.assertTrue(all(arm.depth_hook is None for arm in robot.arms.values()))
        self.assertEqual({key for key in robot.observation_features if key.endswith("_depth")},
                         set(hook.image_features))

    def test_execution_checks_both_buses(self):
        followers = list(self.robot.arms.values())
        for follower in followers:
            follower.arm.get_joint_enable_status = lambda _: True
        with patch("real_test.leader_frames_present", side_effect=[False, True]):
            with self.assertRaisesRegex(RuntimeError, "Leader CAN frames"):
                check_execution_ready(followers, "v121")
        followers[1].arm.get_joint_enable_status = lambda _: False
        with patch("real_test.leader_frames_present", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not enabled"):
                check_execution_ready(followers, "v121")

    def test_connect_failure_closes_both_arms(self):
        for arm in self.robot.arms.values():
            arm.arm = arm.gripper = None
        left, right = self.robot.arms.values()
        with patch.object(left, "connect"), patch.object(right, "connect", side_effect=RuntimeError("CAN failure")), \
             patch.object(left, "disconnect") as close_left, patch.object(right, "disconnect") as close_right:
            with self.assertRaisesRegex(RuntimeError, "CAN failure"):
                self.robot.connect()
            close_left.assert_called_once()
            close_right.assert_called_once()

    def test_config_rejects_bus_alias_and_tcp(self):
        with self.assertRaises(ValueError):
            NeroDualFollowerConfig(right_channel="can0")
        with self.assertRaises(ValueError):
            NeroDualLeaderConfig(right_channel="can0")
        with self.assertRaises(ValueError):
            NeroDualFollowerConfig(action_space="tcp")

    def test_observation_and_four_camera_names(self):
        with patch.object(self.robot.arms["left"], "get_observation", return_value={"joint1.pos": 1}), \
             patch.object(self.robot.arms["right"], "get_observation", return_value={"joint1.pos": 2}):
            self.assertEqual(self.robot.get_observation(), {"left.joint1.pos": 1, "right.joint1.pos": 2})
        rgb, depth = required_cameras({f"observation.images.{name}": None for name in
                                      ("front", "left_wrist", "right_wrist", "side", "left_wrist_depth", "side_depth")}, dual=True)
        self.assertEqual(rgb, {"front", "left_wrist", "right_wrist", "side"})
        self.assertEqual(depth, {"left_wrist", "side"})
        with self.assertRaises(ValueError):
            required_cameras({"observation.images.side": None})

    def test_training_saved_selection_and_live_inputs(self):
        base = FakeDataset()
        base.meta.features["action"].update(names=list(DUAL_FEATURES), shape=[16])
        names = [f"{side}.{key}" for side in SIDES for key in OBSERVATION_FEATURES]
        base.meta.features["observation.state"].update(names=names, shape=[len(names)])
        base.meta.stats["observation.state"]["mean"] = torch.arange(len(names))
        import pyarrow as pa
        base.reader.hf_dataset.data = pa.table({"action": [[0.0] * 16]})
        base.meta.features["observation.images.side"] = dict(base.meta.features["observation.images.front"])
        selection = InputSelection(base.meta, "front,side", "joints,pose")
        dataset = SelectedDataset(base, selection)
        self.assertEqual(dataset.meta.features["action"]["shape"], [16])
        self.assertEqual(len(selection.state_names), 28)
        self.assertEqual(len(dataset.meta.stats["action"]["mean"]), 16)
        config = SimpleNamespace(output_features={"action": SimpleNamespace(shape=[16])})
        self.assertEqual(validate_checkpoint_action(config, selection.to_dict(), dual=True), "joints")
        with self.assertRaises(ValueError):
            validate_checkpoint_action(config, selection.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input_selection.json"
            path.write_text(json.dumps(selection.to_dict()))
            raw = dict.fromkeys(names, 0.0)
            raw["front"] = np.zeros((2, 2, 3), dtype=np.uint8)
            raw["side"] = np.full((2, 2, 3), 255, dtype=np.uint8)
            features = {"observation.state": SimpleNamespace(shape=[28]),
                        "observation.images.front": SimpleNamespace(shape=[3, 2, 2]),
                        "observation.images.side": SimpleNamespace(shape=[3, 2, 2])}
            live = build_live_input(base.meta, features, raw, path, "task")
            self.assertEqual(live["observation.state"].shape, (28,))
            self.assertEqual(live["observation.images.front"].shape, (3, 2, 2))
            self.assertTrue(torch.all(live["observation.images.side"] == 1))
        with self.assertRaises(ValueError):
            InputSelection(base.meta, "front", "joints", "tcp")


if __name__ == "__main__":
    unittest.main()
