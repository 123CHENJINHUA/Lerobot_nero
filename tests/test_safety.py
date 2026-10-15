"""Hardware-free checks for stale feedback and Nero command limits."""

import time
import unittest
from types import SimpleNamespace

from lerobot_robot_nero import NeroFollower, NeroFollowerConfig
from nero_common import FEATURES, LIMITS, complete_joints


class FakeArm:
    def __init__(self):
        self._config = {"comm": {"can": {"channel": "can1"}}}
        self._parser = SimpleNamespace(**{
            key: SimpleNamespace(timestamp=time.time())
            for key in ("joint_12", "joint_34", "joint_56", "joint_7")
        })
        self.sent = []

    def get_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7)

    def move_j(self, joints):
        self.sent.append(joints)


class FakeGripper:
    def __init__(self):
        self.sent = []

    def get_gripper_status(self):
        return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(mode="width", value=0.02))

    def move_gripper_m(self, value, force):
        self.sent.append((value, force))


class SafetyTest(unittest.TestCase):
    def setUp(self):
        self.robot = NeroFollower(NeroFollowerConfig(allow_motion=True, control_mode="host"))
        self.robot.arm = FakeArm()
        self.robot.gripper = FakeGripper()
        self.action = dict.fromkeys(FEATURES, 0.1)
        self.action["gripper.pos"] = 0.05

    def tearDown(self):
        self.robot.arm = None
        self.robot.gripper = None

    def test_limits_and_step_cap(self):
        sent = self.robot.send_action(self.action)
        self.assertAlmostEqual(sent["joint1.pos"], 0.06)
        self.assertAlmostEqual(sent["gripper.pos"], 0.026)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.assertEqual(len(self.robot.gripper.sent), 1)

    def test_rejects_out_of_range_without_motion(self):
        self.action["joint1.pos"] = 10.0
        with self.assertRaises(ValueError):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_small_joint_overshoot_is_clamped_within_limits(self):
        self.robot.config.max_step_rad = 0.01
        for index, (lo, hi) in enumerate(LIMITS):
            for edge, direction in ((lo, -1), (hi, 1)):
                with self.subTest(joint=index + 1, edge=edge):
                    present = [0.0] * 7
                    present[index] = edge - direction * 0.005
                    self.robot.arm.get_joint_angles = lambda: SimpleNamespace(msg=present)
                    self.robot._last_sent = None
                    action = dict.fromkeys(FEATURES, 0.0)
                    action["gripper.pos"] = 0.02
                    action[FEATURES[index]] = edge + direction * 0.009
                    sent = self.robot.send_action(action)
                    self.assertAlmostEqual(sent[FEATURES[index]], edge)
                    self.assertLessEqual(abs(sent[FEATURES[index]] - present[index]), 0.01)
                    for key, (lower, upper) in zip(FEATURES[:7], LIMITS, strict=True):
                        self.assertGreaterEqual(sent[key], lower)
                        self.assertLessEqual(sent[key], upper)

    def test_joint_overshoot_beyond_one_step_rejected_without_motion(self):
        self.robot.config.max_step_rad = 0.01
        for index, (lo, hi) in enumerate(LIMITS):
            for edge, direction in ((lo, -1), (hi, 1)):
                with self.subTest(joint=index + 1, edge=edge):
                    self.action[FEATURES[index]] = edge + direction * 0.011
                    with self.assertRaises(ValueError):
                        self.robot.send_action(self.action)
                    self.action[FEATURES[index]] = 0.1
                    self.assertFalse(self.robot.arm.sent)
                    self.assertFalse(self.robot.gripper.sent)

    def test_rejects_stale_part_without_motion(self):
        self.robot.arm._parser.joint_56.timestamp = time.time() - 2
        with self.assertRaises(RuntimeError):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_rejects_invalid_gripper_without_motion(self):
        for width in (-0.001, 0.081, float("nan")):
            with self.subTest(width=width):
                self.action["gripper.pos"] = width
                with self.assertRaises(ValueError):
                    self.robot.send_action(self.action)
                self.assertFalse(self.robot.arm.sent)
                self.assertFalse(self.robot.gripper.sent)

    def test_rejects_large_initial_difference(self):
        self.action["joint6.pos"] = 0.3
        with self.assertRaisesRegex(
            RuntimeError,
            r"joint6\.pos.*current=0\.0000rad target=0\.3000rad error=0\.3000rad limit=0\.2500rad",
        ):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)
        self.assertFalse(self.robot.gripper.sent)

    def test_external_mode_never_sends_commands(self):
        self.robot.config.control_mode = "external"
        self.robot.config.allow_motion = False
        self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)
        self.assertFalse(self.robot.gripper.sent)

    def test_real_test_threshold_keeps_step_cap(self):
        self.robot.config.max_initial_error_rad = 0.25
        self.robot.config.max_step_rad = 0.01
        self.robot.config.max_gripper_step_m = 0.002
        self.action["joint2.pos"] = 0.25
        sent = self.robot.send_action(self.action)
        self.assertAlmostEqual(sent["joint2.pos"], 0.01)
        self.assertAlmostEqual(sent["gripper.pos"], 0.022)

        self.action["joint2.pos"] = 0.2501
        with self.assertRaises(RuntimeError):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.assertEqual(len(self.robot.gripper.sent), 1)

    def test_reversing_targets_keeps_adjacent_command_step_cap(self):
        self.robot.config.max_step_rad = 0.01
        self.robot.config.max_gripper_step_m = 0.002
        first = self.robot.send_action(self.action)
        for key in FEATURES[:7]:
            self.action[key] = -0.1
        self.action["gripper.pos"] = 0.0
        second = self.robot.send_action(self.action)
        for key in FEATURES[:7]:
            self.assertLessEqual(abs(second[key] - first[key]), 0.01000001)
            self.assertLessEqual(abs(second[key]), 0.01000001)
        self.assertLessEqual(abs(second["gripper.pos"] - first["gripper.pos"]), 0.00200001)

    def test_feedback_jump_rejected_without_new_commands(self):
        self.robot.config.max_step_rad = 0.01
        self.robot.send_action(self.action)
        self.robot.arm.get_joint_angles = lambda: SimpleNamespace(msg=[0.1] * 7)
        with self.assertRaisesRegex(RuntimeError, "feedback moved too far from last command"):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.assertEqual(len(self.robot.gripper.sent), 1)


if __name__ == "__main__":
    unittest.main()
