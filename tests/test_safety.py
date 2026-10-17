"""Hardware-free checks for stale feedback and Nero command limits."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot_robot_nero import NeroFollower, NeroFollowerConfig
from nero_common import FEATURES, LIMITS, complete_joints
from action_space import TCP_FEATURES, limit_tcp_pose, pose_distance


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

    def test_clamping_keeps_initial_difference_guard(self):
        self.action["joint1.pos"] = 10.0
        with self.assertRaisesRegex(RuntimeError, "target too far from follower"):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)
        self.assertFalse(self.robot.gripper.sent)

    def test_joint_overshoot_is_clamped_within_limits(self):
        self.robot.config.max_step_rad = 0.01
        for index, (lo, hi) in enumerate(LIMITS):
            for edge, direction in ((lo, -1), (hi, 1)):
                for overshoot in (0.009, 0.05211542, 10.0):
                    with self.subTest(joint=index + 1, edge=edge, overshoot=overshoot):
                        present = [0.0] * 7
                        present[index] = edge - direction * 0.005
                        self.robot.arm.get_joint_angles = lambda: SimpleNamespace(msg=present)
                        self.robot._last_sent = None
                        action = dict.fromkeys(FEATURES, 0.0)
                        action["gripper.pos"] = 0.02
                        action[FEATURES[index]] = edge + direction * overshoot
                        sent = self.robot.send_action(action)
                        self.assertAlmostEqual(sent[FEATURES[index]], edge)
                        self.assertLessEqual(abs(sent[FEATURES[index]] - present[index]), 0.01)
                        for key, (lower, upper) in zip(FEATURES[:7], LIMITS, strict=True):
                            self.assertGreaterEqual(sent[key], lower)
                            self.assertLessEqual(sent[key], upper)

    def test_joint4_clamping_still_approaches_limit_in_small_steps(self):
        self.robot.config.max_step_rad = 0.01
        present = [0.0] * 7
        present[3] = LIMITS[3][1] - 0.05
        self.robot.arm.get_joint_angles = lambda: SimpleNamespace(msg=present.copy())
        action = dict.fromkeys(FEATURES, 0.0)
        action["joint4.pos"] = 2.198870
        action["gripper.pos"] = 0.02
        for _ in range(8):
            previous = present[3]
            sent = self.robot.send_action(action)
            self.assertLessEqual(abs(sent["joint4.pos"] - previous), 0.01000001)
            self.assertLessEqual(sent["joint4.pos"], LIMITS[3][1])
            present[:] = [sent[key] for key in FEATURES[:7]]
        self.assertAlmostEqual(present[3], LIMITS[3][1])

    def test_nonfinite_joint_targets_rejected_without_motion(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.action["joint4.pos"] = value
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.robot.send_action(self.action)
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

    def test_gripper_feedback_gap_reports_values_without_new_commands(self):
        self.robot.config.max_gripper_step_m = 0.002
        self.robot.send_action(self.action)
        self.robot.gripper.get_gripper_status = lambda: SimpleNamespace(
            timestamp=time.time(), msg=SimpleNamespace(mode="width", value=0.03))
        with self.assertRaisesRegex(
            RuntimeError,
            r"gripper\.pos.*current=0\.030000m last_sent=0\.022000m "
            r"gap=0\.008000m step=0\.002000m target=0\.050000m; motion stopped",
        ):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.assertEqual(len(self.robot.gripper.sent), 1)

    def test_gripper_feedback_outside_range_reports_actual_cause(self):
        self.robot.config.max_gripper_step_m = 0.002
        self.robot._last_sent = [0.0] * 7 + [0.093611]
        self.robot.gripper.get_gripper_status = lambda: SimpleNamespace(
            timestamp=time.time(), msg=SimpleNamespace(mode="width", value=0.093611))
        with self.assertRaisesRegex(
            RuntimeError,
            r"gripper\.pos feedback 0\.093611m outside configured range "
            r"\[0\.000000, 0\.080000\]m",
        ):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)
        self.assertFalse(self.robot.gripper.sent)


class TCPSafetyTest(unittest.TestCase):
    def setUp(self):
        self.robot = NeroFollower(NeroFollowerConfig(
            allow_motion=True, control_mode="host", action_space="tcp",
            tcp_workspace=[[0.2, 0.4], [-0.1, 0.1], [0.3, 0.5]],
            max_step_rad=0.01, max_gripper_step_m=0.002,
        ))
        self.robot.arm, self.robot.gripper = FakeArm(), FakeGripper()
        for name in ("ik_joint_12", "ik_joint_34", "ik_joint_56", "ik_joint_7"):
            setattr(self.robot.arm._parser, name, SimpleNamespace(timestamp=time.time()))
        self.pose = np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
        self.robot.arm.fk = lambda q: self.pose.tolist()
        self.robot.arm.get_arm_status = lambda: SimpleNamespace(
            timestamp=time.time(), msg=SimpleNamespace(arm_status=0, err_code=0))
        self.robot.arm.get_ik_joint_angles = lambda: SimpleNamespace(timestamp=time.time(), msg=[0.0] * 7)
        self.robot.arm.get_tcp2flange_pose = lambda pose: pose
        self.robot.arm.move_p = self.robot.arm.sent.append
        self.telemetry = patch("lerobot_robot_nero.nero_follower.follower_telemetry",
                               side_effect=lambda *args: (np.zeros(7), self.pose.copy(), time.time()))
        self.telemetry.start()
        self.action = dict(zip(TCP_FEATURES, [0.34, 0.0, 0.4, 0.0, 0.0, 0.2, 0.05]))

    def tearDown(self):
        self.telemetry.stop()
        self.robot.arm = self.robot.gripper = None

    def test_tcp_step_limits_and_reversal(self):
        first = self.robot.send_action(self.action)
        self.assertAlmostEqual(first[TCP_FEATURES[0]], 0.302)
        self.assertAlmostEqual(first[TCP_FEATURES[5]], 0.02)
        self.assertAlmostEqual(first["gripper.pos"], 0.022)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.action[TCP_FEATURES[0]] = 0.26
        self.action[TCP_FEATURES[5]] = -0.2
        second = self.robot.send_action(self.action)
        distance, angle = pose_distance([first[k] for k in TCP_FEATURES[:6]],
                                        [second[k] for k in TCP_FEATURES[:6]])
        self.assertLessEqual(distance, 0.00200001)
        self.assertLessEqual(angle, 0.02000001)

    def test_rotation_wrap_uses_shortest_path(self):
        current = [0.3, 0, 0.4, 0, 0, np.pi - 0.01]
        target = [0.3, 0, 0.4, 0, 0, -np.pi + 0.03]
        sent = limit_tcp_pose(target, current, current, 0.002, 0.02)
        self.assertAlmostEqual(pose_distance(sent, current)[1], 0.02)
        self.assertAlmostEqual(pose_distance(sent, target)[1], 0.02)

    def test_tcp_bad_targets_never_send(self):
        for key, value in ((TCP_FEATURES[0], float("nan")),
                           (TCP_FEATURES[0], 0.5), (TCP_FEATURES[5], 1.0),
                           ("gripper.pos", 0.081)):
            action = {**self.action, key: value}
            with self.assertRaises((ValueError, RuntimeError)):
                self.robot.send_action(action)
            self.assertFalse(self.robot.arm.sent)
            self.assertFalse(self.robot.gripper.sent)

    def test_tcp_status_failure_never_sends(self):
        for code in (2, 3, 4, 7):
            self.robot.arm.get_arm_status = lambda: SimpleNamespace(
                timestamp=time.time(), msg=SimpleNamespace(arm_status=code, err_code=0))
            with self.assertRaisesRegex(RuntimeError, "controller status"):
                self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_tcp_ik_limit_failure_stops_new_commands(self):
        self.robot.send_action(self.action)
        self.robot.arm.get_ik_joint_angles = lambda: SimpleNamespace(
            timestamp=time.time(), msg=[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "IK feedback outside"):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)
        self.assertEqual(len(self.robot.gripper.sent), 1)

    def test_tcp_partial_or_stale_ik_stops_new_commands(self):
        self.robot.send_action(self.action)
        del self.robot.arm._parser.ik_joint_34
        with self.assertRaisesRegex(RuntimeError, "ik_joint_34"):
            self.robot.send_action(self.action)
        self.robot.arm._parser.ik_joint_34 = SimpleNamespace(timestamp=time.time() - 2)
        with self.assertRaisesRegex(RuntimeError, "ik_joint_34"):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)

    def test_tcp_workspace_clipping_stays_bounded(self):
        self.pose[0] = 0.399
        self.action[TCP_FEATURES[0]] = 0.6
        sent = self.robot.send_action(self.action)
        self.assertAlmostEqual(sent[TCP_FEATURES[0]], 0.4)

    def test_tcp_workspace_and_feedback_jump_guards(self):
        self.robot.config.tcp_workspace = None
        with self.assertRaisesRegex(ValueError, "tcp_workspace"):
            self.robot.send_action(self.action)
        self.robot.config.tcp_workspace = [[0.2, 0.4], [-0.1, 0.1], [0.3, 0.5]]
        self.pose[0] = 0.19
        with self.assertRaisesRegex(RuntimeError, "outside the training workspace"):
            self.robot.send_action(self.action)
        self.pose[0] = 0.3
        self.robot.send_action(self.action)
        self.robot.arm.get_joint_angles = lambda: SimpleNamespace(msg=[0.1] * 7)
        with self.assertRaisesRegex(RuntimeError, "joint feedback jumped"):
            self.robot.send_action(self.action)
        self.assertEqual(len(self.robot.arm.sent), 1)

    def test_tcp_stale_status_never_sends(self):
        self.robot.arm.get_arm_status = lambda: SimpleNamespace(
            timestamp=time.time() - 2, msg=SimpleNamespace(arm_status=0, err_code=0))
        with self.assertRaisesRegex(RuntimeError, "controller status"):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_tcp_calibration_mismatch_never_sends(self):
        self.robot.arm.fk = lambda q: [0.32, 0.0, 0.4, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(RuntimeError, "FK differs"):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_tcp_stale_feedback_never_sends(self):
        self.robot.arm._parser.joint_12.timestamp = time.time() - 2
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)

    def test_tcp_external_preview_never_sends(self):
        self.robot.config.control_mode = "external"
        self.robot.config.allow_motion = False
        self.robot.send_action(self.action)
        self.assertFalse(self.robot.arm.sent)
        self.assertFalse(self.robot.gripper.sent)


if __name__ == "__main__":
    unittest.main()
