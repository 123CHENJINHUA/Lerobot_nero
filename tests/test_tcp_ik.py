"""Local IK must resolve a TCP target without sending a Cartesian command."""

import unittest
from unittest.mock import patch

import numpy as np

from action_space import nero_mdh, pose_distance, solve_tcp_ik
from nero_common import LIMITS
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh


class LocalIKTest(unittest.TestCase):
    def setUp(self):
        self.joints = np.array([0.12, -0.03, 0.04, 2.05, -0.04, -0.02, 0.58])
        self.mdh = nero_mdh()

    def fk(self, joints):
        return fk_from_mdh(self.mdh, list(joints))

    def test_reachable_target_obeys_both_joint_step_bounds(self):
        previous = self.joints + 0.002
        target = self.fk(self.joints + np.array([0.004, 0, -0.003, 0, 0.002, 0, 0]))
        result = np.array(solve_tcp_ik(target, self.joints, previous, 0.01, self.fk))
        position, angle = pose_distance(target, self.fk(result))
        self.assertLessEqual(position, 0.0002)
        self.assertLessEqual(angle, 0.002)
        self.assertTrue(np.all(np.abs(result - self.joints) <= 0.01000001))
        self.assertTrue(np.all(np.abs(result - previous) <= 0.01000001))
        self.assertTrue(np.all(result >= np.asarray(LIMITS)[:, 0]))
        self.assertTrue(np.all(result <= np.asarray(LIMITS)[:, 1]))

    def test_equivalent_far_posture_does_not_change_joint_branch(self):
        original = np.array([0.1229759007692337, -0.028117254376411438, 0.037978366017341614,
                             2.0573418140411377, -0.03851941600441933, -0.01904154196381569,
                             0.5810724496841431])
        alternative = [1.4932481063919136, -0.10211096038592185, -1.2535423762066997,
                       2.0573418140411395, -0.09146238201304545, 0.09594767121124109,
                       0.588557617612471]
        self.assertGreater(max(abs(original - alternative)), 1.3)
        target = self.fk(alternative)
        self.assertLess(max(pose_distance(target, self.fk(original))), 1e-10)
        result = solve_tcp_ik(target, original, original, 0.01, self.fk)
        self.assertLess(max(abs(np.array(result) - original)), 1e-5)

    def test_angle_wrap_is_equivalent_without_turning_joint(self):
        target = self.fk(self.joints)
        target[5] += 2 * np.pi
        result = solve_tcp_ik(target, self.joints, self.joints, 0.01, self.fk)
        self.assertLess(max(abs(np.array(result) - self.joints)), 1e-5)

    def test_unreachable_target_is_rejected(self):
        target = self.fk(self.joints)
        target[0] += 1.0
        with self.assertRaises(RuntimeError):
            solve_tcp_ik(target, self.joints, self.joints, 0.01, self.fk)

    def test_disjoint_command_bounds_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "feedback moved too far"):
            solve_tcp_ik(self.fk(self.joints), self.joints, self.joints + 0.1, 0.01, self.fk)

    def test_joint_fixed_by_intersecting_bounds_is_supported(self):
        seed = self.joints.copy()
        seed[0] = 0.125
        step = 1 / 64  # Exact binary endpoints exercise an exactly fixed joint.
        previous = seed.copy()
        previous[0] -= 2 * step
        target_joints = seed.copy()
        target_joints[0] -= step
        result = solve_tcp_ik(self.fk(target_joints), seed, previous, step, self.fk)
        self.assertAlmostEqual(result[0], target_joints[0])
        self.assertLess(max(pose_distance(self.fk(result), self.fk(target_joints))), 0.0002)

    def test_nonfinite_target_or_fk_is_rejected(self):
        target = self.fk(self.joints)
        with self.assertRaises(ValueError):
            solve_tcp_ik([float("nan")] * 6, self.joints, self.joints, 0.01, self.fk)
        with self.assertRaisesRegex(RuntimeError, "invalid FK"):
            solve_tcp_ik(target, self.joints, self.joints, 0.01, lambda q: [float("nan")] * 6)

    def test_timeout_has_no_unbounded_fallback(self):
        target = self.fk(self.joints)
        with patch("action_space.time.monotonic", side_effect=[0.0, 1.0]):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                solve_tcp_ik(target, self.joints, self.joints, 0.01, self.fk)


if __name__ == "__main__":
    unittest.main()
