"""Rotation encoding must be continuous at pi and reject invalid predictions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from action_space import decode_tcp_action, encode_tcp_action, pose_distance


class Rotation6DTest(unittest.TestCase):
    def test_round_trip_batches_keep_position_gripper_and_rotation(self):
        rng = np.random.default_rng(17)
        rotations = Rotation.random(20, random_state=rng)
        action = np.c_[rng.normal(size=(20, 3)), rotations.as_euler("xyz"), rng.uniform(0, 0.08, 20)]
        action = action.reshape(2, 10, 7)
        encoded = encode_tcp_action(action, "6d")
        self.assertEqual(encoded.shape, (2, 10, 10))
        decoded = decode_tcp_action(encoded, "6d")
        self.assertTrue(np.allclose(decoded[..., :3], action[..., :3]))
        self.assertTrue(np.allclose(decoded[..., -1], action[..., -1]))
        error = (rotations.inv() * Rotation.from_euler("xyz", decoded[..., 3:6].reshape(-1, 3))).magnitude()
        self.assertLess(float(error.max()), 1e-12)

    def test_columns_are_in_the_documented_order(self):
        pose = [0.1, 0.2, 0.3, 0.2, -0.3, 0.4, 0.05]
        matrix = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
        encoded = encode_tcp_action(pose, "6d")
        self.assertTrue(np.allclose(encoded[3:6], matrix[:, 0]))
        self.assertTrue(np.allclose(encoded[6:9], matrix[:, 1]))

    def test_average_across_pi_stays_near_pi(self):
        poses = np.array([[0, 0, 0, 0, 0, np.deg2rad(angle), 0.03] for angle in (179, -179)])
        encoded = encode_tcp_action(poses, "6d")
        self.assertLess(np.linalg.norm(encoded[0] - encoded[1]), 0.05)
        average = decode_tcp_action(encoded.mean(axis=0), "6d")
        self.assertAlmostEqual(pose_distance(average[:6], [0, 0, 0, 0, 0, np.pi])[1], 0.0)

    def test_zero_parallel_and_nonfinite_axes_rejected(self):
        valid = encode_tcp_action([0, 0, 0, 0, 0, 0, 0.03], "6d")
        for vectors in ([0] * 6, [1, 0, 0, 2, 0, 0], [1, 0, 0, -1, 0, 0], [float("nan")] * 6):
            action = valid.copy()
            action[3:9] = vectors
            with self.assertRaises(ValueError):
                decode_tcp_action(action, "6d")


if __name__ == "__main__":
    unittest.main()
