"""CAN identification with simulated feedback; no hardware access."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import can_test_dual as identify


class CanIdentifyTest(unittest.TestCase):
    def test_discover_can_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (("can0", "280"), ("can1", "280"), ("eth0", "1")):
                (root / name).mkdir()
                (root / name / "type").write_text(kind)
            (root / "gone").mkdir()
            self.assertEqual(identify.can_channels(root), ["can0", "can1"])

    def test_reject_missing_ambiguous_or_no_movement(self):
        self.assertEqual(identify.moving_channel({"can0": 5, "can1": 0.1}, 3), "can0")
        for scores in ({"can0": 5, "can1": None}, {"can0": 5, "can1": 4},
                       {"can0": 0, "can1": 0}):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                identify.moving_channel(scores, 3)

    def test_sample_detects_leader_only_feedback(self):
        arms = {"can0": object(), "can1": object()}
        counts = dict.fromkeys(arms.values(), 0)

        def read(arm, leader, firmware, max_age):
            if not leader:
                raise RuntimeError("No follower feedback")
            counts[arm] += 1
            return [0.1 * (counts[arm] - 1) if arm is arms["can0"] else 0.0] * 7

        with patch.object(identify, "complete_joints", side_effect=read), \
             patch.object(identify.time, "monotonic", side_effect=[0, 0, 0, 0, 0.1, 0.1, 0.2]), \
             patch.object(identify.time, "sleep"), redirect_stdout(io.StringIO()):
            scores = identify.sample_motion(arms, "v121", 0.15)
        self.assertEqual(identify.moving_channel(scores, 3), "can0")
        self.assertEqual(scores["can1"], 0)

    def test_guided_mapping_and_cleanup(self):
        left_arm, right_arm = object(), object()
        output = io.StringIO()
        with patch("sys.argv", ["identify-can", "--channels", "can0", "can1"]), \
             patch("builtins.input", return_value=""), \
             patch.object(identify, "validate_channel"), \
             patch.object(identify, "open_arm", side_effect=[(left_arm, None), (right_arm, None)]), \
             patch.object(identify, "close_arm") as close, \
             patch.object(identify, "sample_motion", side_effect=[
                 {"can0": 0, "can1": 6}, {"can0": 7, "can1": 0}]), redirect_stdout(output):
            identify.main()
        self.assertIn('LEFT_CHANNEL="can1"\nRIGHT_CHANNEL="can0"', output.getvalue())
        self.assertEqual([call.args[0] for call in close.call_args_list], [right_arm, left_arm])

    def test_connection_failure_closes_previous_arm(self):
        arm = object()
        with patch("sys.argv", ["identify-can", "--channels", "can0", "can1"]), \
             patch.object(identify, "validate_channel"), \
             patch.object(identify, "open_arm", side_effect=[(arm, None), RuntimeError("CAN failed")]), \
             patch.object(identify, "close_arm") as close, redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            identify.main()
        self.assertEqual(error.exception.code, 1)
        close.assert_called_once_with(arm)


if __name__ == "__main__":
    unittest.main()
