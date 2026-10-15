"""Passive Nero leader; connect and reads send no motion/mode CAN commands."""

from lerobot.teleoperators.teleoperator import Teleoperator

from nero_common import FEATURES, close_arm, complete_joints, gripper_width, open_arm, wait_feedback
from .config_nero import NeroLeaderConfig


class NeroLeader(Teleoperator):
    config_class = NeroLeaderConfig
    name = "nero_leader"

    def __init__(self, config: NeroLeaderConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self.gripper = None

    @property
    def action_features(self):
        return dict.fromkeys(FEATURES, float)

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return self.arm is not None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise RuntimeError("Nero leader already connected")
        try:
            self.arm, self.gripper = open_arm(self.config.channel, self.config.firmware)
            wait_feedback(self.get_action)
        except BaseException:
            self.disconnect()
            raise

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError("Nero leader is not connected")
        joints = complete_joints(self.arm, True, self.config.firmware, self.config.max_feedback_age_s)
        width = gripper_width(self.gripper, True, self.config.max_feedback_age_s)
        return dict(zip(FEATURES, (*joints, width), strict=True))

    def send_feedback(self, feedback):
        if feedback:
            raise ValueError("Nero leader does not support force feedback")

    def disconnect(self):
        if self.arm is not None:
            close_arm(self.arm)
        self.arm = self.gripper = None
