"""LeRobot Nero follower. Joint values are radians; gripper width is metres."""

from __future__ import annotations

import math

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from nero_common import (
    FEATURES, LIMITS, OBSERVATION_FEATURES, POSE_FEATURES, TORQUE_FEATURES,
    VELOCITY_FEATURES, TCPVelocity, close_arm, complete_joints, follower_telemetry,
    gripper_width, open_arm, wait_feedback,
)
from .config_nero import NeroFollowerConfig


class NeroFollower(Robot):
    config_class = NeroFollowerConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.arm = None
        self.gripper = None
        self._last_sent = None
        self.tcp_velocity = TCPVelocity()
        from depth_record import ACTIVE_HOOK
        self.depth_hook = ACTIVE_HOOK

    @property
    def observation_features(self):
        return {**dict.fromkeys(OBSERVATION_FEATURES, float), **{
            name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()
        }, **(self.depth_hook.image_features if self.depth_hook is not None else {})}

    @property
    def action_features(self):
        return dict.fromkeys(FEATURES, float)

    @property
    def is_connected(self):
        return self.arm is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self):
        return True  # Nero encoder calibration belongs to the hardware/SDK.

    def calibrate(self):
        pass

    def _read_state(self):
        joints = complete_joints(self.arm, False, self.config.firmware, self.config.max_feedback_age_s)
        width = gripper_width(self.gripper, False, self.config.max_feedback_age_s)
        return dict(zip(FEATURES, (*joints, width), strict=True))

    def connect(self, calibrate: bool = True):
        if self.arm is not None:
            raise RuntimeError("Nero follower already connected")
        try:
            self.arm, self.gripper = open_arm(self.config.channel, self.config.firmware)
            wait_feedback(self._read_state)
            wait_feedback(lambda: follower_telemetry(self.arm, self.config.max_feedback_age_s))
            for camera in self.cameras.values():
                camera.connect()
            if self.depth_hook is not None:
                self.depth_hook.attach()
            self.configure()
        except BaseException:
            self.disconnect()
            raise

    def configure(self):
        if self.config.control_mode == "host" and self.config.allow_motion and self.config.manage_mode:
            self.arm.set_follower_mode()
            if not self.arm.enable():
                raise RuntimeError("Nero joint enable failed; inspect hardware status")

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError("Nero follower is not connected")
        obs = self._read_state()
        torques, pose, pose_timestamp = follower_telemetry(self.arm, self.config.max_feedback_age_s)
        velocity = self.tcp_velocity.update(pose, pose_timestamp)
        obs.update(zip(TORQUE_FEATURES, torques, strict=True))
        obs.update(zip(POSE_FEATURES, pose, strict=True))
        obs.update(zip(VELOCITY_FEATURES, velocity, strict=True))
        for name, camera in self.cameras.items():
            obs[name] = camera.async_read()
        if self.depth_hook is not None:
            obs.update(self.depth_hook.snapshot_images())
        return obs

    def send_action(self, action):
        if not self.is_connected:
            raise RuntimeError("Nero follower is not connected")
        if set(action) != set(FEATURES):
            raise ValueError(f"Expected action keys: {FEATURES}")
        target = [float(action[key]) for key in FEATURES]
        if not all(math.isfinite(v) for v in target):
            raise ValueError("Action contains a non-finite value")
        if self.config.control_mode == "external":
            self._read_state()  # ensure the hardware follower is still publishing feedback
            return dict(zip(FEATURES, target, strict=True))
        if not self.config.allow_motion:
            raise RuntimeError("Host motion is disabled; set --robot.allow_motion=true explicitly")
        for i, (value, (lo, hi)) in enumerate(zip(target[:7], LIMITS, strict=True), 1):
            if not lo - self.config.max_step_rad <= value <= hi + self.config.max_step_rad:
                raise ValueError(f"joint{i} target {value:.3f} outside [{lo:.3f}, {hi:.3f}] rad")
            target[i - 1] = max(lo, min(hi, value))
        if not 0 <= target[7] <= self.config.gripper_max_m:
            raise ValueError(
                f"Gripper width {target[7]:.6f}m outside configured range "
                f"[0, {self.config.gripper_max_m:.6f}]m"
            )
        current = self._read_state()
        present = [current[key] for key in FEATURES]
        index = max(range(7), key=lambda i: abs(target[i] - present[i]))
        error = abs(target[index] - present[index])
        if error > self.config.max_initial_error_rad:
            raise RuntimeError(
                f"{FEATURES[index]} target too far from follower: current={present[index]:.4f}rad "
                f"target={target[index]:.4f}rad error={error:.4f}rad "
                f"limit={self.config.max_initial_error_rad:.4f}rad; align arms before starting"
            )
        previous = self._last_sent if self._last_sent is not None else present
        sent = []
        for i, value in enumerate(target):
            step = self.config.max_step_rad if i < 7 else self.config.max_gripper_step_m
            lo, hi = LIMITS[i] if i < 7 else (0.0, self.config.gripper_max_m)
            lower = max(lo, present[i] - step, previous[i] - step)
            upper = min(hi, present[i] + step, previous[i] + step)
            if lower > upper:
                raise RuntimeError(f"{FEATURES[i]} feedback moved too far from last command; motion stopped")
            sent.append(max(lower, min(upper, value)))
        self.arm.move_j(sent[:7])
        self.gripper.move_gripper_m(sent[7], self.config.gripper_force_n)
        self._last_sent = sent
        return dict(zip(FEATURES, sent, strict=True))

    def disconnect(self):
        if self.depth_hook is not None:
            self.depth_hook.detach()
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.arm is not None:
            close_arm(self.arm)
        self.arm = self.gripper = None
        self._last_sent = None
