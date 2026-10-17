"""LeRobot Nero follower. Joint values are radians; gripper width is metres."""

from __future__ import annotations

import math
import time

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from nero_common import (
    FEATURES, LIMITS, OBSERVATION_FEATURES, POSE_FEATURES, TORQUE_FEATURES,
    VELOCITY_FEATURES, TCPVelocity, close_arm, complete_joints, follower_telemetry,
    gripper_width, open_arm, wait_feedback,
)
from .config_nero import NeroFollowerConfig
from action_space import action_names, limit_tcp_pose, pose_distance


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
        self._last_tcp_joints = None
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
        return dict.fromkeys(action_names(self.config.action_space), float)

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
        names = action_names(self.config.action_space)
        if set(action) != set(names):
            raise ValueError(f"Expected action keys: {names}")
        target = [float(action[key]) for key in names]
        if not all(math.isfinite(v) for v in target):
            raise ValueError("Action contains a non-finite value")
        if self.config.control_mode == "external":
            self._read_state()  # ensure the hardware follower is still publishing feedback
            return dict(zip(names, target, strict=True))
        if not self.config.allow_motion:
            raise RuntimeError("Host motion is disabled; set --robot.allow_motion=true explicitly")
        if self.config.action_space == "tcp":
            return self._send_tcp_action(target)
        for i, (value, (lo, hi)) in enumerate(zip(target[:7], LIMITS, strict=True), 1):
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
            if i == 7 and not lo <= present[i] <= hi:
                raise RuntimeError(
                    f"gripper.pos feedback {present[i]:.6f}m outside configured range "
                    f"[{lo:.6f}, {hi:.6f}]m; check gripper calibration and actual travel "
                    "before execution; motion stopped"
                )
            lower = max(lo, present[i] - step, previous[i] - step)
            upper = min(hi, present[i] + step, previous[i] + step)
            if lower > upper:
                unit = "rad" if i < 7 else "m"
                raise RuntimeError(
                    f"{FEATURES[i]} feedback moved too far from last command: "
                    f"current={present[i]:.6f}{unit} last_sent={previous[i]:.6f}{unit} "
                    f"gap={abs(present[i] - previous[i]):.6f}{unit} "
                    f"step={step:.6f}{unit} target={value:.6f}{unit}; motion stopped"
                )
            sent.append(max(lower, min(upper, value)))
        self.arm.move_j(sent[:7])
        self.gripper.move_gripper_m(sent[7], self.config.gripper_force_n)
        self._last_sent = sent
        return dict(zip(FEATURES, sent, strict=True))

    def _send_tcp_action(self, target):
        if self.config.tcp_workspace is None:
            raise ValueError("TCP execution requires the checkpoint's tcp_workspace")
        if not 0 <= target[-1] <= self.config.gripper_max_m:
            raise ValueError("Gripper width outside configured range")
        state = self._read_state()
        joints = [state[key] for key in FEATURES[:7]]
        for value, (lo, hi) in zip(joints, LIMITS, strict=True):
            if not lo - 0.002 <= value <= hi + 0.002:
                raise RuntimeError("TCP joint feedback outside joint limits; motion stopped")
        status = self.arm.get_arm_status()
        if (status is None or not 0 <= time.time() - float(status.timestamp) <= self.config.max_feedback_age_s
                or int(status.msg.arm_status) != 0 or int(status.msg.err_code) != 0):
            raise RuntimeError("Missing, stale or abnormal TCP controller status (IK/limits/collision); motion stopped")
        _, current, _ = follower_telemetry(self.arm, self.config.max_feedback_age_s)
        distance, angle = pose_distance(current, self.arm.fk(joints))
        if distance > 0.01 or angle > 0.05:
            raise RuntimeError("SDK FK differs from TCP feedback; check calibration/TCP offset before execution")
        if self._last_tcp_joints is not None:
            if max(abs(a - b) for a, b in zip(joints, self._last_tcp_joints)) > 2 * self.config.max_step_rad:
                raise RuntimeError("TCP joint feedback jumped; motion stopped")
            for name in ("ik_joint_12", "ik_joint_34", "ik_joint_56", "ik_joint_7"):
                part = getattr(self.arm._parser, name, None)
                if part is None or not 0 <= time.time() - float(part.timestamp) <= self.config.max_feedback_age_s:
                    raise RuntimeError(f"Missing or stale TCP {name} feedback; motion stopped")
            ik = self.arm.get_ik_joint_angles()
            if (ik is None or not 0 <= time.time() - float(ik.timestamp) <= self.config.max_feedback_age_s
                    or len(ik.msg) != 7 or not all(math.isfinite(v) for v in ik.msg)):
                raise RuntimeError("Missing or stale TCP IK feedback; motion stopped")
            for index, (value, (lo, hi)) in enumerate(zip(ik.msg, LIMITS, strict=True)):
                if not lo <= value <= hi or abs(value - joints[index]) > self.config.max_initial_error_rad:
                    raise RuntimeError("TCP IK feedback outside joint limits or too far from follower; motion stopped")
        bounds = np.asarray(self.config.tcp_workspace)
        if np.any(current[:3] < bounds[:, 0]) or np.any(current[:3] > bounds[:, 1]):
            raise RuntimeError("Current TCP is outside the training workspace; align before execution")
        target = list(target)
        target[:3] = np.clip(target[:3], bounds[:, 0], bounds[:, 1]).tolist()
        distance, angle = pose_distance(current, target[:6])
        if distance > self.config.max_tcp_initial_error_m or angle > self.config.max_tcp_initial_error_rad:
            raise RuntimeError(f"TCP target too far from follower: {distance:.4f}m {angle:.4f}rad; align before execution")
        previous = self._last_sent if self._last_sent is not None else [*current, state["gripper.pos"]]
        pose = limit_tcp_pose(target[:6], current, previous[:6],
                              self.config.max_tcp_step_m, self.config.max_tcp_step_rad)
        width = state["gripper.pos"]
        if not 0 <= width <= self.config.gripper_max_m:
            raise RuntimeError("Gripper feedback outside configured range; motion stopped")
        step = self.config.max_gripper_step_m
        lower = max(0.0, width - step, previous[-1] - step)
        upper = min(self.config.gripper_max_m, width + step, previous[-1] + step)
        if lower > upper:
            raise RuntimeError("Gripper feedback moved too far from last command; motion stopped")
        sent = [*pose, max(lower, min(upper, target[-1]))]
        # IK is performed inside the controller; feedback guards cannot preflight its first solution.
        self.arm.move_p(self.arm.get_tcp2flange_pose(pose.copy()))
        self.gripper.move_gripper_m(sent[-1], self.config.gripper_force_n)
        self._last_sent = sent
        self._last_tcp_joints = joints
        return dict(zip(action_names("tcp"), sent, strict=True))

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
        self._last_tcp_joints = None
