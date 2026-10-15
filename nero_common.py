"""Shared Nero hardware helpers for the local LeRobot 0.6.1 adapters."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

JOINTS = tuple(f"joint{i}.pos" for i in range(1, 8))
FEATURES = (*JOINTS, "gripper.pos")
TORQUE_FEATURES = tuple(f"joint{i}.torque" for i in range(1, 8))
POSE_FEATURES = tuple(f"ee.pose.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
VELOCITY_FEATURES = tuple(f"ee.velocity.{axis}" for axis in ("vx", "vy", "vz", "wx", "wy", "wz"))
OBSERVATION_FEATURES = (*FEATURES, *TORQUE_FEATURES, *POSE_FEATURES, *VELOCITY_FEATURES)
LIMITS = (
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
)
FIRMWARES = {"default", "v111", "v112", "v120", "v121"}
_CONNECTIONS = {}


def validate_channel(channel: str) -> None:
    if not channel or "/" in channel or channel.startswith("."):
        raise ValueError(f"Invalid SocketCAN channel: {channel!r}")
    flags = Path("/sys/class/net") / channel / "flags"
    try:
        up = int(flags.read_text().strip(), 16) & 1
    except OSError as exc:
        raise RuntimeError(f"{channel} does not exist; configure the CAN adapter first") from exc
    if not up:
        raise RuntimeError(f"{channel} is DOWN; configure it at 1 Mbps first")


def open_arm(channel: str, firmware: str):
    if firmware not in FIRMWARES:
        raise ValueError(f"Unsupported firmware profile: {firmware}")
    validate_channel(channel)
    key = (channel, firmware)
    if key in _CONNECTIONS:
        arm, gripper, users = _CONNECTIONS[key]
        _CONNECTIONS[key] = (arm, gripper, users + 1)
        return arm, gripper
    import pyAgxArm as sdk

    cfg = sdk.create_agx_arm_config(
        robot=sdk.ArmModel.NERO,
        firmeware_version=firmware,  # spelling is the SDK's public API
        interface="socketcan",
        channel=channel,
        auto_connect=False,
        enable_check_can=False,
    )
    arm = sdk.AgxArmFactory.create_arm(cfg)
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    arm.connect()
    _CONNECTIONS[key] = (arm, gripper, 1)
    return arm, gripper


def close_arm(arm) -> None:
    for key, (existing, gripper, users) in tuple(_CONNECTIONS.items()):
        if existing is arm:
            if users > 1:
                _CONNECTIONS[key] = (existing, gripper, users - 1)
            else:
                del _CONNECTIONS[key]
                arm.disconnect()
            return
    # Test doubles or a partially opened connection.
    arm.disconnect()


def complete_joints(arm, leader: bool, firmware: str, max_age_s: float):
    if leader:
        names = (tuple(f"leader_joint_{i}" for i in range(1, 8))
                 if firmware in {"default", "v111"} else
                 ("leader_joint_12", "leader_joint_34", "leader_joint_56", "leader_joint_7"))
        getter = arm.get_leader_joint_angles
    else:
        names = ("joint_12", "joint_34", "joint_56", "joint_7")
        getter = arm.get_joint_angles
    now = time.time()
    for name in names:
        part = getattr(arm._parser, name, None)
        if part is None or not 0 <= now - float(part.timestamp) <= max_age_s:
            raise RuntimeError(f"Missing or stale {name} feedback")
    message = getter()
    if message is None:
        raise RuntimeError("Joint feedback is unavailable")
    values = [float(v) for v in message.msg]
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("Invalid joint feedback")
    return values


def gripper_width(gripper, leader: bool, max_age_s: float):
    message = (gripper.get_gripper_ctrl_states() if leader else gripper.get_gripper_status())
    if message is None or not 0 <= time.time() - float(message.timestamp) <= max_age_s:
        raise RuntimeError("Missing or stale gripper feedback")
    state = message.msg
    mode = ("angle" if int(state.status_code) & 4 else "width") if leader else str(state.mode)
    if mode != "width":
        raise RuntimeError("AGX gripper must be in width mode (metres)")
    value = float(state.value)
    if not math.isfinite(value) or not 0 <= value <= 0.10:
        raise RuntimeError(f"Invalid gripper width: {value}")
    return value


def follower_telemetry(arm, max_age_s: float):
    """Read current-derived joint torque and the TCP pose from follower feedback."""
    now = time.time()
    torques = []
    for index in range(1, 8):
        state = arm.get_motor_states(index)
        if state is None or not 0 <= now - float(state.timestamp) <= max_age_s:
            raise RuntimeError(f"Missing or stale follower motor state {index}")
        value = float(state.msg.torque)
        if not math.isfinite(value):
            raise RuntimeError(f"Invalid follower torque {index}")
        torques.append(value)

    for part_name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"):
        part = getattr(arm._parser, part_name, None)
        if part is None or not 0 <= now - float(part.timestamp) <= max_age_s:
            raise RuntimeError(f"Missing or stale follower {part_name}")
    pose_feedback = arm.get_tcp_pose()
    if pose_feedback is None:
        raise RuntimeError("Follower TCP pose is unavailable")
    pose = np.asarray(pose_feedback.msg, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError("Invalid follower TCP pose")
    return torques, pose, float(pose_feedback.timestamp)


def _rpy_quaternion(rpy: np.ndarray) -> np.ndarray:
    """XYZ extrinsic Euler angles to an (x, y, z, w) quaternion."""
    cr, cp, cy = np.cos(rpy / 2)
    sr, sp, sy = np.sin(rpy / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _world_angular_delta(previous_rpy: np.ndarray, current_rpy: np.ndarray) -> np.ndarray:
    previous = _rpy_quaternion(previous_rpy)
    current = _rpy_quaternion(current_rpy)
    # current * inverse(previous), giving angular displacement in the base frame.
    px, py, pz, pw = -previous[0], -previous[1], -previous[2], previous[3]
    x, y, z, w = current
    vector = np.array([w * px + x * pw + y * pz - z * py,
                       w * py - x * pz + y * pw + z * px,
                       w * pz + x * py - y * px + z * pw])
    scalar = w * pw - x * px - y * py - z * pz
    if scalar < 0:
        vector, scalar = -vector, -scalar
    magnitude = np.linalg.norm(vector)
    return np.zeros(3) if magnitude < 1e-12 else (2 * math.atan2(magnitude, scalar) / magnitude) * vector


class TCPVelocity:
    """Estimate world-frame TCP linear/angular velocity from consecutive SDK poses."""

    def __init__(self):
        self.previous_pose = None
        self.previous_timestamp = None
        self.last_velocity = np.zeros(6, dtype=np.float64)

    def update(self, pose: np.ndarray, timestamp: float) -> np.ndarray:
        if self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            if dt <= 1e-4:
                return self.last_velocity.copy()
            if dt <= 1.0:
                linear = (pose[:3] - self.previous_pose[:3]) / dt
                angular = _world_angular_delta(self.previous_pose[3:], pose[3:]) / dt
                self.last_velocity = np.concatenate((linear, angular))
            else:
                self.last_velocity = np.zeros(6, dtype=np.float64)
        self.previous_pose = pose.copy()
        self.previous_timestamp = timestamp
        if not np.isfinite(self.last_velocity).all():
            raise RuntimeError("Invalid follower TCP velocity")
        return self.last_velocity.copy()


def wait_feedback(read, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            return read()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"No complete, fresh Nero feedback within {timeout_s}s: {last_error}")
