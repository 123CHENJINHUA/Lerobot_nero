"""Nero action definitions shared by training and guarded inference."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nero_common import FEATURES, POSE_FEATURES

TCP_FEATURES = (*POSE_FEATURES, "gripper.pos")
TCP_6D_FEATURES = (*POSE_FEATURES[:3],
                   *(f"ee.rotation6d.{column}.{axis}" for column in ("c1", "c2") for axis in "xyz"),
                   "gripper.pos")


class IKUnreachableError(RuntimeError):
    """The requested TCP step does not fit the allowed joint command interval."""


def action_names(space: str, tcp_rotation: str = "rpy"):
    if space not in {"joints", "tcp"}:
        raise ValueError("ACTION_SPACE must be joints or tcp")
    if tcp_rotation not in {"rpy", "6d"}:
        raise ValueError("TCP_ROTATION must be rpy or 6d")
    return FEATURES if space == "joints" else (TCP_FEATURES if tcp_rotation == "rpy" else TCP_6D_FEATURES)


def encode_tcp_action(action, tcp_rotation="rpy"):
    """Encode base-frame xyz/RPY/gripper; 6D stores matrix column 1 then column 2."""
    action_names("tcp", tcp_rotation)
    values = np.asarray(action, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected finite xyz/RPY/gripper actions")
    if tcp_rotation == "rpy":
        return values
    matrix = Rotation.from_euler("xyz", values[..., 3:6].reshape(-1, 3)).as_matrix()
    rotation = np.concatenate((matrix[:, :, 0], matrix[:, :, 1]), axis=-1).reshape(*values.shape[:-1], 6)
    return np.concatenate((values[..., :3], rotation, values[..., -1:]), axis=-1)


def decode_tcp_action(action, tcp_rotation="rpy"):
    """Reject degenerate axes instead of silently inventing an execution rotation."""
    size = len(action_names("tcp", tcp_rotation))
    values = np.asarray(action, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != size or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {size}-dimensional TCP actions")
    if tcp_rotation == "rpy":
        return values
    first, second = values[..., 3:6], values[..., 6:9]
    norms = [np.linalg.norm(vector, axis=-1, keepdims=True) for vector in (first, second)]
    if any(not np.isfinite(norm).all() or np.any(norm < 1e-6) for norm in norms):
        raise ValueError("Degenerate 6D rotation: zero/invalid axis; no command sent")
    first, second = first / norms[0], second / norms[1]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate 6D rotation: parallel axes; no command sent")
    second = second / norm
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    rpy = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_euler("xyz").reshape(*values.shape[:-1], 3)
    return np.concatenate((values[..., :3], rpy, values[..., -1:]), axis=-1)


def nero_mdh():
    from pyAgxArm.utiles.mdh_kinematics import get_mdh

    return [list(link) for link in get_mdh("nero")]


def joint_action_to_tcp(action, mdh, tcp_rotation="rpy"):
    """Convert absolute demonstration targets, retaining arbitrary chunk dimensions."""
    import torch
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh

    values = torch.as_tensor(action)
    if values.shape[-1] != 8 or not torch.isfinite(values).all():
        raise ValueError("Expected finite 8-dimensional demonstration actions")
    flat = values.detach().cpu().reshape(-1, 8).numpy()
    converted = np.asarray([fk_from_mdh(mdh, row[:7].tolist()) + [float(row[7])]
                            for row in flat])
    converted = encode_tcp_action(converted, tcp_rotation)
    return values.new_tensor(converted).reshape(*values.shape[:-1], len(action_names("tcp", tcp_rotation)))


def pose_distance(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return (float(np.linalg.norm(a[:3] - b[:3])),
            float((Rotation.from_euler("xyz", a[3:]).inv()
                   * Rotation.from_euler("xyz", b[3:])).magnitude()))


def limit_tcp_pose(target, current, previous, translation_step, rotation_step):
    """Limit geometric motion relative to feedback AND the previous command."""
    result = np.asarray(target, dtype=np.float64).copy()
    for reference in (previous, current):
        reference = np.asarray(reference)
        delta = result[:3] - reference[:3]
        result[:3] = reference[:3] + delta * min(1.0, translation_step / max(np.linalg.norm(delta), 1e-12))
        origin = Rotation.from_euler("xyz", reference[3:])
        rotation = (origin.inv() * Rotation.from_euler("xyz", result[3:])).as_rotvec()
        rotation *= min(1.0, rotation_step / max(np.linalg.norm(rotation), 1e-12))
        result[3:] = (origin * Rotation.from_rotvec(rotation)).as_euler("xyz")
    for reference in (previous, current):
        distance, angle = pose_distance(result, reference)
        if distance > translation_step + 1e-8 or angle > rotation_step + 1e-8:
            raise RuntimeError("TCP feedback moved too far from last command; motion stopped")
    return result.tolist()


def solve_tcp_ik(target, joints, previous, max_step, fk, timeout_s=0.03,
                 position_tolerance=0.0002, rotation_tolerance=0.002):
    """Solve only inside the joint command limits, with no hardware side effects."""
    from nero_common import LIMITS

    started = time.monotonic()
    target, seed, previous = (np.asarray(value, dtype=np.float64) for value in (target, joints, previous))
    if (target.shape != (6,) or seed.shape != (7,) or previous.shape != (7,)
            or not all(np.isfinite(value).all() for value in (target, seed, previous))
            or not np.isfinite(max_step) or max_step <= 0
            or not np.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("Local IK requires finite pose/joints and positive step/time limits")
    limits = np.asarray(LIMITS)
    lower = np.maximum.reduce((limits[:, 0], seed - max_step, previous - max_step))
    upper = np.minimum.reduce((limits[:, 1], seed + max_step, previous + max_step))
    if np.any(lower > upper):
        raise RuntimeError("Joint feedback moved too far from last command; local IK refused")
    origin = np.clip(seed, lower, upper)
    free = upper - lower > 1e-10
    goal_rotation = Rotation.from_euler("xyz", target[3:])

    def residual(values):
        if time.monotonic() - started > timeout_s:
            raise RuntimeError(f"Local IK exceeded {timeout_s:.3f}s; no command sent")
        candidate = origin.copy()
        candidate[free] = values
        pose = np.asarray(fk(candidate.tolist()), dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("Local IK received invalid FK results; no command sent")
        rotation = (goal_rotation.inv() * Rotation.from_euler("xyz", pose[3:])).as_rotvec()
        # A weak posture penalty chooses the nearby solution in the redundant DOF.
        return np.r_[pose[:3] - target[:3], 0.1 * rotation, 1e-4 * (candidate - seed)]

    solution = origin.copy()
    if free.any():
        result = least_squares(residual, origin[free], bounds=(lower[free], upper[free]),
                               max_nfev=30, ftol=1e-9, xtol=1e-9, gtol=1e-10)
        solution[free] = result.x
    error = residual(solution[free])
    position_error = float(np.linalg.norm(error[:3]))
    rotation_error = float(np.linalg.norm(error[3:6]) / 0.1)
    if (position_error > position_tolerance or rotation_error > rotation_tolerance
            or not np.isfinite(solution).all() or np.any(solution < lower) or np.any(solution > upper)):
        raise IKUnreachableError(f"Local IK cannot reach the limited TCP target within the joint step: "
                                 f"position_error={position_error:.6f}m rotation_error={rotation_error:.6f}rad; "
                                 "no command sent")
    if time.monotonic() - started > timeout_s:
        raise RuntimeError(f"Local IK exceeded {timeout_s:.3f}s; no command sent")
    return solution.tolist()


def find_selection(checkpoint: Path) -> Path | None:
    return next((directory / "input_selection.json" for directory in (checkpoint, *checkpoint.parents)
                 if (directory / "input_selection.json").is_file()), None)


def validate_checkpoint_action(config, saved: dict, requested: str | None = None,
                               requested_rotation: str | None = None) -> str:
    space = saved.get("action_space", "joints")
    rotation = saved.get("tcp_rotation", "rpy")
    names = action_names(space, rotation)
    if requested is not None and requested != space:
        raise ValueError(f"ACTION_SPACE={requested} differs from checkpoint action_space={space}; load the matching model")
    if space == "tcp" and requested_rotation is not None and requested_rotation != rotation:
        raise ValueError(f"TCP_ROTATION={requested_rotation} differs from checkpoint tcp_rotation={rotation}")
    feature = config.output_features.get("action")
    if feature is None or tuple(feature.shape) != (len(names),):
        raise ValueError("Checkpoint action dimension differs from its action definition; keep input_selection.json with the model")
    if space == "tcp":
        if saved.get("tcp_frame") != "base_flange_zero_tcp" or saved.get("mdh") != nero_mdh():
            raise ValueError("TCP checkpoint kinematics/frame differs from the current SDK")
        workspace = np.asarray(saved.get("tcp_workspace"), dtype=float)
        if workspace.shape != (3, 2) or not np.isfinite(workspace).all() or np.any(workspace[:, 0] >= workspace[:, 1]):
            raise ValueError("TCP checkpoint is missing valid workspace bounds")
    return space
