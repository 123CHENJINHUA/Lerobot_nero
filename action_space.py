"""Nero action definitions shared by training and guarded inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nero_common import FEATURES, POSE_FEATURES

TCP_FEATURES = (*POSE_FEATURES, "gripper.pos")


def action_names(space: str):
    if space not in {"joints", "tcp"}:
        raise ValueError("ACTION_SPACE must be joints or tcp")
    return FEATURES if space == "joints" else TCP_FEATURES


def nero_mdh():
    from pyAgxArm.utiles.mdh_kinematics import get_mdh

    return [list(link) for link in get_mdh("nero")]


def joint_action_to_tcp(action, mdh):
    """Convert absolute demonstration targets, retaining arbitrary chunk dimensions."""
    import torch
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh

    values = torch.as_tensor(action)
    if values.shape[-1] != 8 or not torch.isfinite(values).all():
        raise ValueError("Expected finite 8-dimensional demonstration actions")
    flat = values.detach().cpu().reshape(-1, 8).numpy()
    converted = np.asarray([fk_from_mdh(mdh, row[:7].tolist()) + [float(row[7])]
                            for row in flat])
    return values.new_tensor(converted).reshape(*values.shape[:-1], 7)


def pose_distance(first, second):
    from scipy.spatial.transform import Rotation

    a, b = np.asarray(first), np.asarray(second)
    return (float(np.linalg.norm(a[:3] - b[:3])),
            float((Rotation.from_euler("xyz", a[3:]).inv()
                   * Rotation.from_euler("xyz", b[3:])).magnitude()))


def limit_tcp_pose(target, current, previous, translation_step, rotation_step):
    """Limit geometric motion relative to feedback AND the previous command."""
    from scipy.spatial.transform import Rotation

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


def find_selection(checkpoint: Path) -> Path | None:
    return next((directory / "input_selection.json" for directory in (checkpoint, *checkpoint.parents)
                 if (directory / "input_selection.json").is_file()), None)


def validate_checkpoint_action(config, saved: dict, requested: str | None = None) -> str:
    space = saved.get("action_space", "joints")
    names = action_names(space)
    if requested is not None and requested != space:
        raise ValueError(f"ACTION_SPACE={requested} differs from checkpoint action_space={space}; load the matching model")
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
