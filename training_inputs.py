"""Select Nero training inputs without rewriting the recorded LeRobot dataset."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from nero_common import FEATURES, LIMITS, POSE_FEATURES, TORQUE_FEATURES, VELOCITY_FEATURES
from action_space import action_names, joint_action_to_tcp, nero_mdh

STATE_GROUPS = {
    "joints": FEATURES,  # seven joints and the gripper
    "pose": POSE_FEATURES,
    "torque": TORQUE_FEATURES,
    "velocity": VELOCITY_FEATURES,
}
SELECTION_FILE = "input_selection.json"


def _items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate training input in {value!r}")
    return items


class InputSelection:
    def __init__(self, meta, images: str, state: str, action_space: str = "joints", tcp_rotation: str = "rpy"):
        if tuple(meta.features["action"]["names"]) != FEATURES:
            raise ValueError("Dataset action order does not match Nero joint and gripper channels")
        self.action_limits = [list(limit) for limit in (*LIMITS, (0.0, 0.08))]
        self.action_space = action_space
        self.tcp_rotation = tcp_rotation if action_space == "tcp" else "rpy"
        action_names(action_space, self.tcp_rotation)
        self.mdh = nero_mdh() if action_space == "tcp" else None
        self.tcp_workspace = None
        image_names = _items(images)
        self.state_groups = _items(state)
        unknown = set(self.state_groups) - set(STATE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown TRAIN_STATE group(s): {sorted(unknown)}; choose from {list(STATE_GROUPS)}")

        self.image_keys = [f"observation.images.{name}" for name in image_names]
        available = {key for key in meta.features if key.startswith("observation.images.")}
        missing = set(self.image_keys) - available
        if missing:
            raise ValueError(f"Images missing from dataset: {sorted(missing)}; available: {sorted(available)}")
        if not self.image_keys and not self.state_groups:
            raise ValueError("Select at least one image or state group for training")

        original_names = meta.features["observation.state"]["names"]
        requested = {name for group in self.state_groups for name in STATE_GROUPS[group]}
        missing = requested - set(original_names)
        if missing:
            raise ValueError(f"State values missing from dataset: {sorted(missing)}")
        self.state_indices = [i for i, name in enumerate(original_names) if name in requested]
        self.state_names = [original_names[i] for i in self.state_indices]
        self.original_state_size = len(original_names)

    def to_dict(self) -> dict:
        result = {
            "image_keys": self.image_keys,
            "state_groups": self.state_groups,
            "state_indices": self.state_indices,
            "state_names": self.state_names,
            "original_state_size": self.original_state_size,
            "action_limits": self.action_limits,
            "action_space": self.action_space,
        }
        if self.action_space == "tcp":
            result.update(mdh=self.mdh, tcp_frame="base_flange_zero_tcp",
                          tcp_workspace=self.tcp_workspace)
            if self.tcp_rotation != "rpy":
                result["tcp_rotation"] = self.tcp_rotation
        return result


def transform_action(action, saved):
    if "action_limits" in saved:
        action = _clip_action(action, saved["action_limits"])
    if saved.get("action_space", "joints") == "tcp":
        action = joint_action_to_tcp(action, saved["mdh"], saved.get("tcp_rotation", "rpy"))
    return action


def _clip_action(action, limits):
    bounds = action.new_tensor(limits)
    return action.clamp(min=bounds[:, 0], max=bounds[:, 1])


class SelectedMetadata:
    def __init__(self, base, selection: InputSelection):
        self._base = base
        self.info = copy.deepcopy(base.info)
        self.info["features"] = {
            key: value for key, value in self.info["features"].items()
            if not key.startswith("observation.images.") or key in selection.image_keys
        }
        self.info["features"]["action"]["names"] = list(action_names(selection.action_space, selection.tcp_rotation))
        self.info["features"]["action"]["shape"] = [len(action_names(selection.action_space, selection.tcp_rotation))]
        if selection.state_indices:
            feature = self.info["features"]["observation.state"]
            feature["names"] = selection.state_names
            feature["shape"] = [len(selection.state_indices)]
        else:
            self.info["features"].pop("observation.state")

        self.stats = {
            key: values for key, values in base.stats.items()
            if key in self.info["features"]
        }
        if selection.state_indices and "observation.state" in self.stats:
            self.stats["observation.state"] = {
                name: value[..., selection.state_indices]
                if getattr(value, "ndim", 0) and value.shape[-1] == selection.original_state_size else value
                for name, value in self.stats["observation.state"].items()
            }

    @property
    def features(self):
        return self.info["features"]

    @property
    def video_keys(self):
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]

    @property
    def image_keys(self):
        return [key for key, feature in self.features.items() if feature["dtype"] == "image"]

    @property
    def camera_keys(self):
        return self.video_keys + self.image_keys

    @property
    def names(self):
        return {key: feature.get("names") for key, feature in self.features.items()}

    @property
    def shapes(self):
        return {key: tuple(feature["shape"]) for key, feature in self.features.items()}

    def __getattr__(self, name):
        return getattr(self.__dict__.get("_base"), name)


class ReaderMetadata:
    """Keep the original Parquet schema while decoding only selected videos."""

    def __init__(self, base, selection: InputSelection):
        self._base = base
        self._images = set(selection.image_keys)

    @property
    def video_keys(self):
        return [key for key in self._base.video_keys if key in self._images]

    @property
    def camera_keys(self):
        return [key for key in self._base.camera_keys if key in self._images]

    def __getattr__(self, name):
        return getattr(self.__dict__.get("_base"), name)


class SelectedDataset:
    def __init__(self, base, selection: InputSelection):
        self.base = base
        self.selection = selection
        self.meta = SelectedMetadata(base.meta, selection)
        import torch
        from lerobot.datasets.compute_stats import get_feature_stats

        actions = np.asarray(base.reader.hf_dataset.data.column("action").to_pylist(), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != len(FEATURES) or not np.isfinite(actions).all():
            raise ValueError("Training actions must be finite Nero joint/gripper vectors")
        bounds = np.asarray(selection.action_limits)
        clipped = np.clip(actions, bounds[:, 0], bounds[:, 1])
        self.clipped_action_frames = int(np.count_nonzero(np.any(clipped != actions, axis=1)))
        if selection.action_space == "tcp":
            clipped = joint_action_to_tcp(torch.as_tensor(clipped), selection.mdh, selection.tcp_rotation).numpy()
            if selection.tcp_workspace is None:
                # A training-coverage box is an additional guard, not a reachability test.
                selection.tcp_workspace = np.stack(
                    (clipped[:, :3].min(axis=0) - 0.02, clipped[:, :3].max(axis=0) + 0.02), axis=1,
                ).tolist()
        self.meta.stats["action"] = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in get_feature_stats(clipped, axis=0, keepdims=False).items()
        }
        if base.reader is not None:
            base.reader._meta = ReaderMetadata(base.meta, selection)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        frame = dict(self.base[index])
        frame["action"] = transform_action(frame["action"], self.selection.to_dict())
        for key in tuple(frame):
            if key.startswith("observation.images.") and key not in self.selection.image_keys:
                del frame[key]
        if self.selection.state_indices:
            frame["observation.state"] = frame["observation.state"][..., self.selection.state_indices]
        else:
            frame.pop("observation.state", None)
        return frame

    def __getattr__(self, name):
        return getattr(self.__dict__.get("base"), name)


def apply_saved_selection(frame: dict, meta, path: Path) -> dict:
    """Use the same state channels during offline inference as during training."""
    saved = json.loads(path.read_text())
    names = meta.features["observation.state"]["names"]
    indices = saved["state_indices"]
    if len(names) != saved["original_state_size"] or [names[i] for i in indices] != saved["state_names"]:
        raise ValueError(f"Dataset state layout differs from the model selection in {path}")
    missing = set(saved["image_keys"]) - set(meta.features)
    if missing:
        raise ValueError(f"Dataset is missing model images: {sorted(missing)}")
    result = {key: value for key, value in frame.items()
              if not key.startswith("observation.images.") or key in saved["image_keys"]}
    if "action" in result:
        result["action"] = transform_action(result["action"], saved)
    if indices:
        result["observation.state"] = frame["observation.state"][..., indices]
    else:
        result.pop("observation.state", None)
    return result


def selected_dataset_from_saved(base, saved):
    selection = InputSelection(base.meta, ",".join(key.removeprefix("observation.images.")
                                                for key in saved["image_keys"]),
                               ",".join(saved["state_groups"]), saved.get("action_space", "joints"),
                               saved.get("tcp_rotation", "rpy"))
    if (selection.state_indices != saved["state_indices"] or selection.state_names != saved["state_names"]
            or selection.original_state_size != saved["original_state_size"]):
        raise ValueError("Dataset state layout differs from the checkpoint selection")
    selection.action_limits = saved.get("action_limits", selection.action_limits)
    if selection.action_space == "tcp":
        selection.mdh = saved["mdh"]
        selection.tcp_workspace = saved["tcp_workspace"]
    return SelectedDataset(base, selection)


def install_training_input_selection(images: str, state: str, action_space: str = "joints", tcp_rotation: str = "rpy"):
    """Patch only this process's dataset factory before importing lerobot_train."""
    from lerobot.datasets import factory

    original_make_dataset = factory.make_dataset
    original_resolve = factory.resolve_delta_timestamps

    def make_selected_dataset(cfg):
        def resolve_selected_timestamps(policy_cfg, meta):
            timestamps = original_resolve(policy_cfg, meta)
            if timestamps is None:
                return None
            selected_images = {f"observation.images.{name}" for name in _items(images)}
            selected_state = bool(_items(state))
            result = {
                key: value for key, value in timestamps.items()
                if not key.startswith("observation.")
                or key in selected_images
                or (key == "observation.state" and selected_state)
            }
            return result or None

        factory.resolve_delta_timestamps = resolve_selected_timestamps
        try:
            base = original_make_dataset(cfg)
        finally:
            factory.resolve_delta_timestamps = original_resolve

        selection = InputSelection(base.meta, images, state, action_space, tcp_rotation)
        result = SelectedDataset(base, selection)
        # Rebuild inputs from this dataset, including when finetuning another robot's policy.
        cfg.policy.input_features = {}
        cfg.policy.output_features = {}
        selection_path = Path(cfg.output_dir) / SELECTION_FILE
        if selection_path.exists():
            saved = json.loads(selection_path.read_text())
            saved.setdefault("action_space", "joints")
            if saved != selection.to_dict():
                raise ValueError(f"Training inputs differ from the existing run: {selection_path}")
        else:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n")
        print(f"Training images: {selection.image_keys}")
        print(f"Training state ({len(selection.state_names)}): {selection.state_names}")
        print(f"Training action_space={selection.action_space} tcp_rotation={selection.tcp_rotation}: "
              f"{action_names(selection.action_space, selection.tcp_rotation)}")
        print(f"Training action limits: {selection.action_limits}; "
              f"clipped {result.clipped_action_frames}/{len(base)} recorded frames")
        return result

    factory.make_dataset = make_selected_dataset
    # Also cover a caller that imported the LeRobot training module earlier.
    import sys
    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is not None:
        train_module.make_dataset = make_selected_dataset
