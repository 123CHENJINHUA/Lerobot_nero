"""Select Nero training inputs without rewriting the recorded LeRobot dataset."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from nero_common import FEATURES, LIMITS, POSE_FEATURES, TORQUE_FEATURES, VELOCITY_FEATURES

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
    def __init__(self, meta, images: str, state: str):
        if tuple(meta.features["action"]["names"]) != FEATURES:
            raise ValueError("Dataset action order does not match Nero joint and gripper channels")
        self.action_limits = [list(limit) for limit in (*LIMITS, (0.0, 0.08))]
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
        return {
            "image_keys": self.image_keys,
            "state_groups": self.state_groups,
            "state_indices": self.state_indices,
            "state_names": self.state_names,
            "original_state_size": self.original_state_size,
            "action_limits": self.action_limits,
        }


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
        frame["action"] = _clip_action(frame["action"], self.selection.action_limits)
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
    if "action_limits" in saved and "action" in result:
        result["action"] = _clip_action(result["action"], saved["action_limits"])
    if indices:
        result["observation.state"] = frame["observation.state"][..., indices]
    else:
        result.pop("observation.state", None)
    return result


def install_training_input_selection(images: str, state: str):
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

        selection = InputSelection(base.meta, images, state)
        result = SelectedDataset(base, selection)
        # Rebuild inputs from this dataset, including when finetuning another robot's policy.
        cfg.policy.input_features = {}
        selection_path = Path(cfg.output_dir) / SELECTION_FILE
        if selection_path.exists():
            if json.loads(selection_path.read_text()) != selection.to_dict():
                raise ValueError(f"Training inputs differ from the existing run: {selection_path}")
        else:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n")
        print(f"Training images: {selection.image_keys}")
        print(f"Training state ({len(selection.state_names)}): {selection.state_names}")
        print(f"Training action limits: {selection.action_limits}; "
              f"clipped {result.clipped_action_frames}/{len(base)} recorded frames")
        return result

    factory.make_dataset = make_selected_dataset
    # Also cover a caller that imported the LeRobot training module earlier.
    import sys
    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is not None:
        train_module.make_dataset = make_selected_dataset
