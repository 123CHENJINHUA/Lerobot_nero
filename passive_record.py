"""Receive-only Nero leader recorder into a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs.video import RGBEncoderConfig

from lerobot_teleoperator_nero import NeroLeader, NeroLeaderConfig
from nero import ROOT, inside_project
from nero_common import FEATURES


def camera_spec(raw: str, fps: int, width: int, height: int):
    if "=" not in raw:
        raise ValueError("Use --camera name=/dev/videoX")
    name, device = raw.split("=", 1)
    if not name.isidentifier() or not device:
        raise ValueError("Invalid camera name or device")
    return name, OpenCVCameraConfig(index_or_path=int(device) if device.isdigit() else Path(device),
                                    fps=fps, width=width, height=height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--camera", action="append", default=[], help="name=/dev/videoX; repeat for multiple RGB cameras")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-s", type=float, default=30)
    parser.add_argument("--root", default=str(ROOT / "datasets" / "nero_passive"))
    parser.add_argument("--repo-id", default="local/nero_passive")
    parser.add_argument("--task", default="Nero leader demonstration")
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or args.episodes < 1 or args.episode_s <= 0:
        parser.error("fps must be 1..60, episodes >= 1, episode-s > 0")
    root = inside_project(args.root)
    if root.exists():
        parser.error(f"Dataset already exists: {root}; choose a new --root")
    cameras_cfg = dict(camera_spec(raw, args.fps, args.width, args.height) for raw in args.camera)
    if len(cameras_cfg) != len(args.camera):
        parser.error("Camera names must be unique")
    cameras = {name: OpenCVCamera(cfg) for name, cfg in cameras_cfg.items()}
    leader = NeroLeader(NeroLeaderConfig(channel=args.channel, firmware=args.firmware))
    dataset = None
    try:
        leader.connect()
        for camera in cameras.values():
            camera.connect()
        obs_features = {**dict.fromkeys(FEATURES, float), **{
            name: (cfg.height, cfg.width, 3) for name, cfg in cameras_cfg.items()
        }}
        features = {**hw_to_dataset_features(obs_features, "observation"),
                    **hw_to_dataset_features(dict.fromkeys(FEATURES, float), "action")}
        dataset = LeRobotDataset.create(
            args.repo_id, args.fps, features, root=root, robot_type="nero_passive",
            use_videos=True, image_writer_threads=max(1, len(cameras) * 2),
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        )
        interval = 1 / args.fps
        for episode in range(args.episodes):
            input(f"Episode {episode + 1}/{args.episodes}: set the scene, then press Enter to record...")
            start = time.monotonic()
            frame_index = 0
            try:
                while time.monotonic() - start < args.episode_s:
                    tick = time.monotonic()
                    action = leader.get_action()
                    obs = dict(action)  # leader state; no follower feedback in passive mode
                    for name, camera in cameras.items():
                        obs[name] = camera.async_read()
                    frame = {
                        **build_dataset_frame(features, obs, "observation"),
                        **build_dataset_frame(features, action, "action"),
                        "task": args.task,
                    }
                    dataset.add_frame(frame)
                    frame_index += 1
                    time.sleep(max(0, start + frame_index * interval - time.monotonic()))
            except KeyboardInterrupt:
                print("Episode interrupted; discarding its partial frames")
                dataset.clear_episode_buffer()
                break
            dataset.save_episode()
            print(f"Saved episode {episode + 1}: {frame_index} frames")
    finally:
        if dataset is not None:
            dataset.finalize()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
