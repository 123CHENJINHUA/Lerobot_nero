"""Run a Nero policy on live follower feedback, with explicit guarded motion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nero import inside_project
from nero_common import FEATURES, validate_channel
from training_inputs import SELECTION_FILE, apply_saved_selection


def leader_frames_present(arm, firmware: str, max_age_s: float = 0.5) -> bool:
    """Detect a leader still sending commands on the follower's CAN bus."""
    names = (tuple(f"leader_joint_{i}" for i in range(1, 8)) if firmware in {"default", "v111"}
             else ("leader_joint_12", "leader_joint_34", "leader_joint_56", "leader_joint_7"))
    now = time.time()
    return any(
        0 <= now - float(part.timestamp) <= max_age_s
        for name in names if (part := getattr(arm._parser, name, None)) is not None
    )


def find_selection(checkpoint: Path) -> Path | None:
    return next(
        (parent / SELECTION_FILE for parent in checkpoint.parents if (parent / SELECTION_FILE).is_file()),
        None,
    )


def required_cameras(input_features: dict) -> tuple[set[str], set[str]]:
    image_keys = {key.removeprefix("observation.images.") for key in input_features
                  if key.startswith("observation.images.")}
    rgb = {name for name in image_keys if not name.endswith("_depth")}
    depth = {name.removesuffix("_depth") for name in image_keys if name.endswith("_depth")}
    unknown = (rgb | depth) - {"front", "camera2"}
    if unknown:
        raise ValueError(f"Unknown model camera(s): {sorted(unknown)}")
    return rgb, depth


def build_live_input(meta, input_features: dict, raw: dict, selection_path: Path | None, task: str):
    """Match the tensors returned by LeRobotDataset.__getitem__ for this policy."""
    import torch
    from lerobot.utils.feature_utils import build_dataset_frame

    wanted = {key: feature for key, feature in meta.features.items()
              if key in input_features or key == "observation.state"}
    frame = build_dataset_frame(wanted, raw, "observation")
    if selection_path is not None:
        frame = apply_saved_selection(frame, meta, selection_path)

    result = {}
    for key, feature in input_features.items():
        if key not in frame:
            raise KeyError(f"Live observation is missing {key}")
        value = np.asarray(frame[key])
        if key.startswith("observation.images."):
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"{key} must be an HWC uint8 RGB image, got {value.shape} {value.dtype}")
            tensor = torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float() / 255.0
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        if tuple(tensor.shape) != tuple(feature.shape):
            raise ValueError(f"{key} shape {tuple(tensor.shape)} != checkpoint {tuple(feature.shape)}")
        result[key] = tensor
    result["task"] = task
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="datasets/nero_demo_01")
    parser.add_argument("--dataset-repo-id", default="local/nero_demo_01")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v121")
    parser.add_argument("--front-serial", default="262723060852")
    parser.add_argument("--second-serial", default="262723060939")
    parser.add_argument("--allow-camera-replacement", action="store_true",
                        help="Allow replacement camera serials with the training depth encoding")
    parser.add_argument("--act-temporal-ensemble", action="store_true",
                        help="Blend overlapping ACT predictions; runs inference every frame")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds of live inference")
    parser.add_argument("--task", default="抓取并放置物体")
    parser.add_argument("--execute", action="store_true", help="Send policy targets to the follower")
    parser.add_argument("--no-display", action="store_true", help="Do not open the Rerun window")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    checkpoint = inside_project(args.checkpoint)
    dataset_root = inside_project(args.dataset_root)
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Trained model not found: {checkpoint}")
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_root}")
    validate_channel(args.channel)

    import torch
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.visualization_utils import (
        init_visualization, log_visualization_data, shutdown_visualization,
    )
    from camera_serial import rgb_device_for_serial
    from lerobot_robot_nero import NeroFollower, NeroFollowerConfig

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use TEST_DEVICE=cpu for prediction")

    dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, video_backend="torchcodec")
    if tuple(dataset.meta.features["action"]["names"]) != FEATURES:
        raise ValueError("Checkpoint dataset action order does not match Nero follower joints")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    config.pretrained_path = checkpoint
    if args.act_temporal_ensemble:
        if config.type != "act":
            parser.error("--act-temporal-ensemble requires an ACT checkpoint")
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = 0.01
    if not config.input_features:
        raise ValueError("Checkpoint does not declare input features")
    rgb_names, depth_names = required_cameras(config.input_features)
    serials = {"front": args.front_serial, "camera2": args.second_serial}
    cameras = {}
    for name in sorted(rgb_names):
        height, width, channels = dataset.meta.features[f"observation.images.{name}"]["shape"]
        if channels != 3:
            raise ValueError(f"RGB camera {name} must have three channels")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(rgb_device_for_serial(serials[name])),
            fps=dataset.fps, width=width, height=height,
        )

    if depth_names:
        from depth_record import install
        depth_meta_path = dataset_root / "meta" / "nero_depth_encoding.json"
        if not depth_meta_path.is_file():
            raise FileNotFoundError(f"Depth conversion metadata missing: {depth_meta_path}")
        depth_meta = json.loads(depth_meta_path.read_text())
        for name in depth_names:
            training_serial = depth_meta["cameras"][f"{name}_depth"]
            if training_serial != serials[name]:
                if not args.allow_camera_replacement:
                    raise ValueError(
                        f"Depth camera serial does not match training data: {name}; "
                        "use --allow-camera-replacement for an intentional camera replacement"
                    )
                print(f"Replacement depth camera {name}: "
                      f"{training_serial} -> {serials[name]}; "
                      "using training depth encoding; verify camera placement and depth units", flush=True)
        install(None, {name: serials[name] for name in sorted(depth_names)},
                int(depth_meta["width"]), int(depth_meta["height"]), dataset.fps,
                int(depth_meta["near_raw"]), int(depth_meta["far_raw"]))

    robot = NeroFollower(NeroFollowerConfig(
        channel=args.channel, firmware=args.firmware, cameras=cameras,
        control_mode="host" if args.execute else "external",
        allow_motion=args.execute, manage_mode=False,
        max_initial_error_rad=0.5, max_step_rad=0.01, max_gripper_step_m=0.002,
    ))
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval()
    policy.reset()
    print(f"policy={config.type} checkpoint={checkpoint} fps={dataset.fps} "
          f"n_action_steps={getattr(config, 'n_action_steps', None)} "
          f"temporal_ensemble_coeff={getattr(config, 'temporal_ensemble_coeff', None)}", flush=True)
    selection_path = find_selection(checkpoint)
    display_started = False
    try:
        robot.connect()
        if args.execute:
            time.sleep(0.6)
            if leader_frames_present(robot.arm, args.firmware):
                raise RuntimeError("Leader CAN frames detected; turn off/disconnect the leader before execution")
            if not robot.arm.get_joint_enable_status(255):
                raise RuntimeError("Follower joints are not enabled; enable them with the Nero hardware procedure")
        if config.type == "diffusion" and args.device == "cuda":
            warmup_start = time.monotonic()
            raw = robot.get_observation()
            observation = build_live_input(dataset.meta, config.input_features, raw, selection_path, args.task)
            with torch.inference_mode():
                postprocessor(policy.select_action(preprocessor(observation)))
            torch.cuda.synchronize()
            policy.reset()
            print(f"Diffusion CUDA warmup={time.monotonic() - warmup_start:.3f}s; "
                  "warmup actions discarded", flush=True)
        if not args.no_display:
            init_visualization("rerun", session_name="nero_real_test")
            display_started = True
        print("实时预测" + ("并向从臂发送动作" if args.execute else "（只读，不发送运动指令）"))
        start = time.monotonic()
        next_report = start
        previous_tick = start
        longest_period = 0.0
        frame_index = 0
        with torch.inference_mode():
            while time.monotonic() - start < args.duration:
                tick = time.monotonic()
                longest_period = max(longest_period, tick - previous_tick)
                previous_tick = tick
                raw = robot.get_observation()
                observation = build_live_input(dataset.meta, config.input_features, raw,
                                               selection_path, args.task)
                # shortcut: chunk inference blocks control; use async inference if it causes pauses.
                predicted = postprocessor(policy.select_action(preprocessor(observation)))
                action = torch.as_tensor(predicted).detach().cpu().reshape(-1)
                if action.numel() != len(FEATURES) or not torch.isfinite(action).all():
                    raise ValueError("Policy produced a non-finite or wrong-size Nero action")
                command = dict(zip(FEATURES, action.tolist(), strict=True))
                command["gripper.pos"] = min(max(command["gripper.pos"], 0.0), robot.config.gripper_max_m)
                elapsed = time.monotonic() - tick
                sent = None
                if args.execute:
                    if leader_frames_present(robot.arm, args.firmware):
                        raise RuntimeError("Leader CAN frames appeared during execution; motion stopped")
                    if not robot.arm.get_joint_enable_status(255):
                        raise RuntimeError("Follower joints are no longer enabled; motion stopped")
                    if elapsed > 1.0:
                        raise RuntimeError(f"Inference/observation too slow ({elapsed:.2f}s); no stale action sent")
                    sent = robot.send_action(command)
                if display_started:
                    log_visualization_data("rerun", observation=raw, action=sent if sent is not None else command)
                if time.monotonic() >= next_report:
                    worst_joint = max(FEATURES[:7], key=lambda key: abs(command[key] - float(raw[key])))
                    error = abs(command[worst_joint] - float(raw[worst_joint]))
                    print(f"frame={frame_index} inference_cycle={elapsed:.3f}s max_joint_error={error:.3f}rad "
                          f"loop_max={longest_period:.3f}s "
                          f"worst_joint={worst_joint} current={float(raw[worst_joint]):.4f}rad "
                          f"mode={'execute' if args.execute else 'preview'} "
                          f"gripper_raw={action[-1].item():.6f}m "
                          f"target={[round(command[key], 3) for key in FEATURES]} "
                          f"sent={[round(sent[key], 4) for key in FEATURES] if sent is not None else None}", flush=True)
                    next_report = time.monotonic() + 1.0
                    longest_period = 0.0
                frame_index += 1
                time.sleep(max(0.0, tick + 1 / dataset.fps - time.monotonic()))
    except KeyboardInterrupt:
        print("用户停止真机测试")
    finally:
        try:
            if display_started:
                shutdown_visualization("rerun")
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
