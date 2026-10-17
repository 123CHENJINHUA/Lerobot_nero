"""Project-local launcher for LeRobot 0.6.1 and Nero adapters."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / ".cache" / "lerobot"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache" / "xdg"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("WANDB_DIR", str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "vendor"))


def inside_project(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside {ROOT}: {value}")
    return path


def option(args: list[str], name: str, default: str | None = None):
    prefix = f"--{name}="
    return next((arg[len(prefix):] for arg in reversed(args) if arg.startswith(prefix)), default)


def add_default(args: list[str], name: str, value: str):
    if option(args, name) is None:
        args.append(f"--{name}={value}")


def install_manual_record_controls():
    """Keep both LeRobot phases open until the operator advances them."""
    from lerobot.scripts import lerobot_record

    original_listener = lerobot_record.init_keyboard_listener
    original_loop = lerobot_record.record_loop

    def manual_loop(*args, **kwargs):
        kwargs["control_time_s"] = float("inf")
        return original_loop(*args, **kwargs)

    def manual_listener():
        listener, events = original_listener()

        def stop():
            events["stop_recording"] = True
            events["exit_early"] = True

        def interrupt(_signal_number, _frame):
            print("\n保存当前 episode 并停止录制...")
            stop()

        signal.signal(signal.SIGINT, interrupt)

        print("手动录制：右方向键或 n 结束当前阶段；重置环境后再按一次保存并开始下一段；Esc、q 或 Ctrl+C 保存并退出")
        return listener, events

    lerobot_record.record_loop = manual_loop
    lerobot_record.init_keyboard_listener = manual_listener


def run_lerobot(command: str, args: list[str]):
    import lerobot_robot_nero  # noqa: F401 - registers draccus subtype
    import lerobot_teleoperator_nero  # noqa: F401

    if command in {"record", "teleoperate"}:
        add_default(args, "robot.type", "nero_follower")
        add_default(args, "teleop.type", "nero_leader")
        shared_bus = option(args, "robot.channel", "can0") == option(args, "teleop.channel", "can0")
        control_mode = option(args, "robot.control_mode", "external")
        if shared_bus and option(args, "robot.firmware", "v121") != option(args, "teleop.firmware", "v121"):
            raise ValueError("Shared CAN bus requires the same firmware profile for leader and follower")
        if shared_bus and control_mode == "host":
            raise ValueError("Host motion on a shared CAN bus could address both arms; use external mode")
        if shared_bus and option(args, "robot.manage_mode", "false").lower() == "true":
            raise ValueError("Do not broadcast mode/enable commands on a shared CAN bus")
        if control_mode == "host" and option(args, "robot.allow_motion", "false").lower() != "true" and "--help" not in args:
            raise ValueError("Real motion requires --robot.allow_motion=true")
        if "--help" not in args:
            from nero_common import validate_channel
            validate_channel(option(args, "teleop.channel", "can0"))
            validate_channel(option(args, "robot.channel", "can0"))
            camera_arg = option(args, "robot.cameras")
            if camera_arg:
                import yaml
                cameras = yaml.safe_load(camera_arg)
                if not isinstance(cameras, dict):
                    raise ValueError("--robot.cameras must be a camera configuration dictionary")
                for name, camera in cameras.items():
                    device = camera.get("index_or_path") if isinstance(camera, dict) else None
                    if isinstance(device, str) and device.startswith("/dev/") and not Path(device).exists():
                        raise FileNotFoundError(f"Camera {name} does not exist: {device}")
    if command == "record":
        manual_record = os.environ.get("NERO_MANUAL_RECORD") == "1"
        add_default(args, "dataset.repo_id", "local/nero_demo")
        add_default(args, "dataset.root", str(ROOT / "datasets" / "nero_demo"))
        add_default(args, "dataset.single_task", "Nero demonstration")
        add_default(args, "dataset.fps", "30")
        add_default(args, "dataset.num_episodes", str(sys.maxsize) if manual_record else "10")
        add_default(args, "dataset.episode_time_s", "1e12" if manual_record else "30")
        add_default(args, "dataset.reset_time_s", "1e12" if manual_record else "15")
        add_default(args, "dataset.push_to_hub", "false")
        add_default(args, "dataset.no_stamp", "true")
        add_default(args, "play_sounds", "true" if manual_record else "false")
        dataset_root = inside_project(option(args, "dataset.root"))
        if "--help" not in args and option(args, "resume", "false").lower() != "true" and dataset_root.exists():
            raise FileExistsError(f"Dataset already exists: {dataset_root}; use --resume=true or a new root")
        depth_cameras = os.environ.get("NERO_DEPTH_CAMERAS")
        if depth_cameras and "--help" not in args:
            cameras = dict(item.split(":", 1) for item in depth_cameras.split(","))
            if not cameras or any(not name.isidentifier() or not serial.isdigit()
                                  for name, serial in cameras.items()):
                raise ValueError("NERO_DEPTH_CAMERAS must look like front:123,camera2:456")
            from depth_record import install
            install(dataset_root, cameras, 640, 480, int(option(args, "dataset.fps")),
                    int(os.environ.get("NERO_DEPTH_NEAR_RAW", "200")),
                    int(os.environ.get("NERO_DEPTH_FAR_RAW", "4000")))
        if manual_record and "--help" not in args:
            install_manual_record_controls()
        from lerobot.scripts.lerobot_record import main
    elif command == "teleoperate":
        from lerobot.scripts.lerobot_teleoperate import main
    elif command == "train":
        add_default(args, "dataset.repo_id", "local/nero_demo")
        add_default(args, "dataset.root", str(ROOT / "datasets" / "nero_demo"))
        if not any(arg == "--policy.path" or arg.startswith("--policy.path=") for arg in args):
            add_default(args, "policy.type", "act")
        policy_type = option(args, "policy.type", "pretrained")
        if policy_type == "act":
            add_default(args, "policy.pretrained_backbone_weights", "null")
        add_default(args, "policy.push_to_hub", "false")
        add_default(args, "wandb.enable", "false")
        add_default(args, "output_dir", str(ROOT / "runs" / f"{policy_type}_nero"))
        inside_project(option(args, "dataset.root"))
        inside_project(option(args, "output_dir"))
        info = inside_project(option(args, "dataset.root")) / "meta" / "info.json"
        if "--help" not in args:
            if not info.is_file():
                raise FileNotFoundError(f"Record a dataset first: {info}")
            features = json.loads(info.read_text())["features"]
            if policy_type in {"act", "diffusion"} and not any(
                key.startswith("observation.images.") or key == "observation.environment_state"
                for key in features
            ):
                raise ValueError(f"{policy_type} needs an RGB camera or environment state in the dataset")
            selected_images = os.environ.get("NERO_TRAIN_IMAGES")
            selected_state = os.environ.get("NERO_TRAIN_STATE")
            from training_inputs import STATE_GROUPS, install_training_input_selection

            if selected_images is not None or selected_state is not None:
                images = selected_images if selected_images is not None else "front,camera2"
                state = selected_state if selected_state is not None else "joints,pose"
            else:
                images = ",".join(key.removeprefix("observation.images.") for key in features
                                  if key.startswith("observation.images."))
                state = ",".join(
                    group for group, names in STATE_GROUPS.items()
                    if set(names).issubset(features["observation.state"]["names"])
                )
            install_training_input_selection(images, state, os.environ.get("ACTION_SPACE", "joints"),
                                             os.environ.get("TCP_ROTATION", "6d"))
        from lerobot.scripts.lerobot_train import main
    else:
        raise ValueError(f"Unknown command: {command}")
    sys.argv = [sys.argv[0], *args]
    return main()


def doctor():
    import importlib.metadata
    import can
    import pyAgxArm

    print(f"project={ROOT}")
    print(f"LeRobot={importlib.metadata.version('lerobot')} pyAgxArm={pyAgxArm.__version__} python-can={can.__version__}")
    for channel in ("can0",):
        try:
            from nero_common import validate_channel
            validate_channel(channel)
            print(f"{channel}=UP")
        except RuntimeError as exc:
            print(f"{channel}={exc}")
    print("doctor only checks imports and CAN interface flags; no commands sent")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: ./nero.sh {doctor|camera-path|passive-record|record|teleoperate|train|offline-test|real-test} [options]")
        return
    command = sys.argv[1]
    if command == "doctor":
        doctor()
    elif command == "camera-path":
        from camera_serial import main as camera_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        camera_main()
    elif command == "passive-record":
        from passive_record import main as passive_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        passive_main()
    elif command == "offline-test":
        from offline_test import main as offline_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        offline_main()
    elif command == "real-test":
        from real_test import main as real_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        real_main()
    else:
        run_lerobot(command, sys.argv[2:])


if __name__ == "__main__":
    main()
