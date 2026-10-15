"""Read-only policy inference on frames from a local Nero dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nero import inside_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained policy on recorded frames without robot motion")
    parser.add_argument("--dataset-repo-id", default="local/nero_demo_01")
    parser.add_argument("--dataset-root", default="datasets/nero_demo_01")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint/pretrained_model directory")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    root = inside_project(args.dataset_root)
    checkpoint = inside_project(args.checkpoint)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset not found: {root}")
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Trained model not found: {checkpoint}")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set TEST_DEVICE=cpu")

    dataset = LeRobotDataset(args.dataset_repo_id, root=root, video_backend="torchcodec")
    if len(dataset) == 0:
        raise ValueError(f"Dataset has no frames: {root}")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    config.pretrained_path = Path(checkpoint)
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval()

    count = min(args.samples, len(dataset))
    indices = [(i * len(dataset)) // count for i in range(count)]
    from training_inputs import SELECTION_FILE, apply_saved_selection
    selection_path = next(
        (directory / SELECTION_FILE for directory in checkpoint.parents if (directory / SELECTION_FILE).is_file()),
        None,
    )
    if selection_path is None and "observation.state" in config.input_features:
        expected = config.input_features["observation.state"].shape[0]
        recorded = dataset.meta.features["observation.state"]["shape"][0]
        if expected != recorded:
            raise FileNotFoundError(
                f"Model expects {expected} state values but dataset has {recorded}; "
                f"keep {SELECTION_FILE} beside the training run when moving the checkpoint"
            )
    errors = []
    with torch.inference_mode():
        for index in indices:
            frame = dataset[index]
            if selection_path is not None:
                frame = apply_saved_selection(frame, dataset.meta, selection_path)
            observation = {key: frame[key] for key in config.input_features if key in frame}
            if "task" in frame:
                observation["task"] = frame["task"]
            policy.reset()
            action = postprocessor(policy.select_action(preprocessor(observation)))
            predicted = torch.as_tensor(action).detach().cpu().reshape(-1)
            recorded = torch.as_tensor(frame["action"]).detach().cpu().reshape(-1)
            if predicted.shape != recorded.shape or not torch.isfinite(predicted).all():
                raise ValueError(f"Invalid predicted action at frame {index}: {tuple(predicted.shape)}")
            errors.append(torch.mean(torch.abs(predicted - recorded)).item())

    print(f"policy={config.type} checkpoint={checkpoint}")
    print(f"frames_checked={count} action_dim={recorded.numel()}")
    print(f"mean_abs_action_difference={sum(errors) / len(errors):.6f}")
    print("Offline single-step check only; this value is not a task success rate. No robot commands were sent.")


if __name__ == "__main__":
    main()
