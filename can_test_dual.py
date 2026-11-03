"""Identify left/right Nero CAN buses by observing joint movement only."""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import time

from nero_common import FIRMWARES, close_arm, complete_joints, open_arm, validate_channel
from can import CanError


def can_channels(root=Path("/sys/class/net")):
    channels = []
    for entry in sorted(root.iterdir()):
        try:
            if (entry / "type").read_text().strip() == "280":  # ARPHRD_CAN
                channels.append(entry.name)
        except OSError:
            continue
    return channels


def moving_channel(scores, threshold_deg):
    if any(value is None for value in scores.values()):
        raise ValueError("有端口缺少新鲜完整反馈，请检查接线、供电和固件，或用 --channels 指定待测端口")
    moving = [channel for channel, value in scores.items() if value >= threshold_deg]
    if len(moving) != 1:
        raise ValueError(f"需要恰好一个端口的关节变化达到 {threshold_deg:g}°，当前为 {moving}；请每次只移动一侧")
    return moving[0]


def sample_motion(arms, firmware, seconds):
    bounds = {}
    scores = dict.fromkeys(arms)
    deadline = time.monotonic() + seconds
    next_report = 0.0
    while time.monotonic() < deadline:
        for channel, arm in arms.items():
            spans = []
            for leader in (False, True):
                try:
                    joints = complete_joints(arm, leader, firmware, 0.5)
                except RuntimeError:
                    continue
                key = channel, leader
                lo, hi = bounds.setdefault(key, (joints.copy(), joints.copy()))
                for i, value in enumerate(joints):
                    lo[i], hi[i] = min(lo[i], value), max(hi[i], value)
                spans.append(math.degrees(max(high - low for low, high in zip(lo, hi, strict=True))))
            scores[channel] = max(spans) if spans else None
        if time.monotonic() >= next_report:
            print("  " + " | ".join(f"{channel}: " + ("无新鲜反馈" if value is None else f"关节变化 {value:.2f}°")
                                     for channel, value in scores.items()), flush=True)
            next_report = time.monotonic() + 1.0
        time.sleep(0.05)
    return scores


def main():
    parser = argparse.ArgumentParser(description="只读识别左右机械臂对应的 CAN 口，不发送运动或使能指令")
    parser.add_argument("--channels", nargs="+", help="待测端口，例如 can0 can1；默认扫描系统 CAN 接口")
    parser.add_argument("--firmware", choices=sorted(FIRMWARES), default="v121")
    parser.add_argument("--seconds", type=float, default=8.0, help="每一侧的观察时间，默认 8 秒")
    parser.add_argument("--threshold-deg", type=float, default=3.0, help="判定移动的最小关节角变化，默认 3 度")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.seconds, args.threshold_deg)):
        parser.error("观察时间和角度阈值必须为有限正数")
    channels = args.channels if args.channels is not None else can_channels()
    if len(channels) < 2 or len(set(channels)) != len(channels):
        parser.error("至少需要两个不同 CAN 端口，可使用 --channels can0 can1")

    print("本脚本只读反馈，不改变 CAN 配置或机械臂模式。左右位置需要通过人工移动确认。", flush=True)
    print("请按现有安全操作方式小幅改变指定侧的关节姿态，其余侧保持静止；不要强扳已使能的从臂。", flush=True)
    try:
        with ExitStack() as stack:
            arms = {}
            for channel in channels:
                validate_channel(channel)
                arm, _ = open_arm(channel, args.firmware)
                stack.callback(close_arm, arm)
                arms[channel] = arm
                print(f"已监听 {channel}", flush=True)
            mapping = {}
            for side, label in (("left", "左"), ("right", "右")):
                input(f"\n保持其他侧静止，按 Enter 开始 {args.seconds:g} 秒观察，随后只移动{label}侧主臂（或可安全拖动的从臂）：")
                scores = sample_motion(arms, args.firmware, args.seconds)
                mapping[side] = moving_channel(scores, args.threshold_deg)
                print(f"{label}侧检测到：{mapping[side]}", flush=True)
            if mapping["left"] == mapping["right"]:
                raise ValueError("两次识别到了同一个端口，无法确定左右映射，请重新测试")
            print("\n请核对并将以下设置填写到 dual_config.sh：")
            print(f'LEFT_CHANNEL="{mapping["left"]}"\nRIGHT_CHANNEL="{mapping["right"]}"')
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "\n已停止 CAN 识别，配置未修改。\n")
    except (RuntimeError, ValueError, OSError, CanError) as exc:
        parser.exit(1, f"\n识别未完成：{exc}；配置未修改。\n")


if __name__ == "__main__":
    main()
