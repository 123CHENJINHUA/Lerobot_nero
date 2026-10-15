#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 主从臂在 can0 上自动跟随；右方向键结束录制，重置环境后再按右方向键保存并录下一段。
DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_demo_01}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nero_demo_01}"
TASK="${TASK:-抓取并放置物体}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-262723060852}"  # 原 D435i
SECOND_CAMERA_SERIAL="${SECOND_CAMERA_SERIAL:-262723060939}"  # 新 D435i
DEPTH_NEAR_RAW="${DEPTH_NEAR_RAW:-200}"
DEPTH_FAR_RAW="${DEPTH_FAR_RAW:-4000}"
export NERO_DEPTH_CAMERAS="front:${FRONT_CAMERA_SERIAL},camera2:${SECOND_CAMERA_SERIAL}"
export NERO_DEPTH_NEAR_RAW="$DEPTH_NEAR_RAW"
export NERO_DEPTH_FAR_RAW="$DEPTH_FAR_RAW"
export NERO_MANUAL_RECORD="${NERO_MANUAL_RECORD:-1}"

if [[ "${1:-}" == "--help" ]]; then
  exec "$PROJECT_ROOT/nero.sh" record --help
fi

FRONT_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$FRONT_CAMERA_SERIAL")"
SECOND_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$SECOND_CAMERA_SERIAL")"
CAMERAS="{front: {type: opencv, index_or_path: $FRONT_CAMERA_DEVICE, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: $SECOND_CAMERA_DEVICE, width: 640, height: 480, fps: 30}}"

exec "$PROJECT_ROOT/nero.sh" record \
  --robot.channel=can0 --teleop.channel=can0 \
  --robot.cameras="$CAMERAS" \
  --display_data=true --play_sounds=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.fps=30 --dataset.streaming_encoding=true \
  --dataset.rgb_encoder.vcodec=h264 --dataset.no_stamp=true --dataset.push_to_hub=false \
  "$@"
