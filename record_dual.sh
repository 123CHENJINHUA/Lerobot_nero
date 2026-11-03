#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/dual_config.sh"
if [[ "${1:-}" == --help ]]; then
  exec "$PROJECT_ROOT/nero.sh" record --robot.type=nero_dual_follower --teleop.type=nero_dual_leader --help
fi
check_dual_cameras
FRONT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$FRONT_CAMERA_SERIAL")"
LEFT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$LEFT_WRIST_CAMERA_SERIAL")"
RIGHT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$RIGHT_WRIST_CAMERA_SERIAL")"
SIDE_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$SIDE_CAMERA_SERIAL")"
CAMERAS="{front: {type: opencv, index_or_path: $FRONT_DEVICE, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_DEVICE, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: $RIGHT_DEVICE, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: $SIDE_DEVICE, width: 640, height: 480, fps: 30}}"
export NERO_DEPTH_CAMERAS="front:${FRONT_CAMERA_SERIAL},left_wrist:${LEFT_WRIST_CAMERA_SERIAL},right_wrist:${RIGHT_WRIST_CAMERA_SERIAL},side:${SIDE_CAMERA_SERIAL}"
export NERO_DEPTH_NEAR_RAW="${DEPTH_NEAR_RAW:-200}"
export NERO_DEPTH_FAR_RAW="${DEPTH_FAR_RAW:-4000}"
export NERO_MANUAL_RECORD="${NERO_MANUAL_RECORD:-1}"
exec "$PROJECT_ROOT/nero.sh" record \
  --robot.type=nero_dual_follower --teleop.type=nero_dual_leader \
  --robot.left_channel="$LEFT_CHANNEL" --robot.right_channel="$RIGHT_CHANNEL" \
  --teleop.left_channel="$LEFT_CHANNEL" --teleop.right_channel="$RIGHT_CHANNEL" \
  --robot.cameras="$CAMERAS" --display_data=true --play_sounds=true \
  --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="${TASK:-双臂协作抓取并放置物体}" \
  --dataset.fps=30 --dataset.streaming_encoding=true \
  --dataset.rgb_encoder.vcodec=h264 --dataset.no_stamp=true --dataset.push_to_hub=false "$@"
