#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/dual_config.sh"
if [[ "${1:-}" == --help ]]; then
  exec "$PROJECT_ROOT/nero.sh" teleoperate --robot.type=nero_dual_follower --teleop.type=nero_dual_leader --help
fi
check_dual_cameras
FRONT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$FRONT_CAMERA_SERIAL")"
LEFT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$LEFT_WRIST_CAMERA_SERIAL")"
RIGHT_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$RIGHT_WRIST_CAMERA_SERIAL")"
SIDE_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$SIDE_CAMERA_SERIAL")"
CAMERAS="{front: {type: opencv, index_or_path: $FRONT_DEVICE, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_DEVICE, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: $RIGHT_DEVICE, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: $SIDE_DEVICE, width: 640, height: 480, fps: 30}}"
exec "$PROJECT_ROOT/nero.sh" teleoperate \
  --robot.type=nero_dual_follower --teleop.type=nero_dual_leader \
  --robot.left_channel="$LEFT_CHANNEL" --robot.right_channel="$RIGHT_CHANNEL" \
  --teleop.left_channel="$LEFT_CHANNEL" --teleop.right_channel="$RIGHT_CHANNEL" \
  --robot.cameras="$CAMERAS" --display_data=true "$@"
