#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-262723060852}"
SECOND_CAMERA_SERIAL="${SECOND_CAMERA_SERIAL:-262723060939}"

if [[ "${1:-}" == "--help" ]]; then
  exec "$PROJECT_ROOT/nero.sh" teleoperate --help
fi

FRONT_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$FRONT_CAMERA_SERIAL")"
SECOND_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$SECOND_CAMERA_SERIAL")"
CAMERAS="{front: {type: opencv, index_or_path: $FRONT_CAMERA_DEVICE, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: $SECOND_CAMERA_DEVICE, width: 640, height: 480, fps: 30}}"

exec "$PROJECT_ROOT/nero.sh" teleoperate \
  --robot.channel=can0 --teleop.channel=can0 \
  --robot.cameras="$CAMERAS" \
  --display_data=true \
  "$@"
