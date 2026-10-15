#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 仅录主臂，数据集里不包含从臂实际反馈。
DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_passive_01}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nero_passive_01}"
TASK="${TASK:-抓取并放置物体}"
NUM_EPISODES="${NUM_EPISODES:-10}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-262723060852}"
SECOND_CAMERA_SERIAL="${SECOND_CAMERA_SERIAL:-262723060939}"

if [[ "${1:-}" == "--help" ]]; then
  exec "$PROJECT_ROOT/nero.sh" passive-record --help
fi

FRONT_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$FRONT_CAMERA_SERIAL")"
SECOND_CAMERA_DEVICE="$("$PROJECT_ROOT/nero.sh" camera-path "$SECOND_CAMERA_SERIAL")"

exec "$PROJECT_ROOT/nero.sh" passive-record \
  --channel can0 --camera "front=$FRONT_CAMERA_DEVICE" --camera "camera2=$SECOND_CAMERA_DEVICE" \
  --fps 30 --episodes "$NUM_EPISODES" --episode-s 30 \
  --root "$DATASET_ROOT" --repo-id "$DATASET_REPO_ID" \
  --task "$TASK" "$@"
