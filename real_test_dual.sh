#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/dual_config.sh"
if [[ -f "$PROJECT_ROOT/$OUTPUT_DIR/pretrained_model/config.json" || -f "$OUTPUT_DIR/pretrained_model/config.json" ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/pretrained_model}"
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints/last/pretrained_model}"
fi
if [[ "${1:-}" != --help ]]; then check_dual_cameras; fi
exec "$PROJECT_ROOT/nero.sh" real-test-dual \
  --dataset-repo-id="$DATASET_REPO_ID" --dataset-root="$DATASET_ROOT" \
  --checkpoint="$CHECKPOINT_DIR" --device="${TEST_DEVICE:-cpu}" --action-space=joints \
  --duration="${TEST_SECONDS:-10}" --task="${TASK:-双臂协作抓取并放置物体}" \
  --left-channel="$LEFT_CHANNEL" --right-channel="$RIGHT_CHANNEL" \
  --front-serial="${FRONT_CAMERA_SERIAL:-}" \
  --left-wrist-serial="${LEFT_WRIST_CAMERA_SERIAL:-}" \
  --right-wrist-serial="${RIGHT_WRIST_CAMERA_SERIAL:-}" \
  --side-serial="${SIDE_CAMERA_SERIAL:-}" "$@"
