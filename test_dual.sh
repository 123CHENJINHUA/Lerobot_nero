#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/dual_config.sh"
if [[ -f "$PROJECT_ROOT/$OUTPUT_DIR/pretrained_model/config.json" || -f "$OUTPUT_DIR/pretrained_model/config.json" ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/pretrained_model}"
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints/last/pretrained_model}"
fi
exec "$PROJECT_ROOT/nero.sh" offline-test-dual \
  --dataset-repo-id="$DATASET_REPO_ID" --dataset-root="$DATASET_ROOT" \
  --checkpoint="$CHECKPOINT_DIR" --device="${TEST_DEVICE:-cpu}" --action-space=joints \
  --samples="${TEST_SAMPLES:-10}" "$@"
