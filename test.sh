#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 离线测试：加载 checkpoint 和采集数据，只计算动作，不连接机械臂。
POLICY_TYPE="${POLICY_TYPE:-act}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_demo_01}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nero_demo_01}"
TRAIN_IMAGES="${TRAIN_IMAGES-front,camera2}"
TRAIN_STATE="${TRAIN_STATE-joints,pose}"
INPUT_TAG="${TRAIN_IMAGES//,/+}_${TRAIN_STATE//,/+}"
INPUT_TAG="${INPUT_TAG// /}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${POLICY_TYPE}_nero_01_${INPUT_TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints/last/pretrained_model}"
TEST_DEVICE="${TEST_DEVICE:-cpu}"
TEST_SAMPLES="${TEST_SAMPLES:-10}"

exec "$PROJECT_ROOT/nero.sh" offline-test \
  --dataset-repo-id="$DATASET_REPO_ID" \
  --dataset-root="$DATASET_ROOT" \
  --checkpoint="$CHECKPOINT_DIR" \
  --device="$TEST_DEVICE" --samples="$TEST_SAMPLES" "$@"
