#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认只读取真机并显示模型预测；增加 --execute 才向单独连接的从臂发送动作。
POLICY_TYPE="${POLICY_TYPE:-act}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_demo_01}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nero_demo_01}"
TRAIN_IMAGES="${TRAIN_IMAGES-front,camera2}"
TRAIN_STATE="${TRAIN_STATE-joints,pose}"
INPUT_TAG="${TRAIN_IMAGES//,/+}_${TRAIN_STATE//,/+}"
INPUT_TAG="${INPUT_TAG// /}"
action_args=()
if [[ -n "${ACTION_SPACE:-}" ]]; then action_args=(--action-space="$ACTION_SPACE"); fi
if [[ "${ACTION_SPACE:-joints}" == tcp ]]; then INPUT_TAG="${INPUT_TAG}_tcp"; fi
OUTPUT_DIR="${OUTPUT_DIR:-runs/${POLICY_TYPE}_nero_01_${INPUT_TAG}}"
if [[ -f "$PROJECT_ROOT/$OUTPUT_DIR/pretrained_model/config.json" || -f "$OUTPUT_DIR/pretrained_model/config.json" ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/pretrained_model}"
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints/last/pretrained_model}"
fi
TEST_DEVICE="${TEST_DEVICE:-cpu}"
TEST_SECONDS="${TEST_SECONDS:-10}"
ROBOT_CHANNEL="${ROBOT_CHANNEL:-can0}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-262723060852}"  #302623061771, 262723060852
SECOND_CAMERA_SERIAL="${SECOND_CAMERA_SERIAL:-262723060939}"

exec "$PROJECT_ROOT/nero.sh" real-test \
  --dataset-repo-id="$DATASET_REPO_ID" --dataset-root="$DATASET_ROOT" \
  --checkpoint="$CHECKPOINT_DIR" --device="$TEST_DEVICE" \
  --duration="$TEST_SECONDS" --channel="$ROBOT_CHANNEL" \
  --front-serial="$FRONT_CAMERA_SERIAL" --second-serial="$SECOND_CAMERA_SERIAL" \
  "${action_args[@]}" "$@"
