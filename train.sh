#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 改 POLICY_TYPE 为 act、diffusion 或本机 LeRobot 支持的其他策略。
POLICY_TYPE="${POLICY_TYPE:-act}"
export ACTION_SPACE="${ACTION_SPACE:-joints}"
case "$ACTION_SPACE" in joints|tcp) ;; *) echo 'ACTION_SPACE 必须是 joints 或 tcp' >&2; exit 2 ;; esac
DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_demo_01}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nero_demo_01}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

# 训练输入：相机可选 front,camera2,front_depth,camera2_depth；状态可选 joints,pose,torque,velocity。
# 默认使用两路彩色图 + 关节/夹爪 + 末端位姿；想用全部数据可把两项都写全。
TRAIN_IMAGES="${TRAIN_IMAGES-front,camera2}"
TRAIN_STATE="${TRAIN_STATE-joints,pose}"
INPUT_TAG="${TRAIN_IMAGES//,/+}_${TRAIN_STATE//,/+}"
INPUT_TAG="${INPUT_TAG// /}"
if [[ "$ACTION_SPACE" == tcp ]]; then INPUT_TAG="${INPUT_TAG}_tcp"; fi
OUTPUT_DIR="${OUTPUT_DIR:-runs/${POLICY_TYPE}_nero_01_${INPUT_TAG}}"
export NERO_TRAIN_IMAGES="$TRAIN_IMAGES"
export NERO_TRAIN_STATE="$TRAIN_STATE"

if [[ "${1:-}" == "--help" ]]; then
  exec "$PROJECT_ROOT/nero.sh" train --policy.type="$POLICY_TYPE" --help
fi

policy_args=()
policy_type_args=(--policy.type="$POLICY_TYPE")
for arg in "$@"; do
  if [[ "$arg" == --policy.path || "$arg" == --policy.path=* ]]; then
    policy_type_args=()
    break
  fi
done
case "$POLICY_TYPE" in
  act) ;;
  diffusion)
    policy_args=(--policy.resize_shape='[128,128]' --policy.down_dims='[64,128,256]')
    ;;
esac

exec "$PROJECT_ROOT/nero.sh" train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  "${policy_type_args[@]}" \
  --policy.device="$POLICY_DEVICE" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$TRAIN_STEPS" --batch_size="$BATCH_SIZE" --save_freq=5000 \
  "${policy_args[@]}" "$@"
