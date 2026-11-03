#!/usr/bin/env bash
# 双臂硬件配置：只需在这里填写，录制、遥操作和实机测试统一读取。
FRONT_CAMERA_SERIAL="262723060939"        # 中间摄像头序列号
LEFT_WRIST_CAMERA_SERIAL="302623061771"   # 左臂摄像头序列号
RIGHT_WRIST_CAMERA_SERIAL="262723060852"  # 右臂摄像头序列号
SIDE_CAMERA_SERIAL="303523061448"        # 侧面摄像头序列号
LEFT_CHANNEL="can1"           # 左侧主从臂所在的 CAN 总线
RIGHT_CHANNEL="can0"          # 右侧主从臂所在的 CAN 总线

# 以下数据集和训练默认值仍可通过环境变量覆盖。
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/nero_dual_02}"
export DATASET_ROOT="${DATASET_ROOT:-datasets/nero_dual_02}"
export TRAIN_IMAGES="${TRAIN_IMAGES-front,left_wrist,right_wrist,side}"
export TRAIN_STATE="${TRAIN_STATE-joints,pose}"
export ACTION_SPACE="${ACTION_SPACE:-joints}"
if [[ "$ACTION_SPACE" != joints ]]; then
  echo '双臂当前只支持 ACTION_SPACE=joints' >&2
  exit 2
fi
INPUT_TAG="${TRAIN_IMAGES//,/+}_${TRAIN_STATE//,/+}"
INPUT_TAG="${INPUT_TAG// /}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/${POLICY_TYPE:-act}_nero_dual_02_${INPUT_TAG}}"

check_dual_cameras() {
  : "${FRONT_CAMERA_SERIAL:?请设置中间摄像头 FRONT_CAMERA_SERIAL}"
  : "${LEFT_WRIST_CAMERA_SERIAL:?请设置左腕摄像头 LEFT_WRIST_CAMERA_SERIAL}"
  : "${RIGHT_WRIST_CAMERA_SERIAL:?请设置右腕摄像头 RIGHT_WRIST_CAMERA_SERIAL}"
  : "${SIDE_CAMERA_SERIAL:?请在 dual_config.sh 设置侧面摄像头 SIDE_CAMERA_SERIAL}"
  if [[ "$FRONT_CAMERA_SERIAL" == "$LEFT_WRIST_CAMERA_SERIAL" ||
        "$FRONT_CAMERA_SERIAL" == "$RIGHT_WRIST_CAMERA_SERIAL" ||
        "$LEFT_WRIST_CAMERA_SERIAL" == "$RIGHT_WRIST_CAMERA_SERIAL" ||
        "$SIDE_CAMERA_SERIAL" == "$FRONT_CAMERA_SERIAL" ||
        "$SIDE_CAMERA_SERIAL" == "$LEFT_WRIST_CAMERA_SERIAL" ||
        "$SIDE_CAMERA_SERIAL" == "$RIGHT_WRIST_CAMERA_SERIAL" ]]; then
    echo '四个摄像头必须使用不同序列号' >&2
    exit 2
  fi
}
