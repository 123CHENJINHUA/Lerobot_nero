# Nero × LeRobot 0.6.1

本目录提供 Nero 7 关节 + AGX 夹爪的 LeRobot 适配器，支持数据采集、ACT / Diffusion / SmolVLA 训练和真机推理。使用源工程 `TP-KMPs-nero-isaacsim` 中的 `pyAgxArm` 1.0.0 和 `python-can` 4.6.1；相关代码与许可证保留在 `vendor/`。

## 安装

当前验证环境为 Linux x86_64、Python 3.12、LeRobot 0.6.1。先克隆仓库并进入目录，再安装依赖：

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

深度采集和视频解码还需要系统 `ffmpeg`；Ubuntu 可执行 `sudo apt install ffmpeg`。GPU 训练需要可用的 NVIDIA 驱动和匹配的 PyTorch；安装后用 `python -c "import torch; print(torch.cuda.is_available())"` 检查。SmolVLA 预训练权重首次使用时从 Hugging Face 下载，pip 镜像不会下载这些权重。

相机序列号需要在脚本顶部或通过 `FRONT_CAMERA_SERIAL`、`SECOND_CAMERA_SERIAL` 环境变量设置为自己的设备。`vendor/` 包含本项目实际使用的 SDK、第三方许可证及平台相关组件，请保留目录；其他系统或 Python 版本未验证。

## 硬件对应

| 接口 | 设备 | 读写 |
| --- | --- | --- |
| `can0` | Nero 主臂和自动跟随的从臂共用总线 | 只读主臂 7 关节/0x159 夹爪目标及从臂实际反馈 |

关节字段为 `joint1.pos` 至 `joint7.pos`（弧度），`gripper.pos` 为夹爪**开度**（米）。示教臂的夹爪值是控制目标回显；执行臂的夹爪值是实际状态。固件默认 `v121`，可用 `--robot.firmware` / `--teleop.firmware` 改为 `v120`、`v112`、`v111` 或 `default`。若 CAN 接口与表不同，用 `--robot.channel` / `--teleop.channel` 指定。

默认 `external` 模式对应你当前的硬件主从跟随：电脑只读取主臂动作、从臂反馈和相机图像，**不会发模式、使能或运动 CAN 命令**。两种状态都要求反馈完整且未过期。电脑主动控制单列为 `host` 模式，仅允许主从臂使用不同 CAN 接口并显式加 `--robot.allow_motion=true`；这时还检查 Nero SDK 关节限位、初始姿态差（0.25 rad）和每步变化（0.06 rad / 6 mm）。本项目不在共享 `can0` 上发送主从模式配置。断开连接只关闭通信，不自动断电或释放电机。

## 运行脚本

先进入本目录并激活已安装 LeRobot 的 conda 环境；所有脚本使用当前环境的 `python`：

```bash
conda activate lerobot
cd /path/to/lerobot_nero
```

| 脚本 | 作用 | 默认输出 |
| --- | --- | --- |
| `./doctor.sh` | 检查 LeRobot、Nero SDK 和 `can0`，不发 CAN 报文 | 终端 |
| `./tele.sh` | 只读查看主从臂跟随和 D435i RGB | 终端 |
| `./record.sh` | 主臂动作 + 从臂状态/力矩/末端运动 + 两路 D435i 彩色与深度采集 | `datasets/nero_demo_01/` |
| `./train.sh` | 用采集数据训练 ACT 或其他策略 | `runs/<模型名>_nero_01_<输入组合>/` |
| `./train_all.sh` | 依次训练 ACT、Diffusion、SmolVLA | `runs/three_policies_<时间>/` |
| `./test.sh` | 加载 checkpoint，在已录数据上离线预测动作 | 终端 |
| `./real_test.sh` | 从臂真机实时预测；显式加 `--execute` 才发送动作 | 终端及 Rerun |
| `./passive_record.sh` | 仅主臂的备用采集模式 | `datasets/nero_passive_01/` |

按流程依次运行 `./doctor.sh`、`./tele.sh`、`./record.sh`、`./train.sh`、`./test.sh`。`tele.sh` 和 `record.sh` 默认打开 LeRobot 的 Rerun 可视化窗口；`train.sh` 和 `test.sh` 在终端运行。每个脚本都可单独运行；配置集中在对应脚本的顶部，`nero.sh` 和 `nero.py` 只是内部启动与适配代码。

两台 D435i 的序列号分别为 `262723060852`（原相机）和 `262723060939`（新相机）。三个采集/预览脚本顶部只配置 `FRONT_CAMERA_SERIAL` 和 `SECOND_CAMERA_SERIAL`；两台相机默认都会用于预览和录制。`camera_serial.py` 在每次启动时按序列号查找当前彩色、深度视频节点，因此重新插拔导致的视频编号变化无需修改脚本。当前 conda 环境没有 `pyrealsense2`：彩色仍由 LeRobot/OpenCV 读取，`record.sh` 用系统 `ffmpeg` 读取两路原始 Z16 深度，不需要额外安装 Python 包。第二路在数据集中叫 `camera2`，两路都配置为 `640×480`、30 FPS。

新增深度输入会改变数据集特征；若旧数据集已经建立，在 `record.sh` 顶部换新的 `DATASET_REPO_ID` 和 `DATASET_ROOT`，然后把 `train.sh`、`test.sh` 的对应配置改成同一值。四路图像均为 `640×480`。若 `can0` 在重启后为 DOWN，先按 Nero 设备说明配置 CAN 接口。`record.sh` 使用已自动跟随的主从臂，电脑只读取数据；`action` 来自主臂，`observation.state` 来自从臂。

`record.sh` 按 LeRobot 的两阶段流程手动控制：录完一段按 **右方向键**，会提示 “Reset the environment” 并进入不写入数据集的重置阶段；整理好场景后再按 **右方向键**，才会保存当前 episode 并开始下一段。**左方向键**用于重录，**Esc** 用于保存当前段并结束整个会话，Ctrl+C 也会保存并退出。LeRobot 0.6.1 在终端提供键盘监听，也可直接按 `n`、`r`、`q`，分别对应右方向键、左方向键、Esc，无需回车。录制和重置阶段都由按键结束，不会按固定秒数自动跳过；开始录制、进入重置和结束时默认启用 LeRobot 的语音提示。录制时使用流式 H.264 编码。

### 新增的从臂与深度数据

`record.sh` 的 `observation.state` 现在是 27 维，顺序写在数据集的 `meta/info.json` 中：前 8 维为从臂 7 个关节角与夹爪开度，接着 7 维 `joint1.torque`～`joint7.torque`（N·m），6 维 `ee.pose.{x,y,z,rx,ry,rz}`（TCP 在基坐标系中的米与弧度），最后 6 维 `ee.velocity.{vx,vy,vz,wx,wy,wz}`（米/秒与弧度/秒，基坐标系）。力矩由 Nero SDK 根据电机电流及标定系数估计；末端速度由相邻位姿反馈及其时间戳计算，首帧为零。未设置 TCP 偏移时，TCP 就是法兰位姿。新格式须录到新数据集，不能与旧的 8 维状态数据集混录。

两路深度现在是 LeRobot 原生视觉特征 `observation.images.front_depth` 和 `observation.images.camera2_depth`，与两路 RGB 一起保存在数据集 `videos/` 中。采集时把 Z16 原始深度的固定范围映射为三通道 8 位灰度图；默认范围为 200～4000 原始单位，0 表示无效值，超出范围的值会被截断。可在 `record.sh` 顶部修改 `DEPTH_NEAR_RAW`、`DEPTH_FAR_RAW`，实际映射和相机序列号另记在 `meta/nero_depth_encoding.json`。H.264 编码有损，深度图适合作为模型视觉输入，不能用于恢复精确的 Z16 数值或可靠的无效像素掩码。两路彩色与深度按采集帧配对，未做硬件时间同步或深度到彩色的像素对齐。

```python
import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path.cwd() / ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(Path.cwd() / ".cache/huggingface/datasets")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("local/nero_demo_01", root="datasets/nero_demo_01")
frame = dataset[0]
print(frame["observation.images.front_depth"].shape)  # (3, 480, 640)
```

`train.sh` 顶部的 `TRAIN_IMAGES` 和 `TRAIN_STATE` 决定模型输入，不影响录制数据。可选项如下：

| 变量 | 可选值 | 默认值 |
| --- | --- | --- |
| `TRAIN_IMAGES` | `front,camera2,front_depth,camera2_depth` 中的相机名，用逗号分隔 | `front,camera2` |
| `TRAIN_STATE` | `joints,pose,torque,velocity` 中的状态组，用逗号分隔 | `joints,pose` |
| `ACTION_SPACE` | `joints`（8 维关节/夹爪）或 `tcp`（7 维末端位姿/夹爪） | `joints` |

`joints` 包含 7 个关节角和夹爪开度（8 维），`pose`、`torque`、`velocity` 分别为 6、7、6 维。只用两路彩色图与末端位姿时设 `TRAIN_STATE="pose"`；使用全部输入时设 `TRAIN_IMAGES="front,camera2,front_depth,camera2_depth"` 和 `TRAIN_STATE="joints,pose,torque,velocity"`。`TRAIN_STATE=""` 表示只用图像。也可以在命令前临时设置，例如 `TRAIN_STATE=pose ./train.sh`。

`TRAIN_STATE` 决定输入，`ACTION_SPACE` 决定输出，二者独立。切换输出方式不需要重新采集数据：

```bash
# 用现有数据训练末端位姿 action；切换模型可将 act 改为 diffusion 或 smolvla。
ACTION_SPACE=tcp POLICY_TYPE=act ./train.sh
ACTION_SPACE=tcp POLICY_TYPE=act TEST_DEVICE=cuda ./test.sh
ACTION_SPACE=tcp POLICY_TYPE=act TEST_DEVICE=cuda TEST_SECONDS=5 ./real_test.sh --no-display
# 只读预演确认后，单独连接从臂执行。
ACTION_SPACE=tcp POLICY_TYPE=act TEST_DEVICE=cuda TEST_SECONDS=5 ./real_test.sh --execute --no-display

# 保留原来的关节 action。
ACTION_SPACE=joints POLICY_TYPE=act ./train.sh
```

如果使用新数据集，以上命令都加 `DATASET_REPO_ID=local/nero_demo_02 DATASET_ROOT=datasets/nero_demo_02`，或修改三个脚本中的默认数据集。TCP 默认输出目录在对应关节目录后加 `_tcp`，例如 `runs/act_nero_01_front+camera2_joints+pose_tcp`；也可以用 `OUTPUT_DIR` 指定独立目录。两种 action 的模型不能混用或互相恢复训练。测试从 `input_selection.json` 自动识别动作类型；显式设置 `ACTION_SPACE` 时会校验，类型不一致就报错。

TCP 标签由原 action 中的主臂关节目标通过 SDK FK 计算，夹爪目标保持不变。延续原有处理顺序：先将关节/夹爪目标裁剪到项目限位，再做 FK；不会用同帧从臂实际位姿替代目标。转换、动作统计重算均发生在读取时，不改原始数据。此处 `tcp` 明确定义为**基坐标系下、零工具偏移的法兰位姿**，与当前录制的 `ee.pose` 定义一致，并不代表夹爪尖端。输出顺序是 `[x,y,z,roll,pitch,yaw,gripper]`，单位为米/弧度/米，使用绝对位姿。欧拉角在 ±π 附近仍有表示不连续的问题；执行限速使用几何旋转距离，避免将角度跨界当作整圈旋转。

TCP 使用控制器的 `move_p` 进行逆解与运动。默认每条命令平移不超过 `0.002 m`、旋转不超过 `0.02 rad`，夹爪不超过 `0.002 m`；同时受当前反馈和上次发送目标限制。目标位置先裁剪到训练目标覆盖范围加 `0.02 m` 的包围盒，当前位置必须在盒内，目标与当前位置误差不得超过 `0.05 m` / `0.35 rad`。这些是逐命令限制，不是实时速度保证。执行还检查新鲜控制器状态、IK 反馈、实际关节限位/跳变，以及 SDK FK 与末端反馈的一致性。控制器报告无解、奇异点、限位或碰撞等异常时停止后续发送。

包围盒不代表可达性或避障验证；SDK 没有独立的离线 IK 求解接口，因此第一次位姿指令的逆解无法预检，也不能保证每个关节的目标变化都不超过原关节模式的 `0.01 rad`。TCP 的冗余关节路径和连续跟踪须在真机小范围验证；程序停止后的保持行为仍由控制器决定。复制模型时须保留 `input_selection.json`，其中还包含 FK 参数、坐标定义和训练工作空间。

TCP 执行检查需要固件 `v112` 或更新版本提供完整 IK 反馈，默认 `v121` 已满足 SDK 接口要求。位姿距离与旋转插值使用 SciPy，已包含在 `requirements.txt` 的 `scipy-dep` 中。

训练时会一起筛选图像、状态及其归一化统计量；原始数据集不变。选择会写入输出目录的 `input_selection.json`，`test.sh` 自动读取。默认输出目录带有选择的输入名称；运行 `test.sh` 时设置相同的 `TRAIN_IMAGES`、`TRAIN_STATE`，或直接指定相同的 `OUTPUT_DIR`。之前的 `runs/act_nero_01` 模型会保留；若要测试它，使用 `OUTPUT_DIR=runs/act_nero_01 ./test.sh`。

`record.sh` 的 Rerun 同时显示四路图像和状态；`tele.sh` 仍只显示 RGB 和状态。深度由原始 Z16 缩减为 8 位并经 H.264 压缩，文件占用显著小于逐帧保存 16 位 PNG。

`train.sh` 默认 `POLICY_TYPE=act`，改为 `diffusion` 就会使用脚本中的小型 Diffusion 配置；`DATASET_REPO_ID`、`DATASET_ROOT` 必须与 `record.sh` 一致。其他 LeRobot 0.6.1 策略可填相应模型名，并在 `train.sh` 的 `policy_args` 中加入其专属参数；模型可能另需依赖或预训练权重。没有 CUDA 时把 `POLICY_DEVICE` 改成 `cpu`。所有输出留在本目录。

默认训练 50000 步、batch size 为 2，每 5000 步保存。可使用环境变量和命令行调整：

```bash
POLICY_TYPE=act TRAIN_STEPS=50000 OUTPUT_DIR=runs/act_demo ./train.sh --save_freq=5000
POLICY_TYPE=diffusion OUTPUT_DIR=runs/diffusion_demo ./train.sh
POLICY_TYPE=smolvla OUTPUT_DIR=runs/smolvla_demo ./train.sh \
  --policy.path=lerobot/smolvla_base --policy.load_vlm_weights=false
```

每次从头训练使用新的输出目录。SmolVLA 加载完整预训练策略，输入特征会依据当前选择的数据集重建；`load_vlm_weights=false` 避免另行下载已包含在策略中的骨干权重。

从云端复制模型时，除完整的 `pretrained_model/`（包含权重、配置、预处理器和归一化统计文件）外，还须复制该次训练输出目录根部的 `input_selection.json`，放在本地 `pretrained_model/` 的上一级。例如 `runs/smolvla_last/pretrained_model/` 对应 `runs/smolvla_last/input_selection.json`。它记录状态通道的名称和顺序；仅复制权重可能导致原始 27 维状态与训练选用的 14 维状态不匹配。

也可以运行 `./train_all.sh` 顺序训练三种模型，每种使用独立输出目录和日志；任一训练失败会停止队列。`runs/latest_three_policies.txt` 记录该次输出目录，目录内的 `status.log` 记录队列状态，`act.log`、`diffusion.log`、`smolvla.log` 记录各模型进度。该脚本在前台运行，`Ctrl+C` 可停止；查看日志时按 `Ctrl+C` 只会停止查看日志。

`test.sh` 默认读取 `train.sh` 对应输出目录的 `checkpoints/last/pretrained_model`，抽取 10 帧验证模型能预测 8 维关节动作或 7 维 TCP 动作。TCP 分别显示位置、旋转和夹爪误差，关节模式显示平均绝对差。这是单步离线检查，不代表任务成功率，也不会连接或驱动机械臂。若训练换了 `POLICY_TYPE`、输入组合、`ACTION_SPACE`、`OUTPUT_DIR` 或数据集，测试脚本的对应配置也要一致。

### 真机测试：只连接从臂

将主臂关机并从共享 CAN 线路断开，只让**从臂**连接电脑的 `can0`；从臂须按 Nero 的硬件流程上电、使能，测试场地留出活动空间。主臂在原来的 leader 模式是零力拖动示教，无法同时由模型主动驱动并继续作主臂自动带动从臂。只连接从臂时不需要另加 `can1`。先运行 `./doctor.sh` 确认 `can0` 为 UP，再运行：

```bash
# 默认只读：从真机相机和从臂反馈推理、显示预测动作，不发送运动指令。
./real_test.sh

# 现有完整输入 ACT checkpoint 的只读预演。
OUTPUT_DIR=runs/act_nero_01 ./real_test.sh

# 确认主臂已关闭/断开、只读预测正常后，短时间让模型控制从臂。
OUTPUT_DIR=runs/act_nero_01 TEST_SECONDS=5 ./real_test.sh --execute
```

`real_test.sh` 默认打开 Rerun；无图形界面时加 `--no-display`。如需 GPU 推理，可在命令前设置 `TEST_DEVICE=cuda`。`Ctrl+C` 停止测试。执行模式不自动切换、使能或断电；它检查从臂已使能、总线上没有新鲜主臂帧，限制每步关节和夹爪变化，并拒绝过慢或异常的模型输出。程序停止时不会自动回原点或断电；停止后的保持行为取决于控制器。训练动作来自主臂角度和夹爪目标，执行前仍须在只读预演中核对主从臂关节方向与角度映射；模型成功完成离线预测不代表能在真机完成任务。

`real_test.sh` 默认模型目录现在与 `train.sh` 一致，读取 `checkpoints/last/pretrained_model`；显式 `OUTPUT_DIR` 下若存在直接复制的 `pretrained_model/config.json` 则优先读取该目录。仍可通过 `CHECKPOINT_DIR` 指定完整 checkpoint 路径。

真机推理将夹爪模型预测截断到 `0～0.08 m` 后再做每步限幅；终端的 `gripper_raw` 显示截断前的预测开度，`target` 显示截断后的目标。

执行时，关节和夹爪目标同时受当前位置与上一次发送目标的步长限制；反馈跳变导致无法满足限幅时停止发送。终端 `target` 是模型目标，`sent` 是实际发送的限幅目标（只读模式为 `None`），`loop_max` 是上次报告以来的最大循环间隔。执行模式的 Rerun 动作曲线显示实际发送目标。

执行端将有限的关节预测逐项截到 SDK 上下限，再施加当前位置和上一条指令的步长限制。例如 joint4 预测 `2.198870 rad` 时，有效目标为上限 `2.146755 rad`，逐步接近该上限。实际发送目标始终位于 SDK 关节范围内，真机测试的关节步长为 `0.01 rad`；初始姿态差、反馈时效、NaN/Inf 等检查仍生效。原始模型目标与限幅后的实际发送目标分别显示为终端 `target` 和 `sent`。训练端保留原有动作裁剪与统计重算，可使用全部演示，但超限轨迹被裁剪后可能影响任务完成效果。

`train.sh` 读取数据时会将关节动作裁剪到 SDK 限位、夹爪动作裁剪到 `0～0.08 m`，并通过 LeRobot 重新计算裁剪后的动作归一化统计。原始 Parquet、图像、状态反馈和旧模型不修改。动作限位写入新运行的 `input_selection.json`，离线测试使用相同的动作裁剪。修复前的模型应从头重新训练到新输出目录，例如：`POLICY_TYPE=diffusion OUTPUT_DIR=runs/diffusion_nero_02_limited ./train.sh`；不要修改旧 checkpoint 的统计或用它恢复这次训练。

ACT 可加 `--act-temporal-ensemble`，使用 LeRobot 的时间集成合并对同一时刻的重叠预测，无需重新训练。它每帧重新推理；先只读检查运行速度，30 FPS 的周期预算约为 `0.033 s`，不能把超过周期预算的模型直接当作 30 Hz 控制使用。例如：`OUTPUT_DIR=runs/act_nero_01_front+camera2_joints+pose TEST_DEVICE=cuda ./real_test.sh --act-temporal-ensemble --no-display`。

Diffusion 的 CUDA 推理会先预热一次，不发送预热动作；预热后清空动作队列，控制循环重新读取观测。控制循环中的超时阈值为 `1.0 s`；同步生成新动作段仍会暂停控制，需根据实际推理耗时判断运行是否平稳。

更换推理相机时，在 `real_test.sh` 中设置新的 `FRONT_CAMERA_SERIAL` / `SECOND_CAMERA_SERIAL`，并加 `--allow-camera-replacement`。例如：`FRONT_CAMERA_SERIAL=302623061771 TEST_DEVICE=cuda ./real_test.sh --allow-camera-replacement`。保留训练数据中的原始相机序列号和深度编码元信息；新相机的安装视角和深度原始单位应与训练时一致，先只读预演，再执行动作。

`passive_record.sh` 不读取从臂反馈；其 `observation.state` 和 `action` 都来自主臂，适合单臂采集或接口检查。它生成的数据集与 `record.sh` 的双臂反馈数据集含义不同。

## 文件布局

- `lerobot_robot_nero/`、`lerobot_teleoperator_nero/`：LeRobot Robot/Teleoperator 插件实现。
- `nero_common.py`：SDK 加载、CAN 与反馈检查、从臂力矩/末端运动计算、字段/限位。
- `passive_record.py`：单示教臂只读采集。
- `record.sh`、`train.sh`、`test.sh`、`real_test.sh`、`tele.sh`、`doctor.sh`、`passive_record.sh`：独立运行入口。
- `nero.py`、`nero.sh`：内部 LeRobot 命令入口和本地路径设置。
- `offline_test.py`：只读离线策略推理与动作维度检查。
- `real_test.py`：真机实时模型推理与显式执行模式，使用与训练相同的输入筛选。
- `camera_serial.py`：通过 RealSense 序列号查找当前彩色/深度视频节点。
- `depth_record.py`：两路 Z16 深度流转换为 LeRobot 三通道视觉输入。
- `training_inputs.py`：训练时按配置选择图像和状态维度，并保存离线测试所需的选择信息。
- `action_space.py`：关节目标到 TCP 标签的转换、动作定义校验与几何位姿限速。
- `vendor/`：从源工程复制的 Nero SDK、python-can 及其依赖。
- `datasets/`、`runs/`、`.cache/`、`calibration/`：运行时自动创建，均在当前目录内。

本适配器按官方 [LeRobot 0.6.1](https://github.com/huggingface/lerobot/tree/v0.6.1) 的 API 更新。`record.sh` 指定 H.264 编码器并关闭自动添加数据集时间戳，让录制、训练和测试继续使用相同的 `DATASET_REPO_ID`。现有 v3.0 数据集与旧 ACT checkpoint 已通过离线读取和推理检查；真实双臂采集仍需连接 `can0` 与相机后验证。

## 提交到 GitHub

提交源代码、Shell 脚本、测试、`requirements.txt`、README 和完整的 `vendor/`（含第三方许可证）。`.gitignore` 已排除本地数据集、模型、训练日志、下载缓存、标定文件、Python 缓存、虚拟环境和本地凭据目录；这些文件仍保留在本机。仓库不包含可直接推理的训练权重或示教数据，需自行采集、训练或另外下载。

在已初始化的 Git 仓库中，先执行 `git status --short` 和 `git add --dry-run .` 检查待提交文件，再提交。`.gitignore` 不会自动移除已经被 Git 跟踪的文件；如旧仓库曾提交模型或数据，需要另行处理其索引或历史。
