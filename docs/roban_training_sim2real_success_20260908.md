# Roban S17：HEFT 三阶段训练到真机部署成功记录

记录日期：2026-09-08。工作区：`/data/liyinkun/project_HEFT`。

本文记录本次实际走通的流程、配置和排障结论，作为后续训练与部署的复现基线。
真机结果来自操作者在现场完成测试后反馈的“sim2real 完全成功”；本工作区保存了
训练、模型导出和仿真验证证据，但没有新增现场全量评测数据，不能把这次成功扩展为
所有动作、所有速度或长时间运行均已验证。

## 1. 成功方案的准确边界

最终方案是：

```text
本项目最新 Roban XML + 50K 动作数据
    ↓ Teacher 50000 rollout
    ↓ Adapt 2543 rollout + 加载权重后额外 5000 rollout
    ↓ Finetune 5000 rollout
    ↓ 导出完整 Student ONNX
官方 AMP → 官方切换控制流程 → 本项目 Student
    ↓ 官方关节映射、PD、动作滤波和状态历史初始化
Roban S17 真机
```

必须同时记住以下事实：

1. AMP 始终使用官方模型，HEFT 跟踪网络替换为我们的 Student。
2. 实际成功的 Adapt 总预算为 **7543 rollout**，不是当前一键脚本默认的 5000。
3. 真机成功时**保留了官方滤波和历史初始化**，没有改成训练环境逐物理子步滤波、历史清零的行为。
4. 训练 XML 将胸部设为浮动根；旧部署 XML 将骨盆设为浮动根。胸部 IMU 位置语义一致。
5. 真机配置保留腰关节方向 `-1`；不能因为训练使用新 XML 就直接删除这个映射。
6. 新上传包没有编译好的程序；现场复用了原有 `leju_heft` 目录中的 ARM64 二进制。
7. 最终现场只保留了站立和 `walk_forward_stop` 两条参考动作。这是上传后在机器人上做的修改，原上传包仍包含更多动作。

## 2. 代码和环境基线

训练分支：`roban/leju-heft-align-0902`。本次功能提交依次为：

| Commit | 内容 |
| --- | --- |
| `86b7719` | 大数据集生成/校验、CPU 参考数据、Roban 仿真容量 |
| `ac5c746` | 非有限值隔离、归一化保护、DDP 同步 |
| `5b8c87f` | 三阶段脚本、本地 checkpoint、TensorBoard、关闭 W&B |
| `a044e0c` | Teacher 评估、checkpoint 校验和诊断测试 |

- 训练使用仓库 `.venv/bin/python` 和 `.venv/bin/torchrun`。
- GPU：物理 `3,4,7`；DDP 3 个进程，每进程 4096 个环境，共 12288 个环境。
- `MEMPATH=/data/liyinkun/project_HEFT/dataset`。
- 后续训练统一 `wandb.mode=disabled`，TensorBoard 开启。
- 历史 Teacher 运行使用 W&B **offline** 保存本地产物，这是旧运行的事实，不是后续训练要求。
- 修改后，本地配置、周期 checkpoint 和最终 checkpoint 的保存不再依赖 W&B。
- `setuptools<81` 保留以满足当前 TensorBoard 的兼容需求；无需为此更换正在工作的环境。

### 2.1 数据与机器人资产

数据配置：

```yaml
task:
  num_envs: 4096
  profile:
    command:
      dataset:
        keep_on_cpu: true
        groups:
          - name: formal_50k
            teacher_mem_path: roban/main/train_50000_v1
            weight: 1.0
```

“50K”是所选源文件规模，不代表处理后的有效动作段恰好等于 50000。
数据选择、过滤和生成记录应以该数据目录内 manifest、conversion report 为准。
完整参考张量放在 CPU，仅当前环境所需的有界窗口缓存进入 GPU。

训练资产：

```text
active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml
```

Roban 配置入口：`cfg/task/profile/Roban_tracking.yaml`。
策略关节 21 个；手球、固定头部来自该具体资产。关键仿真配置为：

```yaml
step_dt: 0.02
mujoco_physics_dt: 0.005
nconmax: 200
njmax: 768
```

`njmax=768` 是 Roban 大动作集的约束容量配置。它不意味着可以删除非有限值检查。
本次训练保留了恢复后的再次检查、异常观测隔离及 DDP 同步保护。

## 3. 本次真实训练链路

下面的 `total_frames` 是配置预算；训练按完整 rollout 取整。

| 阶段 | 新执行 rollout | 每轮步数 | 配置 total_frames | 保存间隔 | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher | 50000 | 32 | 19660800000 | 1000 | offline，历史运行 |
| Adapt 首轮 | 2543 | 16 | 500000000 | 150 | disabled |
| Adapt 追加 | 5000 | 16 | 983040000 | 250 | disabled |
| Finetune | 5000 | 32 | 1966080000 | 250 | disabled |

计算方式：

```text
Teacher / Finetune 每轮 = 3 × 4096 × 32 = 393216 transition
Adapt 每轮             = 3 × 4096 × 16 = 196608 transition
首轮 Adapt 实际采集     = 2543 × 196608 = 499974144 transition
```

### 3.1 实际 checkpoint 路径

以下路径均相对于 `/data/liyinkun/project_HEFT`：

```text
Teacher:
dataset/_audit/roban_ominisoma/roban_teacher_50000_r50000_v3/wandb/wandb/offline-run-20260905_141941-roban_teacher_50000_r50000_v3/files/checkpoint_final.pt

Adapt 首轮:
dataset/_audit/roban_ominisoma/roban_adapt_50000_20260907_203222/checkpoint_final.pt

Adapt 追加后:
dataset/_audit/roban_ominisoma/roban_adapt_50000_continue5000_v1/checkpoint_final.pt

最终 Finetune:
dataset/_audit/roban_ominisoma/roban_pipeline_20260908_091456/finetune/checkpoint_final.pt
```

每个后一阶段的 `cfg.yaml` 中 `checkpoint_path` 指向上一项。

### 3.2 三个阶段分别做什么

- **Teacher**：使用特权信息编码器和 Teacher actor 控制，PPO 更新 Teacher/编码器/critic，适应模块同时学习预测 latent。
- **Adapt**：加载 Teacher 后将 Teacher actor 复制给 Student actor；Student 控制采样，只更新适应模块。
- **Finetune**：加载 Adapt，冻结适应模块，PPO 更新 Student actor 和 critic。
- Roban 的 Student latent 为 40 维；使用 `tanh × 4.499549865722656` 后取整的量化链，量化器自身没有可训练权重。

checkpoint 保存网络、归一化、配置及部分训练元信息，**不保存完整优化器状态**。
因此 Adapt 追加属于继承权重和相关已保存状态、重新初始化优化器，不是精确断点续训。
不要将“累计更新次数”与新运行 TensorBoard 从 1 开始的横轴混淆。

### 3.3 这次训练结果带来的经验

- Teacher 最后两个千轮窗口平均奖励为 `0.24472 → 0.24562`，达到预算结束时仍有收益。
- Adapt 追加后，平均回合约 28.74 步，高度超限仍占约 99.58%。仅凭 estimator loss 下降不能判断控制已达标。
- Finetune 后，最后 250 轮平均回合约 211.05 步；高度超限约 1.07%，动作播放结束约 98.39%。
- Finetune 最后两个 250 轮窗口平均奖励 `0.23226 → 0.23300`，Student 控制能力明显改善。

因此，本次不能把 Adapt 末段控制较差直接解释为训练链路失败；Finetune 对最终控制质量非常关键。
终止比例是已结束回合中的触发比例，可能重叠，且 `train/stats/*` 来自 rank 0。

## 4. 后续训练怎么启动

### 4.1 一键三阶段

```bash
cd /data/liyinkun/project_HEFT
DRY_RUN=1 bash train.sh
# 确认预算和设备后：
bash train.sh
```

默认 Teacher/Adapt/Finetune 为 `50000 / 5000 / 5000` 轮。
这是一套后续训练预算，不是本次 `50000 / (2543+5000) / 5000` 的精确重放。

自定义示例：

```bash
CUDA_VISIBLE_DEVICES=3,4,7 NPROC=3 PER_RANK_ENVS=4096 \
TRAIN_ROLLOUTS=50000 ADAPT_ROLLOUTS=5000 FINETUNE_ROLLOUTS=5000 \
OUTPUT_ROOT=/absolute/path/to/new_pipeline \
bash train.sh
```

将 Adapt 设置为 7543 可以对齐总轮数，但仍不等价于中间重建优化器的两次运行。
要重放本次方案，应按第 3 节分两次 Adapt 执行。

### 4.2 从已有 Adapt 开始 Finetune

```bash
cd /data/liyinkun/project_HEFT
CUDA_VISIBLE_DEVICES=3,4,7 NPROC=3 PER_RANK_ENVS=4096 \
START_STAGE=finetune \
CHECKPOINT_PATH="$PWD/dataset/_audit/roban_ominisoma/roban_adapt_50000_continue5000_v1/checkpoint_final.pt" \
FINETUNE_ROLLOUTS=5000 FINETUNE_SAVE_INTERVAL=250 \
bash train.sh
```

新产物在 `roban_pipeline_<timestamp>/finetune/`，不覆盖旧运行。
`START_STAGE=adapt` 表示从 Teacher 进入 Adapt 并继续 Finetune，不是仅执行一次 Adapt 追加。

### 4.3 单独追加 5000 轮 Adapt

```bash
cd /data/liyinkun/project_HEFT
export CUDA_VISIBLE_DEVICES=3,4,7
export MEMPATH="$PWD/dataset"
export WANDB_MODE=disabled
ADAPT_PARENT=/absolute/path/to/adapt/checkpoint_final.pt
ADAPT_OUTPUT="$PWD/dataset/_audit/roban_ominisoma/adapt_more_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ADAPT_OUTPUT"
set -o pipefail

.venv/bin/torchrun --standalone --nproc_per_node=3 scripts/train.py \
  task=tracking +exp=roban_adapt task.num_envs=4096 \
  'task.profile.command.dataset.groups=[{name:formal_50k,teacher_mem_path:roban/main/train_50000_v1,weight:1.0}]' \
  checkpoint_path="$ADAPT_PARENT" total_frames=983040000 start_iter=0 save_interval=250 \
  wandb.mode=disabled tensorboard.enabled=true tensorboard.log_dir="$ADAPT_OUTPUT/tensorboard" \
  headless=true app.headless=true app.enable_cameras=false eval_render=false \
  hydra.run.dir="$ADAPT_OUTPUT" hydra.job.chdir=false hydra.output_subdir=null \
  2>&1 | tee "$ADAPT_OUTPUT/console.log"
```

### 4.4 TensorBoard 与保存

```bash
.venv/bin/tensorboard --logdir dataset/_audit/roban_ominisoma --port 6006
```

每个运行保留 `cfg.yaml`、`ppo.py`、日志、TensorBoard 和 checkpoint。
`save_interval` 的单位为 rollout；周期文件名取从零开始的循环索引，例如
`checkpoint_250.pt` 在第 251 次循环更新后保存。正常结束还会保存 `checkpoint_final.pt`。

`rollout/terminal_count_global` 统计所有 `done`，包含动作结束和超时，不等于跌倒次数。
Teacher/Finetune 与 Adapt 每轮步数不同，也不能直接比较每轮结束总数。

## 5. 部署网络与导出

真机只需要最终 Student 推理模型，不需要另上传 Teacher/Adapt checkpoint。
完整推理链为：

```text
原始 policy 观测
 → 当前 checkpoint 的观测归一化
 → adapt_module
 → 40 维 latent 量化
 → 将量化 latent 与归一化 policy 观测拼接
 → Student actor 确定性均值输出
 → action
```

Teacher actor、Teacher 特权编码器、critic、训练探索噪声不进入最终动作推理图。

| 项目 | 本次导出值 |
| --- | --- |
| 模型文件 | `policy_project_heft_finetune.onnx` |
| 输入 | `policy`, float32, `[1,1313]` |
| 输出 | `action`, float32, `[1,21]` |
| ONNX opset | 17 |
| 外部权重 | 无，单文件模型，约 19 MiB |
| 输出语义 | 未经外部 clip、EMA、scale、joint_direction 的确定性 action |

不要在部署端再次归一化，也不要把输出直接当作硬件关节角或力矩。

### 5.1 模型身份

最终 checkpoint SHA-256：

```text
258692caf9c5f5adbd6a85e560543aa8060906c85f034f3c51698dd6182606b6
```

ONNX SHA-256：

```text
809c987a3f21db4d50c8c38f1186845b2870a9d64ede85def5c916845dbf233e
```

### 5.2 导出与一致性验证产物

```text
dataset/_audit/roban_ominisoma/deploy_our_heft_20260908/
├── export_student.py
├── validate_replay.py
├── policy_project_heft_finetune.onnx
├── policy_project_heft_finetune.json
├── onnx_replay_validation.json
├── installed_model_validation.json
├── installation.json
└── roban_project_heft_patch.tar.gz
```

导出脚本依赖同一工作区中
`official_amp_our_heft_20260908/student_policy.py`，通过原生层和严格 state_dict 加载恢复 Student。
这些审计目录受 Git 忽略；只 clone 训练代码并不会得到它们，复现必须同时归档这些文件。
已安装 ONNX 则不依赖 Python 导出脚本。

验证结果：128 组构造输入最大动作绝对误差 `2.1458e-6`；821 帧真实回放输入最大误差
`2.3842e-6`。ONNX 检查、配置解析、模型加载均通过。821 帧回放期间由 ONNX 实际控制。

## 6. 新旧运动树、IMU 和腰关节坐标

| 物理部位 | 本项目新 XML | 官方旧 XML |
| --- | --- | --- |
| 胸部 | `base_link`，浮动根 | `torso` |
| 骨盆 | `waist_yaw_link`，腰关节子节点 | `base_link`，浮动根 |
| IMU | 胸部 `base_link/imu` | 胸部 `torso/imu` |

两者胸部使用同一 `torso` 网格及相同胸部碰撞几何，IMU site 都位于胸部局部
`pos="0 0 0"`，无额外局部旋转。不能依据链接名字把新 `base_link` 当作骨盆。

```text
旧：骨盆根 → 腰关节 → 胸部 + IMU
新：胸部 + IMU 根 → 腰关节 → 骨盆
```

本次数值验证采用：新根世界位姿 = 旧 `torso` 世界位姿；新腰角 = −旧腰角；其他关节角不变。
50 组随机姿态、每组六个身体配对的位置误差最大 `4.02e-16 m`，旋转矩阵误差最大 `5.55e-16`。
这是运动学对应关系验证，不是新旧质量和惯性完全相同的证明。

**部署要点：**

- 胸部姿态/陀螺仪直接作为跟踪观测来源，不给 HEFT 跟踪输入额外减去腰角。
- 官方 planner 的 `getPlannerRootQuat()` 有自己的去腰旋转用途，不应套到跟踪网络输入上。
- 旧硬件坐标通过 `joint_direction[0]=-1` 转到策略坐标，速度同样变号。
- 输出目标通过相同方向系数转回硬件坐标；新 XML 仿真适配还需处理物理坐标的反向映射，避免乘两次错误的负号。
- 上机时并未把旧部署 XML 替换成新训练 XML。真机关键是电机和 IMU 坐标，不是仿真根节点名字。

## 7. 本次成功所用的输入输出与控制参数

### 7.1 观测顺序

以下为原始网络输入拼接顺序，区间采用 Python 左闭右开表示：

| 字段 | 维数 | 区间 |
| --- | ---: | --- |
| 启动指示 | 1 | `[0:1]` |
| 参考位置相对位移 | 36 | `[1:37]` |
| 参考姿态相对当前胸部姿态，rot6d | 78 | `[37:115]` |
| 参考关节位置 | 273 | `[115:388]` |
| 参考关节位置减当前位置 | 273 | `[388:661]` |
| 参考根高度 | 13 | `[661:674]` |
| 参考重力方向 | 39 | `[674:713]` |
| 胸部角速度历史 | 27 | `[713:740]` |
| 胸部重力方向历史 | 27 | `[740:767]` |
| 关节位置历史 | 189 | `[767:956]` |
| 关节速度历史 | 189 | `[956:1145]` |
| 动作历史 | 168 | `[1145:1313]` |

- 参考帧：`[0,1,2,3,4,5,6,7,8,-1,-2,-4,-8]`，共 13 帧。
- 状态历史：`[0,1,2,3,4,8,12,16,20]`，共 9 帧，最新状态偏移为 0。
- 动作历史：最近 8 帧，最新在前。
- 四元数为 `wxyz`；rot6d 是旋转矩阵前两列按列展开。
- 位置相对位移使用当前参考姿态坐标；姿态差使用当前机器人胸部坐标，并由官方 dummy-world yaw 处理初始航向。

### 7.2 滤波与历史：已验证的差异也必须保留记录

| 项目 | 训练环境 | 本次真机使用的官方部署行为 |
| --- | --- | --- |
| 策略周期 | 20 ms | 20 ms |
| 动作滤波更新 | 每 5 ms 物理子步 | 每次推理完成后一次 |
| EMA α | 每回合 `[0.8,1.0]` 采样 | 固定默认 `0.9` |
| 滤波初值 | 零 | 首次推理输出 |
| 状态历史初始化 | 清零，再逐步写入 | 当前状态填满全部历史槽位 |
| 动作历史初始化 | 零 | 零 |
| 延迟 | 0–2 个物理子步的随机延迟模型 | 实际异步推理/通信时序 |
| 动作历史语义 | 最后子步的 applied action | 官方滤波后的 action cache |

这次成功并不证明两边滤波等价，也不意味着应立即修改成功基线。
训练 α=0.9 在无延迟时连续四次更新的等效系数为 `0.9999`，不等于官方每周期一次的 `0.9`。
若后续改变这些行为，应作为独立实验重新比较。

### 7.3 动作和 PD

HEFT 零默认关节位置、不使用参考残差动作：

```text
a_clip = clamp(network_action, -10, 10)
a_filter = 0.1 × previous_filtered + 0.9 × a_clip
q_target_hardware = joint_direction × (a_filter × action_scale)
tau = clip(kp × (q_target_hardware − q_hardware) − kd × dq_hardware, ±torque_limit)
```

首次滤波输出直接取 `a_clip`。模式为 CST 软件 PD；官方代码把底层额外 kp/kd 设为零，
避免重复 PD/阻尼。不要同时在 ONNX 和驱动端重复应用 scale、符号或滤波。

下方是最终现场应使用的 HEFT 配置参考。它由打包时配置生成，只保留现场确认的两条 motion；
`action_alpha: 0.9` 显式写出了原 C++ 默认值。其余控制参数保持官方值。

```yaml
HumanoidRobotCfg:
  inference_engine: onnxruntime
  loop_dt: 0.001
  policy_path: ../policy/heft/policy_project_heft_finetune.onnx
  motion:
    default_motion: default_stand
    motions:
    - {name: default_stand, file: ../motion_data/roban_default_stand_10s_deploy.csv}
    - {name: walk_forward_stop, file: ../motion_data/main_220728_walk_forward_stop_002__A044_deploy.csv}
  env:
    policy_dt: 0.02
    heft:
      future_steps: [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -4, -8]
      history_steps: [0, 1, 2, 3, 4, 8, 12, 16, 20]
      prev_action_steps: 8
      boot_indicator_max: 25.0
      action_clip: 10.0
      output_name: action
      expected_obs_dim: 1313
      live: {topic: /rt/pico/heft_reference, robot_version: roban_s17, schema_version: 1,
        cursor_delay: 0.16, fresh_timeout: 0.15, hold_timeout: 0.5, buffer_capacity: 2.0,
        fallback_controller: amp}
      planner: {onnx_path: ../policy/heft/best.onnx, exec_steps: 2, command_offset: 150,
        ode_steps: 5, t_start: 0.7, control_hz: 50.0, horizon_frames: -1, intra_op_threads: 2}
      action_alpha: 0.9
    robot:
      residual_action: false
      enable_arm_controller: false
      enable_waist_controller: false
      joint_names: [waist_yaw_joint, leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint,
        leg_l5_joint, leg_l6_joint, leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint,
        leg_r5_joint, leg_r6_joint, zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint,
        zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint]
      joint_direction: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      joint_default_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      joint_torque_limit: [80.0, 150.0, 150.0, 70.0, 150.0, 74.0, 74.0, 150.0, 150.0,
        70.0, 150.0, 74.0, 74.0, 14.1, 37.0, 37.0, 37.0, 14.1, 37.0, 37.0, 37.0]
      actuator_kp: [40.1792, 90.1792, 150.0984, 40.1792, 150.0984, 34.2506, 34.2506,
        90.1792, 150.0984, 40.1792, 150.0984, 34.2506, 34.2506, 14.2506, 14.2506,
        14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506]
      actuator_kd: [3.5579, 3.5579, 8.3088, 3.5579, 8.3088, 2.9072, 2.9072, 3.5579,
        8.3088, 3.5579, 8.3088, 2.9072, 2.9072, 1.9072, 1.9072, 1.9072, 1.9072, 1.9072,
        1.9072, 1.9072, 1.9072]
      actuator_control_mode: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0]
      action_scale: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### 7.4 官方 AMP 参数快照

AMP 模型路径保持官方版本。此处保留影响静止接管和 PD 的关键参数；完整 AMP YAML 不作改写。

```yaml
policy_path: ../policy/amp/06-09_15-39-29_amp_agent_72000.onnx
loop_dt: 0.001
policy_dt: 0.02
robot:
  enable_arm_controller: true
  enable_waist_controller: false
  enable_amp_arm_enhance_v17: true
  enable_back_enhance_v17: true
  enable_standup_enhance_v17: true
  enable_turn_in_place_enhance_v17: true
  joint_names: [waist_yaw_joint, leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint,
    leg_l5_joint, leg_l6_joint, leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint,
    leg_r5_joint, leg_r6_joint, zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint,
    zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint]
  joint_direction: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  joint_default_pos: [0.0, -0.285, -0.0194, -0.201, 0.404, -0.221, 0.0, 0.285, 0.0194,
    0.201, 0.404, -0.221, 0.0, 0.0, 0.05, -0.0, -0.2, 0.0, -0.05, 0.0, -0.2]
  start_stand_pos: [0.0, -0.185, -0.0194, -0.201, 0.454, -0.221, 0.0, 0.185, 0.0194,
    0.201, 0.454, -0.221, 0.0, 0.0, 0.05, -0.0, -0.2, 0.0, -0.05, 0.0, -0.2]
  joint_torque_limit: [80.0, 150.0, 150.0, 70.0, 150.0, 74.0, 74.0, 150.0, 150.0,
    70.0, 150.0, 74.0, 74.0, 14.1, 37.0, 37.0, 37.0, 14.1, 37.0, 37.0, 37.0]
  actuator_kp: [40.179, 90.179, 150.098, 40.179, 150.098, 34.25, 34.25, 90.179, 150.098,
    40.179, 150.098, 34.25, 34.25, 14.25, 14.25, 14.25, 14.25, 14.25, 14.25, 14.25,
    14.25]
  actuator_kd: [2.557, 3.557, 8.308, 3.557, 8.308, 1.907, 1.907, 3.557, 8.308, 3.557,
    8.308, 1.907, 1.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907]
  actuator_control_mode: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0]
  action_scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
observations:
  history_length: 1
  stack_order: classic
  terms:
    base_ang_vel:
      scale: 1.0
      clip: [-5000.0, 5000.0]
    projected_gravity:
      scale: 1.0
      clip: [-5000.0, 5000.0]
    velocity_commands:
      scale: 1.0
      clip: [-5000.0, 5000.0]
    cmd_stance:
      scale: 1.0
      clip: [0.0, 1.0]
    joint_pos:
      scale: 1.0
      clip: [-5000.0, 5000.0]
    joint_vel:
      scale: 1.0
      clip: [-5000.0, 5000.0]
    actions:
      scale: 1.0
      clip: [-5000.0, 5000.0]
```

## 8. 仿真验证链与结果

### 8.1 本项目 Finetune 在训练环境中回放

目录：`dataset/_audit/roban_ominisoma/finetune_video_20260908/`。

- 16 环境，1000 步，20 秒，确定性动作；视频显示环境 0。
- 77 个结束回合均为动作播放结束，未触发高度/姿态终止，恢复次数为零。
- 根 XY 平均误差 7.08 cm，高度平均误差 1.94 cm，关节平均 RMSE 约 4.0°。
- 视频：`roban_finetune_tracking.mp4`。

这是训练环境短时采样，不与下一节单条 CSV 直接混合统计。

### 8.2 官方 AMP → 官方 HEFT 基线

目录：`dataset/_audit/roban_ominisoma/official_amp_heft_20260908/`。

- 官方 AMP：`06-09_15-39-29_amp_agent_72000.onnx`。
- 官方 HEFT：`policy_sym.onnx`；`best.onnx` 是 planner，不是动作跟踪网络。
- 本项目新 XML + 平地；保留该 XML 的质量、惯性、碰撞和被动关节参数。
- 参考 CSV：`main_220728_walk_forward_stop_002__A044_deploy.csv`，571 帧，50 Hz。
- 0–3 秒 AMP 零指令站立；3–5 秒双策略五次曲线混合；陀螺仪满足阈值后约 5.498 秒起播。
- 切换过程不 reset、不瞬移、不对机器人施加参考位姿强制控制。
- 最终站稳走完，根水平位移约 2.59 m。
- 视频：`official_amp_to_heft_walk.mp4`。

### 8.3 官方 AMP → 我们的 Student

目录：`dataset/_audit/roban_ominisoma/official_amp_our_heft_20260908/`。

仅替换 HEFT 推理模块及其归一化，其他流程相同。切换前 3 秒 AMP 仿真 trace 差异为零。

| 行走区间指标 | 官方 HEFT | 我们的 Student |
| --- | ---: | ---: |
| 平均根 XY 误差 | 6.76 cm | 25.77 cm |
| 根 XY P95 | 10.57 cm | 36.74 cm |
| 平均根高度误差 | 1.31 cm | 1.35 cm |
| 平均关节 RMSE | 2.00° | 3.12° |
| 全程水平位移 | 2.59 m | 2.33 m |
| 跌倒 | 无 | 无 |

视频：`official_amp_to_our_heft_walk.mp4`，19.44 秒，1280×480，25 FPS，486 帧全部解码通过。
左侧是动力学实际结果，右侧才是运动学参考。误差统计排除了 AMP、混合阶段和末帧保持。

这是单进程确定性复现：官方独立 MuJoCo 的物理默认周期为 2 ms，推理为 20 ms；
控制线程的异步 1 kHz 调度、DDS 和网络抖动未仿真。它不能取代完整真机验证。

## 9. 最小部署改动与打包

### 9.1 改了哪些文件

两个官方解压目录均完成了同样的适配：

```text
Self_Add/leju的heft部署代码/leju_heft_contract/
Self_Add/leju的heft部署代码/extracted/lejulab_platform-heft-dev/
```

在各自 `src/leju-controllers/leju-rl-controller/config/17/` 下：

1. 新增 `policy/heft/policy_project_heft_finetune.onnx`。
2. 新增 `policy/heft/policy_project_heft_finetune.json`。
3. `controllers/config_heft.yaml` 只将 `policy_path` 改为新模型。

原官方模型保留。C++、AMP、关节符号、PD、滤波和历史初始化均未修改。
`heft`、`heft_planner`、`heft_pico` 共用该 HEFT YAML，所以三种模式都绑定新的跟踪模型；
本次验证以 CSV 为准，不把 planner/Pico 列为现场成功范围。

### 9.2 本次上传的包

```text
/data/liyinkun/project_HEFT/Self_Add/leju的heft部署代码/lejulab_platform-heft-dev_project-heft_20260908.tar.gz
```

- 大小：598912238 bytes，约 571 MiB。
- 顶层目录：`lejulab_platform-heft-dev/`，与原始包一致。
- SHA-256：

```text
46c9c56cb12d92eb6144c4b152dcfe3520f2929393862c605f0d3515b8bf03f3
```

仅上传 `.tar.gz` 即包含源码、配置和模型，不需要另传训练 checkpoint；`.sha256` 文件可选。
但这个包**不含可执行程序**，不是独立可运行的 ARM64 二进制发布包。

原始 `lejulab_platform-heft-dev.tar.gz` 未覆盖。上面这个成功上传包中的动作列表仍是
原来的多动作列表；“仅站立/行走”是在机器人端后续修改的。若重新制作两动作包，
必须使用新名称和新哈希，不能继续套用此处包哈希。

### 9.3 解压及检查

机器人上的操作目录：

```text
/home/lab/lejulab_platform_0908/
```

```bash
cd /home/lab/lejulab_platform_0908
tar -xzf lejulab_platform-heft-dev_project-heft_20260908.tar.gz
cd lejulab_platform-heft-dev
CFG=src/leju-controllers/leju-rl-controller/config/17

grep -n 'policy_path:' "$CFG/controllers/config_amp.yaml"
grep -n 'policy_path:' "$CFG/controllers/config_heft.yaml"
sha256sum "$CFG/policy/heft/policy_project_heft_finetune.onnx"
grep -n -A4 -E 'name:.*(amp|heft)' "$CFG/controller_manager.yaml"
```

预期 AMP 使用官方 `72000.onnx`，HEFT 使用 `policy_project_heft_finetune.onnx`，
模型哈希与第 5 节一致。注释掉的 `52800.onnx` 不生效。

## 10. 机器人端实际启动方式

本次机器人 `uname -m` 为 `aarch64`。
第一次从新源码目录启动报错：`binary 'leju-hardware' not found`。
原因是没有构建产物，不是 ONNX 错误。现场查找发现已有可执行程序，最终复用：

```text
/home/lab/leju_heft/lejulab_platform/build_cmake/src/leju-hardware/leju-hardware
/home/lab/leju_heft/lejulab_platform/build_cmake/src/leju-controllers/leju-rl-controller/run_rl_controller
```

启动命令如下。应在确认上一套控制程序已退出、机器人处于合适支撑条件下执行，
避免多个硬件控制程序同时运行。

```bash
sudo su
cd /home/lab/lejulab_platform_0908/lejulab_platform-heft-dev

export ROBOT_VERSION=17
export LEJULAB_PROJECT_ROOT="$PWD"
export BUILD_DIR=/home/lab/leju_heft/lejulab_platform/build_cmake

CFG="$PWD/src/leju-controllers/leju-rl-controller/config/17"

bash src/leju_launch/scripts/launch_real.sh \
  --config="$CFG/controller_manager.yaml" \
  --teleop-config="$CFG/teleop_bindings.yaml"
```

目录职责：

- `LEJULAB_PROJECT_ROOT` 决定新源码/配置/模型目录。
- `BUILD_DIR` 指向已有 ARM64 构建产物。
- `--config` 显式指定本次 controller manager。
- `--teleop-config` 显式指定本次手柄绑定，避免默认读取 `/root/.config/lejuconfig/teleop_bindings.yaml`。

启动脚本依次从构建目录、安装目录和 PATH 查找每个节点。这里只确认现场已有的硬件和 RL
二进制路径；其他辅助节点实际来源应以 launcher 日志为准。复用的二进制没有提供 git
版本或 SHA-256，不能仅因路径名相似就假定它与打包源码逐字一致。

本次现场日志已经实际证明 START、X、A 和双推理切换能够工作。将来更换二进制或部署
环境时，应重新核对加载路径和按键行为，而不是只检查新目录中的 YAML。

## 11. 手柄操作及最终两动作配置

### 11.1 最终现场修改

文件：

```text
src/leju-controllers/leju-rl-controller/config/17/controllers/config_heft.yaml
```

只保留以下 motion 块，**保留其后的完整 `env:` 内容**：

```yaml
  motion:
    default_motion: default_stand
    motions:
      - name: default_stand
        file: "../motion_data/roban_default_stand_10s_deploy.csv"
      - name: walk_forward_stop
        file: "../motion_data/main_220728_walk_forward_stop_002__A044_deploy.csv"
```

两个动作文件均为官方文件。`walk_forward_stop` 是仿真验证过的前进行走后停止动作。
不要只留下 YAML 前几行而删除 PD、关节映射和 HEFT 参数。

修改前可备份：

```bash
FILE=src/leju-controllers/leju-rl-controller/config/17/controllers/config_heft.yaml
cp "$FILE" "$FILE.bak.$(date +%Y%m%d_%H%M%S)"
```

修改后重新启动才能读取新配置。不要在运行中的程序旁再启动一份实例。

### 11.2 已确认的操作顺序

以直立初始姿态为例：

1. 启动 `launch_real.sh`，等待日志提示“机器人就绪，等待 start”。
2. 单独按 **START**：软件进入 Running，当前控制器为官方 AMP。
3. 速度指令回零并站稳后，单独按 **A**：请求切换到 `heft`，使用我们的 ONNX。
4. 约 2 秒双策略过渡后，gyro 连续 0.5 秒低于 0.3 rad/s，自动播放 `default_stand`。
5. 站立动作完成后，再单独按 **A**：选择并播放 `walk_forward_stop`。
6. 行走完成后，再按 **A**：选择站立参考。

没有“切入 HEFT 后再按另外一个键起播”的必要，因为 A 的绑定包含
`auto_start_motion: true`。播放结束不会自动串行执行下一条，必须由下一次操作选择。

| 键 | 当前绑定 |
| --- | --- |
| START | 启动生命周期；已经 Running 时再次按会被忽略 |
| X | 请求官方 AMP；若已在 AMP，保持当前控制器 |
| A | 进入 HEFT CSV 并自动起播；已在 HEFT 时轮换下一条 |
| Y | HEFT planner，不是本次 CSV 行走验证模式 |

X 不是急停：动作播放中、切换中或不满足站立条件时，切换可能被拒绝。
Ctrl+C 是退出程序，硬件退出日志包含电机 Disable；它也不是“切回 AMP”。
需要结束程序时，先确保机器人不会因失去控制而倒下。

## 12. 本次排障：看似“没反应”时实际发生了什么

现场完整日志给出了以下证据：

```text
JoyTeleopAdapter: START button pressed
Lifecycle: Start triggered, entering running state
ControlLogic: Current controller before entry: amp
```

说明 START 已收到、状态机已运行。反复 START 出现：

```text
Lifecycle: Start trigger ignored (state: kRunning)
```

这是重复启动被忽略，不是输入丢失。按 X 出现：

```text
Controller 'amp' is already active, skip switch
```

说明原本就在 AMP。按 A 后出现：

```text
Processing switch_controller to 'heft' (auto_start=1)
Starting RL->RL dual inference switch: amp -> heft
CommitSwitch complete: amp -> heft
CommitSwitch: robot stabilized ... auto-starting motion
Motion 'default_stand' playback started
```

说明控制器切换和自动起播均成功。机器人看起来只是站着，是因为默认动作就是站立。
日志中残留的“Press guide”提示不能覆盖后面已发生的自动起播事实。

以下信息并未被这份日志证明为控制故障根因：

- `mic wake-up retry`：麦克风唤醒重试。
- `Hand state unavailable`：手部状态不可用，不等于腿部控制没有启动。
- Ctrl+C 之后 `pgrep` 仅剩常驻手柄服务：对应日志里的正常控制程序退出过程。

只读进程检查：

```bash
pgrep -af 'leju-hardware|run_rl_controller|leju-joystick'
```

软件 Running 也不单独证明电机输出有效；仍需区分主动保持站姿与完全松软。
本次后续由操作者确认实际进入 AMP、我们的 HEFT，并最终反馈 sim2real 成功。

## 13. 归档与后续复现

建议同时保留：

- 第 3 节的四个 checkpoint 及各自配置、日志。
- 训练代码 commit 与环境锁文件。
- 最终 ONNX、JSON、SHA-256。
- 导出、回放脚本及原始官方 ONNX/CSV；这些在 Git 忽略的目录，需单独备份。
- 原始官方包、这次候选上传包、机器人端最终两动作 YAML 和显式 teleop 配置。
- 机器人实际使用的可执行程序版本/哈希和启动日志；本次尚未采集这些哈希。

主要可点击入口：

- [通用 Roban 训练说明](使用说明.md)
- [三阶段入口](../train.sh)
- [导出与校验目录](../dataset/_audit/roban_ominisoma/deploy_our_heft_20260908/)
- [官方 AMP → 我们 Student 的视频](../dataset/_audit/roban_ominisoma/official_amp_our_heft_20260908/official_amp_to_our_heft_walk.mp4)
- [官方/我们 Student 对比指标](../dataset/_audit/roban_ominisoma/official_amp_our_heft_20260908/comparison.json)
- [新旧运动树验证](../dataset/_audit/roban_ominisoma/official_amp_heft_20260908/tree_mapping.json)

本记录建立的是**已成功的部署基线**。后续优化训练预算、控制滤波、历史初始化或资产
动力学参数时，每次单独改变一类因素，与此基线作对照。不要将尚未做过的改动写入
“本次成功配置”，也不要把单条行走成功解释为全部动作已经通过。
