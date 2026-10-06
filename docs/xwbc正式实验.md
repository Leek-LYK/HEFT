# XWBC 正式实验：单 B200、无负载、直接 PPO

研究入口使用 HEFT 的机器人资产、仿真、PD 控制及跟踪奖励，训练一个支持人体/G1/Roban 命令和 G1/Roban 执行对象的策略。研究模型直接进行 PPO，训练流程为数据准备 → 策略训练 → 六组合评估，不调用 Teacher/Adapt/Finetune 三阶段脚本。

研究环境的随机化注册表为空：不挂载负载、不增加额外质量、不启用 WPC、外力或推扰。动作力矩采用资产标定范围，不使用原有放大力矩课程。旧 HEFT 入口保持原样。

v0.1 的共享模型 seed 0 已在 `logs/v0.1/shared_seed0_run01` 完成 2,500 次更新。
对应源码、数据和权重摘要及 CPU 验证方法见 [v0.1 可复现基线](xwbc_v0.1基线.md)。
第一轮完整六组合评估仍待开展。当前用户已安排 v0.1 第二轮训练，采用下文的宽窗口配置；第一轮基线通过 `CONFIG=cfg/xwbc/train.yaml` 复现，历史证据保留。

## 运行环境与默认预算

在仓库根目录使用 `.venv`，单进程占用一张 B200。两个机型环境池位于同一设备，分别采集数据，共同优化共享参数。

第二轮默认每机型 8,192 个环境、每次采集 32 步、10,000 次更新，合计 5,242,880,000 transitions；每个机型获得相同份额。此规模已完成 B200 有限预算运行验证。预算、环境数和输出目录通过环境变量覆盖；训练曲线和验证评估用于后续预算调整。

主要配置：`cfg/xwbc/heft_context.yaml`，通过 `_base_` 继承并覆盖第一轮的 `cfg/xwbc/train.yaml`。统一参考预览最远 400 ms、本体历史最远 400 ms、8 个历史动作，Encoder 输出 256 维；`history_steps` 优先于兼容旧模型的 `history` 字段。详见 [第二轮方案](../科研/v0.1/04_第二轮训练方案与记录.md)。默认方法为 `shared`，种子为 `0`。三种方法的网络输入、奖励和每机型采样预算保持一致。

## 正式入口

只预览命令，不启动程序：

```bash
DRY_RUN=1 bash scripts/run_xwbc.sh train
DRY_RUN=1 bash scripts/run_xwbc.sh suite
```

启动第二轮共享模型：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT="$PWD/logs/v0.1/shared_seed0_run02" \
bash scripts/run_xwbc.sh train
```

启动完整对照实验（按顺序占用同一张卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT="$PWD/logs/v0.1/comparison_run01" \
bash scripts/run_xwbc.sh suite
```

`suite` 默认依次运行 `shared`、`independent`、`no_align`，每种方法使用种子 `0 1 2`，共九次训练。每次训练结束，对冻结的最终 checkpoint 分别执行验证集和测试集的六组合评估，最后生成跨种子汇总。当前协议预先选择最终 checkpoint，测试结果不参与 checkpoint 选择。

选择部分对照或调整预算：

```bash
VARIANTS="shared no_align" SEEDS="0" \
NUM_ENVS=2048 UPDATES=40000 \
OUTPUT_ROOT="$PWD/logs/v0.1/selected_comparison" \
bash scripts/run_xwbc.sh suite
```

`NUM_ENVS` 是每机型环境数；上述示例与默认设置的总 transitions 相同。单模型可用 `VARIANT=independent` 或 `VARIANT=no_align`。所有实验选择新输出目录，已有结果保留。

## 数据准备

`train` 和 `suite` 自动检查派生数据是否已准备；也可单独执行：

```bash
WORKERS=8 bash scripts/run_xwbc.sh prepare
```

默认位置：

- 候选索引：`dataset/xwbc/manifests/paired_10k.jsonl`。
- 派生数据：`dataset/xwbc/paired_10k_v1/`。
- `DATASET` 和 `MANIFEST` 可指定其他版本或候选清单。

抽样按命名家族隔离训练/验证/测试，固定类别配额并优先覆盖不同家族，总量 8,000/1,000/1,000。编译器检查三侧帧数、字段、采样率约定、数值和人体旋转表示，统一机器人单位、关节顺序，并利用当前机器人 MJCF 生成命令空间特征。G1 CSV 按 BONES-SEED 的厘米、角度和外禀 xyz 约定读取；Roban 按已确认的源顺序重排，腰部符号保持不变。

同一片段的三种命令和两个目标参考一起进入 shard，时间索引一致。格式不合格片段共同退出本次派生数据，在 `dataset.json` 中记录原因。物理几何摘要另存于各 shard 的 `motions.json`；当前版本的准入是格式/数值检查，原始动作及其难度覆盖保持可追溯。

数据目录包含 `dataset.json`、完成标记 `READY`、分片命令数组和两个机器人的 HEFT memdataset。加载时核对完成摘要及资产版本，源动作保持原样。当前已编译版本包含 10,000 个动作、4,028,269 帧。

## 小规模完整链路验证

```bash
OUTPUT_ROOT="$PWD/logs/v0.1/smoke_run01" \
bash scripts/run_xwbc.sh smoke
```

默认每个划分只使用 12 个片段，每机型 32 环境、2 次更新，然后在验证集的 4 条动作上运行六组合评测。可通过 `SMOKE_ENVS`、`SMOKE_UPDATES`、`SMOKE_DATASET` 和 `VARIANT` 调整。默认运行墙钟上限为 10 分钟，从训练循环开始计时。

使用正式数据进行有限预算验证：

```bash
UPDATES=2 MAX_MINUTES=10 \
OUTPUT_ROOT="$PWD/logs/v0.1/full_data_short_run" \
bash scripts/run_xwbc.sh train
```

训练退出会保存实际完成次数和最终 checkpoint。达到时间预算提前结束时，`result.json` 会标记 `budget_exhausted`，保留实际预算信息。

## 单独评估

```bash
CHECKPOINT="$PWD/logs/v0.1/shared_seed0_run02/checkpoint_final.pt" \
OUTPUT_ROOT="$PWD/logs/v0.1/shared_seed0_run02_eval" \
SPLIT=test EVAL_ENVS=64 \
bash scripts/run_xwbc.sh evaluate
```

每条动作从目标机器人参考的起始帧初始化，统一抬升 2 厘米，不加初态噪声；在三种来源下保持相同初态。策略使用确定性动作，参考及本体历史按回合正确重置。完成片段优先于普通同帧失败，非有限状态单列为失败。提前终止前的有效步仍计入误差。

输出包括：

- `clips.jsonl`：逐片段的来源、机型、类别、家族、成功、完成比例、误差与失败原因。
- `matrix.csv`：六组合指标表。
- `summary.json`：逐组合、逐类别、类别等权成功率及动作家族 bootstrap 区间。
- `*_environment.json`：评分使用的奖励、终止、关键点和初始化配置。

评价包含世界坐标和根相对关键点位置误差、速度误差、根位置与旋转误差。对照使用相同关键点集合及终止实现；位置误差的参考由当前目标资产 FK 生成。杂技等家族较少的类别同时展示家族数量，便于解释统计覆盖。

汇总已有实验：

```bash
.venv/bin/python -B scripts/summarize_xwbc.py --root logs/v0.1/comparison_run01
```

汇总生成每种方法、每个 split、每个路由的跨种子均值与标准差。`comparison.csv` 使用新文件写入；重复汇总请保留旧结果后使用新的输出位置或自行管理已有汇总文件。

## checkpoint 与继续训练

```bash
CONFIG="$PWD/cfg/xwbc/train.yaml" \
RESUME="$PWD/logs/v0.1/shared_seed0_run01/checkpoint_1000.pt" \
UPDATES=2500 \
OUTPUT_ROOT="$PWD/logs/v0.1/shared_seed0_continue" \
bash scripts/run_xwbc.sh train
```

`UPDATES` 是累计目标更新次数。继续训练时 `CONFIG` 选择与 checkpoint 相同的配置；第一轮使用 `CONFIG="$PWD/cfg/xwbc/train.yaml"`，第二轮使用 `CONFIG="$PWD/cfg/xwbc/heft_context.yaml"`。继续训练加载网络、优化器和随机数状态，使用相同数据版本、网络与训练配置，并从新仿真回合继续；checkpoint 不保存仿真器的完整内部状态。各次运行的起点、实际预算和继续语义写入记录。

新 checkpoint 使用 `xwbc-v1` 格式，包含全部来源编码器、机型适配模块与共享策略。旧 HEFT 三阶段 checkpoint 属于另一套模型接口。

## 模型信息边界与方案差异

actor 只使用所选来源的命令和执行机器人的本体历史。目标机器人参考用于奖励、终止、初始化及特权 critic。其他来源的配对命令参与训练对齐损失，独立的路由测试验证它们不影响已选来源的 actor 输出。

当前模型使用固定物理量缩放及 LayerNorm；三个命令编码器输出统一 token，MLP 主体共享，动作头保持目标机型的实际动作维度。`independent` 为两套架构匹配的独立主体及命令编码器；`no_align` 仅去掉对齐项。

命令采用骨盆中心、水平朝向归一化的空间表示，并保留根高度及转向角速度；窗口为当前与邻近过去/未来参考帧。本体历史只含关节位置、速度、角速度、重力和已施加动作。人体与机器人根的语义、各刚体局部轴差异由明确的数据处理和来源编码器承担。

本版提供离线仿真研究闭环。主干采用 MLP；Transformer 对照作为后续研究扩展单独设计。

## 日志与复现

每次训练目录保存完整配置、两个环境配置、数据摘要、源码快照、Git 信息、逐更新 JSONL、TensorBoard、checkpoint 和重载检查结果。脚本同时保留控制台输出；训练原始数据不进入运行目录。

```bash
.venv/bin/tensorboard --logdir logs/v0.1
```

关注六组合跟踪结果，以及训练中的 reward、KL、value loss、命令对齐、token 方差、来源采样数量和仿真恢复计数。`peak_torch_cuda_bytes` 仅统计 PyTorch 分配，MuJoCo Warp 显存另看设备监测。

实验记录与科研分析存放于 `科研/v0.1/`；长期工作方式继续遵循 `.SKILL/SKILL.md`。
