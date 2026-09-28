# 全量 Ominisoma 数据训练

数据范围为 `Self_Add/dataset/npz/ominisoma_npz_train` 下六个类别的全部 NPZ。
输出为 `dataset/roban/full/train_v1`，原始文件及既有 50K 数据保持不变。
只有 `full_report.json` 的状态为 PASS 才可启动训练。

## 本次转换结果（2026-09-09）

全量扫描和转换已完成，`full_report.json` 为 PASS：

| 类别 | 源 NPZ | 保留 | 过滤 |
| --- | ---: | ---: | ---: |
| acrobatic | 270 | 196 | 74 |
| crawl | 2,880 | 2,465 | 415 |
| jump | 10,833 | 6,440 | 4,393 |
| kneel | 1,920 | 1,764 | 156 |
| main | 108,193 | 97,809 | 10,384 |
| squat | 6,718 | 6,387 | 331 |
| 合计 | 130,814 | 115,061 | 15,753 |

最终 27 个 shard、115,061 个 motion、44,952,911 帧。参考张量磁盘 FP16
约 2.34 GiB；转换结束时数据及审计文件共约 2.67 GiB（后续日志另计）。
联合 loader 为 CPU FP32，参考张量占 5,034,726,032 bytes，即约 4.69 GiB/rank；
这不是训练进程总内存，加载拼接还会产生临时副本。

拒绝原因计数：少于 150 帧 15,145 次、关节速度尖峰 357 次、全片最大 body
高度过低 187 次、body 穿地超过 0.10 m 76 次。同一文件可命中多个规则。
全部 130,814 个源文件转换前后 SHA-256 一致。27 个 shard 分别通过
100-motion FK 检查，合计抽查 2,700 个 motion、10,800 帧；联合 CPU loader
全张量 finite 检查通过。过滤是排除出新训练集，原始文件未删除。

## 质量规则

使用现有 Roban adapter 验证 50 Hz、必需字段、shape、数值 finite、quaternion
有效性及 21 维 joint order。root=base_link、wxyz、waist yaw 不取反。

物理质量规则复用 `generate_dataset.py::check_motion`：

- 少于 150 帧（3 秒）拒绝；
- 所有 body 同时高于地面 0.2 m 持续超过 1 秒拒绝；
- 全片最大 body 高度不超过 0.2 m 拒绝；
- root 线速度超过 8 m/s、角速度超过 15 rad/s、任一关节速度超过 20 rad/s 拒绝（0.1 秒窗口）；
- 任意 body 位置 z < -0.10 m 拒绝。

本版采用**整文件剔除**：只要文件存在上述坏区间，就不保留该文件的其他片段。
因此“质量差”指不符合当前训练数据规则，并非对所有运动任务通用的质量评级。
不按文件名删除 jump、crawl 等类别，也不会把 source quaternion 的等价 q/-q 翻转
当作错误；适配器正常处理符号连续性。

## 转换及产物

```bash
cd /data/liyinkun/project_HEFT
export MEMPATH=$PWD/dataset
.venv/bin/python -m scripts.data_process.build_roban_full_dataset \
  --source-root Self_Add/dataset/npz/ominisoma_npz_train \
  --output dataset/roban/full/train_v1
```

输出已存在时命令拒绝覆盖；重新转换请使用新版本目录。最多 5000 文件/shard，
每个 shard 单独转换、验证，最后执行多 shard 联合 loader 验证和全部源文件哈希复查。
最终目录包含：

- `selection_manifest.json`：六类全部源文件及每类内 SHA-256 排序；
- `source_manifest.jsonl`：逐文件源哈希、帧数、质量结论和原因；
- `failed_files.jsonl`：所有被拒绝的文件及原因、坏区间；
- `<category>_shard_NNN/`：训练 memdataset、转换报告和 100-motion FK 验证；
- `dataset_groups.json`：训练需要的所有 shard 路径与权重；
- `full_report.json`：总量、按类统计、过滤原因、联合 loader 检查结果；
- `logs/`：逐 shard 完整转换及验证日志。

磁盘为 float16，loader 为 CPU float32；body pose/velocity 在 GPU 的有界窗口内通过 FK
生成。多 shard loader 会在 CPU 拼接全部参考张量，因此主机仍需足够内存。

## 全量采样

原 50K 配置在 episode reset 时不重抽 motion。全量配置明确开启
`resample_on_reset=true`：新 episode 重新抽取 motion 并刷新 cache；需要 rewind 的
episode 保持原 motion，避免把旧时间索引用到新 motion 上。

每个 shard 权重等于其中有效 motion 数，shard 内均匀抽样，得到全局 motion 均匀
采样。不是六类等权；main 的份额由有效文件数决定。旧训练配置默认保持不重采样。

## Teacher（单 B200，GPU 0）

启动新的 Teacher：

```bash
bash scripts/train_roban_full.sh
```

脚本自动定位仓库，默认 GPU `0`、单进程、16384 个环境、50000 rollout、
32 steps/rollout，`total_frames=26214400000`（按单卡计算）。
W&B disabled，TensorBoard 和本地 checkpoint 开启；启动时清除继承的
`LD_LIBRARY_PATH`，避免 Conda CUDA 库冲突。
输出目录为 `dataset/_audit/roban_ominisoma/teacher_full_r50000_<timestamp>`。
可覆盖 `CUDA_VISIBLE_DEVICES`（仅一个设备）、`MEMPATH`、`PER_RANK_ENVS`、
`ROLLOUTS`、`OUTPUT`；`DRY_RUN=1` 只打印命令，不创建训练目录。

仅进行短验证，保留正式课程预算：

```bash
STOP_AFTER_ROLLOUTS=100 \
OUTPUT="$PWD/dataset/_audit/roban_ominisoma/teacher_full_validation_v2" \
bash scripts/train_roban_full.sh
```

如需继承已有 Teacher 权重，用 `CHECKPOINT=/absolute/path/checkpoint_final.pt`。
当前 checkpoint 不保存完整优化器状态，因此属于加载权重继续学习，而非精确断点恢复。

## 历史三卡训练验证（4090）

本次先完成了全量 Teacher 的 100-rollout 三卡验证（从头初始化），未启动
50000-rollout 长任务，也未运行 Adapt/Finetune。验证输出：
`dataset/_audit/roban_ominisoma/teacher_full_validation_v1`。
详细指标见其中的 `training_validation_report.json`。

- 100 rollout / 39,321,600 global frames，正常退出；
- TensorBoard 118 个 tag、6,263 个标量点全部 finite；
- 第 1、25、50、75、100 rollout 的 DDP 参数和归一化状态差异均为 0；
- 累计 simulator recovery 17,434 次；未触发 PPO loss/gradient finite gate；
- PPO 采集与更新合计吞吐约 89,622 transitions/s（不含启动和 checkpoint I/O）；
- 第约 70 rollout 的 nvidia-smi 快照为 GPU 3/4/7：21,472/19,872/19,872 MiB，
  不是全程显存峰值；rank 0 CPU 峰值 RSS 约 11,742.5 MiB；
- 保存 `checkpoint_25.pt`、`checkpoint_50.pt`、`checkpoint_75.pt`、`checkpoint_final.pt`。
- 最终 checkpoint 两次 reload 的 deterministic action roundtrip 为 PASS，
  参考数据均保持在 CPU；详见同目录 `roundtrip_report.json`。

静态过滤不保证零 simulator NaN；本次未启用逐事件溯源，因此 TensorBoard
simulator events 使用 recovery 计数，不是独立事件审计。这是训练链路短验证，
不代表策略收敛，也未对优化器内部全部状态做逐步审计。

## 三阶段 pipeline

全量入口通过 `MODE=pipeline` 调用 `train.sh`，自动传入全量 groups；
Teacher/Adapt/Finetune 都使用相同过滤数据及重采样配置：

```bash
MODE=pipeline DRY_RUN=1 bash scripts/train_roban_full.sh
```

检查打印的三阶段命令后启动：

```bash
MODE=pipeline \
OUTPUT_ROOT="$PWD/dataset/_audit/roban_ominisoma/pipeline_full_v1" \
bash scripts/train_roban_full.sh
```

全量入口默认阶段预算：Teacher 50000、Adapt 10000、Finetune 10000。
可分别设置 `TRAIN_ROLLOUTS`、`ADAPT_ROLLOUTS`、`FINETUNE_ROLLOUTS`；
直接调用 `train.sh` 的 Teacher 默认预算仍为 100000。
pipeline 输出使用 `OUTPUT_ROOT`，从中间阶段启动使用 `START_STAGE` 和
`CHECKPOINT_PATH`（见使用说明）。

单 B200 全量三阶段短验证（每阶段两轮，自动衔接 checkpoint）：

```bash
MODE=pipeline STOP_AFTER_ROLLOUTS=2 \
OUTPUT_ROOT="$PWD/dataset/_audit/roban_ominisoma/b200_full_pipeline_validation" \
bash scripts/train_roban_full.sh
```

`STOP_AFTER_ROLLOUTS` 只限制实际运行轮数，不缩短各阶段课程的总预算。

## 单 B200 三阶段实测（2026-09-10）

使用上述短验证配置，16384 个环境、全量 27 shard / 115061 motion，
Teacher → Adapt → Finetune 正常退出，各完成两次更新并保存最终 checkpoint。
各阶段均验证 CPU 参考数据、reset 重采样和前一阶段 checkpoint 路径。

| 阶段 | 实际 frames | TensorBoard 标量点 | 仿真恢复累计次数 |
| --- | ---: | ---: | ---: |
| Teacher | 1048576 | 147 | 1262 |
| Adapt | 524288 | 63 | 725 |
| Finetune | 1048576 | 145 | 1241 |

三份 checkpoint 的 policy / vecnorm 各 79 个张量及所有标量点均为有限值。
验证报告：`dataset/_audit/roban_ominisoma/b200_full_pipeline_validation_20260910/validation_report.json`；
同目录各阶段包含 `console.log`、`cfg.yaml`、TensorBoard 和 `checkpoint_final.pt`。
启动脚本的 6 项 unittest 通过；额外配置测试因 `.venv` 缺少 pytest 未运行。
本次只验证训练链路与默认环境规模可运行，不证明策略收敛或长时间稳定性；
仿真仍发生非有限状态恢复，正式训练应继续观察恢复率和跟踪效果。

已成功的旧 50K Adapt 实际累计预算为 7543；需要匹配该预算可设置 `ADAPT_ROLLOUTS=7543`。
导出和部署仍按当前使用说明执行。
