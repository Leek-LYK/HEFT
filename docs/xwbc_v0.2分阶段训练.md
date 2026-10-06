# XWBC v0.2 分阶段训练

v0.2 先建立本机型控制能力，再把两种机型迁入共享 Human 学生，最后学习机器人命令入口。共享学生保留 v0.1 第二轮的 256 维 token、MLP `[512,512,256]`、12 个参考时刻、9 个本体历史时刻和 8 个实际动作历史。

当前独立工作树为 `/root/kube-user/lyk/project/HEFT_XWBC_v0.2`，分支 `research/xwbc/v0.2`。`.venv`、`dataset`、`logs` 通过链接使用原项目入口。v0.1 的工作树、进程、checkpoint 和数据保持原样。

## 数据准备

```bash
cd /root/kube-user/lyk/project/HEFT_XWBC_v0.2
WORKERS=2 bash scripts/run_xwbc_staged.sh prepare
```

候选清单位于 `dataset/xwbc/manifests/simple_5k_v02.jsonl`，派生数据位于 `dataset/xwbc/simple_5k_v02/`。一条 motion_id 对应三份同步动作。按与 v0.1 相同的命名家族和 seed=31 分配 train/val/test，再从满足简单动作条件的样本中抽取 4,000/500/500 条。

名称筛选排除跳跃、踢腿、翻滚、跪爬、坐躺、受伤、楼梯等动作；数值筛选检查时长、三侧帧数、根速度、关节速度、骨盆高度与躯干倾斜。上身动作额外限制根运动，浅蹲额外限制两种机器人的膝角。具体阈值和逐文件结论写入候选清单旁的 summary/audit 文件。

本次实际组成：

| 类别 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 走路 | 2,101 | 263 | 263 |
| 跑步 | 1,051 | 132 | 132 |
| 上身动作 | 840 | 105 | 105 |
| 浅蹲 | 8 | 0 | 0 |

满足本轮整段浅蹲条件的样本来自同一训练家族，全部留在训练划分。验证/测试评估覆盖前三类；浅蹲单独标注为有限训练覆盖，后续扩展时补充独立家族。总计 5,000 条，源文件和既有 v0.1 派生数据均保留。

## 阶段 A：两个独立特权教师

```bash
ROBOT=g1 OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/teacher_g1" \
  bash scripts/run_xwbc_staged.sh teacher
ROBOT=roban OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/teacher_roban" \
  bash scripts/run_xwbc_staged.sh teacher
```

两条命令可在独立终端并行执行。默认各 4,096 环境、32 步 rollout、10,000 次更新，每个教师约 13.1 亿 transitions。只使用对应机型命令库，不共享网络或优化器，不计算来源对齐损失，也不启用负载。

教师输入为本机型参考窗口、实际本体历史和特权状态。特权量包括参考关节/根误差、真实根线速度/高度及当前关键点位姿、速度。教师 actor/critic 复用 HEFT 的 `make_mlp`（LayerNorm、Mish）；actor 隐层为 `[1024,1024,512]`，输出本机型的高斯动作均值和可学习标准差。继续复用 HEFT 环境、PD、动作缩放、FK、奖励、终止和状态恢复；PPO 复用现有 GAE/裁剪损失接口。

评估教师（每份只评本机型路径）：

```bash
CHECKPOINT="$PWD/logs/v0.2/staged_run01/teacher_g1/checkpoint_final.pt" \
OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/teacher_g1/eval_val" \
  bash scripts/run_xwbc_staged.sh evaluate
```

Roban 同样执行。输出 `clips.jsonl`、`summary.json` 和 `gate.json`。2026-10-04 按用户明确的跨机型跟踪目标调整验收：完整验证集成功率 ≥90%、完成比例 ≥95%、恢复片段比例 ≤1%、根相对 MPKPE ≤0.10 m。配置 `acceptance.world_error_is_diagnostic: true` 将世界坐标 MPKPE 的 0.15 m 阈值保留为诊断项，超出时写入 gate 的 `diagnostics`，不阻止进入下一阶段；指标仍须为有限值。缺少此开关的历史配置继续执行原严格门槛。原 `eval_val/` 结果保持原义，本轮复用相同评分、按新验收规则生成 `eval_val_success_priority/`，记录原报告摘要及变更原因。

阶段 A 接受本实现的独立特权教师 checkpoint；现有 v0.1 共享学生不作为教师自动复用。复用已有阶段 A 教师时校验模型、数据、动作契约和完整验证集结果。

## 阶段 B：Human 共享学生与 DAgger

```bash
G1_TEACHER="$PWD/logs/v0.2/staged_run01/teacher_g1/checkpoint_final.pt" \
ROBAN_TEACHER="$PWD/logs/v0.2/staged_run01/teacher_roban/checkpoint_final.pt" \
G1_GATE="$PWD/logs/v0.2/staged_run01/teacher_g1/eval_val_success_priority/gate.json" \
ROBAN_GATE="$PWD/logs/v0.2/staged_run01/teacher_roban/eval_val_success_priority/gate.json" \
OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/student_human" \
  bash scripts/run_xwbc_staged.sh student
```

入口重新计算并校验阶段门槛，绑定 checkpoint SHA、数据版本、时间上下文和验证集 motion_id。部分片段 smoke 或未达标教师不会进入 B。

仅训练 Human 命令编码器、两机型本体编码器、共享主干和两个动作头。机器人命令编码器、critic 和策略标准差保持冻结，PPO 更新数为零。

每个控制步使用一个真实目标机器人状态：先读取本体历史，再分别计算学生 Human 动作和该机型教师的原生参考动作均值。教师额外状态也从同一环境池取得。标签是教师控制均值，随后通过同一 HEFT 动作接口执行；数据集参考关节角仅供教师输入和环境参考。

每种机型独立按有效动作维度求 MSE，然后对两机型等权平均。一个环境池只执行一个动作：由教师接管或学生执行决定，历史由 HEFT 实际施加动作缓冲生成，不维护教师影子轨迹。

默认前 20 次更新使用教师执行，随后 500 次更新将接管概率线性降至零。每轮都在实际访问状态上重新查询冻结教师，将状态与动作标签加入每机型 32,768 行的 uniform reservoir，再进行监督更新。缓冲保留累计访问状态的有界样本，并随 checkpoint 保存。该设计借鉴 [DAgger 的在线数据聚合思路](https://proceedings.mlr.press/v15/ross11a.html)，减少只在教师轨迹上拟合造成的状态分布偏移。

默认 B 为每机型 1,024 环境、16 步采集、2,000 次迭代，每轮 32 次监督更新。`mixture_reward` 是当时执行混合策略的奖励；纯学生效果以独立评估为准。额外记录当前访问状态上的动作 MSE、接管比例和累计聚合样本数。

B 评估只执行 `Human→G1/Roban`，没有教师接管。相对对应教师，成功率和完成比例分别允许下降最多 5 个百分点，根相对 MPKPE 允许增加 0.02 m，同时保持恢复片段比例 ≤1%。世界 MPKPE 增量超过 0.05 m 时记录诊断；当前配置不将其作为硬门槛。

## 阶段 C：固定 Human anchor，只训练来源入口

2026-10-04 正式 C 使用独立配置 `cfg/xwbc/v02/source_adaptation.yaml`。按用户确认的能力优先目标，B→C 准入采用两路成功率 ≥90%、完成比例 ≥95%、恢复片段 ≤1%、根相对 MPKPE ≤0.10 m；世界误差及相对原生教师的退化保留为诊断。原 B 评估保留，新判定基于同一完整验证生成 `student_human/eval_val_capability/`，并绑定原报告、动作评分和最终 checkpoint。这个绝对准入设置仅作用于 B；C 的六组合仍与已验收 B 的 Human 路径作相对比较。

```bash
CONFIG="$PWD/cfg/xwbc/v02/source_adaptation.yaml" \
STUDENT="$PWD/logs/v0.2/staged_run01/student_human/checkpoint_final.pt" \
STUDENT_GATE="$PWD/logs/v0.2/staged_run01/student_human/eval_val_capability/gate.json" \
OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/source_encoders" \
  bash scripts/run_xwbc_staged.sh sources
```

B 通过完整验证后才能进入 C。冻结 Human 编码器、两种本体编码器、共享主干和两个动作头，只更新 G1/Roban 命令编码器。模型使用固定物理量缩放与 LayerNorm，没有额外更新的运行均值/方差；冻结范围包括相关 LayerNorm 参数。

两项损失共同作用：

- 同一配对窗口的机器人 token 对齐固定 Human 的**原始 token**，保留幅值约束。
- 同一目标状态上，两种新入口分别拟合 Human 路径动作，按目标有效动作维度求均值并平衡两机型。

冻结执行模块仍参与前向和对输入 token 的反向传播，参数保持不变。每次保存与结束时校验冻结参数摘要。初始 token/action 权重均为 1，按验证结果调整研究配置。

闭环采集让 G1/Roban 新入口实际执行，在这些状态上查询冻结 Human 路径并继续聚合监督；Human 接管概率在前 300 次更新中降至零。默认容量与 B 相同。C 评估六组合，结果与已验证 B 的对应 Human 路径比较，使用相同退化容限。

## 阶段 D 与流程边界

D 已实现为显式 `finetune` 入口，使用 `cfg/xwbc/v02/finetune.yaml`；用户授权后从已通过完整六组合验证的 C 权重初始化。保留三个命令编码器，更新本体编码器、共享主干、动作头与 critic。B/C 继续使用原监督流程。

```bash
POLICY="$PWD/logs/v0.2/staged_run01/source_encoders/checkpoint_final.pt" \
POLICY_GATE="$PWD/logs/v0.2/staged_run01/source_encoders/eval_val/gate.json" \
OUTPUT_ROOT="$PWD/logs/v0.2/staged_run01/finetune_ppo" \
  bash scripts/run_xwbc_staged.sh finetune
```

首轮预算为每机型 1,024 环境、32 步采集、600 轮，其中前 100 轮只训练 critic，后 500 轮进行 PPO。actor 名义学习率 2e-5、critic 学习率 1e-4；固定动作标准差 0.1，clip 0.1，对冻结 C 动作均值的维度归一化 MSE 权重为 0.1。每个机型内环境槽按 `env_id % 3` 固定来源，使每轮六路都有样本，三来源份额最多相差一个环境；两个机型的损失等权。

PPO 使用每个机型的有效动作维度。对固定标准差的高斯策略计算解析 KL，并在每次 actor 更新后回溯参数增量，检查实际应用的 minibatch KL ≤0.02。必要时停止本轮余下 minibatch；完全拒绝的 actor 步同时恢复 actor Adam 状态。critic 独立更新。前 100 轮核验 actor 均值参数未改变，训练全程校验来源编码器与探索标准差冻结。

D checkpoint 保存两组优化器和 RNG，可用相同 `finetune` 入口及 `RESUME` 在新目录续训；只允许调整总更新目标与 checkpoint 间隔。恢复重开仿真回合。D 使用独立的 C 动作基线，不读取 A 教师监督或 B/C replay。

训练结束进行同一验证集的六组合评分。D 每条路与 C 的**同一路**比较，`summary.json` 另存 `comparison_to_C`。`gate.json` 继续检查完整性、退化容限及恢复，世界误差为诊断；通过门槛说明保持能力，是否取得改进还要比较实际成功率和困难类别结果。

TensorBoard 选择 `finetune_ppo/tensorboard`，先看 `critic_warmup`、`ppo_update`、`value_loss`、`explained_variance`；PPO 开始后看 `max_post_step_kl`、`actor_step_scale`、`anchor_action_mse` 和逐来源训练回合统计。`train_episode_success_rate` 使用随机训练起点，与完整片段验证的范围不同。

实现与正式启动记录见 [阶段 D 记录](../科研/v0.2/06_阶段D实现与正式训练记录.md)。

## 运行、恢复与复现

主要配置为 `cfg/xwbc/v02/staged.yaml`，`CONFIG` 可以选择实验配置；`NUM_ENVS`、`UPDATES`、`SEED`、`MAX_MINUTES` 可覆盖运行预算。`DRY_RUN=1` 只预览命令。`RESUME` 使用同阶段 checkpoint，恢复网络、优化器、聚合缓冲及随机状态，仿真回合重新开始；教师或 Human anchor 摘要变化时使用新实验。

每份运行保存配置、源码快照、动作契约、逐轮 JSONL、TensorBoard 和 checkpoint。评估记录协议版本、完整/部分范围、各类指标、模型和数据摘要。阶段门槛与纯策略评估分开于混合采样训练曲线。

```bash
.venv/bin/tensorboard --logdir logs/v0.2/staged_run01 --host 127.0.0.1 --port 6008
```

`tests/test_xwbc_staged.py` 验证阶段约束；`scripts/verify_xwbc_staged.py` 是有界 GPU 工程检查，使用未验收模型验证数据流，不产生阶段通过标记或正式研究结论。
