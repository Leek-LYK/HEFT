# XWBC 六组合评估协议与证据边界

本说明定义现有 `xwbc-v1` 实现的评分含义、比较条件和验收边界，供后续冻结模型评估复用。运行入口见 [正式实验说明](xwbc正式实验.md)，历史实现及运行证据见 [v0.1 基线](xwbc_v0.1基线.md)。本说明不变更成功定义、抽样、指标字段或历史统计。

实现位置：[六路评估及聚合](../active_adaptation/learning/xwbc.py)、[环境及指标](../active_adaptation/envs/xwbc.py)、[配对命令库](../active_adaptation/utils/xwbc_data.py)。CPU 语义验收入口：[test_xwbc_evaluation.py](../tests/test_xwbc_evaluation.py)。

## 片段、成功与完成比例

同一冻结策略在 `human/g1/roban -> g1/roban` 六路上评分。每条片段从参考第 0 帧初始化，使用确定性动作；当前初始化为目标参考加 2 厘米抬升、无姿态/速度噪声。环境配置随评分保存，目标资产定义关键点。CPU 合成测试检验评分逻辑；真实初始化及物理轨迹仍需仿真验收。

令片段帧数为 L、当前参考帧为 t。每个进入评分循环时仍 active 的行先执行一步，再读取指标、累计误差并增加 `steps`。该行在 `done` 或 `t >= L-1` 时结束，仅输出一次。

| 当步状态 | success | reason |
|---|---|---|
| 完成，无恢复；即使同帧普通终止或超时 | true | complete |
| 恢复事件；包括同帧完成 | false | nonfinite |
| 未完成，普通终止 | false | 触发的 termination 名称，以逗号连接 |
| 未完成，仅超时、无 termination 名称 | false | 空字符串（现有输出） |

`completion = min(1, steps / max(1, L-1))`。它记录已计入控制步的比例，不替代成功判定；同帧完成并恢复时可以是 `completion=1` 且 `success=false`。成功按完成优先规则判断，不要求同帧没有普通失败信号。因此它不能单独解释为全程无失败信号。

纯提前超时的空 `reason` 是现有分类信息限制，不能按空字符串推断成功；必须读取 `success` 和 `completion`。当前评估的最大回合长度设为最长片段帧数加 2；CPU 测试注入提前超时以锁定边界语义，不证明真实数据评估会产生提前超时。

## 误差、权重与提前失败

每片段误差为 `sum(active 步的指标读数) / steps`。聚合误差为 `sum(片段均值 * steps) / sum(steps)`，也就是被计入控制步的等权平均；较长轨迹权重更大。这里“有效步”仅指进入累计器的 active 步，并不保证它是未恢复的物理跟踪步。

成功率与平均完成比例则按片段等权。比如 9 步、误差 4 的片段与 1 步、误差 1 的片段，聚合误差是 3.7，片段等权误差才是 2.5。提前失败只留下已执行前缀，缺失后续误差不会受到惩罚；低误差不能独立说明完整片段跟踪更好。比较时联合报告成功率、完成比例、误差、步数及失败分类。

恢复终止步也计入 `steps`、完成比例和全部误差。`XWBCPool.step()` 返回后才调用 `metrics()`，因此读取的是恢复边界后的可见状态。底层恢复会重置/恢复部分状态并刷新全环境；不能把该读数当成未经恢复的失效轨迹误差。CPU 脚本注入前一步读数 2、恢复终止步读数 20，确认均值为 11；这证明计数时序，不证明 GPU 上实际恢复读数的值、偏差方向或偏差大小。

`recovery_clips` 是 `reason == nonfinite` 的片段数，不是训练累计恢复事件数，也不是模拟器底层恢复尝试次数。

## 指标名称、参考点与坐标轴

设关键点实际位置为 p、根位置为 r，参考为 p_ref、r_ref，环境原点为 o。参考位置先加 o 与实际世界坐标对齐。每步关键点指标是选定关键点的欧氏距离均值。

| 输出字段 | 每步定义与单位 |
|---|---|
| mpkpe_world_m | mean(norm(p - (p_ref + o)))，米 |
| mpkpe_root_relative_m | mean(norm((p-r) - (p_ref-r_ref)))，米，两个差向量仍在世界坐标轴表达 |
| velocity_mps | 世界坐标关键点线速度差的范数均值，米/秒 |
| root_rotation_rad | 2 * acos(clamp(abs(dot(q, q_ref)), 0, 1))，弧度；wxyz 单位四元数，q 与 -q 等价 |
| root_position_m | norm(r-r_ref-o)，米 |

`mpkpe_root_relative_m` 消除根平移及环境原点，保留朝向差；不旋转到各自根坐标系。把根与关键点共同平移 [3,4,0]，世界及根位置误差是 5，根相对误差是 0。保持相同根位置，将长 1 米的关键点臂绕 Z 转 90 度，即使各自局部姿态一致，世界轴根相对误差仍为 sqrt(2)。CPU 测试直接调用真实指标函数验证这些数值。

字段名为历史兼容名称。`root_position_m` 在 `clips.jsonl` 和 `summary.json` 中保留，现有 `matrix.csv` 不含该列；其他四项误差列与 JSON 聚合一致。后续增加根朝向对齐指标应使用独立名称和清楚公式，并继续保留以上字段。

## 前 N 条、类别覆盖与统计

`CommandBank` 依 schema 的 shard 顺序及各 `motions.json` 行顺序拼接，不在加载时按类别重排。`--max-motions N` 使用这个顺序的前 N 条，超过可用条数时取全部；它没有类别均衡或随机抽样含义。编译小数据时可采用类别轮转，但这不能替代评估入口对任意已有库的覆盖检查。

`categories` 只包含实际被评估类别，`macro_category_success` 是这些类别成功率的等权均值；缺失类别不自动补零。例如 main 两条成功率 0.5、jump 一条成功率 1，全体成功率是 2/3、类别宏均值是 0.75；只取前两条时，宏均值为 0.5 且只覆盖 main。不能把这种小样本宏均值称作完整类别宏均值。

正式比较必须使用相同 split、动作 ID 集合及顺序、类别/家族覆盖、环境设置、确定性策略和初始化条件。先核对每路片段数量及逐类别/家族数量；小样本结果用于链路与边界验收，其覆盖不能外推为完整验证集或测试集结果。数据摘要相同也不能替代前 N 范围一致。

现有置信区间仅针对成功率：以动作家族为 cluster，固定 RNG 42 重采样 1000 次，并在抽到的家族内保留全部片段，再按片段计成功率；它不是家族成功率的等权平均，也不是误差的置信区间。家族数少时仍需明确覆盖和解释范围。

## CPU 证据与 GPU 验收

| 关注点 | 可由 CPU 回归建立的事实 | 仍需实证的范围 |
|---|---|---|
| 同帧事件优先级 | 六路均按完成优先、恢复例外评分；反驳“同帧普通终止必然失败”和“完成比例 1 必然成功” | 真实片段事件频率、阈值与物理稳定性 |
| 误差聚合 | 按计入步数加权，失败前缀与终止步均参与 | 真实失败前缀引入的排序偏差有多大 |
| 恢复终止步 | 返回的当步读数计入均值及分母 | GPU 恢复状态、误差偏差方向/大小、是否仍有异常读数 |
| 前 N 覆盖 | 保留库顺序；缺失类别不进入宏均值 | 实际评估范围的类别/家族覆盖，需获授权后核对 |
| 根相对指标 | 去平移、保留世界轴朝向差；反驳“已在各自根坐标系对齐” | 各真实机型/动作的朝向误差来源 |
| 已结束/补齐行 | 不直接产生额外记录或累计误差；评分掩码有效 | 这些行仍随整个池 step；跨环境接触、全局 forward/sense、恢复刷新或其他共享状态的影响，必须留作 GPU 经验问题 |

CPU mock 不能证明 inactive 行不会污染 active 行，也不能证明存在这种污染。GPU 验收应在独立新运行目录、固定模型与片段下，对相同 active 轨迹比较单环境与批量环境、不同批大小/补齐安排以及其他行提前终止/恢复的条件，并关联恢复诊断。检验事件分类、完成度和轨迹/误差差异；先规定可接受数值容差并记录仿真非确定性。该设计需要单独授权的 GPU 预算，本 CPU 验收不执行它。

## 兼容性与回归入口

修改评分代码前先运行下面的合成 CPU unittest，不执行训练、checkpoint 重载或真实验证数据。临时输出及缓存使用新建日志目录；保留历史运行。

```bash
mkdir -p logs/v0.2
XWBC_CPU_RUN=$(mktemp -d "$PWD/logs/v0.2/cpu_eval_semantics_XXXXXXXX")
mkdir -p "$XWBC_CPU_RUN"/{tmp,cache,warp,torch,torch_extensions,triton,cuda,matplotlib,numba}
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
TMPDIR="$XWBC_CPU_RUN/tmp" TMP="$XWBC_CPU_RUN/tmp" TEMP="$XWBC_CPU_RUN/tmp" \
XDG_CACHE_HOME="$XWBC_CPU_RUN/cache" WARP_CACHE_PATH="$XWBC_CPU_RUN/warp" \
TORCH_HOME="$XWBC_CPU_RUN/torch" TORCH_EXTENSIONS_DIR="$XWBC_CPU_RUN/torch_extensions" \
TRITON_CACHE_DIR="$XWBC_CPU_RUN/triton" CUDA_CACHE_PATH="$XWBC_CPU_RUN/cuda" \
MPLCONFIGDIR="$XWBC_CPU_RUN/matplotlib" NUMBA_CACHE_DIR="$XWBC_CPU_RUN/numba" \
.venv/bin/python -B -m unittest discover -s tests -p test_xwbc_evaluation.py -v \
> "$XWBC_CPU_RUN/unittest.log" 2>&1
```

验收要求：六路事件评分、提前终止解释、恢复终止步计数、步数加权、补齐/结束行输出掩码、前 N/宏均值覆盖、命令库顺序与真实指标几何全部通过。测试对评分入口的 CUDA 初始化、策略加载及环境池进行替换；不实例化真实模拟器，合成 shard 使用 CPU。

后续如要排除恢复终止步、改变成功定义、采用均衡抽样或增加朝向对齐指标，应先明确协议版本与比较规则。保留现有字段和旧协议结果，独立报告新定义并对同一冻结模型/片段做配对比较。旧逐片段均值通常不能还原恢复前逐步误差；不能只重算一个总表就声称具有新协议结果。已保存的 v0.1 统计及运行保持原义。
