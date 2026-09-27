# 大规模机器人强化学习训练中的 NaN 处理实践

本文面向其他机器人训练项目的开发人员，完整说明一套已用于大规模并行强化学习训练的非有限值处理方式。案例为 Roban 的 5 万源动作文件训练，整理日期为 2026-09-17。下文同时覆盖 NaN、正无穷和负无穷，统一称为“非有限值”。

文档以该训练实际实现为准，独立展开处理流程、关键代码和验证结果，不依赖原项目的目录、导入路径或外部代码链接。代码中仅对框架接口作通用化命名；凡涉及调度或框架适配的片段均标为流程摘录，不构成可直接运行的完整训练器。示例没有额外加入尚未实现的自动跳过更新、参数回滚或全面数值修复。

## 1. 处理原则与适用边界

实际方案分为两类行为：

| 异常来源 | 实际处理 |
| --- | --- |
| 个别并行环境的仿真状态、传感器或接触力异常 | 标记、隔离、有限次数恢复，并终止该环境当前回合 |
| 策略输出、关节目标、控制输入异常 | 报错停止，不能当作普通仿真异常掩盖 |
| 未标记异常的原始观测、归一化输入或输出异常 | 报错停止 |
| PPO 或 estimator 的损失、梯度异常 | 在相应反向传播或优化器更新之前报错停止 |
| 恢复最多尝试 4 次仍不能得到有限仿真边界 | 报错停止 |

整体数据流为：

```text
源数据校验与四元数规范化
  → 仿真容量配置
  → 每个物理子步前后检查
  → 异常环境隔离、恢复、全局复查
  → 策略步末尾刷新后再次检查
  → 异常 reward / 原始观测清零、强制终止
  → 排除异常行的观测归一化
  → PPO / estimator 损失与梯度检查
  → 更新与跨进程一致性检查
```

目标是阻止局部仿真异常扩散到训练状态。成功运行不代表仿真从此不产生 NaN；恢复次数本身就是需要持续观察的指标。

## 2. 训练数据入口：先保证参考数据有限

转换阶段检查必需字段、数组形状、帧率及数值有限性。对四元数另外执行：

1. 拒绝 NaN/Inf。
2. 拒绝模长不大于 `1e-6` 的四元数。
3. 拒绝超过配置容差的严重模长偏差。
4. 对可接受的四元数归一化。
5. 修正相邻帧符号，使相邻单位四元数的点积非负。

核心计算如下，`severe_norm_deviation` 由转换器配置提供：

```python
import numpy as np

def normalize_quaternions(quat, severe_norm_deviation):
    # quat: [T, ..., 4]，分量顺序必须由数据接口明确。
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim < 2 or quat.shape[-1] != 4:
        raise ValueError("invalid quaternion shape")
    if not np.isfinite(quat).all():
        raise ValueError("quaternion contains NaN/Inf")

    norms = np.linalg.norm(quat, axis=-1)
    if np.any(norms <= 1e-6):
        raise ValueError("near-zero quaternion")
    if np.max(np.abs(norms - 1.0)) > severe_norm_deviation:
        raise ValueError("severe quaternion norm deviation")

    normalized = quat / norms[..., None]
    adjacent_dot = np.sum(normalized[1:] * normalized[:-1], axis=-1)
    sign_steps = np.where(adjacent_dot < 0.0, -1.0, 1.0)
    factors = np.concatenate([
        np.ones((1,) + sign_steps.shape[1:]), sign_steps
    ], axis=0)
    normalized *= np.cumprod(factors, axis=0)[..., None]
    return normalized.astype(np.float32)
```

转换之后还要检查最终加载张量和运动学结果。案例中 50,000 个源文件经过处理，形成 45,213 个有效动作段、18,026,422 帧；最终关节位置、根位置、根四元数张量均通过有限性检查。转换报告的源文件读取/转换失败数为 0，另有 4,802 个源文件被既有过滤流程跳过，不能把这些跳过文件解释成“含 NaN 的坏文件”。

数据有限只是入口条件；有限的参考动作仍可能引发仿真求解异常。

## 3. 仿真容量：解决约束溢出，但不把它当成唯一根因

案例中的大动作集曾需要 578 条约束，超过原默认 `njmax=500`，因此针对该机器人设置：

```yaml
sim:
  nconmax: 200
  njmax: 768
```

实际策略步长为 `0.02 s`，物理步长为 `0.005 s`，每个策略步对应 4 个物理子步。

`768` 是案例配置，不是其他机器人通用的安全值。后续故障在没有约束溢出警告时仍然发生，因此扩大容量后仍保留全部非有限值检查。

## 4. 检查范围与状态生命周期

### 4.1 每个环境维护一个隔离标记

```python
recovery_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
recovery_count = 0
```

`recovery_mask` 在每个策略步的仿真更新开始时清零，在该步所有物理子步、最终刷新和恢复过程中累计。一个环境一旦被标记，就不会在该 transition 中提前解除隔离。

累计恢复计数在某环境本次 transition 首次被标记时增加。同一环境在该 transition 中反复恢复，不等于增加多个首次异常事件。

### 4.2 真实检查字段

| 字段 | 含义 |
| --- | --- |
| `qpos` | 广义位置，包括浮动根位姿 |
| `qvel` | 广义速度 |
| `qacc` | 广义加速度 |
| `qacc_warmstart` | 求解器热启动加速度状态 |
| `ctrl` | 写入仿真器的控制输入 |
| `sensordata` | 仿真传感器缓冲区 |
| `action` | 策略动作 |
| `joint_target` | 关节位置目标 |
| `actuator_force` | 执行器力 |
| `contact_force` | 接触力 |

各字段首维必须对应环境数。快速检查按环境拼接并归约，只有异常出现时再计算逐字段诊断信息：

```python
import torch

def all_finite_by_env(fields, num_envs):
    flattened = [value.reshape(num_envs, -1) for value in fields.values()]
    return torch.isfinite(torch.cat(flattened, dim=1)).all(dim=1)
```

接触力传感器可能缓存上一轮拼接结果。实际实现会在边界检查前调用可用的缓存失效接口，再读取当前力张量，避免检查到旧数据。没有可用力张量时，以 `[num_envs, 0]` 空张量代表该项，不凭空构造力值。

### 4.3 检查顺序

以下为实际顺序的流程摘录，`read_fields()`、回调和数据写入由仿真框架适配：

```text
策略步开始：清空 recovery_mask
对每个物理子步：
    保存 qpos_before
    应用动作、执行 pre-step 回调、写入仿真控制
    将已隔离环境的 ctrl 清零
    检查 action / joint_target / ctrl；非有限则立即报错
    检查全部字段；对新异常环境标记并恢复
    sim.step()
    检查全部字段；对所有异常环境恢复，对新异常环境计数
    scene.update()
    执行 post-step 回调，包括传感器/历史更新

保存刷新前 qpos
sim.forward()
sim.sense()
检查全部字段；标记并恢复异常环境
随后才继续命令、奖励和观测计算
```

检测与恢复必须发生在对应的传感器历史回调消费异常缓冲区之前。最终 `forward()/sense()` 也可能独立产生非有限值，因此不能只检查 `sim.step()` 的输出。

## 5. 最关键的恢复修复：全局刷新后重新检查所有环境

### 5.1 故障传播链

历史故障中，训练在完成 7 个 rollout 后触发了策略步前控制输入检查。当时没有约束溢出警告。更早的复现检查在私有观测的足部接触状态中截获了异常。

定位出的传播路径是：

```text
环境 A 出现异常
  → 恢复 A
  → forward / sense 重算所有环境
  → 环境 B，或本 transition 中此前恢复过的环境，产生新异常
  → 单次恢复没有发现这些异常
  → 接触力 / 传感器历史读入异常
  → 异常进入观测、归一化和策略输入
```

因此，“重置刚才报错的环境”不足以建立一个有限的仿真边界。

### 5.2 实际恢复算法

下面展开核心实现。省略了事件日志写入，保留状态修改、检查、累计隔离和失败行为。调用者应在首次调用之前标记初始异常环境并增加计数。

```python
def recover_nonfinite_envs(ctx, env_ids, qpos_before):
    """流程摘录；ctx 提供 sim、scene、read_fields 等框架接口。"""
    for attempt in range(4):
        ctx.sim.reset(env_ids)

        # 只恢复确实有限的旧位置；其余保留 sim.reset 给出的状态。
        finite_old_qpos = torch.isfinite(qpos_before[env_ids]).reshape(
            env_ids.numel(), -1
        ).all(dim=1)
        if bool(finite_old_qpos.any().item()):
            restore_ids = env_ids[finite_old_qpos]
            ctx.sim.data.qpos[restore_ids] = qpos_before[restore_ids]

        ctx.sim.data.qvel[env_ids] = 0.0
        ctx.scene.reset(env_ids)
        ctx.sim.forward()
        ctx.sim.sense()

        # 清理累计隔离集合，不能只清理本轮刚重置的 env_ids。
        quarantined = ctx.recovery_mask.nonzero().reshape(-1)
        ctx.sim.data.qacc[quarantined] = 0.0
        ctx.sim.data.qacc_warmstart[quarantined] = 0.0
        ctx.sim.data.sensordata[quarantined] = 0.0

        fields = ctx.read_fields()
        invalid = ~all_finite_by_env(fields, ctx.num_envs)
        if not bool(invalid.any().item()):
            return

        bad_controls = [
            name for name in ("action", "joint_target", "ctrl")
            if not bool(torch.isfinite(fields[name]).all().item())
        ]
        if bad_controls:
            raise FloatingPointError(
                f"nonfinite control during recovery: {bad_controls}"
            )

        newly_invalid = invalid & ~ctx.recovery_mask
        if bool(newly_invalid.any().item()):
            ctx.recovery_mask.logical_or_(newly_invalid)
            ctx.recovery_count += int(newly_invalid.sum().item())
            # 诊断模式在此记录 recovery_refresh_nonfinite 事件。

        env_ids = invalid.nonzero().reshape(-1)

    raise FloatingPointError(
        "simulator recovery did not reach a finite boundary after 4 attempts"
    )
```

这段逻辑不是把所有状态粗暴替换为零：有限的旧位置尽可能保留；速度归零；特定加速度和传感器临时缓冲区只对已隔离环境清理；其他异常仍需通过复查并恢复。

结束当前回合并重置环境时，框架先执行正常环境重置和 reset 回调，然后对本次恢复过的环境再次清零 `qacc/qacc_warmstart/sensordata`，防止临时状态进入替代回合。

## 6. 终止样本处理：reward 和观测隔离，切断回合递推

### 6.1 清零异常行

各奖励组完成计算后，将恢复行清零，再执行时间步缩放和统计累计。奖励组内部写入的奖励统计也有恢复行屏蔽。

各原始观测组完成计算后，同样对恢复行清零，并检查其余数据：

```python
reward = reward.masked_fill(recovery_mask.unsqueeze(1), 0.0)

observation.masked_fill_(recovery_mask.unsqueeze(1), 0.0)
if not bool(torch.isfinite(observation).all().item()):
    raise FloatingPointError("raw observation nonfinite before VecNorm")

vecnorm_valid = (~recovery_mask).unsqueeze(1)
```

原始观测错误报告还会包含观测组、时间戳、部分异常索引和异常组件名称，便于继续定位。

清零的是原始终止观测；零经过均值/标准差变换后不一定仍为零。要求是最终归一化输出有限，而不是要求最终输出恒等于零。

### 6.2 恢复环境立即终止

实际普通终止条件有前 5 步宽限期，但恢复事件必须绕过它：

```python
terminated = (
    ordinary_terminated & (episode_length > 5).unsqueeze(1)
) | recovery_mask.unsqueeze(1)

done = terminated | truncated
```

恢复异常标记为 `terminated`，不能只记作时间限制 `truncated`。在正常有限输入下，GAE 的计算为：

```python
nonterm = 1.0 - terminated.float()
nondone = 1.0 - done.float()
gae = 0
for t in reversed(range(num_steps)):
    next_v = next_value[:, t] * nonterm[:, t]
    gamma_t = discount[:, t] * gamma
    delta = reward[:, t] + gamma_t * next_v - value[:, t]
    gae = delta + gamma_t * gae_lambda * nondone[:, t] * gae
    advantages[:, t] = gae
returns = advantages + value
```

这会切断下一状态价值 bootstrap 和跨回合优势递推。但乘以零无法消除一个已经存在的 NaN，因此该逻辑依赖前面的有限观测边界；它不是独立的 NaN 修复器。

### 6.3 没有把整个异常 transition 从 PPO 删除

当前策略是形成一个 reward 为零、终止观测有限的终止样本，并从观测归一化统计中排除该行。PPO 的有效样本掩码仍使用 `~is_init`，不是将 `recovery_mask` 作为完整样本删除条件。

这意味着恢复会改变局部轨迹的训练语义，不能将其描述为对训练数据完全无影响。

## 7. 观测归一化：阻止单行异常污染全局统计

### 7.1 先检查所有输入，再修改任何统计

归一化包装层读取有效行标记，并在调用基础归一化之前处理所有输入键：

```python
def guarded_normalize(batch, input_keys, output_keys, valid, normalize_impl):
    """流程摘录：normalize_impl 的统计更新必须使用同一个 valid。"""
    for key in input_keys:
        value = batch.get(key)
        if value is None:
            continue
        if valid is not None and value.shape[0] == valid.numel():
            mask = valid.reshape(
                valid.numel(), *([1] * (value.ndim - 1))
            ).bool()
            value = torch.where(mask, value, 0.0)
            batch[key] = value
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"nonfinite input to VecNorm: {key}")

    result = normalize_impl(batch)
    for key in output_keys:
        value = result.get(key)
        if value is not None and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"nonfinite output from VecNorm: {key}")
    return result
```

关键是所有输入键都先检查完，再开始更新运行统计。否则前一个键已经更新，后一个键才失败，会留下部分修改的归一化状态。

### 7.2 用选择操作排除异常行

不能写 `value * valid` 来屏蔽 NaN，因为 `NaN * 0 = NaN`。以下展示二维观测 `[N, D]` 对应的统计计算；实际包装层还支持按参考统计形状归约：

```python
def masked_moments(value, valid):
    # valid: [N, 1]
    mask = valid.reshape(value.shape[0], 1).bool()
    safe = torch.where(mask, value, torch.zeros_like(value))
    value_sum = safe.sum(dim=0)
    value_ssq = safe.square().sum(dim=0)
    count = mask[:, 0].sum().to(value.dtype)
    return value_sum, value_ssq, count
```

归一化统计的更新规则为：

```python
running_sum = decay * running_sum + value_sum
running_ssq = decay * running_ssq + value_ssq
running_count = decay * running_count + count

mean = running_sum / running_count
std = (running_ssq / running_count - mean.square()).clamp_min(eps).sqrt()
normalized = (value - mean) / std.clamp_min(eps)
```

被隔离的零行既不参与数值累计，也不增加样本数。对称归一化还会将有效统计作镜像变换：数值和应用带符号变换，平方和仅作对应排列，样本数翻倍。

当前实现没有在上述 `running_count` 的除法处另加零计数保护。如果从未建立有效统计且全批被排除，不能假设一定能够正常归一化；输出有限性检查会暴露异常。这里保留实际边界，不在示例中偷偷补充新的行为。

## 8. 策略、损失、梯度：使用硬失败保护

### 8.1 控制输入检查不依赖诊断模式

每个物理子步写入控制之后执行：

```python
bad = [name for name, value in {
    "action": action,
    "joint_target": joint_target,
    "ctrl": ctrl,
}.items() if not bool(torch.isfinite(value).all().item())]

if bad:
    raise FloatingPointError(f"non-finite pre-step control tensors: {bad}")
```

这类错误不能用重置仿真器掩盖。该检查属于普通训练保护，即使没有逐事件日志记录器也会执行。

### 8.2 损失与梯度检查使用跨 rank 共同判定

```python
import torch.distributed as dist

def distributed_enabled():
    return dist.is_available() and dist.is_initialized()

def require_all_ranks_finite(label, value):
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if distributed_enabled():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f"non-finite {label} on at least one rank")

def require_finite_gradients(label, modules, device):
    finite = torch.ones((), device=device, dtype=torch.int32)
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    finite.zero_()
                    break
        if not bool(finite.item()):
            break
    if distributed_enabled():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f"non-finite {label} gradient on at least one rank")
```

参与同一训练分支的 rank 必须按一致顺序调用集合通信。环境侧的异常检查仍是本地抛异常，并未被全部改造成这样的 collective；不能宣称所有异常都具有同步退出协议。

### 8.3 实际优化顺序

PPO 更新顺序如下：

```text
清空 actor / critic 优化器梯度
→ 跨 rank 检查 PPO 总损失
→ backward
→ 跨 rank 检查 actor、critic、参与更新的 encoder 梯度
→ 对参与更新的 encoder 梯度作最大范数 1.0 裁剪
→ 对 actor 梯度作最大范数 1.0 裁剪
→ actor 优化器 step
→ 将 actor 标准差参数限制在初始化噪声上限以内
→ 对 critic 梯度作最大范数 1.0 裁剪
→ critic 优化器 step
```

标准差限制是上限截断，不能将它描述成完整的上下界约束或 NaN 修复。

estimator 分支则执行：清空梯度 → 检查损失 → backward → 检查梯度 → optimizer step。当前该分支没有上述 PPO 的梯度裁剪步骤。

没有实现“损失异常时跳过本批继续训练”，也没有实现“优化器更新后发现异常再自动回滚”。梯度裁剪前有逐元素有限性检查，但当前没有额外将裁剪返回的总范数作为独立硬门控。

## 9. 多 GPU 统计同步与一致性验证

参与训练的 Teacher actor、Student actor、特权信息 encoder、critic 和适配模块使用 DDP。这里的统计同步与状态检查用于避免不同 rank 的训练状态漂移，不能替代 NaN 检查。

### 9.1 VecNorm 同步

每次训练更新开始前，遍历归一化的嵌套统计张量；浮点/复数统计跨 rank 求和再除以 rank 数，其他张量从 rank 0 广播。统计键按固定顺序遍历，保证 collective 顺序一致。

```python
for key in sorted(stats.keys(True, True), key=str):
    value = stats.get(key)
    if not isinstance(value, torch.Tensor):
        continue
    if value.is_floating_point() or value.is_complex():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(dist.get_world_size())
    else:
        dist.broadcast(value, src=0)
```

这是当前对运行统计的同步方式，并非每次环境步后立即同步。

### 9.2 ValueNorm 同步

ValueNorm 更新先计算本地 return 的和、平方和、样本数，再分别跨 rank 求和，以全局样本数计算均值和二阶矩：

```python
batch_sum = returns.sum(dim=sample_dims)
batch_sq_sum = returns.square().sum(dim=sample_dims)
# batch_count 是被 sample_dims 归约的样本数，不包括特征维。
for stat in (batch_sum, batch_sq_sum, batch_count):
    dist.all_reduce(stat, op=dist.ReduceOp.SUM)
batch_mean = batch_sum / batch_count.clamp_min(1)
batch_sq_mean = batch_sq_sum / batch_count.clamp_min(1)
```

运行统计采用指数更新，去偏分母下限为 `1e-5`，去偏方差下限为 `1e-2`。这些下限避免有限数值情况下的极小分母，不会把 NaN 自动变成有限值。

### 9.3 一致性检查的真实含义

初始化附近和每 25 次更新检查参数与归一化状态的跨 rank 差异；检测到非零差异时报错，不静默覆盖其他 rank 来掩盖问题。

检查使用聚合签名：参数和、平方和、绝对最大值以及参数加权和；归一化状态使用和、平方和、绝对最大值。所谓 `max_gap=0` 是这些签名的差异为零，不是对全部参数逐元素做了相等证明。

## 10. 可选事件诊断：追踪来源与异常频率

### 10.1 正式训练与诊断训练的区别

正式训练开启仿真恢复、观测保护和控制/损失/梯度硬检查。逐事件诊断是显式启用的 pilot 功能，不是正式长训练默认配置。

默认诊断配置如下：

```yaml
nonfinite_diagnostics:
  enabled: true
  detailed_event_limit: 200
  rate_min_policy_transitions: 100000
  max_event_rate: 1.0e-4
  max_motion_frame_repeats: 10
  max_events_per_iteration: 20
```

上述阈值仅描述诊断配置默认值；具体实验可以覆盖，不能反推所有历史运行都采用相同阈值。

### 10.2 每条事件记录什么

记录内容包括：

- 策略迭代、策略帧、物理子步、全局物理 tick、环境编号和 episode 长度；
- 动作编号、源动作标识、参考动作帧和命令时间；
- 根高度、根四元数模长、最低身体高度；
- 原始/应用动作、关节目标、控制量、速度、加速度和力的最大绝对值；
- 仿真前后各字段是否有限、异常字段列表、首个异常字段及其展平索引；
- 是否只涉及 `qacc_warmstart`，是否涉及核心状态；
- 是否真的执行过仿真步，以及异常发生阶段。

阶段包括 `pre_step_nonfinite`、`post_step_new_nonfinite`、`post_refresh_new_nonfinite` 和 `recovery_refresh_nonfinite`。其中“首个异常字段”是固定字段遍历顺序中的第一项，不是时间分辨率足以证明的物理根因。

详细事件写入 JSONL，汇总写入 JSON。最多写入前 200 条详细事件，但后续事件仍参与汇总计数。非有限标量以字符串等严格 JSON 可表示形式保留，不将非法 JSON 数字写入文件。

### 10.3 停止阈值与统计口径

事件率定义为：

```text
事件率 = 累计事件数 / 累计环境 transition 数
```

每执行一批策略 transition，分母增加 `num_envs`；每执行一批物理子步，物理子步统计也增加 `num_envs`。因此它不是“异常数 / rollout 数”。

默认行为是累计至少 100,000 个环境 transition 后，事件率超过 `1e-4` 请求停止；同一动作/参考帧次数超过 10，或单轮事件超过 20，也请求停止。诊断记录器还区分仿真前控制异常并请求停止；普通训练本身已有独立控制硬检查，可能在写入逐事件记录之前就抛出异常。

正式训练在未开启记录器时，仍记录累计恢复数。某些训练指标的“累计 simulator events”在没有记录器时使用累计恢复数作为替代来源，因此不能拿它当成完整逐事件溯源统计。

## 11. 保存、续训和历史验证

### 11.1 保存训练状态

配套修复包括：checkpoint 显式保存 VecNorm；验证重新加载后的确定性动作一致；本地周期和最终 checkpoint 不再依赖 W&B；启动器保留完整标准输出和错误输出。

历史 Teacher 使用过 W&B offline，本地保存与后续关闭 W&B 的训练方式需要区分。W&B 开关不属于 NaN 修复机制。

当前 checkpoint 不包含完整优化器恢复状态，不能把加载模型继续训练描述为精确断点恢复；也没有将 checkpoint 用作每次 NaN 后的自动回滚机制。

### 11.2 回归测试覆盖的故障

专门构造的测试覆盖：

1. 首次恢复环境 A 时，全局刷新让环境 B 的传感器产生 NaN；恢复必须继续处理 B。
2. 再次刷新时，让已隔离的 A 再产生加速度/传感器 NaN；累计隔离必须继续有效。
3. 未标记 NaN 进入 VecNorm 时，在任何运行统计变化前报错。
4. 已标记 NaN 被排除，正常样本的统计仍然正确，输出保持有限。

例如，正常行观测为 `2.0`、异常行为 NaN，有效掩码为 `[True, False]` 时，均值应为 `2.0`，不能把异常行替换成零之后按两个样本计算成 `1.0`。

### 11.3 保留的历史验证结果

使用实际 5 万源文件数据和正式 50,000-rollout curriculum，只通过停止条件将验证限制为 100 个 rollout；没有通过缩短总预算来改变课程进度。

| 项目 | 历史验证结果 |
| --- | --- |
| GPU 进程数 | 3 |
| 每进程环境数 | 4096 |
| 执行 rollout | 100 |
| 全局环境 transition 数 | 39,321,600 |
| 累计恢复数 | 11,956 |
| 已记录标量 | 91 类，全部有限 |
| 参数/VecNorm 一致性指标 | 报告最大差异均为 0 |
| 约束溢出警告 | 未出现 |
| 原始观测有限性检查失败 | 未出现 |
| 退出码 | 0 |

结果表明该验证窗口内恢复机制能够处理大量局部异常，并保持已记录训练指标有限。不能由此推断所有动作都稳定、所有内部张量始终有限，或任意长训练都会通过。本文整理时未重新运行上述训练验证。

## 12. 当前实现仍然存在的数值边界

以下限制必须随方法一起传递，避免其他项目将其误读为已完成的全面防护：

| 边界 | 当前事实 |
| --- | --- |
| reward → GAE → return/advantage → ValueNorm | 没有在每个入口分别设置独立 finite 硬检查 |
| 优势归一化 | 仍使用乘法掩码，无法消除输入中已经存在的 NaN |
| GAE 终止掩码 | 也是乘法逻辑，依赖输入有限 |
| estimator 的样本损失掩码 | 同样采用乘法；后续总损失检查负责暴露异常 |
| ValueNorm 更新 | 同步全局统计，但没有独立 return 有限性门控；非有限值可能先进入统计再被下游检查发现 |
| 优化器内部状态 | 诊断工具检查过有限性，普通训练没有每步完整扫描全部优化器状态 |
| 梯度裁剪 | 检查梯度元素有限后裁剪，没有额外覆盖所有裁剪范数溢出情形 |
| 跨 rank 签名一致性 | 是聚合检测，不是逐元素完全一致证明 |
| 异常样本 | 没有整体从 PPO 删除，而是构造受控终止样本 |
| 长训练诊断 | 不默认开启逐事件追踪，也不默认执行 pilot 事件率停止阈值 |
| checkpoint | 不是完整优化器精确恢复，也不是自动回滚 |

以当前优势归一化为例，核心统计仍包含：

```python
count = mask.sum()
value_sum = (advantage * mask).sum()
value_sq_sum = (advantage * advantage * mask).sum()
# 多卡时对上述统计求全局和。
count = count.clamp_min(1)
mean = value_sum / count
var = value_sq_sum / count - mean.square()
std = var.clamp_min(0.0).sqrt().clamp_min(1e-5)
```

方差和标准差下限防止有限数值的退化情况，但不解决 `NaN * 0`。本文没有将其改写成一个尚未部署的更强方案。

## 13. 迁移到其他项目时需要保持的接口语义

其他项目可以按本文实现自己的适配层，但需要明确以下对应关系：

- 哪个维度表示并行环境；哪些控制量是策略输出，哪些是经过执行器处理的最终输入。
- `reset/forward/sense` 是只影响选中环境，还是会全局重算；传感器缓存何时失效、历史何时更新。
- 隔离掩码必须覆盖整个 transition，恢复复查必须先于相应历史消费。
- 清零 reward 和原始观测后，异常环境仍需立即真正终止，且不得参与观测归一化统计。
- 控制、损失和梯度异常必须保留硬失败行为，不能统一转换成普通仿真恢复。
- DDP 集合通信顺序、统计同步时机及计数口径必须一致。
- 机器人容量、步长、诊断阈值是案例配置，不能仅复制数字就认为迁移完成。

验收时同时检查恢复数量、原始观测/归一化边界、损失与梯度、跨 rank 状态，以及回合长度和控制表现。异常恢复是对失败环境的受控处理，恢复率过高仍需要继续检查动作分布、接触状态和仿真配置。
