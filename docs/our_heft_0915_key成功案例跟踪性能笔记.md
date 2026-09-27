# 0915 key 批次 our_heft 成功案例：关节与根相对身体跟踪性能

更新日期：2026-09-19。本文已修正上一版的评价口径：大脑参考片段的 root X、Y 均为 0，世界坐标中的根部水平位移不用于本批动作跟踪质量评价。以下使用关节 RMSE，以及身体节点在各自根坐标系中的相对位置/姿态误差。

## 1. 核心结果

原批次共 45 个案例，41 个完整播放且未触发停止条件（91.11%）。以下只统计原先这 41 个成功案例；未以新指标重新筛选案例。

| 指标 | 成功案例平均 | 案例均值中位数 | 案例均值最小–最大 |
|---|---:|---:|---:|
| Joint RMSE (rad) | **0.0879** | 0.0839 | 0.0444–0.2014 |
| Joint RMSE (°) | **5.0384** | 4.8093 | 2.5437–11.5374 |
| Root-relative Body Pos (mm) | **14.0066** | 13.2887 | 4.6521–37.2762 |
| Root-relative Body Ori (rad) | **0.0839** | 0.0730 | 0.0284–0.2473 |

位置误差平均为 **14.01 mm**，描述相对于胸部根节点的身体构型跟踪；关节 RMSE 平均为 **5.04°**。这两项不惩罚世界坐标中的整体平移。根相对指标还消除了整体刚体朝向差异，因此不能用它们评价全球轨迹、根倾角或绝对朝向的准确性。

## 2. 指标如何定义

### 2.1 Joint RMSE

每个采样点对 21 个策略关节计算 `sqrt(mean_j((q_actual[j] - q_ref[j])²))`，再对动作有效窗口求时间平均，最后对 41 个案例等权平均。角度差沿用原实现直接相减；不是把所有时刻合并后一次计算 RMSE。原报告的平均值在重放中复核，不使用策略输出目标值代替真实关节状态。

### 2.2 Root-relative Body Pos

根节点为胸部 `base_link`。对于实际机器人和参考机器人分别计算：

```text
p_local_actual[b] = R_root_actualᵀ · (p_world_actual[b] − p_world_actual[root])
p_local_ref[b]    = R_root_refᵀ    · (p_world_ref[b] − p_world_ref[root])
e_pos[t,b] = 1000 · ||p_local_actual[t,b] − p_local_ref[t,b]||₂
Body Pos = mean_cases(mean_time(mean_bodies(e_pos)))
```

这是各自根局部坐标系中的误差，既去除根平移，也去除根旋转；不是仅减去 root XY 后继续在世界坐标中比较。位置取 MuJoCo 的 body frame 原点 `xpos`，不是刚体质心、mesh 表面距离或控制目标位置。Body Pos 是欧氏距离的平均，不是位置坐标的 RMSE。

### 2.3 Root-relative Body Ori

```text
R_local_actual[b] = R_root_actualᵀ · R_body_actual[b]
R_local_ref[b]    = R_root_refᵀ    · R_body_ref[b]
e_ori[t,b] = angle(R_local_ref[t,b]ᵀ · R_local_actual[t,b])
Body Ori = mean_cases(mean_time(mean_bodies(e_ori)))
```

单位 rad，是相对旋转的最短角，取值范围 [0, π]。这项是补充的身体相对姿态指标，区别于旧笔记中的 root orientation。

### 2.4 节点集合与权重

采用训练配置 `required_motion_body_patterns` 对应的身体节点集合，排除根节点自身，共 14 个 body frame：

```text
waist_yaw_link
zhead_2_link
l_handball
r_handball
zarm_l3_link
zarm_r3_link
zarm_l4_link
zarm_r4_link
leg_l1_link
leg_r1_link
leg_l4_link
leg_r4_link
leg_l6_link
leg_r6_link
```

各节点等权，不按质量或肢体长度加权。这里是指定的 14 个训练相关身体节点，并非 XML 中所有相机、雷达及辅助刚体，也不是 21 个关节各自的轴心集合。固定头部等近零误差节点仍保留，根节点自身不参与，避免引入必然为零的根误差。由于节点选择会影响均值，跨项目比较时必须使用相同集合。

## 3. 数据补算和一致性验证

原 `trace.npy` 没有保存逐帧全部关节和根四元数，因此无法从旧轨迹直接恢复身体相对误差。本次对 41 个原成功案例进行无渲染 CPU 重放：使用原 `simulate.py` 控制逻辑、原 CSV、相同最终 Finetune checkpoint 和原机器人 XML；动作推理、PD、滤波、混合及稳定起播逻辑不变。

每个原误差采样时刻复制实际 qpos 到独立 MuJoCo 数据对象，再在另一个对象中使用参考 CSV 的根位姿和 21 个关节角；分别执行 `mj_kinematics` 得到身体坐标。测量不修改动力学仿真的状态。补算产物写入新子目录，原报告、视频和轨迹不覆盖。

- 所有 41 个案例仍完整播放且未跌倒。
- 原有四项指标的 mean/P95 与原报告的最大绝对差：`0`（要求小于 `1e-9`）。
- 重放完整 trace 与原 trace 的最大绝对差：`0`（要求小于 `1e-9`）。
- 每个输入 CSV 均验证 root XY 全部为零。
- 每例 298 个有效样本，对应参考帧 0–297；41 例共 12,218 个时间样本、171,052 个 body-time 误差。
- 指标采样在 50 Hz 控制 tick、物理积分之前；排除 AMP、混合、等待起播和最终帧保持阶段。

## 4. 尾部误差与节点表现

| 指标 | 所有 body-time 元素的 P95 | 每帧先平均节点后的全样本 P95 |
|---|---:|---:|
| Body Pos (mm) | 67.7131 | 38.6669 |
| Body Ori (rad) | 0.3188 | 0.2410 |

上述 P95 直接从此次保存的逐帧逐节点样本计算，不是各案例 P95 的平均。两列分位数的统计对象不同。

| 节点 | 平均相对位置误差 (mm) | 平均相对姿态误差 (rad) |
|---|---:|---:|
| waist_yaw_link | 0.0000 | 0.0336 |
| zhead_2_link | 0.0000 | 0.0000 |
| l_handball | 6.7267 | 0.0569 |
| r_handball | 10.1555 | 0.0686 |
| zarm_l3_link | 0.0000 | 0.0446 |
| zarm_r3_link | 0.0000 | 0.0446 |
| zarm_l4_link | 4.9910 | 0.0569 |
| zarm_r4_link | 6.9612 | 0.0686 |
| leg_l1_link | 4.1749 | 0.0780 |
| leg_r1_link | 4.1749 | 0.0682 |
| leg_l4_link | 22.8358 | 0.1726 |
| leg_r4_link | 22.1276 | 0.1602 |
| leg_l6_link | 60.9101 | 0.1478 |
| leg_r6_link | 53.0344 | 0.1734 |

## 5. 成功案例明细

所有值都是对应案例的时间均值；Body 指标先对 14 个节点平均。

| ID | 动作 | Joint RMSE (°) | Body Pos (mm) | Body Ori (rad) |
|---|---|---:|---:|---:|
| 001 | a person crosses arms and stands still | 2.854 | 6.346 | 0.0365 |
| 002 | a person is running | 5.120 | 12.221 | 0.0762 |
| 003 | a person is standing at a place and making various hand gesture | 3.015 | 5.506 | 0.0345 |
| 004 | a person is walking forward slowly | 3.477 | 11.634 | 0.0599 |
| 005 | a person is walking forward slowly in short strides while lifting a crate with both hands in front | 5.145 | 22.691 | 0.0954 |
| 006 | a person jogs forward and then stops | 5.019 | 14.639 | 0.0847 |
| 008 | a person marches in place with high knees | 5.551 | 16.624 | 0.0876 |
| 009 | a person raises both arms above the head | 2.932 | 5.837 | 0.0493 |
| 010 | a person speaks, gesturing with both hands | 2.675 | 4.652 | 0.0318 |
| 011 | a person squats deeply | 11.535 | 37.276 | 0.2473 |
| 012 | a person stands up from the chair, then stretches the arms | 5.702 | 11.161 | 0.0696 |
| 013 | a person walks forward, turns around, and walks back | 5.236 | 17.549 | 0.1043 |
| 014 | a person waves with the left hand | 3.002 | 6.308 | 0.0473 |
| 015 | a person waves with the right hand | 3.062 | 7.028 | 0.0360 |
| 016 | a person stands, then sits on a chair and drinks with their left hand | 5.814 | 16.179 | 0.1004 |
| 017 | a person waves with the right hand, then bows politely | 4.123 | 7.923 | 0.0589 |
| 018 | a person walks with both hands on hips | 4.809 | 14.300 | 0.1121 |
| 019 | a person jumps in place lightly | 6.621 | 17.270 | 0.0979 |
| 020 | a person jumps forward lightly, taking two steps | 6.080 | 18.738 | 0.0882 |
| 021 | a person stretches the whole body side to side | 5.838 | 21.286 | 0.0921 |
| 022 | a person bends forward to touch the toes and then stands up straight | 9.314 | 28.674 | 0.2054 |
| 023 | a person performs a vogue dance and stands up while waving the body | 11.537 | 36.823 | 0.2158 |
| 024 | a person rolls both shoulders while standing in place | 3.629 | 11.468 | 0.0504 |
| 025 | a person nods the head as if saying yes | 3.587 | 7.360 | 0.0474 |
| 026 | a person shakes the head as if saying no | 6.817 | 15.701 | 0.1045 |
| 027 | a person stands and holds both hands in front as if carrying a tray | 2.544 | 5.923 | 0.0284 |
| 028 | a person bends down, picks up an object with both hands, and stands up | 11.510 | 22.836 | 0.1430 |
| 029 | a person extends both hands forward as if pushing a door open | 4.944 | 16.397 | 0.0899 |
| 030 | a person holds a box with both hands in front of the chest while standing | 2.741 | 7.336 | 0.0357 |
| 032 | a person walks slowly | 3.267 | 10.576 | 0.0593 |
| 033 | a person walks forward at a slow and careful pace while staying in place | 3.484 | 10.096 | 0.0625 |
| 034 | a person standing in place waves hello with the right hand several times | 3.032 | 4.949 | 0.0419 |
| 035 | a person walks backward slowly | 5.260 | 19.458 | 0.0949 |
| 036 | a person sidesteps to the left | 4.586 | 16.192 | 0.0652 |
| 037 | a person sidesteps to the right | 5.215 | 18.663 | 0.0694 |
| 038 | a person turns around to the left while standing in place | 4.752 | 15.871 | 0.0795 |
| 039 | a person is walking forward at a normal pace | 3.971 | 13.433 | 0.0730 |
| 040 | a person stands still with both hands on hips | 2.906 | 5.684 | 0.0381 |
| 041 | a person claps both hands together while standing | 7.816 | 13.289 | 0.1884 |
| 042 | a person points forward with the right hand | 3.204 | 6.559 | 0.0457 |
| 043 | a person salutes with the right hand | 4.848 | 11.817 | 0.0896 |

按相对位置误差排序，较大的 5 个成功案例为：

- `011_a_person_squats_deeply`：Body Pos 37.28 mm，关节 RMSE 11.53°。
- `023_a_person_performs_a_vogue_dance_and_stands_up_while_waving_the_body`：Body Pos 36.82 mm，关节 RMSE 11.54°。
- `022_a_person_bends_forward_to_touch_the_toes_and_then_stands_up_straight`：Body Pos 28.67 mm，关节 RMSE 9.31°。
- `028_a_person_bends_down,_picks_up_an_object_with_both_hands,_and_stands_up`：Body Pos 22.84 mm，关节 RMSE 11.51°。
- `005_a_person_is_walking_forward_slowly_in_short_strides_while_lifting_a_crate_with_both_hands_in_front`：Body Pos 22.69 mm，关节 RMSE 5.15°。

## 6. 解释边界

- “成功”表示没有触发胸部高度低于 0.4 m 或 qpos 非有限等停止条件，并完成播放，不是根据上述误差阈值判定。
- 原失败案例 000、007、031、044 不进入本表；本表是成功子集统计，不代表全部 45 例总体质量。
- 根局部坐标消除了整体旋转，不能据此判断机器人相对重力的姿态是否正确；稳定性仍需结合跌倒判定和视频。
- 对输入参考的构型跟踪准确，不等于文本任务完成准确，例如原地生成的走路片段不能用此指标证明真实路径走对。
- 仍是单进程仿真，未模拟真实 DDS/异步线程时延，不能直接推断实机误差。

## 7. 产物与复现

- [逐案例汇总 JSON](../dataset/_audit/roban_ominisoma/brain_amp_heft_20260915_key/root_relative_tracking_20260919/summary.json)。
- [补算脚本](../scripts/analyze_brain_root_relative_tracking.py)。
- 补算目录各案例内保存 `root_relative_samples.npz`（位置、姿态、帧号、节点名）、`relative_metrics.json`、重放脚本、日志及重放的报告和轨迹。
- 原始批次：[README](../dataset/_audit/roban_ominisoma/brain_amp_heft_20260915_key/README.md)。

项目根目录复现命令（CPU、无渲染；覆盖补算目录内同名产物，不修改原始案例目录）：

```bash
.venv/bin/python -B scripts/analyze_brain_root_relative_tracking.py
```
