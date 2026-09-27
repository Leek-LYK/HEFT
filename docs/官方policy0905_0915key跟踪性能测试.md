# 官方 policy0905.onnx：0915 key 跟踪性能测试

测试日期：2026-09-19。使用同一批 45 个大脑动作，CPU 推理、无渲染，直接替换 HEFT 策略为用户提供的 ONNX。原始案例、our_heft 重放结果和报告未覆盖。

## 1. 结果

官方策略完整播放 **42/45（93.33%）**；our_heft 原批次完整播放 **41/45（91.11%）**。以下数值越小越好。成功定义为已起播、完整播放且未触发跌倒/异常停止。

| 统计集合 | 策略 | 案例数 | 21 关节 RMSE (°) | 根相对 Body Pos (mm) | 根相对 Body Ori (rad) |
|---|---|---:|---:|---:|---:|
| 各自成功集合 | 官方 policy0905 | 42 | 4.7802 | 12.1862 | 0.0841 |
| 各自成功集合 | our_heft | 41 | 5.0384 | 14.0066 | 0.0839 |
| 双方共同成功集合 | 官方 policy0905 | 40 | 4.3249 | 11.6963 | 0.0764 |
| 双方共同成功集合 | our_heft | 40 | 4.8760 | 13.4248 | 0.0798 |

各自成功集合可能不同，因此横向判断跟踪质量应优先看“双方共同成功集合”。失败案例的截断前误差没有混入成功均值；未起播案例不能视为零误差。成功率与误差需要一起看，不能只用成功子集误差判断整体能力。

## 2. 直接替换及控制条件

- 模型：`dataset/_audit/roban_ominisoma/官方heft/policy0905.onnx`。
- 模型 SHA-256：`b4de8ec26d56b38baa90710b371caebeb8f22ec81bd7683089110fafddb00cb4`。
- ONNX 输入 `policy [1,1313]`，使用输出 `action [1,21]`；其余中间输出不使用。
- 直接将原观测向量输入 ONNX，不增加外部归一化、关节重排、符号变换或接口适配。
- 原控制流程中的动作限幅、EMA、PD、关节映射和历史处理原样保留，与 our_heft 测试一致；这些不是为官方策略新增的处理。
- 同一机器人 XML、平地、初始站姿和参考 CSV；0–3 s 官方 AMP，3–5 s 混合，陀螺仪范数小于 0.3 rad/s 连续 0.5 s 后起播，末帧保持 2 s。
- 动力学 500 Hz，策略/参考 50 Hz；胸部高度低于 0.4 m 或 qpos 非有限则停止，20 s 未起播则超时。
- 无渲染 CPU 测试，未使用 GPU、未修改训练或部署中的模型文件。

## 3. 三项指标口径

- **21 关节 RMSE**：每帧先对 21 个实际关节角与参考角的差计算 RMSE，再对时间平均；表中为各成功案例均值的等权平均。不是策略目标角误差。
- **根相对 Body Pos**：分别计算实际/参考的 `R_root.T @ (p_body - p_root)`，求两者三维欧氏距离，转为 mm 后对 14 个节点、时间和案例平均。
- **根相对 Body Ori**：分别计算 `R_root.T @ R_body`，求实际/参考相对旋转的最短角（rad），再对节点、时间和案例平均。
- 根为胸部 `base_link`；节点集合与 our_heft 补算一致，根自身不计入，位置使用身体 frame 原点而非质心。
- 参考 root XY 为零，因此不评价世界坐标根部水平位移。根局部指标同时移除了整体平移和旋转，不能解释为绝对路径、根朝向或重力方向跟踪指标。
- 每个成功案例采样参考帧 0–297，共 298 帧；仅起播后的物理步前采样，排除 AMP、混合、等待与末帧保持。

节点集合：

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

## 4. 所有案例明细

失败案例仅给出状态和样本数，其截断前指标保存在 JSON，不作为完整动作成绩。

| ID | 动作 | 官方状态 | our_heft 原状态 | 官方有效样本 | 官方 Joint RMSE (°) | 官方 Body Pos (mm) | 官方 Body Ori (rad) |
|---|---|---|---|---:|---:|---:|---:|
| 000 | a person crawls on the ground | 跌倒/异常停止 | 跌倒/异常停止 | 76 | — | — | — |
| 001 | a person crosses arms and stands still | 完整播放 | 完整播放 | 298 | 2.322 | 5.177 | 0.0367 |
| 002 | a person is running | 完整播放 | 完整播放 | 298 | 4.966 | 14.448 | 0.0890 |
| 003 | a person is standing at a place and making various hand gesture | 完整播放 | 完整播放 | 298 | 2.466 | 4.750 | 0.0364 |
| 004 | a person is walking forward slowly | 完整播放 | 完整播放 | 298 | 3.488 | 10.919 | 0.0577 |
| 005 | a person is walking forward slowly in short strides while lifting a crate with both hands in front | 完整播放 | 完整播放 | 298 | 4.501 | 17.311 | 0.0796 |
| 006 | a person jogs forward and then stops | 完整播放 | 完整播放 | 298 | 4.742 | 14.944 | 0.0896 |
| 007 | a person kneels down | 跌倒/异常停止 | 跌倒/异常停止 | 298 | — | — | — |
| 008 | a person marches in place with high knees | 完整播放 | 完整播放 | 298 | 4.265 | 14.314 | 0.0784 |
| 009 | a person raises both arms above the head | 完整播放 | 完整播放 | 298 | 2.575 | 7.566 | 0.0527 |
| 010 | a person speaks, gesturing with both hands | 完整播放 | 完整播放 | 298 | 2.433 | 4.745 | 0.0399 |
| 011 | a person squats deeply | 跌倒/异常停止 | 完整播放 | 298 | — | — | — |
| 012 | a person stands up from the chair, then stretches the arms | 完整播放 | 完整播放 | 298 | 3.289 | 8.148 | 0.0614 |
| 013 | a person walks forward, turns around, and walks back | 完整播放 | 完整播放 | 298 | 4.684 | 15.771 | 0.0938 |
| 014 | a person waves with the left hand | 完整播放 | 完整播放 | 298 | 2.683 | 4.869 | 0.0514 |
| 015 | a person waves with the right hand | 完整播放 | 完整播放 | 298 | 2.599 | 8.055 | 0.0378 |
| 016 | a person stands, then sits on a chair and drinks with their left hand | 完整播放 | 完整播放 | 298 | 4.911 | 16.246 | 0.1042 |
| 017 | a person waves with the right hand, then bows politely | 完整播放 | 完整播放 | 298 | 3.345 | 8.023 | 0.0605 |
| 018 | a person walks with both hands on hips | 完整播放 | 完整播放 | 298 | 4.628 | 12.207 | 0.1057 |
| 019 | a person jumps in place lightly | 完整播放 | 完整播放 | 298 | 5.379 | 13.503 | 0.0903 |
| 020 | a person jumps forward lightly, taking two steps | 完整播放 | 完整播放 | 298 | 4.731 | 15.222 | 0.0733 |
| 021 | a person stretches the whole body side to side | 完整播放 | 完整播放 | 298 | 4.805 | 12.548 | 0.0890 |
| 022 | a person bends forward to touch the toes and then stands up straight | 完整播放 | 完整播放 | 298 | 9.914 | 27.230 | 0.2153 |
| 023 | a person performs a vogue dance and stands up while waving the body | 完整播放 | 完整播放 | 298 | 9.460 | 30.684 | 0.1938 |
| 024 | a person rolls both shoulders while standing in place | 完整播放 | 完整播放 | 298 | 2.868 | 9.842 | 0.0451 |
| 025 | a person nods the head as if saying yes | 完整播放 | 完整播放 | 298 | 3.248 | 6.733 | 0.0511 |
| 026 | a person shakes the head as if saying no | 完整播放 | 完整播放 | 298 | 6.292 | 12.197 | 0.1052 |
| 027 | a person stands and holds both hands in front as if carrying a tray | 完整播放 | 完整播放 | 298 | 2.115 | 5.108 | 0.0283 |
| 028 | a person bends down, picks up an object with both hands, and stands up | 完整播放 | 完整播放 | 298 | 12.868 | 21.726 | 0.1538 |
| 029 | a person extends both hands forward as if pushing a door open | 完整播放 | 完整播放 | 298 | 4.058 | 14.475 | 0.0843 |
| 030 | a person holds a box with both hands in front of the chest while standing | 完整播放 | 完整播放 | 298 | 2.096 | 5.304 | 0.0329 |
| 031 | a person reaches up with both hands as if placing an object on a high shelf | 完整播放 | 跌倒/异常停止 | 298 | 16.881 | 22.743 | 0.3333 |
| 032 | a person walks slowly | 完整播放 | 完整播放 | 298 | 3.193 | 9.458 | 0.0536 |
| 033 | a person walks forward at a slow and careful pace while staying in place | 完整播放 | 完整播放 | 298 | 3.901 | 12.827 | 0.0691 |
| 034 | a person standing in place waves hello with the right hand several times | 完整播放 | 完整播放 | 298 | 2.636 | 5.032 | 0.0469 |
| 035 | a person walks backward slowly | 完整播放 | 完整播放 | 298 | 5.399 | 13.433 | 0.0802 |
| 036 | a person sidesteps to the left | 完整播放 | 完整播放 | 298 | 3.794 | 13.201 | 0.0591 |
| 037 | a person sidesteps to the right | 完整播放 | 完整播放 | 298 | 4.049 | 14.236 | 0.0628 |
| 038 | a person turns around to the left while standing in place | 完整播放 | 完整播放 | 298 | 4.087 | 13.033 | 0.0667 |
| 039 | a person is walking forward at a normal pace | 完整播放 | 完整播放 | 298 | 4.157 | 14.572 | 0.0770 |
| 040 | a person stands still with both hands on hips | 完整播放 | 完整播放 | 298 | 2.556 | 5.076 | 0.0406 |
| 041 | a person claps both hands together while standing | 完整播放 | 完整播放 | 298 | 6.748 | 9.639 | 0.1308 |
| 042 | a person points forward with the right hand | 完整播放 | 完整播放 | 298 | 2.521 | 5.783 | 0.0439 |
| 043 | a person salutes with the right hand | 完整播放 | 完整播放 | 298 | 4.225 | 9.496 | 0.0868 |
| 044 | a person bends down to pick up an object with right hand, then stands up straight | 完整播放 | 跌倒/异常停止 | 298 | 10.892 | 21.226 | 0.1436 |

## 5. 产物与复现

- [逐案例原始结果](../dataset/_audit/roban_ominisoma/brain_amp_heft_20260915_key/official_policy0905_tracking_20260919/summary.json)。
- [同集合比较汇总](../dataset/_audit/roban_ominisoma/brain_amp_heft_20260915_key/official_policy0905_tracking_20260919/comparison.json)。
- [评测脚本](../scripts/evaluate_official_brain_tracking.py)，复用 [根相对指标实现](../scripts/analyze_brain_root_relative_tracking.py)。
- 每例保存独立控制脚本、日志、报告、trace 和逐帧逐节点误差 NPZ；输入 CSV 哈希与原批次逐例一致。

在项目根目录执行（会覆盖官方补测目录同名结果，不覆盖原始案例）：

```bash
.venv/bin/python -B scripts/evaluate_official_brain_tracking.py
```

结果适用于这一批 45 条动作及当前单进程控制设置，不是训练全集评估或实机测试；未模拟真实 DDS 和异步线程时延。
