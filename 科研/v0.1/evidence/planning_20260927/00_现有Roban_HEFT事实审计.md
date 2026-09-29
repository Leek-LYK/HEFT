# 00 现有 Roban HEFT 事实审计

本页的“当前”指 2026-09-27、HEAD `52d8d9a5c1500536facbdc162b4e9670363dc5ec`。事实来自本地源码、profile、保存的 checkpoint 配置和产物；历史性能引用仓库报告，未在本轮重跑。推荐设计与已实现行为分开写。源码导航见 [审计来源](evidence/审计方法与来源.md)。

## 1. 机器人与控制

资产为 `active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml`，`assets/roban.py` 用 BuiltinPositionActuatorCfg 替换源 XML 的 torque motor。CPU 实际编译验证 nq=28、nv=27、nu=21；21 个可控 hinge，另有浮动根，头部固定。模型注册为 roban，S22 交付命名不能替代 S17 接口。root/IMU 为胸部 `base_link`，骨盆为 `waist_yaw_link`。

策略顺序：waist_yaw；左腿 leg_l1…6；右腿 leg_r1…6；左臂 zarm_l1…4；右臂 zarm_r1…4，全部带 `_joint`。不是 29 DOF G1。

`JointPosition.__call__`：网络 action → clamp[-10,10] → delay buffer → 每物理子步 EMA → default+offset+scale×applied_action → position actuator。腰/腿 scale=0.5、臂=1.0；默认 q=0，非 reference residual、非 torque policy。动作 history 8 帧为 applied。策略 50 Hz、物理 200 Hz（0.02/0.005 s）；训练默认 alpha=[0.8,1.0]、delay 0–2 物理步，Teacher torque limit 课程 4→1。boot 前 2 子步可使用 clean reset pose 保护。

PD：腰 40.1792/3.5579/80；腿1 90.1792/3.5579/150、腿2/4 150.0984/8.3088/150、腿3 40.1792/3.5579/70、腿5/6 34.2506/2.9072/74；臂 kp/kd=14.2506/1.9072，臂1 effort=14.1、其余37。按 kp/kd/effort 顺序。训练内不翻转腰，hardware 边界才 direction=-1。

成功部署的官方 EMA 每 policy tick 一次、history 初始化填满当前状态；训练每子步滤波、history 清零。两者不能称为等价，v0.1 主实验保持训练控制链，部署复核单独列层。

## 2. 真实 observation 与网络

保存的 Finetune 权重矩阵直接验证：policy=1313、priv=4270、priv_critic=90；critic 总输入=5673。维度不是从论文猜测。

| policy 部分 | 维度 | 来源 |
|---|---:|---|
| boot indicator | 1 | reset 后倒计时/25 |
| command root position / orientation | 36+78 | student reference 相对位移和当前 IMU 下 rot6d |
| target q 与 target q−current q | 273+273 | 13×21，各为 raw reference；current q 可用共享本体噪声 |
| target root z / gravity | 13+39 | student reference |
| gyro / gravity history | 27+27 | 9 taps |
| joint q / dq history | 189+189 | 9×21，各自 |
| applied action history | 168 | 8×21 |
| 合计 | 1313 | Teacher 与 Student 共用 |

Student reference offsets `[0,1,2,3,4,5,6,7,8,-1,-2,-4,-8]`；state history taps `[0,1,2,3,4,8,12,16,20]`。未来 +8 对应 160 ms reference buffer，既有部署协议已经要求它，不能声称零前瞻/仅当前帧。

priv：clean q（含 q−actual）、root z/gravity、root 相对位置/姿态/线角速度目标；当前 keypoint 位姿/线角速度；clean keypoint 位置/姿态及差值；无噪 root linvel/angvel/gravity、joint q/dq histories；applied action/torque；clean-relative body-z 与 gravity 终止提示（含计数）；实际脚接触与 clean 参考脚接触。Teacher offsets `[0,1,2,4,8,12,16,20,-1,-2,-4,-8]`，12 帧，最多未来400 ms。并非仅 clean motion 的压缩。

priv_critic：motor params、material、joint offset、gravity domain 参数，仅 critic。Critic 同时接收 policy+priv+priv_critic。移除 randomization 对象会影响对应 observation provider，因此 nominal 应优先把范围退化为名义常量，而不是直接删空 randomization。

| 模块 | 实际结构 |
|---|---|
| encoder_priv | 4270→512→40；隐藏层 Linear+LayerNorm+Mish |
| adapt_module | 1313→1024→512→40 |
| Teacher Actor | concat(policy,z_T)=1353→1024→1024→512→21 mean；另有21维可学习 Gaussian std |
| Student Actor | 相同主体、独立参数，输入 policy+Q(z_S) |
| critic | 5673→1024→512→512→1 |

Teacher `z_T=priv_feature` 为连续、未量化40维；Adapter 输出 `z_S=priv_pred`；Student Actor 使用 `Q(z_S)=round(tanh(z_S)×4.499549865722656)`，带 STE，无训练参数，离散值 -4…4。Estimator MSE 比较 **z_S 与 z_T**，不是 Q(z_S) 与 z_T。Teacher Actor 与 Student Actor 仅在从 train checkpoint 加载时硬复制；Finetune 后不能当作同一参数。

VecNorm 在 train 更新，adapt/finetune 固定；export 将 policy normalization 包进 ONNX。网络是 MLP，history 来自输入拼接，不是 RNN。Encoder 没有 motion reconstruction head。

## 3. 三阶段实际实现

入口 `train.sh → scripts/train.py → cfg/exp/roban_{train,adapt,finetune}.yaml → PPOPolicy`。算法 config 通过 `ppo.py` ConfigStore 注册，不在 cfg/algo/*.yaml。

| 阶段 | rollout | 更新 | 固定/不 step |
|---|---|---|---|
| train | encoder_priv+Teacher Actor | Teacher Actor、priv encoder、critic 的 PPO；另更新 Adapter | Student Actor |
| adapt | Adapter+量化+Student Actor | 仅 Adapter 的连续 latent MSE，2 个 estimator epoch | Teacher Actor、priv encoder、Student Actor、critic、VecNorm |
| finetune | Adapter+量化+Student Actor | Student Actor 与 critic 的 PPO | Adapter（no_grad）、priv encoder、Teacher Actor、VecNorm |

这里“固定”指不进入对应 optimizer step / no_grad，不是所有模块都执行了 requires_grad_(False)。Adapt 是 **Student on-policy 状态上的 Teacher latent 标注**，不是只回放 Teacher 数据。

Teacher PPO 总 loss 含 clipped policy loss、entropy、value MSE、symmetry mean/std loss，以及 `progress×0.2×MSE(stopgrad(z_S),z_T)`；这一项更新 encoder，让 representation 易于估计。随后 estimator 用 stopgrad(z_T) 更新 Adapter。Finetune 不做 latent alignment。默认 PPO 3 epochs/8 minibatches，GAE γ=.99 λ=.95，optimizer=muon（头层按实现分组），actor 起始1e-4、critic/estimator5e-4，KL 调 LR。PPO reward 先合并各组再 clamp_min(0)。

`task.student_train` 还改变：root XY reward 变为1秒滑动目标、课程进度按末期处理、delay/torque limit 课程行为。它不是单纯的 actor switch。因而跨阶段 return 不直接代表相同目标下的性能 gap。

脚本默认50K/5K/5K rollout，3卡×4096 env，horizon32/16/32；成功旧链实际为50K/(2543+5000)/5000。预算不是收敛判据。

## 4. Reward、termination 与已有噪声

参考 target 一直主要取 clean `_motion`，raw 在 `_motion_student`。Reward `exp(-error/sigma)`，不是统一的 squared-error exponential。root pos/rot/linvel/angvel 权重0.5/0.5/1/1；keypoint pos/vel/rot/angvel 2/1/2/1；joint pos/vel 1/0.5。joint reward 用21关节绝对误差均值，不等于 evaluator Joint RMSE。

loco 包括 survival3、joint_vel_l2 5e-4、action_rate_l2（Teacher .02，Finetune .1）、feet_air_time_ref5、joint soft-limit1、torque soft-limit .01；Roban dense toe reward关闭。Teacher loco scale从.5→1；Student按最终课程。keypoint reward 使用各自 root-local body coordinates。

`motion_timeout` 和 episode timeout 是 truncated；body-z 与 gravity mismatch 连续5帧终止。body-z 相对 clean target 阈值[-.25,+.35]m，腾空 lower bound 有放宽；gravity vector差阈值1。另有 nonfinite 和参考 XY范围>15m截断。terminal count不等于跌倒。

旧配置的 `_transform_student_motion` 有 joint jitter(.01)、腿 joint bias(.1)、root位置/旋转噪声与漂移；其 joint bias 实际是均匀采样幅值，不能因字段叫 std 就当 Gaussian。物理 DR 包括 COM、friction/material、motor、joint offset、root push、wrench、gravity。现有 paired raw 加载后仍经过该 transform；新 paired 实验必须显式关闭它，避免双重污染。

## 5. 50K 数据与训练采样

真实目录 `dataset/roban/main/train_50000_v1`：50000 个选中 main 源 NPZ → **45213 motion / 45198 unique source paths / 18026422 frames**。同源可有多个有效段，不能按 segment 独立切 split。源总帧18623173。selection_seed=20260904，SHA-256 排序选源；manifest hash `43ec8a0fb32c3070f0f3e9c4023c66a548ab1c92eafbe600aea7b94cb69e7375`。

Ominisoma 源50 Hz，root=body[0] base_link、wxyz，joint21为旧交错顺序。adapter重排到 policy order，waist不取反；检查 quaternion、finite、质量并筛段。标准 GMR入口的 root_rot 是 xyzw，不能套在 Ominisoma 上。

memdataset仅保存 FP16 `root_pos_w[T,3] / root_quat_w[T,4] / joint_pos[T,21]`，meta starts/ends、joint_names，id_label提供source/segment；不保存 runtime velocity/body target。loader每rank CPU FP32完整加载并cat；GPU有界cache通过当前资产FK和差分构造 MotionData。约1.009 GB磁盘张量、约2.019 GB FP32/rank，加载临时副本和 body caches另计。

默认 skip前5帧、request[-20,24)、fill[-20,100)。global lengths因此每motion减5。短期切片在边界夹紧，不跨motion。paired groups 已支持 teacher_mem_path+student_mem_path；检查joint_names、motion数量、长度，**按索引配对，未核验 source/segment身份或独立fps元数据**。

旧 train.sh 的50K分支默认 resample_on_reset=false；构造时给每个env抽一个motion，此后reset只随机起点/失败rewind。不能由“加载了50K”推断每条均被训练；历史实际唯一覆盖需要原motion ID日志，当前证据未恢复。本轮新实验建议所有新对照统一 true并记录覆盖。

仓库有转换 validation_report、smoke/main_shard、45条brain外部评测、另有六类full数据；转换验证不是 ML validation split，45条brain也不是严格held-out 50K test。没有在该50K链里核验到固定、互斥、家族去重的train/validation/test三分清单。

## 6. Checkpoint 与 export 血缘

所有文件已于本轮读取并SHA-256；全路径及配置见 [provenance](evidence/provenance.json)。

| 名称 | phase / updates | SHA-256 |
|---|---|---|
| Teacher final | train / 50000 | `46ef3c507763ee185bcbafae833b154d2e4984e4b5b2e1ca6a3a0f667effc2ff` |
| Adapt首轮 | adapt / 2543 | `f12fbad18a9846fa36683ef522af5101a07362fd81f48511cf77719eb65429e8` |
| Adapt追加 | adapt / 7543累计 | `91180a3e163158007487c102673b280890f67c0dff99d40cc2f79870c04a30be` |
| Finetune final | finetune / 5000 | `258692caf9c5f5adbd6a85e560543aa8060906c85f034f3c51698dd6182606b6` |
| Student ONNX | opset17，1313→21 | `809c987a3f21db4d50c8c38f1186845b2870a9d64ede85def5c916845dbf233e` |

parent为Teacher(null)→Adapt首轮→Adapt追加→Finetune。checkpoint存 policy/env/cfg/vecnorm与部分meta，不存完整optimizer，继续训练是权重续接。load_state_dict对子模块加载失败只warning，未来正式run须额外hard-fail校验键、shape和roundtrip，不能把warning当作兼容通过。

`eval.py --run_path`走W&B下载，不能直接传本地目录。成功本地export脚本在审计目录，硬编码旧checkpoint及输出，后续需参数化到scripts中新入口，不能直接重跑覆盖旧模型。导出只含 policy normalization+Adapter+quantizer+Student mean Actor，无priv/critic；clip/EMA/scale/hardware sign在runtime。

## 7. Evaluation 的事实与限制

`evaluate_roban_teacher.py`是本地checkpoint、optimizer-free入口，默认shard_000、单GPU3限制；随机采样不是逐motion完整播放，root errors读取reward_root_pos_w（Student时会是滑动XY），统计终止前状态。不能直接作为跨阶段全motion公平测试。

已有brain batch对45条参考，our_heft 41/45、policy0905 42/45；共同成功40条的joint RMSE分别4.8760°/4.3249°，root-relative Body Pos 13.4248/11.6963mm。均为历史报告引用，且对参考root XY=0不评价global XY。14 body等权、每帧21关节RMSE再时间/案例平均；不是传统21个关节轴心MPJPE。

训练与brain两种evaluator控制周期、初始化和termination不同，应分层报告。现有工具尚未提供本研究统一的污染矩阵、motion级配对置信区间和全面控制诊断。

## 8. 正式训练前需要处理的实现细节

1. Student量化导致连续latent MSE与Actor实际输入不一致；保留主协议，补连续/量化诊断，不以低MSE认定成功。
2. paired loader先FK再 `_post_process` clamp q/dq，body target可能与clamped q不一致；本轮另取全帧等距100000帧，在XML物理范围外的joint元素约0.391619%。这是描述性样本，非全量越界率，必须审计有效参考和原始参考差异。
3. boot protect用clean初始化参考，评测/部署若沿用会给Student额外辅助；新nominal协议关闭boot actuator override，reset仍可按共同clean初态用于受控仿真，结果注明初始化条件。
4. finite防护保留；reward→GAE/ValueNorm与乘法mask仍非全入口防护，重置/恢复不算任务成功。
5. 从已预训练50K中再划split不能证明pretraining-unseen泛化；必须标记pretraining exposure未知/可能见过。
