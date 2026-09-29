# 03 网络及信息边界

保持原 HEFT 权重形状：policy1313，priv4270，priv_critic90，latent40，action21。

- Teacher Encoder：clean参考+simulator privileged → 连续z_T；不读corruption label。
- Teacher Actor：**原HEFT实现仍为 raw policy1313 + clean privileged z_T**。这不是“整个Teacher Actor只读clean”；保留既有框架的raw直连，不新增Actor或重写观察路由。
- Student：raw policy→Adapter(1024/512/40)→Q(z_S)=round(tanh(z_S)×4.499549865722656)→Student Actor；同时保留policy直连。不读priv/clean/metadata。
- Critic、reward、termination可以读clean意图和仿真信息。Encoder目标在adapt固定，连续z_S/z_T MSE保持原实现。

运行时探针在每阶段固定policy并替换priv/priv_critic，Student mean action逐元素不变。paired窗口C0 joint相等，C1有非零joint差异，root相等；原 `_transform_student_motion` enable=false。不改变普通proprioception噪声，target_joint_pos_obs中noise_std作用于actual shared proprioception，不是第二个reference污染器。

两组关闭clean boot actuator override：task.action.boot_protect=false。训练reset仍用共同clean姿态及原init noise，此初始化条件需随研究结果披露；不新增部署输入。

validation保存连续latent MSE、Q(z_S)九级占用、Student Actor对连续/量化latent的动作差、Teacher/Student动作差。量化gap仅诊断，不修改量化器。body/root/control是主要闭环指标，latent不是主指标。

各阶段参数更新与冻结由checkpoint逐tensor比较核验，见 evidence/smoke_stage_audit.json。旧网络完整说明保存在 evidence/planning_20260927/03_Teacher_Student网络设计.md。
