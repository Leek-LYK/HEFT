# A_T→B_T→E_T：Transformer 主干对照

日期：2026-10-06（北京时间）。用户授权实现并启动正式实验；阶段明确命名A_T、B_T、E_T。对照为当前最佳MLP A→B→E。本实验从A_T重新训练两个教师，随后训练新学生，不复用MLP教师作为监督。因此检验整条路线的actor主干替换，不把结果单独归因于学生Transformer。

## 架构

采用小型模态融合Transformer：d_model=256，2层，4头，FFN=512，Pre-LayerNorm，GELU，残差，dropout=0；可学习CLS和模态类型向量，CLS输出经LayerNorm及必要线性投影接原动作头。使用PyTorch scaled-dot-product attention，无跨batch注意力、无跨episode缓存。

- 教师A_T：本体历史、特权状态、原生参考命令各自经LayerNorm＋Linear得到一个token，加CLS共4个token，替换原actor MLP。critic仍用原MLP，PPO算法、动作标准差学习、输入与奖励保持原设置。
- 学生B_T/E_T：保留Human/G1/Roban命令编码器、两本体编码器和动作头接口。所选命令表征与本体表征各作为一个token，加CLS共3个token；共享Transformer替换执行主干MLP。各token经LayerNorm＋Linear投影，输出256维接目标动作头。
- 历史及参考窗口仍由原编码器处理，本实验不把每个时间步拆成token，不声称测试了长序列时间注意力。没有增加参考未来信息或输入特权。

现有MLP默认构造、权重格式保持兼容；Transformer构造参数保存于checkpoint，阶段身份保存为A_T/B_T/E_T。stage语义归一化仅用于复用原训练与评分逻辑，准入仍核验实际Transformer阶段身份。

## 训练预算与范式

| 阶段 | 训练轮数 | 每目标环境×rollout | transitions | 优化预算 |
|---|---:|---:|---:|---|
| A_T，每教师 | 10000 | 4096×32 | 1,310,720,000 | PPO 3 epochs，minibatch8192，480000次Adam更新 |
| B_T，两机型合计 | 2000 | 1024×16 | 65,536,000 | 每轮32次监督，batch2048/机型，64000次Adam更新 |
| E_T，两机型合计 | 2300 | 1024×16 | 75,366,400 | 前2000轮32次×batch2048，末300轮16次×batch1024，共68800次Adam更新 |

A_T两教师共2,621,440,000 transitions；整条路线总计2,762,342,400。学生B_T＋E_T共132800次优化、534,118,400次回放状态抽样，对齐当前最佳B＋E。工程短检查与评估另计。Transformer参数量、FLOPs、墙钟不要求相等，独立记录参数量与耗时，不能称等算力或等参数对照。

A_T从随机初始化训练G1/Roban原生特权教师，学习率1e-4，其余PPO超参数不变。两教师并行训练后分别完整验证，原成功率/完成/恢复/误差门槛通过后进入B_T；世界误差沿用诊断口径。

B_T从随机Transformer学生开始，使用冻结A_T监督Human两路。前20轮教师接管，随后500轮退火；仅训练Human编码器与执行模块，机器人来源编码器、critic和标准差冻结。学习率1e-4。使用既有B绝对准入（成功≥90%、完成≥95%、恢复≤1%、根相对≤0.10m）。不修改门槛迁就新结构。

E_T加载同一B_T最终学生、创建新Adam与空reservoir。三来源编码器与执行模块联合训练；冻结A_T、critic和标准差，无PPO和教师接管。损失为Teacher动作MSE＋0.1×冻结B_T同来源动作保持MSE，六路等权，学习率1e-5，其余沿用直接E配方。不经过C。

## 固定评估与失败处理

训练前登记各阶段最终预算。A_T/B_T完整验证64环境；E_T预先固定第2300轮，完整验证64环境、完整测试500环境，seed42、每划分500动作×六路，复用同条件MLP A→B→E完整评分，核验身份/环境并独立重算原始分数和家族配对区间。不根据测试重选checkpoint。

A_T或B_T准入失败时，队列写pipeline_failure.json并停止下游，不制造通过记录或放宽门槛。E_T的负结果仍完整测试。单训练种子、现有已查看测试集，不能据此扩展多种子或全架构结论。

## 入口和记录

- 模型：active_adaptation/learning/modules/xwbc_transformer.py。
- 配置：cfg/xwbc/v02/staged_transformer.yaml、refinement_transformer.yaml。
- 正式队列：scripts/run_xwbc_transformer.py。
- 独立预算与身份审计：scripts/audit_xwbc_transformer.py。
- 工程准备：logs/v0.2/transformer_20261006_preparation/。
- 正式目录：logs/v0.2/transformer_20261006/，teacher_g1/teacher_roban对应A_T，student_human对应B_T，E_T对应最终阶段。

复现使用新的输出根目录：

```bash
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -B scripts/run_xwbc_transformer.py \
  --output logs/v0.2/transformer_reproduction
```

## 工程检查与实际启动

50项CPU回归通过，包含原MLP/U兼容、Transformer注意力梯度、train/eval无dropout一致性、A_T/B_T/E_T权重重载、B冻结机器人来源、E全部来源更新和固定anchor不变。

G1/Roban各96环境×32步×3轮真实PPO短检查完成，各9216 transitions、9次Adam更新；独立阶段/预算/权重审计通过，恢复次数0/3。另做真实B_T、E_T各3轮工程监督更新，六路输出有限、冻结模块及固定B_T保持模型不变。该学生短检查不使用或伪造质量准入，不计正式预算。

正式队列于北京时间2026-10-06T13:52:19.329170+08:00启动，驱动PID 167148。启动记录为transformer_20261006_preparation/formal.launch.json；后台日志formal.driver.log。A_T两教师先并行训练，B_T/E_T在真实完整验证准入后自动接续。启动核验时G1/Roban各完成13/12轮（各自目标10000），transitions与4096×32预算一致；已写指标均有限，模型构造与TensorBoard实际曲线核验通过。初期恢复事件G1/Roban为0/209，完整保留。详细快照见formal.startup_check.json。

TensorBoard：http://127.0.0.1:6012/ 。当前A_T两教师对应teacher_g1/tensorboard和teacher_roban/tensorboard；后续B_T为student_human/tensorboard，E_T为E_T/tensorboard。远程IDE转发6012端口。

实际参数量（MLP→Transformer，含critic；不是等参数实验）：

- teacher_g1: 8,259,131 → 5,621,793。
- teacher_roban: 7,796,267 → 5,330,513。
- student: 6,710,314 → 7,241,002。

详细构造与参数计数见transformer_20261006_preparation/architecture_comparison.json。

A_T单路与E_T六路仿真评分接口短检查均完成，阶段身份正确；非完整验证按原规则拒绝质量通过，不把工程短检查作为正式阶段准入。

## 第5600轮同期诊断

已按相同第5600轮、最近500轮窗口比较原MLP与A_T，配置与环境可比性核验通过。当前Transformer训练奖励落后、KL与熵偏高，但曲线仍在改善；未修改原训练超参数或终止队列。完整统计与曲线见[同期训练对比](15_Transformer与MLP同期训练对比.md)。
