# Roban HEFT + RefineTrack-TS v0.1

日期：2026-09-27。状态：**planning / audit complete，研究代码未实现、未启动训练或闭环评测**。

研究问题：在 nominal physics 下，仅 joint reference 出现错误时，能否沿用 HEFT 的 clean privileged guidance → Adapter → Student 三阶段机制，改善闭环跟踪，同时保留 clean tracking。目标是跟踪隐藏的 clean 意图，而不是忠实追随污染；部署端没有 clean 输入。

## 当前基线与工作边界

- 分支：`roban/RefineTrack/v0.1`；审计 Git HEAD：`52d8d9a5c1500536facbdc162b4e9670363dc5ec`，开始时工作区 clean。
- 原 50K Teacher → Adapt 2543+5000 → Finetune 5000 链已存在，文件与 parent 字段本轮逐一核验。
- 最终权重：`dataset/_audit/roban_ominisoma/roban_pipeline_20260908_091456/finetune/checkpoint_final.pt`。
- 最终 ONNX：`dataset/_audit/roban_ominisoma/deploy_our_heft_20260908/policy_project_heft_finetune.onnx`。
- 网络：policy 1313 → Adapter 1024/512/40 → Student quantizer；Actor 接收 1353，输出 21。旧基线并非 clean-only：已使用 reference noise 与 physics DR。
- 本轮仅新增科研资料；未更改训练代码、原始数据、基线配置或模型，未切换分支、commit、push。

## 文件入口

| 文件 | 用途 |
|---|---|
| [00_现有Roban_HEFT事实审计.md](00_现有Roban_HEFT事实审计.md) | 真实 action、observation、网络、数据、训练、评价与局限 |
| [01_研究问题与总体方案.md](01_研究问题与总体方案.md) | 假设、对照、最小改动和实施顺序 |
| [02_数据与污染协议.md](02_数据与污染协议.md) | split、配对、统计驱动的 S0–S3 与复现规则 |
| [03_Teacher_Student网络设计.md](03_Teacher_Student网络设计.md) | 源码对应的信息流、量化差异与信息边界 |
| [04_三阶段训练协议.md](04_三阶段训练协议.md) | 初始化、rollout、loss、更新/冻结和阶段验收 |
| [05_评价与验收协议.md](05_评价与验收协议.md) | 完整评价矩阵、自然波动、clean 保真与统计 |
| [06_实验记录.md](06_实验记录.md) | 本轮审计实录与后续 run 模板 |
| [07_阶段性结论.md](07_阶段性结论.md) | 已有证据支持的结论及边界 |
| [experiment_contract.json](experiment_contract.json) | **DRAFT_NOT_FROZEN**；正式训练前补齐并冻结 |
| [evidence/provenance.json](evidence/provenance.json) | 数据/资产/权重 SHA-256、checkpoint parent 和参数 shape |
| [evidence/motion_statistics_pilot.json](evidence/motion_statistics_pilot.json) | 4096 motion 有界 CPU 描述性统计 |
| [evidence/archive_validation.json](evidence/archive_validation.json) | 文档/链接/JSON、源与模型哈希及改动范围校验 |
| [evidence/审计方法与来源.md](evidence/审计方法与来源.md) | 统计复现方法、源码导航、证据范围 |

## 已完成与下一步

已完成源码链路审计、4 个 checkpoint/ONNX 身份核验、有效数据计数、关节尺度 pilot、方案及协议归档。没有 RefineTrack 性能结果。

下一步先实现 manifest 配对验证、nominal overlay、确定性评价与数据污染；完成 split 家族去重、train-only 标定、S0 一致性和信息隔离测试，再做短 smoke/pilot 与旧基线自然波动测量。补齐合同后才能开始正式三阶段训练。未来实验结果追加到 06/07，不回填修改已冻结合同。
