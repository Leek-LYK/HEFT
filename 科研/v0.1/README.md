# 2026-09-29 迁移交付说明

本分支现用于B200迁移交接；旧4090合同保留为历史基线，提交后的HEAD不再匹配其正式启动gate。**先阅读 [B200迁移说明](12_B200迁移说明.md)，不要直接启动旧正式脚本。** 大型数据/parent需按清单单独传输，B200适配尚未进行。

# RefineTrack-TS v0.1 实施交付

2026-09-28：C0/C1 **4096-env 三阶段短 smoke 通过**，正式长训练未启动。分支 `roban/RefineTrack/v0.1`，HEAD `52d8d9a5c1500536facbdc162b4e9670363dc5ec`；无 commit/push。历史 checkpoint 与原始数据 hash 已复核未变。

主实验仅 C0 vs C1；沿用 HEFT 网络、量化器与 train→adapt→finetune。两组旧 reference transform OFF、荷载标签 OFF、physics DR ON；区别为 Student raw=clean 或 paired joint corruption。

入口：

- [数据与污染](02_数据与污染协议.md)、[网络边界](03_Teacher_Student网络设计.md)、[三阶段协议](04_三阶段训练协议.md)、[评价](05_评价与验收协议.md)。
- [实施记录](09_实施与训练记录.md)、[基线结果](10_基线评估结果.md)、[启动说明](11_训练启动说明.md)。
- [冻结合同](experiment_contract.json)，SHA-256 `99f798e902ef1e9770d51836a706e3eda393ae632ff000a3aad2a2fabba919b0`。
- [运行索引](evidence/run_index_with_evaluations.json)、[阶段更新审计](evidence/smoke_stage_audit.json)、[数据生成统计](evidence/data_generation.json)。

原 2026-09-27 草案保留在 `evidence/planning_20260927/`，其中 nominal训练、80/10/10 split、多消融及候选预算均被本轮合同替代，不是当前执行协议。科研根目录没有平铺版本资料。

Smoke中存在simulator非有限恢复，详见09，不能声称全程无NaN。当前 GPU4 空闲显存不足正式要求，脚本会拒绝启动；等待资源释放。短 smoke Student 的跟踪成绩不是正式实验结果。
