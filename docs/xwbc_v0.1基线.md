# XWBC v0.1 可复现基线

本基线固定 Human/G1/Roban 三来源命令、G1/Roban 两执行对象的共享 MLP、无负载直接 PPO 实现，提供独立策略及无对齐对照入口。保留在 `research/xwbc/v0.1`；后续实现变更应有研究依据，再讨论新版本。长期协作入口为 [项目规范](../.SKILL/SKILL.md)，运行参数、数据契约及评估协议见 [正式实验说明](xwbc正式实验.md)。

## 版本与复现边界

第一轮基线固化继承 HEFT 提交 `959a7e0d2a3941af50b5371906fbc566e7e7593a` 的环境、奖励、机器人配置和资产，固化时不修改模型结构、训练方法或九份核心源码。完成运行的 `provenance.json` 指向上述父提交；当时 XWBC 文件尚未跟踪，实际运行版本由 `code/` 和 `source_hashes.json` 定义。第一轮固化提交为 `fa9d9a2`。当前分支已扩展第二轮时间上下文及编码器容量，配置见 `cfg/xwbc/heft_context.yaml`；复现第一轮源码可读取 `fa9d9a2` 或其运行快照，当前入口通过 `CONFIG=cfg/xwbc/train.yaml` 保留第一轮配置与旧 checkpoint 兼容。

源码与配置进入 Git，数据及运行证据独立保留。单独克隆代码不能获得源动作或已训练权重；复现需要相同源数据、机器人资产与运行环境。这里的复现保证是版本、输入契约、配置及证据可核验；CUDA 仿真及算子未验证逐位重现。

## 完成运行及证据

2026-10-01 在本地只读核验 `logs/v0.1/shared_seed0_run01/`：

- `shared`、seed `0`，每机型 `8192` 环境，rollout `32` 步，`2500` 次更新，总计 `1,310,720,000` transitions，每机型份额相同。
- `result.json` 为 `complete`、实际与请求更新均为 `2500`、`budget_exhausted=false`、`payload_enabled=false`，记录六路确定性 checkpoint 重载检查通过。该检查来自已完成运行，本次未重跑 GPU 检查。
- `metrics.jsonl` 恰有 `2500` 行，更新序号连续 `1..2500`，全部数值指标 finite；最终 transitions 与上述预算一致。`result.json` 记录耗时 `21963.016043597017` 秒。
- 最后一行累计恢复计数：G1 `0`，Roban `273034`。数值恢复事件需要后续按动作与评估结果定位；训练正常结束不能替代跟踪质量判断。
- 该目录没有 `eval*` 结果。正式验证/测试六组合、独立策略/无对齐的正式预算对照及多种子统计尚待完成；GPU 预算待确认。

以下文件留在本地运行目录，不进入 Git。迁移证据时用这些 SHA-256 校验：

| 文件 | SHA-256 |
|---|---|
| `result.json` | `03d5a6c7a0ad293ae27a2eda40d5e481484467a68475fdd4ede4be97591a6689` |
| `config.json` | `8e5d2408bc6aed84db7537e468f9177be01370b203b8cb539c308e93cbaa05a8` |
| `provenance.json` | `bd418765cb13f31c40df9b9e9c4366ff029258e446c954b373eebd9e7d810a60` |
| `source_hashes.json` | `5bf2b1d72497318e7bf084a21fd6c0513157e14d04e3eab7e141317f2ec3003f` |
| `metrics.jsonl` | `02ca13c216cd3c2695b355f5983d9c31ea909e98e7670d412f64ceaf54745941` |
| `checkpoint_final.pt` | `800c071153db2fd56dc7fd555e4311a094625536bfdd6dc9c666359f366121c5` |

2026-10-01 核验时，九份核心文件摘要与 `code/` 快照及 `source_hashes.json` 记录一致。下表固定第一轮历史源码摘要；第二轮修改后的工作区文件不再要求与该表相同。

| 项目相对路径 | SHA-256 |
|---|---|
| `active_adaptation/utils/xwbc_data.py` | `97c9fa3fd6f19e7922b3978e06f89a346567c595f0693e9ee0daddd9ba35f890` |
| `active_adaptation/envs/xwbc.py` | `6acdbc9aeb0c90e5e07ec428a1309adcc5b39caa45cc4adc184d74bb7be912b6` |
| `active_adaptation/learning/modules/xwbc.py` | `41f71ae07fa79eafac50c99d4eb8ecb07bcf8407fbd41f6d98af99e3da40630c` |
| `active_adaptation/learning/xwbc.py` | `bfd937aa5839892d1723c011e822583702b342880e8033ea81c58810c086878b` |
| `scripts/xwbc.py` | `8cdce2624959aee308f31930a6fe0768a0018ef6877c7c1ff82e5db707963c1d` |
| `scripts/run_xwbc.sh` | `85b09cd872d3f057773edeaf49eb58b73ed65a5fcea7cfaa9ef33b4cdb486475` |
| `scripts/data_process/build_xwbc_dataset.py` | `e80a4daa7ace6f53bd4a0dc7607e7b325779fa047b8c3939690ace37635c5db9` |
| `scripts/data_process/select_xwbc_motions.py` | `492ed1aa24d0a148b9c545340109a88d7e470ca73f38d95cae02b0f47d898093` |
| `cfg/xwbc/train.yaml` | `cbf82d40604935af8c455fb94b75a81c0cf7de43998645cfbb4093118fca39e5` |

## 数据与资产

源动作目录为 `dataset/Human_22d_50hz`、`dataset/g1_50hz`、`dataset/roban_50hz`。选样与编译入口已保存，源数据只读；家族分组规则、类别配额和确定性选样顺序在 `scripts/data_process/select_xwbc_motions.py` 中固定。现有候选清单 `dataset/xwbc/manifests/paired_10k.jsonl` 摘要为 `749dced4dadcdd6ac0b8a4c36033901d8c8eda37f59168d5a205c1370f950b7a`。

派生数据 `dataset/xwbc/paired_10k_v1/dataset.json` 摘要为 `0f25195027f2b2a2669fbd4ee22db800b498a1b43bbab6507e95fee18d620aa8`，与 `READY` 及训练 provenance 一致。准入剔除为零。

| 划分 | 动作 | 帧 | shard |
|---|---:|---:|---:|
| train | 8000 | 3285826 | 63 |
| val | 1000 | 384331 | 8 |
| test | 1000 | 358112 | 8 |

数据 schema 记录的 MJCF 摘要已与当前资产重新核验：

- `active_adaptation/assets/G1/g1.xml`：`bc17f1ee20d009896df16d6138640d6e814cd9d5837065ba5f8eaa74c221df52`。
- `active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml`：`05ef4d4946803de21c0445d11187b7c4316e57654d7b929a57943741c4d553ab`。

MJCF 使用的网格及既有机型 Python/YAML 配置由父提交提供。本次没有新增或替换资产。数据转换使用当前目标资产 FK 统一头手参考点，保留源关节及空间语义；有关字段、单位、时间窗口、坐标系及信息路由见正式实验说明和 `active_adaptation/utils/xwbc_data.py`。

## 环境与 CPU 验证

本次检查使用已有 `.venv`，未安装依赖。运行 provenance 记录 B200、PyTorch `2.10.0+cu128`；以下为固化时实际读取的环境版本，其他运行时版本未逐项写入原 provenance，不能据此断言过去全部相同。仓库已有 `pyproject.toml` 和 `uv.lock`；本次未重新解析或安装锁文件。

| 包 | 当前版本 |
|---|---|
| Python | 3.10.12 |
| torch / numpy / scipy | 2.10.0+cu128 / 2.2.6 / 1.15.3 |
| mujoco / mujoco-warp | 3.8.0 / 3.8.0.1 |
| warp-lang / mjlab | 1.13.0.dev20260306 / 1.3.0 |
| hydra-core / omegaconf | 1.4.0.dev1 / 2.4.0.dev4 |
| tensordict / torchrl / tensorboard | 0.11.0 / 0.11.1 / 2.20.0 |

在项目根目录运行以下 CPU 回归命令，将临时文件及缓存限制在忽略的项目日志目录：

```bash
mkdir -p logs/v0.1/baseline_cpu_audit
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/logs/v0.1/baseline_cpu_audit" XDG_CACHE_HOME="$PWD/logs/v0.1/baseline_cpu_audit/cache" WARP_CACHE_PATH="$PWD/logs/v0.1/baseline_cpu_audit/warp" .venv/bin/python -B -m unittest discover -s tests -p test_xwbc.py -v
bash -n scripts/run_xwbc.sh
DRY_RUN=1 PYTHONDONTWRITEBYTECODE=1 OUTPUT_ROOT="$PWD/logs/v0.1/baseline_preview" bash scripts/run_xwbc.sh suite
```

2026-10-01 实际通过 8 项 unittest（21.771 秒），涵盖 actor 来源隔离、共享/独立梯度路由、动作维度与 PPO 概率、对齐、GAE 时间截断/reset、家族泄漏、坐标平移不变性、窗口边界及无负载语义关键点配置。另有九份 Python 文件 `ast.parse`、Bash 语法检查通过；`suite` 预览打印九份训练与各自 val/test 评估命令，没有执行它们。本次没有 GPU 评估、训练或进程启停。

## 提交范围与本地保留

长期纳入：九份核心源码/配置，`scripts/summarize_xwbc.py` 的跨种子汇总，`tests/test_xwbc.py` 的正式行为回归，`.SKILL/SKILL.md` 的长期规范，README、忽略规则及两份复现说明。运行目录中的脚本快照、数据、日志、checkpoint、缓存和临时验证产物均不提交。

2026-10-01 第一轮基线固化时，以下文件完整保留在本地并排除提交：

- `paper/X-WBC.pdf`：11,350,327 字节。PDF 元数据识别为 *X-WBC: A Cross-Embodiment Foundation Model for Humanoid Whole-Body Control*，作者 Juntong Zhang、Chun Gu、Li Zhang，生成器为 arXiv GenPDF；本地未确认下载出处及再分发许可。论文不是代码运行依赖，按精确文件路径忽略，不扩大为整个 paper 目录。
- `active_adaptation/assets/roban/urdf/biped_s17_verified_handball_fixedhead.urdf`：30,748 字节。已有 [Roban 资产说明](../active_adaptation/assets/roban/README.md) 记录来自本地 `roban_s22_0901.zip`、派生自 Kuavo biped_s17，包元数据声明 BSD；新增 URDF 未提供独立的派生记录及许可文本。训练及数据 FK 实际依赖 MJCF，未发现 XWBC 对该 URDF 的引用，按精确文件路径忽略。
- `科研/v0.1/01_实验规划.md`、`科研/v0.1/02_小批次检查记录.md`、`科研/v0.1/03_正式实验实现与验证.md`：保留规划历史和阶段实测的原始记录，本次排除阶段报告。它们包含旧待办及“正式长跑尚未启动”的历史状态；当前基线以本文件和完成目录为准。本次不改写这些历史文件。

第一轮基线固化时的后续优先事项是取得 GPU 预算后，用冻结的最终 checkpoint 完成正式六组合评估并分析恢复事件，再开展架构匹配对照。迁移到其他机器还需要单独提供上述数据与运行证据，并确认资产/源数据使用权限；本次没有网络访问或再分发这些材料。


## 第二轮与科研归档维护

2026-10-07（北京时间）整理：当前 v0.1 将第二轮时间上下文与训练复现、六组合评估协议和 CPU 语义回归、科研归档合并为一次提交。上文的第一轮运行、摘要及核验结果保持历史含义；本次提交不改变已有运行的源码身份。

- [01_实验规划](../科研/v0.1/01_实验规划.md)、[02_小批次检查记录](../科研/v0.1/02_小批次检查记录.md)、[03_正式实验实现与验证](../科研/v0.1/03_正式实验实现与验证.md) 纳入 Git，保留阶段假设与当时状态。
- [04_第二轮训练方案与记录](../科研/v0.1/04_第二轮训练方案与记录.md) 记录宽时间上下文、256 维编码与训练预算；操作入口见 [正式实验说明](xwbc正式实验.md)。
- [六组合评估协议](xwbc评估协议.md) 固定成功判定、误差聚合、类别覆盖和证据边界，已有结果按原定义解读。

### 工作树与资源约定

| 分支 | 研究职责 | 入口 |
|---|---|---|
| `research/xwbc/v0.1` | 共享策略直接 PPO、第一轮基线与第二轮上下文 | `scripts/run_xwbc.sh` |
| `research/xwbc/v0.2` | 教师与学生分阶段训练、纠偏及路线对照 | 对应分支的 `scripts/run_xwbc_staged.sh` 与 `科研/v0.2/` |
| `research/xwbc/v0.3` | 形态条件化关节策略与联合 PPO | 对应分支的 `scripts/run_xwbc_morph.sh` 与 `科研/v0.3/` |

各研究分支保留独立源码目录；公共修复以明确提交按需同步。新运行关联代码提交、完整配置、数据与资产摘要及独立输出目录；历史运行仍由其原始源码快照和摘要界定。

`.venv`、`dataset` 和 `logs` 为本地运行资源入口，保持忽略，可使用目录或软链接；共享资源的变更需兼顾正在运行的实验。许可证、依赖声明、受跟踪资产、源码和稳定文档由各分支维护，跨版本参考注明来源提交，避免依赖相邻工作树的动态文件内容。

本机绝对路径导航及资源别名放在忽略的 `.local/`。原 `科研/v0.2/00_工作区入口.md` 本次移至 `.local/科研/v0.2/00_工作区入口.md`，保留本地访问便利；Git 中的版本关系以上表及对应分支记录为准。训练日志、数据、checkpoint 与缓存继续独立保留。

本次整理验证：21 项现有 CPU unittest 全部通过，另通过 Python 静态解析、Bash 语法检查、新旧配置的命令预览及仓库内文档引用检查。测试输出与缓存隔离在临时目录，未执行 GPU 训练或真实数据仿真评分。历史记录中独立保存的运行产物引用保持原路径，不要求随代码检出存在。
