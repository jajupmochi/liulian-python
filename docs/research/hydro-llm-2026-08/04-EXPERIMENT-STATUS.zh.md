> **语言：** [English](04-EXPERIMENT-STATUS.md) | 中文

# 04 · 实验状态 — 层级、运行中的作业、结果台账（实时更新文档）

属于合并后的 hydro-LLM 文档集的一部分（[README](README.md)）。每当有 cell 完成时更新
本文件。假设见 [00 §4](00-RESEARCH-PLAN.md)；模式定义见
[01](01-ARCHITECTURE-SPEC.md)；要对结果执行的分析见 [03](03-ANALYSIS-PLAN.md)。

## 1. 集群作业（截至 2026-08-05 上午）

| 作业 | 内容 | 配置 | 状态 |
|---|---|---|---|
| ~~11557210~~ | Tier-0，7 个 cell，**50 样本** HPO | 旧版 ETT 描述提示词 | **24h 整点 TIMEOUT,0/7 cell**(2026-08-05)——被 11623379(24 样本匹配对照)取代 |
| ~~11594547~~ | Tier-0 promptfix，7 个 cell，24 样本 HPO | 修复后的瑞士提示词（`prompt_domain: 1`，P3），显式 `--config` | **24h 整点 TIMEOUT**(2026-08-06 01:50 结束,cell 1 HPO 进行中)。由 11840703 续跑。 |
| ~~11623379~~ | Tier-0 ETT 对照，7 个 cell，**24 样本** HPO(匹配) | `configs/tier0_ettcontrol.yaml` = timellm_config.yaml 仅改 `prompt_domain: 0` | **24h 整点 TIMEOUT**(2026-08-06 13:53 结束,cell 1 HPO 进行中)。由 11840705 续跑。 |
| 11840703(+11840704 afterany) | promptfix 续跑,同 tag/环境,`--resume`(manifest 跳过 + Ray Tune 恢复) | 同 11594547 | 2026-08-07 约 14:35 提交,排队中;后继作业用 `--dependency=afterany` 链上,下一个 24h 段自动开跑 |
| 11840705(+11840706 afterany) | ETT 对照续跑,同 tag/环境,`--resume` | 同 11623379 | 2026-08-07 约 14:35 提交,排队中;afterany 后继已链 |

**墙钟实测事实(2026-08-07)**:`gpu` 分区 TIMELIMIT 为 **1-00:00:00(24 小时硬上限)**
(`sinfo -p gpu`),`--time=96:00:00` 在提交时直接被拒;QoS `job_gratis` 无独立 MaxWall
(其限制是 GPU 数量)。此前"gratis 允许 96h"的解读对 GPU 分区不成立。长扫描因此按
**24h 分段 + `--dependency=afterany` 链 + `--resume`** 方式推进。

Harness 时代的锚点数字（n=3，对论文而言已被这些重跑结果取代）：见
[00 §2](00-RESEARCH-PLAN.md)。

## 2. 层级计划（优先级、顺序、消融实验）

状态图例：✅ 已完成 · 🔵 代码就绪，尚未运行 · ⚪ 未实现 · 🧪 消融实验。

### Tier 0 — 旗舰基线（最先在集群上运行，3 个瑞士数据集）

正在 UBELIX gratis 上运行：作业 **11557210**（`hydro-tier0-2026-08-04b`），`--phase full`
（对 `timellm_swiss` 做 Ray Tune HPO），单一种子 2026。`entity_description` 的防护逻辑
会自动跳过 2010/zurich（无站点文本）→ 共 **7 个 cell**（1990 的全部 3 种模式；
2010/zurich 的 none + numeric_embedding）。截至上次轮询，cell 1（1990 none）的 HPO
正在探索搜索空间（试验取值 d_ff∈{32,128,256}，d_model∈{16,32,64}，
lr∈{1e-3,1e-2}，llm_layers∈{3,6}）。全量数据（163968 个训练窗口）+ 50 次 HPO 试验的负载
较重；这 7 个 cell 的扫描预计会跨越多个 24 小时的 gratis 窗口（requeue 时通过
`--resume` 续跑）。

| # | cell | 状态 | 备注 |
|---|---|---|---|
| T0.1 | `none` × {1990,2010,zurich} | 🟡 运行中 | pipeline 处理 2010/zurich 的 NaN |
| T0.2 | `entity_description` × 1990 | 🟡 运行中 | 文本身份标识；2010/zurich 因无站点文本被自动跳过 |
| T0.3 | `numeric_embedding`（可学习）× 3 | 🟡 运行中 | 约 −19% 的效应 |

> 此前 harness 的数字（种子 2026，无 HPO）：1990 none 0.014177，text 0.014485
> （+2.2%），learnable-emb 0.011433（−19.4%），random-emb 0.011569（−18.4%）。这些数字
> 已被 pipeline+HPO 重跑取代（仅作为合理性参考保留；2010/zurich 在 harness 上为
> NaN）。

### Tier 1 — 3 个数据集上 Level A 的其余部分

| # | cell | 状态 | 是否消融？ |
|---|---|---|---|
| T1.1 | `soft_prompt` × 3 | ⚪→🔵 | 缺失的 2×2 cell（learned × prefix） |
| T1.2 | `text_embedding` × 3 | ⚪→🔵 | text × additive 的 cell |
| T1.3 | A2 阶梯：random / onehot / sinusoidal / coordinates × 3 | 🔵 代码全部就绪 | 🧪 区分度 vs 容量（coordinates 已接入，commit `8b58f83`） |

### Tier 2 — 正交轴消融实验

| # | 轴 | 状态 | 是否消融？ |
|---|---|---|---|
| T2.1 | `llm_tuning`：frozen → ln_only → lora，在最佳 Level-A 模式上 | ⚪ | 🧪 可训练性阶梯 |
| T2.2 | `llm_backbone`：GPT2 / LLAMA / BERT，在 `none` + 最佳模式上 | ⚪ | 🧪 主干敏感性 |
| T2.3 | A1 提示词丰富度：minimal / rich / +stats / +coords | ⚪ | 🧪 "文本效果弱是因为提示词质量差吗？" |

### Tier 2.4 — 身份标识 × 可训练性的交互作用（最低优先级，🧪）

应用户要求于 2026-08-03 添加。这是一项解耦消融实验，只在主效应确定后才运行。

在实体信息丰富的数据集（swiss-1990）上做 `{numeric_embedding: on/off} × {llm_tuning: frozen/lora}`
的 2×2 实验：

| | frozen | lora |
|---|---|---|
| 无嵌入 | baseline | none+lora |
| + 嵌入 | embedding+frozen（当前） | embedding+lora |

**为什么它有意义（不是无意义的重复劳动）：** 交互项（加入 LoRA 后，嵌入带来的增益是否
会缩小？）把两个相互混淆的机制区分开来 —— *身份即信号*（增益无论是否微调都持续
存在）vs *身份即冻结接口的权宜之计*（一旦 LoRA 给了 LLM 自己的逐站点适配路径，
增益就会缩小）。这正是论文要回答的机制问题（"reprogramming 接口是不是瓶颈？"），
所以这个 cell 是诊断性的，而非附加性的。**扩展：** 用 `random_embedding` ×
{frozen,lora} 重复该实验，以检验这种交互作用是 *可学习* 容量特有的，还是对纯粹的
*区分度* 也成立。最低优先级：它是在 Tier 0–1 确立主效应之后再精细化机制，而且
LoRA 试验的计算成本较高。

### Tier 3 — 其他 SOTA reprogramming/LLM-TS 模型（task 5）

同一入口 + pipeline，与 Time-LLM 完全相同的接线方式，**仅替换主干**：

| 模型 | 参考文献 | 状态 | 角色 |
|---|---|---|---|
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | ✅ `--arch gpt4ts` | 🧪 阴性对照（无提示词/协变量路径）；仅有加性身份标识 |
| TEMPO | arXiv 2310.04948 | ✅ `--arch tempo`（commit `974c658`）| 分解（趋势+季节性）+ 共享的冻结 GPT-2，求和；加性身份标识；从零实现的适配器，冒烟测试 2/2 通过，8 个测试 |
| AutoTimes | arXiv 2402.02370 | ✅ `--arch autotimes`（commit `8ab418f`）| 自回归时间 token + 因果冻结 GPT-2，逐段解码；加性身份标识；从零实现的适配器，冒烟测试 2/2 通过，9 个测试 |
| CALF | arXiv 2403.07300 | ✅ `--arch calf`（commit `cdf0344`）| 跨模态双分支前向：一个跨模态分支将 patch 重编程进 LLM 词嵌入空间（复用了 timellm 的 ReprogrammingLayer）+ 一个时间分支，两者都经过共享的冻结 GPT-2，再融合。加性身份标识。从零实现的适配器；特征/输出/梯度对齐损失是任务层的扩展（不在前向中）。已完成端到端验证（冒烟测试 2/2 通过，两个分支均有贡献，7 个测试）。 |

在适用的情况下，每个模型都运行相同的 Level-A 模式 → 用于检验身份标识效应是
Time-LLM 特有的，还是对 LLM-TS 模型普遍成立。已完成的三个模型都只有加性通路
（没有提示词路径），因此它们的身份标识效应与 Time-LLM 的提示词路径形成了一个干净
的对照：身份标识是否既能通过数值型加性通道起作用，也能通过 LLM 提示词起作用？

---


## 3. 目标任务执行顺序（截至 2026-08-05 的状态）

1. ✅ task 3：将 `hydro_llm/run_matrix.py` 重新接线 → pipeline + HPO 空间。**（基础性工作）**
2. ✅ task 4：Level A 模式（`soft_prompt`/`text_embedding`/重命名为 `numeric_embedding`），
   A2 子变体（含 **coordinates**，于 2026-08-04 接入）、A1 丰富度（default/minimal/stats；
   coords 文本格式化待完成）、A1.1 LoRA + `ln_only`、多主干（GPT2/BERT/**LLAMA**）。
3. ✅ task 5：其他 SOTA（GPT4TS/TEMPO/AutoTimes/CALF）作为替换主干的适配器。
4. ✅ task 2：harness 的 `run_experiment.py` 已标记为弃用（横幅提示 + 运行时 DeprecationWarning）。
5. ✅ 检查点：已提醒用户进行调试（task 6 的关卡）；用户正在本地调试 `none` cell。
6. 🟡 task 7：完善文档（本文件）— 本轮进行中。
7. 🟡 task 6：集群 — **Tier 0 运行中**（作业 11557210，7 个 cell，`--phase full` HPO）；接下来是 Tier 1。
8. ⚪ 最终完整撰写（待 Tier 0/1 结果落地后）。

剩余尾部工作（非阻塞，按优先级排序）：(a) Tier-0 结果 → 旗舰基线表；
(b) Tier-1 的 Level A 其余部分 + 含 coordinates 的 A2 阶梯；(c) A1 `coords` 文本格式化；
(d) BERT 权重同步 + LLAMA/BERT 主干扫描；(e) Tier-2 消融实验（微调阶梯、主干
敏感性、A1 丰富度）；(f) Tier-2.4 身份标识×可训练性交互（最低优先级）。

---


## 4. 排队任务列表（来自 2026-07-16 升级计划台账，含 GPU-h 估算）

| id | 任务 | GPU-h | 备注 |
|---|---|---|---|
| 1.9 | 提示词质量阶梯（minimal → default → stats → shuffled/symbol） | ~10–20 | ★★ "审稿人一定会要求的消融实验" — 所有分支均已实现，运行并入 Tier 1/2 |
| 1.10 | 在 {none, text, numeric} 上做 frozen vs LoRA 的交叉实验 | ~15–30 | 论文核心创新点的 cell（H2） |
| 1.8 | 归一化范围 × 身份标识的 2×2（min-max vs z-score） | ~5 | 在 swiss 上检验仿射类擦除机制（doc-08 并入内容） |
| 2.1 | 站点 ID 线性探针 + Hewitt–Liang 选择性对照 | ~2 | [03](03-ANALYSIS-PLAN.md) 中的分析 A7 |
| 1.6 | UniTime 作为第二个原生 2×2 主干 | ~30–70 | 若预算受限，GPT4TS/TimeMoE 是更便宜的替代方案 |
| 2.5 | Chronos(-2) 零样本阴性对照 | ~1 | "没有学习到的实体嵌入能走多远" |
| 2.4 | CAMELS-CH-Chem（86 个瑞士小时站点） | 数据处理 1–2 天 | ⚠ 站点 ID 与我们已有的 28 个站点的对齐问题（自我泄漏风险） |
| 1.11 | `timellm_swiss` 的 llm_layers 由 {3,6} 扩为 {3,6,12}（GPT-2 全深度） | +0（HPO 预算不变） | 官方 argparse 默认值是 6，但论文脚本跑的是 llama_layers=32（完整主干）— 所以"用完整主干"才是忠于论文的分支；不得在运行中途改空间（在跑的 Tier-0 cell 采样自 {3,6}）；在下一 tier/重跑时应用 |
| 1.12 | Tier 边界同步包：llm_layers {3,6,12} + `precision: bf16`（Time-LLM 官方混合精度，见 [01 §6.5](01-ARCHITECTURE-SPEC.zh.md)） | 省显存/省时 | 本地已提交，集群同步只在 Tier-0→Tier-1 边界执行 — 中途同步会让同一组对比混用精度/空间；在跑的 Tier-0 分支（11594547/11623379 及其 --resume 续跑）全程保持 fp32+{3,6} |

## 5. 结果台账（cell 落地后填写）

| 运行标签 | cell | 最佳配置 | 验证集反归一化 RMSE | 测试集反归一化 RMSE | 备注 |
|---|---|---|---|---|---|
| （epoch 诊断） | 1990 none, lr 0.01 | 固定的标准配置 | 1.811（最佳 epoch 8） | — | 上限 100 epoch，early-stop 18 |
| （epoch 诊断） | 1990 none, lr 0.001 | 固定的标准配置 | 1.746（最佳 epoch 8） | — | 在 swiss 上 lr 1e-3 优于标准的 1e-2 |
| hydro-tier0-2026-08-04b | … | （HPO 运行中） | … | … | ETT 描述对照分支 |
| hydro-tier0-promptfix-2026-08-04 | … | （HPO 运行中） | … | … | 修复后提示词的论文级分支 |
