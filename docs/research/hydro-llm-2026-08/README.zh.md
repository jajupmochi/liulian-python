> **语言：** [English](README.md) | 中文

# Hydro-LLM 研究 — 合并文档集 (2026-08)

实体标识符（文本提示词 vs 数值嵌入 vs 微调）在 Time-LLM 类冻结 LLM 预测器中的应用，基于瑞士河流水温数据。本文件夹整合并取代了此前的三个文件夹（于 2026-08-05 合并 + 去重 + 消解冲突）：
`2026-07-25-hydro-llm-plan/`、`2026-08-03-hydro-llm-levels/`、`2026-08-04-prompt-design/`，
并纳入了更广泛的 `docs/research/` 语料库中的相关内容（提示词 vs 嵌入的结论、timellm
验证、N 系列分析、通道消融审计、5 篇论文计划、STATUS）。

## 文档地图

| 文档 | 作用 | 何时阅读 |
|---|---|---|
| [00-RESEARCH-PLAN.md](00-RESEARCH-PLAN.md) | 定位、相关工作 + 引用陷阱、预注册假设 H1–H5、数据集、目标期刊/会议 | 设计/撰写论文时 |
| [01-ARCHITECTURE-SPEC.md](01-ARCHITECTURE-SPEC.md) | 已锁定的架构：分类体系（Level A/A1/A1.1/A2 + 设计空间 a1–g 映射）、实现状态、HPO 空间、PEFT 阶梯、入口点决策、epoch 策略、调试指南 | 涉及代码或配置改动时 |
| [02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md) | 提示词内容：占位符/ETT 相关 bug、瑞士数据画像、上游提示词结构剖析、设计原则、P0–P4 候选方案、区分符 vs 内容阶梯 | 修改任何提示词文本或 A1 分支时 |
| [03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md) | 核心分析文档：12 项实验菜单、理论框架、可视化 × 理论、贝叶斯/UQ、agent 方法、指标标准 | 分析结果/撰写分析章节时 |
| [04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md) | 实时更新文档：层级（tiers）、集群作业、含 GPU-h 的排队任务、结果台账 | 查看/记录运行状态时 |
| [figs/](figs/) | epoch 诊断图等 | — |

## 一段话概览当前状态 (2026-08-05)

所有代码轴均已在同一条 pipeline（入口
`experiments/hydro_llm/run_matrix.py`）上实现并完成本地验证：5 种 Level-A 模式、5 种 A2
嵌入子变体（含坐标）、5+2 种 A1 提示词丰富度分支（含区分符对照组
`symbol`/`shuffled`）、frozen/ln_only/lora、GPT2/BERT 主干（外加已在集群上缓存权重的
LLAMA），以及 4 个 SOTA 适配器（GPT4TS/TEMPO/AutoTimes/CALF）。提示词内容相关的 bug
（占位符文件 + 硬编码的 ETT 描述）已修复；`prompt_variant`/`prompt_stats`
开关提供了真正的空提示词分支和统计信息阶梯。两个 Tier-0 HPO 作业正在
UBELIX gratis 队列上运行：ETT 描述对照组（11557210）和修复后提示词的论文级分支
（11594547）。已知待办事项：A1 `coords` 文本格式化、BERT 权重同步、UniTime/
Chronos-2 候选主干、留出站点划分（A11）。

## 已被取代的来源（仅保留在 git 历史中）

`2026-07-25-hydro-llm-plan/00-PLAN.md`（+FEASIBILITY、ENTRYPOINT-DESIGN）— 已合并入 00/01；
`2026-08-03-hydro-llm-levels/00-MASTER-SPEC.md`（+DELIVERY）— 已拆分入 01/04；
`2026-08-04-prompt-design/00-PROMPT-DESIGN.md` — 已拆分入 02/03。合并过程中解决的
冲突：harness 时代的 n=3 数字被标记为已被 pipeline+HPO 重跑取代的锚点；
a1–g 设计空间被映射到已实现的 Level 分类体系上；coords/LLAMA 的 "BLOCKED" 状态已
更正为 DONE；旧计划中的乱序诊断（"P0 zero-GPU"）已标记为 A1 `shuffled` 并标注为
IMPLEMENTED。
