# LIULIAN 实体标识符研究 — 状态总表

> 常驻状态文档。每轮工作后更新此表，并以此格式汇报。
> 最后更新: **2026-06-27** · 集群: UBELIX gratis(免费) · **当前无作业在跑**(队列空)

---

## 1. 总体进度 (workstreams)

| # | 工作流 | 状态 | 关键产出 / commit |
|---|---|---|---|
| W1 | 实体标识符主矩阵 (3域 × 3模型 × 6标识符) | ✅ 完成 | `results-table.pdf` (78 cells) |
| W2 | PatchTST 注入位置消融 (concat vs add_after_patch) | ✅ 完成 | `ablation-patchtst-injection.pdf` |
| W3 | N2 通道冗余分析 | ✅ 完成 | `2026-06-24-N-series-analyses.md §1` |
| W4 | N6 每实体误差离散度分析 (归一化重算) | ✅ 完成 | `N-series §2` |
| W5 | 论文骨架 (controlled-study 定位) | ✅ 完成 | `2026-06-24-paper-skeleton.md` |
| W6 | Time-LLM 复现验证 (ours vs 官方代码) | ✅ 完成 | commit `bfd7469` |
| W7 | Time-LLM H4 文本身份机制 + ETTh1 (n=3) | ✅ 完成 | commit `a5df1e4` |
| W8 | Time-LLM H4 **swiss 头牌** (命名站点, 实体丰富) | ⏸ **等决策 A/B/C/D** | — |

---

## 2. 实验结果总表

### 2a. 实体标识符主矩阵 — RMSE(越低越好)
swiss = °C(反归一化); electricity/traffic = 归一化 RMSE。**粗体**=该数据集全局最佳; <u>下划线</u>=该(模型,数据集)行最佳。

| 数据集 | 模型 | none | embed | onehot | sin | random | coord | 最佳id改善 |
|---|---|---|---|---|---|---|---|---|
| swiss-1990 | lstm | 1.723 | 1.289 | 1.119 | **1.116** | 1.139 | 1.155 | **−35%** (sin) |
| swiss-1990 | patchtst | 1.374 | <u>1.303</u> | 1.319 | 1.325 | 1.326 | 1.353 | −5% (embed) |
| swiss-1990 | dlinear | 1.281 | 1.286 | 1.286 | 1.282 | 1.283 | <u>1.279</u> | ~0 (惰性) |
| swiss-2010 | lstm | 1.642 | 1.368 | **1.201** | 1.224 | 1.222 | 1.255 | **−27%** (onehot) |
| swiss-2010 | patchtst | 1.488 | <u>1.386</u> | 1.423 | 1.451 | 1.424 | 1.459 | −7% (embed) |
| swiss-2010 | dlinear | <u>1.354</u> | 1.356 | 1.356 | 1.358 | 1.361 | 1.357 | ~0 (惰性) |
| swiss-zurich | lstm | 1.553 | 1.378 | **1.237** | 1.259 | 1.249 | 1.276 | **−20%** (onehot) |
| swiss-zurich | patchtst | 1.480 | <u>1.388</u> | 1.427 | 1.434 | 1.441 | 1.475 | −6% (embed) |
| swiss-zurich | dlinear | 1.390 | 1.401 | 1.396 | 1.393 | 1.393 | <u>1.385</u> | ~0 (惰性) |
| traffic | lstm | <u>0.783</u> | 0.784 | 0.784 | 0.783 | 0.784 | — | ~0 (扁平) |
| traffic | patchtst | 0.659 | 0.683 | **0.656** | — | — | — | −0.5% (onehot) |
| traffic | dlinear | <u>0.799</u> | 0.800 | 0.800 | 0.800 | 0.800 | — | ~0 (扁平) |
| electricity | lstm | 0.516 | 0.507 | 0.501 | <u>0.498</u> | 0.511 | — | −3.5% (sin) |
| electricity | patchtst | 0.408 | **0.387** | 0.408 | — | — | — | −5.1% (embed) |
| electricity | dlinear | <u>0.434</u> | 0.435 | 0.434 | — | — | — | ~0 (惰性) |

**结论**: ①**per_entity LSTM 上身份增益最大**(swiss −20~−35%, onehot/sin 最佳); ②**DLinear 对身份惰性**(线性容量用不上); ③**PatchTST 用 embed 最好但增益小**; ④**多通道高冗余(traffic)身份近扁平**; ⑤ electricity 小增益。
注: electricity/traffic 的 dlinear&patchtst 及 traffic lstm 为 May-2026 旧基线(旧HPO/代码),标"初步"。

### 2b. Time-LLM
| 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|
| 复现验证 V1(官方) vs V2(ours) | ETTh1@96 GPT2 fp32 | 逐epoch逐位相同, best **MSE 0.3908 / MAE 0.4159** | ✅ 移植数值忠实 |
| H4 ETTh1 (none vs entity_description, **n=3 seed**) | GPT2, 同配置只翻 mode | none **0.39125±0.0026** / H4 **0.39121±0.0036** (Δ−0.01%) | ❌ **无可检测效应(null)**; 单seed −0.89%是噪声(逐seed符号翻转 −0.89%~+1.40%) |
| H4 **swiss 命名站点** | GPT2, per-entity | **未跑** | ⏸ 等描述源 A/B/C/D |

### 2b-bis. swiss-1990 LSTM 身份增益 — **n=3 多 seed 误差棒** (P2, job 7287420+7364880)
denorm RMSE (°C), seeds {2026, 2027, 2028}:

| mode | mean ± std | %Δ vs none |
|---|---|---|
| none | **1.702 ± 0.026** | — |
| embedding | **1.294 ± 0.007** | −24.0% |
| onehot | **1.128 ± 0.013** | −33.7% |
| sinusoidal | **1.116 ± 0.004** | −34.5% |

**结论**: 身份增益 −24~−35% **跨 seed 高度稳定**(增益 ≈ 20× std → 显著)。与 ETTh1 H4 文本身份的 null(符号翻转)形成鲜明对比 → 实锤"身份增益需 per-entity 域 + 真实判别信号"。这是论文 C3(regime)主张的误差棒证据。

### 2c. 机制消融 + 分析
| 分析 | 结果 | 出处 |
|---|---|---|
| N1 注入位置 (PatchTST swiss, concat vs add_after_patch) | ✅**已核实+修表bug**: pre-norm `concat_to_x` **回归 +32~85%**(swiss-1990 onehot **2.189** vs none 1.374); post-norm `add_after_patch` 恢复(−1.5~−4.4%)。**论文头号主张 C1 成立**(12/12 swiss cells)。原表 concat 列被 builder 覆盖成 add 值=显示bug,已修(`build_entity_id_figures.py:275,292`) | `ablation-patchtst-injection.tex` |
| N2 通道冗余 | 三域均**近秩-1** (participation ratio 1.2–3.1, 通道数 57–862) | `N-series §1` |
| N6 每实体误差离散度 | 身份**移动均值但不改变离散度**(归一化NRMSE; 反归一化版是尺度伪影) | `N-series §2` |

---

## 3. 正在跑的实验
**无。** 集群队列空。(P2 swiss-1990 lstm multi-seed 已 8/8 完成 → §2b-bis。)

---

## 4. 下一步可做内容 (按优先级)

| 优先级 | 任务 | 阻塞? |
|---|---|---|
| **P0** | Time-LLM H4 **swiss 命名站点**(头牌, 实体丰富主张) | ⛔ **等你定描述源 A/B/C/D** |
| ~~P1~~ | ✅**完成** 核对 N1 → C1 主张成立(concat +32-85%回归是真的); 修了表渲染bug。详见 §2c | — |
| ~~P2~~ | ✅**完成** swiss-1990 lstm multi-seed n=3 误差棒 → 增益 −24~−35% 跨seed稳定(§2b-bis)。下一步可扩到 swiss-2010/zurich + 其它模型 | — |
| P3 | traffic/electricity × {dlinear,patchtst} × 5模式 补全真实基线(#39, 20 cells) | 否(自主) |
| P4 | 论文初稿: Figure 1 注入位置图 + 引用 BibTeX 程序化核实 | 否(自主) |
| P5 | Time-LLM **embedding(H2)数值身份**模式(对比 H4 文本身份) | 否(需接 pipeline) |

**swiss 描述源决策(P0)**:
- **A**(推荐) 我抓真实 BAFU/FOEN 站点元数据(河名+位置, 联网, 需你放行)
- **B** 你给站点元数据文件
- **C** 用已有经纬度拼文本(无需外部, 偏弱)
- **D** 到此为止(ETTh1 null + 机制已是完整子结果)
