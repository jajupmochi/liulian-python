# 入口设计：只分实验设计层，不分管道

> **本文档已按用户 2026-07-29 的意见重写。前一版主张"两个入口 + 契约层"，那是错的**——
> 见 §1 的核实结果。现在的方案：**管道统一，只有实验设计层分开**。

---

## 1. 更正：两条"数据层"其实是同一个东西

前一版声称 Time-LLM 的 `channel-independent (N=1)` 与 pipeline 的 `per_entity ConcatDataset`
是两种数据层，因此不能合并。**读码核实后，这个说法不成立**：

| | Time-LLM 侧 | LIULIAN 侧 |
|---|---|---|
| 构造 | `Dataset_Swiss_1990(ConcatDataset)`，子集是 `SequenceWindowedDataset(..., embedding_idx=i, name=station)` | `swiss_river.py` 的 per_entity split，注释原文：*"`torch.utils.data.ConcatDataset`（**in the reference project**）"* |

**同一个构造，一个是原版、一个是移植。** 所谓 `N=1` 并不是另一种数据结构，而是
"每个样本本来就只含一个站点的窗口"的必然结果——ConcatDataset 天生如此。

**结论**：数据层**不该分**。若将来某模型确实需要不同的组织方式（如多通道联合输入），
**按模型自动选择**即可，而不是复制一条平行管道。

---

## 2. 正确的分层：只有实验设计层分

| 层 | 是否分开 | 理由 |
|---|---|---|
| 数据层 | ❌ 统一 | 本来就一样；差异按模型自动选择 |
| 模型层 | ❌ 统一 | `TimeLLMAdapter` 已在 `liulian.models.torch` 内，pipeline 动态导入即可加载 |
| **主管道** | ❌ 统一 | Time-LLM 与 LSTM 走同一条 `pipeline.run_experiment`，**嵌入方案作为参数** |
| **实验设计层** | ✅ **分开** | LLM 需要扫**嵌入方案**（soft prompt / 文本嵌入 / LoRA / prompt 质量阶梯…），与**嵌入模式**（none/文本/数值/随机）是**笛卡尔积**关系，矩阵维度与 LSTM 矩阵根本不同 |

### 关键收益（用户指出的，也是本方案的核心理由）

> **LLM 专项实验跑完后，最优嵌入方案回落为主管道 `timellm` 的一个参数，
> 直接与 LSTM/PatchTST/DLinear 在同一条管道上对比。**

这样"实验架构与代码不一致导致的结果偏移"**从根上消失**，而不是靠一个"结果格式契约层"
事后打补丁——后者只能保证数字长得像，保证不了它们是同一条路算出来的。

---

## 3. 迁移阻力评估（已核实，比预想小）

| 需要的东西 | 现状 |
|---|---|
| `TimeLLMAdapter` 在 pipeline 可达 | ✅ `liulian/models/torch/timellm.py:561`，pipeline 动态导入 `liulian.models.torch.*` |
| pipeline 里的 timellm 分支 | ✅ `pipeline.py:527` |
| swiss 的 per_entity split | ✅ `pipeline.py:112` 默认就是 `per_entity` |
| Time-LLM 需要的 prompt 内容 | ✅ `_load_prompt_content()` 已含 swiss 映射（`swiss-river-1990 → wt-swiss-1990`） |
| 四种身份模式的模型侧实现 | ✅ 已在 `timellm.py`（none / entity_description / embedding / random_embedding） |
| **矩阵注册** | ❌ **缺**：`matrix.py` 的 `MODELS` 无 `timellm`，`BASE_CONFIG_BY_PAIR` 无对应条目 |
| **嵌入方案参数化** | ❌ **缺**：soft_prompt / text_embedding / llm_tuning 尚未实现 |

**只有最后两项要做。**

---

## 4. 执行方案

### 阶段 A · 把 timellm 接进主矩阵（先做，判据明确）

1. `matrix.py`：`MODELS` 增加 `'timellm'`，补 `BASE_CONFIG_BY_PAIR[(swiss-*, 'timellm')]`
   指向一个 pipeline 版的 timellm config。
2. **go/no-go 判据**：pipeline 路径跑 `none` 一格，与 harness 已发表的
   **MSE 0.01457 ± 0.00022** 对照。
   - **一致** ⟹ 两条路等价，harness 退为"官方复现参照"，此后全部实验走主管道。
   - **不一致** ⟹ **就地停下查因**（不是各留一条），因为按 §1 两者本该等价，
     不一致意味着某处存在真实差异，必须定位而不是绕开。
3. 一致后，`experiments/entity_identifier/run.py` 即可跑
   `--models timellm --modes none embedding ...`，与 LSTM 同表。

### 阶段 B · LLM 专项矩阵（`experiments/hydro_llm/`）

保留此入口，但**职责收窄为"实验设计层"**：它只负责枚举 LLM 特有的
**嵌入方案 × 身份模式 × 可训性 × 骨干** 这个笛卡尔积，**底层调用主管道**
（而不是 harness）。产出的最优方案写回主管道参数。

### 阶段 C · 回归主管道

最优嵌入方案成为 `timellm` 的一个配置项，与其它模型在阶段 A 的同一条管道上对比。

---

## 5. 当前状态与下一步

- `experiments/hydro_llm/run_matrix.py` 已存在，但**目前调的是 harness**，属阶段 B 的临时形态；
  阶段 A 完成后应改为调主管道。
- 已在此过程中修掉两个真 bug（都已提交）：
  1. **`results.json` 缺 `rmse`** ⟹ 格子会被图表构建器静默跳过。
  2. **harness 的 YAML 无条件覆盖 CLI 参数** ⟹ `--train_epochs 1` 实际跑 30；
     最危险的是 `--identifier_mode` 也会被吞，导致**格子标签与实际计算不符**。
     已记入 [`debug_verification_guide.md`](../../debug_verification_guide.md) 陷阱清单第 8 条。
- 另发现 `Dataset_Swiss_1990` **接收 `percent` 却从不使用**（dead knob，违反项目
  CLAUDE.md 的"no dead knobs"）——限流需自行实现。

**下一步 = 阶段 A 第 1–2 步**：注册 timellm 进矩阵，跑 `none` 对照 0.01457。
