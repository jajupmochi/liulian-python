# 入口设计：为什么是两个入口、一套契约

> 回答用户的三个追问：① 之后要和 LSTM/PatchTST 比较，入口是否该统一？
> ② LLM 线之后要接进 LIULIAN 应用、用统一 API/命令；③ 要做训练/推理。
>
> **结论：入口分两个，契约只有一套；pipeline 是终局，harness 退为复现参照。**

---

## 1. 为什么现在不能一步统一

两条链路的**数据层根本不同**，这不是工程口味问题：

| | `entity_identifier/run.py` | Time-LLM harness |
|---|---|---|
| 数据 | `liulian/data/` swiss loader，**per_entity ConcatDataset** | `refer_projects/Time-LLM-Revised/data_provider`，**channel-independent** |
| 模型看到的 | 每样本含多站信息 | **每样本一个站点的单变量窗口，`N=1` 恒成立** |
| 身份注入 | `EntityAwareMixin` + wrapper | `entity_id_mark_col` 经 `x_mark` 透传 |
| **已提交的 n=3 Time-LLM 结果** | ❌ | ✅ **全部出自这条** |

**若现在把 Time-LLM 改走 pipeline**：模型脚下的数据层被换掉 ⟹ 已发表的
`none 0.01457 / 文本 0.01430 / 数值 0.01200 / 随机 0.01178`（n=3）**全部失效需重跑**，
且这正是本项目已经栽过一次的「**跨代码时代混用**」——2026-07 的 electricity 补格就因此被
research-critic 拦下。

---

## 2. 分层方案

### 层一 · 契约统一（**现在就做，零风险**）

`hydro_llm/run_matrix.py` 输出**与 entity_identifier 完全一致的 `results.json`**：

```
data.dataset · data.identifier_mode · model.type · metrics.test.{mse,mae}
```

⟹ `tools/build_entity_id_figures.py` 的 `collect_tags()` **能同时吃两条链路**，
LSTM / PatchTST / DLinear / Time-LLM **直接进同一张表**。

并在 `provenance` 里**如实标注**数据层差异：

> "Channel-independent data path (N=1 per sample) … comparable in metric, not in data layer."

**这解决了你的①**：可比性靠契约，不靠共用入口。

### 层二 · 迁进 pipeline（**中期，分步验证**）

**好消息：Time-LLM 已经注册在 pipeline 里**——`liulian/pipeline.py:527` 有 `timellm` 分支，
`liulian/models/torch/timellm.py:561` 有 `TimeLLMAdapter(EntityAwareMixin, TorchModelAdapter)`，
pipeline 用**动态导入**加载任意 `liulian.models.torch.*`。**架构上没有障碍，缺的是配置与验证。**

迁移步骤（每步都可回退）：

1. 在 `matrix.py` 的 `MODELS` 加 `'timellm'`，并补 `BASE_CONFIG_BY_PAIR[(swiss-*, timellm)]`。
2. **先只验证 `none` 一格**：pipeline 路径能否复现 harness 路径的 **MSE ≈ 0.01457**。
   - 复现得上 ⟹ 数据层差异对该模型无实质影响，可安全迁移全部模式。
   - 复现不上 ⟹ **就地停下**，记录差异来源（很可能是 N=1 vs per-entity 的身份语义不同），
     两条链路各自保留，契约层继续承担可比性。
3. 逐模式迁移，每次与 harness 数字对照。

**这一步顺带解决你的②③**：进了 pipeline 就自动获得
`liulian` CLI、`Experiment` 状态机（INIT→TRAIN→EVAL→INFER→COMPLETED）、统一 `results.json`、
checkpoint 约定与推理路径——**不需要为 LLM 单独造一套应用侧 API**。

### 层三 · harness 退为复现参照（长期）

harness 的**不可替代价值**是它与官方 Time-LLM **逐位对齐**（ETTh1@96 MSE 0.3908 已验证）。
迁移完成后它不删除，保留作"官方复现基准"，日常实验全走 pipeline。

---

## 3. 现在的入口用法

```bash
# 列出将要跑的格子，不执行
python experiments/hydro_llm/run_matrix.py --phase dry --modes none entity_description embedding

# 单格 debug：1 epoch + num_workers=0（断点可命中）
python experiments/hydro_llm/run_matrix.py --phase debug --modes none

# 正式跑
python experiments/hydro_llm/run_matrix.py --phase full \
    --modes none entity_description embedding random_embedding --seeds 2021 2022 2023
```

**轴守卫**：未实现的取值会**报错而非静默退回基线**——

- `--modes soft_prompt` → 明确报"PLANNED but not implemented in timellm.py"
- `--tuning lora` → 明确报"LLM is frozen unconditionally at timellm.py:334；需先加 `llm_tuning` 开关 + peft"

这防的是本项目最忌讳的失败模式：**一个看起来是结果、实际是基线的格子**。

---

## 4. 待办

- [ ] 层一：验证 `results.json` 能被 `build_entity_id_figures.py` 读入（需把 run-tag 加进 `RUN_TAGS`）
- [ ] 层二第 1–2 步：pipeline 路径复现 `none` 基线（**这是迁移的 go/no-go 判据**）
- [ ] `llm_tuning ∈ {frozen, ln_only, lora}` 开关（`timellm.py:334`）+ `peft` 依赖
- [ ] soft_prompt / text_embedding 两个模式的模型侧实现
