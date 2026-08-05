> **语言：** [English](01-ARCHITECTURE-SPEC.md) | 中文

# 01 · 架构规格 — 分类体系、锁定设计、实现状态（LOCKED）

隶属于 hydro-LLM 文档合集（[README](README.md)）。实验分层与运行状态见
[04-EXPERIMENT-STATUS.md](04-EXPERIMENT-STATUS.md)；prompt 内容见
[02-PROMPT-DESIGN.md](02-PROMPT-DESIGN.md)；分析见 [03-ANALYSIS-PLAN.md](03-ANALYSIS-PLAN.md)。

## 1. Level 分类体系（identity 注入）

一个站点的 identity 可以通过若干种**载体（carrier）**注入到 Time-LLM 中。
这些载体彼此平级；每个载体下又有子变体。

```
Level A  — 注入 MODE（平级/兄弟模式，每次运行选一个）：
  ├─ none                 基线，prompt 与官方 Time-LLM 逐字节一致
  ├─ entity_description   TEXT identity，注入到 LLM prompt 中
  │     └─ Level A1   prompt 注入的子变体（丰富度 / 附加信息）
  │           ├─ A1: minimal text · rich text · text+statistics · text+coordinates · ...
  │           └─ A1.1: LoRA — prompt 注入的微调对应物
  │                    （对 LLM 做适配，而非/而外地丰富 prompt）
  ├─ numeric_embedding    NUMERIC identity，一个加到 patch embeds 上的逐站点向量
  │     └─ Level A2   embedding 子变体（面向 LLM 的 entity_identifier 阶梯）
  │           └─ A2: learnable · random_embedding · onehot · sinusoidal · coordinates · ...
  ├─ soft_prompt          可学习前缀：预置在前面的逐站点连续 token
  └─ text_embedding       句向量：将描述编码、投影、注入
```

**重命名说明：** 旧的 mode `embedding` 现改名为 **`numeric_embedding`**（其旧的子类型
"learnable" 变为 A2 的一个取值；`random_embedding` 是另一个取值）。这样消除了
"载体本身"与"其某个子类型"之间的命名冲突。

**总览性的 2×2（表示来源 × 注入位置）：**

| | 注入 = PROMPT/PREFIX | 注入 = ADDITIVE（加到 patch embeds 上） |
|---|---|---|
| **来源 = TEXT** | `entity_description`（A） | `text_embedding`（A） |
| **来源 = LEARNED** | `soft_prompt`（A） | `numeric_embedding` / A2（A） |

`none` 位于这个 2×2 之外，作为基线。这就是本文将"三点比较"升级为"完整网格"
所依据的设计空间框架。

### 正交轴（适用于任何 Level-A 模式）

- **`llm_tuning`** — 训练基础 LLM 的多少：
  - `frozen` — 所有 LLM 权重全部冻结（当前默认；只训练 reprogramming layer + 输出头）。
  - `ln_only` — **见 §2.1**。
  - `lora` — 在 attention 投影上加低秩适配器（这就是 Level A1.1）。
- **`llm_backbone`** — 基础 LLM，与原始 Time-LLM 论文一致：`GPT2` · `LLAMA` ·
  `BERT` · …… Time-LLM 必须通过配置接受其中任意一个（task 4）。

---

## 2. LOCKED 架构（task 3 — 不得再变更）

```
experiments/hydro_llm/run_matrix.py          ← 实验入口（矩阵运行器）
        │  构建 cells = {dataset} × {Level-A mode} × {sub-variant} × {llm_tuning}
        │                 × {backbone} × {seed}
        ▼
experiments.run.run_with_config(...)          ← 训练 pipeline
        │  （与 LSTM/PatchTST/DLinear 使用的是同一条 pipeline：liulian.pipeline
        │   → ForecastTrainer → train / valid / eval，加上 Ray Tune HPO，加上 NaN masking）
        ▼
liulian.models.torch.timellm.Model            ← 模型本体（骨干网络可替换）
```

- **`hydro_llm/run_matrix.py` 是入口。pipeline（`run_with_config`）是引擎。
  harness（`experiments/swiss_river/run_experiment.py`）已 RETIRED**——task 3
  落地后不再有任何调用它的地方（见 §5）。这解决了以下问题：（a）harness 上没有
  Ray Tune HPO，（b）harness 对 swiss-2010/zurich 没有 NaN 处理，（c）Time-LLM 的
  数值与其他模型不可比。pipeline 同时解决了这三个问题，因为 Time-LLM 现在与
  LSTM/PatchTST/DLinear 走的是完全相同的 train/valid/HPO 路径。
- **2026-08-03 已验证：** Time-LLM 通过 pipeline 端到端构建并运行成功
  （`entity_identifier/run.py --models timellm --phase smoke` 得到真实的
  test MSE 0.016478 / MAE 0.1025）。管线本身已打通，剩下的是功能build-out。

### 2.1 什么是 `ln_only`？（task 1）

`ln_only` 是一种**参数高效微调（PEFT）**策略：冻结基础 LLM 的*每一个*权重，
**除了 LayerNorm 参数**（逐特征的缩放 `γ` 和偏移 `β`），只训练这些参数。在
GPT-2 中，这对应 `ln_1`、`ln_2` 以及最后的 `ln_f`。

- **存在的理由：** LayerNorm 的仿射参数只占模型极小的一部分（数千 vs 1 亿以上），
  因此训练它们的成本几乎与完全冻结一样低，但对每一层激活值做重新缩放/偏移，
  能让冻结的骨干网络获得真正适应新模态的空间。这是一个公认的强力 PEFT baseline
  （与 BitFit 的 bias-only tuning 同属一个家族），也是介于 `frozen` 与 `lora`
  之间自然的**中间挡位**。
- **在本项目的矩阵中**，它是 `llm_tuning` 轴的一个取值，构成 tuning 阶梯
  `frozen → ln_only → lora`——用于消融"reprogramming 接口需要多少可训练性"。
  它挂载在单一的冻结点
  [timellm.py:333-334](../../../liulian/models/torch/timellm.py:333)（目前是对
  所有 LLM 参数的无条件冻结）。

### 2.2 HPO 搜索空间（task 3 — 已外部化到 YAML）

已按项目规则 3（外部化到配置文件、前端可编辑）外部化到
`liulian/optim/search_spaces.yaml`，并按模式做了门控（按规则 4，不留 dead knob）。
只在该旋钮确实会改变对应代码路径上训练出的模型时才纳入调优：

| 旋钮 | 适用范围 | 范围（初始值） |
|---|---|---|
| `learning_rate` | 所有 timellm cell | log 1e-4 … 3e-3 |
| `d_ff`（reprogramming FFN） | 全部 | {16, 32, 64} |
| `n_heads`（reprogramming attn） | 全部 | {4, 8} |
| `dropout` | 全部 | 0.0 … 0.2 |
| `patch_len` / `stride` | 全部 | patch {8,16,24}（stride=patch/2） |
| `llm_layers` | 全部 | {3, 6} |
| `embedding_size` | A: numeric_embedding + A2 learnable/random | {8, 16, 32} |
| `soft_prompt_len` | A: soft_prompt | {4, 8, 16} |
| `text_proj_dim` | A: text_embedding | {16, 32} |
| `lora_r` / `lora_alpha` | llm_tuning=lora（A1.1） | r {4,8}，alpha {8,16} |

骨干网络（`llm_backbone`）与 Level-A mode 是**矩阵轴，不是 HPO 旋钮**。

---

## 2.3 实现状态（截至 2026-08-04，均已在本地验证）

各轴的代码状态（验证方式 = 能构建 + 能前向 + 输出与 `none` 不同，以证明确实发生了
真实注入）。自 2026-08-03 快照以来，有三项此前标记为 BLOCKED 的工作现已完成：
**A2 coordinates 已接通**（此前"无坐标数据"的判断是一次错误的检索——坐标数据其实
一直都在 `graph_*.pth` 中）、**LLAMA 权重已下载**到集群、以及**配置已重新对齐到
权威上游**（Time-LLM ETTh1 canonical + swiss-benchmark 数据设置，而非已弃用的
harness）。entity-identifier pipeline 测试套件为 33 通过 / 1 跳过（坐标相关新增 2 个），
trainer 测试套件新增 3 个（`pass_entity_ids` 中的 coordinates_embedding）。

| 轴 | 取值 | 代码 | 验证情况 |
|---|---|---|---|
| Level A | none / entity_description / numeric_embedding / soft_prompt / text_embedding | ✅ | 均与 none 不同 |
| A2（embedding 子变体） | learnable / random / onehot / sinusoidal | ✅ | 修复后的代码确实完成了注入（差异 3.08） |
| A2 | coordinates | ✅（`8b58f83`） | 2026-08-04 已接通（此前的"BLOCKED"是一次错误的检索——坐标数据其实存在于 `dataset/swiss_river/graph_*.pth` 中）。`_load_topology` 现在会为 `coordinates_embedding` 触发；feat (28,2) 非零 + 28 行互不相同（no-fake-zero guard 通过）；端到端 smoke 测试 1/1 通过；2 个测试。 |
| A1（prompt 丰富度） | default / minimal / stats | ✅ | `default`=人工撰写的丰富文本，`minimal`=纯位置 id（`adab88e`），`stats`=id + 逐站点、仅取自 TRAIN 的温度统计量，不泄漏（`309cc15`）；均已验证彼此不同 + 端到端 smoke 测试通过 |
| A1（prompt 丰富度） | coords | 🔵 代码部分完成 | 坐标数据本身已接通（`8b58f83`，与 A2 coordinates 同一个 graph .pth 数据源）；只剩文本格式化这一步（把 (x,y) 渲染进 prompt）——不再被数据阻塞 |
| llm_tuning | frozen / ln_only | ✅ | ln_only 解冻了 19968 个 LayerNorm 参数 |
| llm_tuning | lora（A1.1） | ✅ | peft 已安装；可训练参数 50.9M/132.8M 已验证（集群上的 lora 扫描是唯一剩余部分） |
| llm_backbone | GPT2 / BERT | ✅ | BERT 构建+前向正常（vocab 30522） |
| llm_backbone | LLAMA | ✅ | `huggyllama/llama-7b` 权重已于 2026-08-04 下载到集群 HF 缓存（13G，2 个 safetensors 分片）；加载正常（hidden 4096，vocab 32000）。`llm_model: LLAMA` 分支已就绪；集群骨干网络扫描是剩余部分。 |
| HPO | `timellm_swiss` 空间 | ✅ | commit `0b929c3`；以 canonical 为中心（{d_model 16/32/64, d_ff 32/128/256, lr 1e-3/1e-2, llm_layers 3/6}），dead-knob 防护（embedding_size 对 timellm/gpt4ts 跳过），6 个测试。epoch 诊断结论：30 epoch + early-stop 已足够（两个 lr 都在 epoch 8 左右收敛）；在 swiss 单通道上 lr 1e-3 优于 canonical 的 1e-2。 |

另外已落地的工作：entity_ids 这一关键连接点（所有 identity 模式都通过 pipeline
到达模型）、一个 fail-loud 的 tokenizer 防护（退化的 vocab 现在会直接报错，而不是
静默地毁掉 prompt——这曾捕获过一个不完整的本地 gpt2 缓存和 bert 缓存）、以及
`_load_entity_descriptions` 加载器（对没有站点文本的数据集会直接报错）。

集群备注：集群缓存了 gpt2（完整，vocab 50257）以及现已缓存的
`huggyllama/llama-7b`（2026-08-04 下载，加载正常）。BERT 权重在做集群 BERT 扫描前
仍需要同步。


## 2.4 验证锚点（来自 2026-06-24 的验证轮次）

1. **逐位一致的移植**：本项目的 Time-LLM 与官方仓库（GPT-2，ETTh1@96，fp32）——
   逐 epoch 的 Train/Vali/Test loss 完全一致；最优 Test MSE 0.3908 / MAE 0.4159
   （early-stop 约在 e10）。文档记录了一处无害的分歧：本项目在 patch embedding
   处保持 fp32，而官方实现在此处转为 bf16。
2. **骨干网络决策**：GPT-2 124M，`llm_layers=6`（LLaMA-7B 在 gratis RTX4090 上
   此前不可行；其权重现已缓存在集群上，因此一组 LLaMA 敏感性实验现在是可以
   排期的——见 [04](04-EXPERIMENT-STATUS.md)）。
3. **逐样本 identity 机制（H4 接线方式的修正）**：harness/pipeline 在数据层面
   是 channel-independent 的（每个样本都是某一个站点的一个窗口），因此 identity
   必须按逐样本方式穿入（`entity_ids` kwarg / `x_mark` 列）——原来的 `b % N`
   方案是无效的（所有样本都拿到了 description[0]）。pipeline 的 trainer
   会为每一种 identity mode 传入 `entity_ids`（见 [trainer.py] pass_entity_ids）。
4. **Prompt 文本规则（预注册风险点）**：frozen LLM 可能会忽略专有名词；在人工撰写
   描述时，应优先使用描述性文本（"高山河流站点，海拔 1200 米"），而不是裸名称；
   A1 阶梯正是用来度量这一点的。
5. **frozen 骨干网络的注意事项**（entity-id-deep）：在 frozen LLM 下，entity 信号
   只能通过可训练组件起作用（对 Time-LLM 而言是 reprogramming/head，对
   GPT4TS 类的 ln_only 而言是 LayerNorm）——这正是 llm_tuning 轴之所以能提供
   信息量的原因。

## 3. 完整的 identity 注入设计空间（分类体系的来源）

上文的 Level 分类体系是 2026-07-25 调研出的完整设计空间的已实现投影
（每种机制均有已核实的先例）。映射与覆盖情况：

| # | 机制 | 先例 | 本项目的 mode | 状态 |
|---|---|---|---|---|
| a1 | 裸 ID 文本（"station k"） | Time-LLM PaP | A1 `minimal` | ✅ |
| a2 | 领域/数据集指令 | UniTime | `prompt_variant: minimal`（数据集层级） | ✅ |
| a3 | 丰富描述（河流/城镇/坐标） | CHARM channel description | A1 `default` | ✅ |
| a4 | 以自然语言表达的统计量 | Time-LLM PaP stats block | A1 `stats` + `prompt_stats` 旋钮 | ✅ |
| b | 学习得到的逐实体连续前缀 | Prefix-Tuning · P-Tuning v2 · TEST · S²IP-LLM | Level-A `soft_prompt` | ✅ |
| c | 加到 patch/token embeddings 上 | Time-LLM · C-LoRA | Level-A `numeric_embedding`（+A2 子变体） | ✅ |
| d | FiLM / cross-attention 调制 | FiLM · TFT static encoder · CHARM | — | ⚪ 不计划（CHARM 已占据这一空间；在 frozen 骨干下需要侵入式 hook） |
| e | 逐实体 LoRA（identity 即参数） | C-LoRA（CIKM 2024） | — | ⚪ 推迟（与 LoRA 轴共线，边际信息量低） |
| f | 检索 / 原型路由（cluster ID） | CCM（NeurIPS 2024） | — | ⚪ 候选：可检验个体 identity 是否被过度参数化 |
| g | text EMBEDDING 注入（句子编码 → 投影） | CHARM · LETS-C · TimeCMA | Level-A `text_embedding` | ✅ |
| — | 区分器对照 | Min et al. 2022（NLP）· Li et al. 2022（水文学） | A1 `symbol` / `shuffled` + A2 `random` | ✅ |

这个总览性的 2×2 视角（表示：text vs learned × 注入位置：prefix vs additive）
恰好对应 {a1/a3 ↔ b} × {g ↔ c}——即"三点比较升级为完整设计空间"。

## 4. llm_tuning 轴（PEFT 阶梯，已验证配置）

三个挡位，保守优先（28 个站点 × 约 8k 个日尺度步数，对任何更大的模型都是明显的
过拟合风险区间——见 00-RESEARCH-PLAN §算力现实）：

| 挡位 | 训练内容 | 规模 | 先例 |
|---|---|---|---|
| `frozen` | 仅 reprogramming + head | 0 个 LLM 参数 | Time-LLM 默认 |
| `ln_only` | LayerNorm γ/β（+wpe） | 约 20-40k 参数 | GPT4TS（NeurIPS'23 Spotlight，训练约 4.6%） |
| `lora` | r=4, α=8，目标为 `c_attn`，dropout 0.1 | 约 74k（GPT-2 的 0.06%） | CALF（AAAI'25）；Beyond-LoRA（[2409.11302](https://arxiv.org/abs/2409.11302)）：在 Chronos-Tiny 上 rank 2 即已足够；排序为 FourierFT > BitFit > LayerNorm ≈ LoRA |

实现注意事项（已核实）：GPT-2 的 `c_attn` 是一个融合的 `Conv1D(768→2304)`——
peft 作用于它时会把 Q、K、V 一并适配；GPT-2 中不存在 `q_proj`/`v_proj` 这样的
模块名，因此文献中"仅 Q、V"的设置需要手动切片。LoRA 的学习率应作为一个独立的
参数组（1e-4），与 reprogramming/head 的学习率（配置中的 lr）分开。容量上界挡位
（r=8, α=16，加 `c_proj`）已定义但尚未排期。

## 5. 入口点决策（2026-07-29，经用户纠正；2026-08-03 已实现）

一条 pipeline，只在实验设计层面做拆分。此前"两个入口 + 结果契约"的想法是
错误的：Time-LLM 的 channel-independent 的 `Dataset_Swiss_1990(ConcatDataset)`
与 pipeline 的 `per_entity` 划分本质上是同一种构造（一个是另一个的参照移植版本），
因此从来就不曾存在第二个数据层。所有后果（均已实现）：

1. 数据/模型/pipeline 层：统一——timellm 像 LSTM/PatchTST/DLinear 一样运行
   `pipeline.run_experiment`。
2. 实验设计层：拆分——`experiments/hydro_llm/run_matrix.py` 扫描 LLM 相关的
   各个轴（mode × A1/A2 × tuning × backbone × arch），这是一个非 LLM 矩阵不具备
   的笛卡尔空间。
3. 收益：胜出的 identity 方案会收敛回普通的 `timellm` 参数，并在同一条 pipeline
   上与 LSTM/PatchTST/DLinear 比较——不存在跨 harness 的结果偏差。

## 6. Epoch / early-stopping 策略（为什么不采用固定 epoch 数）

不要把"正确的 epoch 数"硬编码。用一个宽松的上限训练，配合**基于验证集的
early stopping**，让验证集来挑选最优 epoch——这也是 Time-LLM 论文以及本项目
harness 的做法：**train_epochs=30，patience=10**（`timellm_config.yaml` 的默认值）。

- `--phase dev`（train_epochs=5）仅用于 PIPELINE 验证，不是一个科学配置，
  也不与任何论文对齐。有证据表明这个数字太少了：在 dev Tier-0 上，
  `best_epoch=4` 恰好落在 5-epoch 的上限上 ⟹ 模型当时仍在提升；early stopping
  从未触发过。
- 与论文/harness 对齐的 BASELINE 运行使用 YAML 中的 30 epoch + patience 10：
  `run_matrix.py --phase dev --train-epochs 30`（dev = 不做 HPO、不做 quick_test；
  `--train-epochs` 会覆盖 5-epoch 的上限；patience 10 来自 YAML）。early stopping
  会挑选最优 epoch。
- 记录逐 epoch 的验证曲线是个好习惯（pipeline 会记录并报告
  `best_epoch`/`best_val_score`）；early stopping 本身就已经编码了
  "最合适的 epoch 数"这一概念。
- `--phase full` 会在 30-epoch 训练之上额外运行 Ray Tune HPO（50 个 trial）——
  这是为最终论文级数值准备的，成本约为前者的 50 倍。

下方 §3 中的 dev-5 Tier-0 数值仅用于验证，已被 30-epoch 的运行取代。

## 7. 调试真实入口（`run_matrix.py`）

调试真正的矩阵入口，而不是自定义脚本（自定义 driver 会偏离真实 pipeline——
例如它构建了不同的 val/test loader——因此在它上面设的断点不能证明任何事）。
`run_matrix.py` 在进程内（`_run_in_process`）执行每个 cell，因此 PyCharm 断点
能在 driver 以及 post-HPO 的 rebuild/retrain（主进程）中命中。

- **快速调试配置：** `experiments/swiss_river/debug.yaml`——与 `timellm_config.yaml`
  对齐但做了缩减（64 个训练窗口，2 个 epoch）。通过 `--config` 直通参数
  （`9b68db0` 中新增）经由真实入口加载它：

  ```
  python experiments/hydro_llm/run_matrix.py --config experiments/swiss_river/debug.yaml \
      --phase full --arch timellm --datasets swiss-river-1990 --modes none \
      --seeds 2026 --hpo-num-samples 2
  ```
  已验证：加载 debug.yaml、应用其上限（max_train_samples 163968→64）、进入真实的
  Ray Tune HPO（`Starting HPO via RayOptimizer`，samples=2）。`--config` 默认为
  `timellm_config.yaml`，因此真实运行不受影响。
- **HPO 编排相关断点**（`build_optimizer`、`resolve_search_space`、ASHA、
  最优配置、rebuild/retrain）：使用 `--phase full`。Ray 2.x 在 worker 进程中
  运行每个 trial 的 trainable，因此 trial 内部的断点不会命中——但一个 trial
  运行的是与 post-HPO retrain（主进程）相同的 `build_model`/`timellm.forecast`/
  `trainer.fit`，因此应在那里设断点。
- **无需等待 HPO 即可命中的模型/训练相关断点**：使用 `--phase dev`——真实
  pipeline，直接在主进程中训练，`build_model`/`forecast`/`fit` 中的断点会
  立即命中。与 HPO trial 使用相同的模型代码，只是没有 HPO 的包装。
- 通过 `--modes` / `--a2`（例如 `--modes numeric_embedding --a2 coordinates`）
  切换被测试的分支。


## 8. 弃用说明

- `experiments/swiss_river/run_experiment.py`（harness）以及
  `experiments/hydro_llm` 中驱动该 harness 的旧代码路径：task 3 落地后**已弃用**。
  harness 会作为官方 Time-LLM 复现参照继续留在代码树中，并附带一条弃用横幅
  （task 2）。任何实验入口都不得再调用它。
