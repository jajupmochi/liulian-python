> **语言：** [English](00-MASTER-SPEC.md) | 中文

# Hydro-LLM 身份研究 — 主规范（架构已锁定）

> entity-identity × Time-LLM × hydrology 研究的唯一权威信源（single source of truth）。
> 创建于 2026-08-03，依据用户的 `/goal`。**§2 中的架构已锁定 ——
> 不要再次"改进"或重新布线。** 此前的反复（harness vs pipeline、两个
> 入口点、seed 漂移）到此终结。

---

## 1. Level 分级体系（身份注入）

站点的身份可以通过若干种**载体（carrier）**注入 Time-LLM。
这些载体彼此并列，各自还有子变体。

```
Level A  — injection MODE (peer/sibling modes, one per run):
  ├─ none                 baseline, prompt byte-identical to official Time-LLM
  ├─ entity_description   TEXT identity, injected into the LLM prompt
  │     └─ Level A1   prompt-injection sub-variants (richness / extra info)
  │           ├─ A1: minimal text · rich text · text+statistics · text+coordinates · ...
  │           └─ A1.1: LoRA — the fine-tuning counterpart of prompt injection
  │                    (adapt the LLM instead of / on top of enriching the prompt)
  ├─ numeric_embedding    NUMERIC identity, a per-station vector added to patch embeds
  │     └─ Level A2   embedding sub-variants (the entity_identifier ladder, for LLM)
  │           └─ A2: learnable · random_embedding · onehot · sinusoidal · coordinates · ...
  ├─ soft_prompt          LEARNABLE PREFIX: per-station continuous tokens prepended
  └─ text_embedding       SENTENCE-VECTOR: encode the description, project, inject
```

**重命名：** 旧模式 `embedding` 更名为 **`numeric_embedding`**（其旧子类型
"learnable" 是 A2 的一个取值；`random_embedding` 是另一个取值）。这样就消除了
"载体本身"与"其某个子类型"之间的命名冲突。

**核心 2×2（表示来源 × 注入位置）：**

| | injection = PROMPT/PREFIX | injection = ADDITIVE (to patch embeds) |
|---|---|---|
| **source = TEXT** | `entity_description` (A) | `text_embedding` (A) |
| **source = LEARNED** | `soft_prompt` (A) | `numeric_embedding` / A2 (A) |

`none` 位于这个 2×2 之外，作为基线。这是本文将设计空间从"三点比较"升级为"完整
网格"所依据的框架。

### 正交轴（适用于任意 Level-A 模式）

- **`llm_tuning`** — 基座 LLM 有多少部分参与训练：
  - `frozen` — 所有 LLM 权重冻结（当前默认；只训练 reprogramming layer + 输出头）。
  - `ln_only` — **见 §2.1**。
  - `lora` — 在注意力投影上加低秩适配器（这正是 Level A1.1）。
- **`llm_backbone`** — 基座 LLM，与原始 Time-LLM 论文一致：`GPT2` · `LLAMA` ·
  `BERT` · ……Time-LLM 必须能通过配置接受其中任意一个（task 4）。

---

## 2. 已锁定架构（task 3 — 不要再改动）

```
experiments/hydro_llm/run_matrix.py          ← THE experiment entry (matrix runner)
        │  builds cells = {dataset} × {Level-A mode} × {sub-variant} × {llm_tuning}
        │                 × {backbone} × {seed}
        ▼
experiments.run.run_with_config(...)          ← THE training pipeline
        │  (the SAME pipeline LSTM/PatchTST/DLinear use: liulian.pipeline
        │   → ForecastTrainer → train / valid / eval, + Ray Tune HPO, + NaN masking)
        ▼
liulian.models.torch.timellm.Model            ← the model (backbone-swappable)
```

- **`hydro_llm/run_matrix.py` 是入口。pipeline（`run_with_config`）是
  引擎。harness（`experiments/swiss_river/run_experiment.py`）已 RETIRED（退役）** ——
  task 3 落地之后不再有任何代码调用它（见 §5）。这修复了三个问题：(a) harness 上没有
  Ray Tune HPO，(b) harness 对 swiss-2010/zurich 没有 NaN 处理，(c) Time-LLM 的
  数字与其他模型不可比。pipeline 一并解决了这三点，因为 Time-LLM 现在走的是与
  LSTM/PatchTST/DLinear 完全相同的 train/valid/HPO 路径。
- **已于 2026-08-03 验证：** Time-LLM 能通过 pipeline 完整构建并端到端运行
  （`entity_identifier/run.py --models timellm --phase smoke` 产出了真实的
  test MSE 0.016478 / MAE 0.1025）。管线打通了，剩下的是功能建设。

### 2.1 `ln_only` 是什么？（task 1）

`ln_only` 是一种**参数高效微调（parameter-efficient fine-tuning, PEFT）**策略：冻结
基座 LLM 的*所有*权重，**只保留 LayerNorm 参数**（逐特征的缩放 `γ` 和偏移 `β`）参与
训练。在 GPT-2 中即 `ln_1`、`ln_2` 以及最后的 `ln_f`。

- **存在的原因：** LayerNorm 的仿射参数只占模型的极小一部分（几千个参数，相较于
  100M+），所以训练它们的开销几乎和完全冻结一样低，但对每一层激活做重新缩放/偏移，
  能让冻结的基座在面对新模态时获得真正的调整空间。这是一个已被广泛验证的强 PEFT
  基线（与 BitFit 的 bias-only 微调同属一族），也是介于 `frozen` 与 `lora` 之间
  自然的**中间档位**。
- **在我们的矩阵中**，它是 `llm_tuning` 轴上的一个取值，构成调优梯度
  `frozen → ln_only → lora`——用来消融"reprogramming 接口到底需要多少可训练性"
  这一问题。它挂载在唯一的冻结点
  [timellm.py:333-334](../../../liulian/models/torch/timellm.py:333)（目前是对所有
  LLM 参数的无条件冻结）。

### 2.2 HPO 搜索空间（task 3 — 已外部化到 YAML）

已按项目规则 3（前端可编辑）外部化到配置文件 `liulian/optim/search_spaces.yaml`
中，并按模式做了门控（按规则 4，杜绝死旋钮）。只在某个旋钮确实会改变该代码路径上
训练出的模型时才对其调优：

| 旋钮 | 适用范围 | 范围（初始） |
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
| `lora_r` / `lora_alpha` | llm_tuning=lora（A1.1） | r {4,8}, alpha {8,16} |

Backbone（`llm_backbone`）和 Level-A 模式是**矩阵的轴（axis），不是 HPO 旋钮**。

---

## 2.3 实现状态（截至 2026-08-04，均已在本地验证）

各轴的代码状态（验证标准 = 能构建 + 能前向 + 输出与 `none` 不同，证明确实发生了
注入）。自 2026-08-03 的快照以来，三项原本标记为 BLOCKED 的条目现已 DONE（完成）：
**A2 坐标已接通**（此前"无坐标数据"的结论是一次错误的搜索所致 —— 坐标其实一直都在
`graph_*.pth` 中）、**LLAMA 权重已下载**到集群，以及**配置已重新对齐到权威上游**
（Time-LLM ETTh1 的规范配置 + swiss-benchmark 的数据设置，而非已弃用的 harness）。
entity-identifier pipeline 套件现为 33 通过 / 1 跳过（坐标相关新增 2 个），trainer
套件新增 3 个（`pass_entity_ids` 中的 coordinates_embedding）。

| 轴 | 取值 | 代码 | 验证 |
|---|---|---|---|
| Level A | none / entity_description / numeric_embedding / soft_prompt / text_embedding | ✅ | 均与 none 不同 |
| A2（embedding 子变体） | learnable / random / onehot / sinusoidal | ✅ | 修复后的代码确实注入了（diff 3.08） |
| A2 | coordinates | ✅（`8b58f83`） | 已于 2026-08-04 接通（此前的"BLOCKED"是一次错误的搜索所致 —— 坐标其实就在 `dataset/swiss_river/graph_*.pth` 里）。`_load_topology` 现在会为 `coordinates_embedding` 触发；feat (28,2) 非零 + 28 行互不相同（no-fake-zero 防护通过）；e2e smoke 1/1 ok；2 个测试。 |
| A1（prompt 丰富度） | default / minimal / stats | ✅ | `default` = 人工撰写的丰富文本，`minimal` = 纯位置 id（`adab88e`），`stats` = id + 每站点仅用 TRAIN 数据算出的温度统计量，无泄漏（`309cc15`）；均已验证互不相同 + 端到端 smoke |
| A1（prompt 丰富度） | coords | 🔵 代码部分完成 | 坐标数据现已接通（`8b58f83`，与 A2 坐标同源自 graph .pth）；只剩文本格式化这一步（把 (x,y) 渲染进 prompt）——不再卡在数据上 |
| llm_tuning | frozen / ln_only | ✅ | ln_only 解冻了 19968 个 LayerNorm 参数 |
| llm_tuning | lora（A1.1） | ✅ | peft 已安装；已验证可训练参数 50.9M/132.8M（唯一剩下的是集群上的 lora sweep） |
| llm_backbone | GPT2 / BERT | ✅ | BERT 构建 + 前向 OK（vocab 30522） |
| llm_backbone | LLAMA | ✅ | `huggyllama/llama-7b` 权重已于 2026-08-04 下载到集群 HF 缓存（13G，2 个 safetensors 分片）；加载 OK（hidden 4096，vocab 32000）。`llm_model: LLAMA` 分支已就绪；剩下的是集群上的 backbone sweep。 |
| HPO | `timellm_swiss` space | ✅ | commit `0b929c3`；以规范配置为中心（{d_model 16/32/64, d_ff 32/128/256, lr 1e-3/1e-2, llm_layers 3/6}），带死旋钮防护（timellm/gpt4ts 会跳过 embedding_size），6 个测试。Epoch 诊断：30 epoch + early stop 已经足够（两个 lr 都在约 epoch 8 收敛）；在 swiss 单通道上 lr 1e-3 优于规范值 1e-2。 |

此外还落地了：entity_ids 这一枢纽机制（所有身份模式都通过 pipeline 到达模型）、
一个 fail-loud 的 tokenizer 防护（vocab 退化时现在会直接报错，而不是默默地把
prompt 弄坏 —— 这曾经抓到过一个不完整的本地 gpt2 缓存以及一个不完整的 bert 缓存）、
以及在数据集没有站点文本时会主动报错的 `_load_entity_descriptions` 加载器。

CLUSTER 备注：集群已缓存 gpt2（完整，vocab 50257），现在也缓存了
`huggyllama/llama-7b`（2026-08-04 下载，加载 OK）。BERT 权重仍需同步后才能在集群上
跑 BERT sweep。

## 3. 实验计划（task 6/7 — 优先级、顺序、消融）

状态图例：✅ 已完成 · 🔵 代码就绪、尚未运行 · ⚪ 尚未实现 · 🧪 消融实验。

### Tier 0 — 旗舰基线（最先在集群上跑，3 个 swiss 数据集）

正在 UBELIX gratis 上运行：job **11557210**（`hydro-tier0-2026-08-04b`），
`--phase full`（对 `timellm_swiss` 做 Ray Tune HPO），单一 seed 2026。
`entity_description` 的防护逻辑会自动跳过 2010/zurich（没有站点文本）→
**7 个 cell（矩阵单元）**（1990 的全部 3 种模式；2010/zurich 的 none +
numeric_embedding）。截至最近一次轮询，cell 1（1990 none）的 HPO 正在探索空间
（试验取值 d_ff∈{32,128,256}，d_model∈{16,32,64}，lr∈{1e-3,1e-2}，
llm_layers∈{3,6}）。完整数据（163968 个训练窗口）加 50 次 HPO 试验开销很大；
这 7 个 cell 的 sweep 大概率要跨越多个 24 小时的 gratis 窗口（`--resume` 会在
requeue 后继续）。

| # | cell | 状态 | 备注 |
|---|---|---|---|
| T0.1 | `none` × {1990,2010,zurich} | 🟡 运行中 | pipeline 处理了 2010/zurich 的 NaN |
| T0.2 | `entity_description` × 1990 | 🟡 运行中 | 文本身份；2010/zurich 自动跳过（无站点文本） |
| T0.3 | `numeric_embedding`（learnable） × 3 | 🟡 运行中 | 那个约 −19% 的效应 |

> 早前 harness 上的数字（seed 2026，无 HPO）：1990 none 0.014177，text 0.014485
> （+2.2%），learnable-emb 0.011433（−19.4%），random-emb 0.011569（−18.4%）。
> 这些数字已被 pipeline+HPO 的重跑结果取代（仅作为一个健全性参考保留；harness 上
> 2010/zurich 是 NaN）。

### Tier 1 — 3 个数据集上 Level A 的其余部分

| # | cell | 状态 | 是否消融？ |
|---|---|---|---|
| T1.1 | `soft_prompt` × 3 | ⚪→🔵 | 缺失的那个 2×2 cell（learned × prefix） |
| T1.2 | `text_embedding` × 3 | ⚪→🔵 | text × additive 那个 cell |
| T1.3 | A2 梯度：random / onehot / sinusoidal / coordinates × 3 | 🔵 全部代码就绪 | 🧪 distinctness-vs-capacity（区分度 vs 容量；coordinates 已在 `8b58f83` 接通） |

### Tier 2 — 正交轴消融

| # | 轴 | 状态 | 是否消融？ |
|---|---|---|---|
| T2.1 | `llm_tuning`：frozen → ln_only → lora，作用于最佳 Level-A 模式 | ⚪ | 🧪 可训练性梯度 |
| T2.2 | `llm_backbone`：GPT2 / LLAMA / BERT，作用于 `none` + 最佳模式 | ⚪ | 🧪 backbone 敏感性 |
| T2.3 | A1 prompt 丰富度：minimal / rich / +stats / +coords | ⚪ | 🧪 "文本效果弱，是不是因为 prompt 写得太差？" |

### Tier 2.4 — 身份 × 可训练性的交互作用（优先级最低，🧪）

按用户要求于 2026-08-03 添加。这是一个解耦（disentanglement）消融，只在主效应
确立之后才跑。

在实体信息丰富的数据集（swiss-1990）上做 `{numeric_embedding: on/off} ×
{llm_tuning: frozen/lora}` 的 2×2：

| | frozen | lora |
|---|---|---|
| 无 embedding | baseline | none+lora |
| + embedding | embedding+frozen（当前） | embedding+lora |

**为什么这不是无意义的凑数（not busywork）：** 交互项（加上 LoRA 之后，embedding
带来的增益是否会缩小？）可以把两个此前混杂在一起的机制分开 —— *身份即信号*
（identity-as-signal，增益不随调优方式改变而持续存在）vs
*身份是对冻结接口的一种绕行*（identity-as-frozen-interface-workaround，一旦
LoRA 给了 LLM 自己的逐站点适配通路，增益就会缩小）。这正好对应本文的机制问题
("reprogramming 接口是不是瓶颈？")，所以这个 cell 是诊断性的，而不是锦上添花。
**扩展：** 用 `random_embedding` × {frozen,lora} 重复一遍，测试这个交互作用是
*learnable* 容量特有的，还是对纯粹的 *distinctness*（区分度）同样成立。优先级
最低：它是在 Tier 0–1 确立主效应之后才用来细化机制的，而且 LoRA 试验的计算开销
很大。

### Tier 3 — 其他 SOTA reprogramming / LLM-TS 模型（task 5）

同一个入口 + pipeline，接线方式与 Time-LLM 完全一致，只是**换了 backbone**：

| model | ref | status | role |
|---|---|---|---|
| GPT4TS（OneFitsAll） | arXiv 2302.11939 | ✅ `--arch gpt4ts` | 🧪 阴性对照（negative control，没有 prompt/covariate 通路）；仅有 additive 身份注入 |
| TEMPO | arXiv 2310.04948 | ✅ `--arch tempo`（`974c658`） | 分解（趋势+季节）+ 共享的冻结 GPT-2，相加；additive 身份注入；从零实现的 adapter，smoke 2/2 ok，8 个测试 |
| AutoTimes | arXiv 2402.02370 | ✅ `--arch autotimes`（`8ab418f`） | 自回归时间 token + 因果冻结 GPT-2，逐段解码（next-segment decode）；additive 身份注入；从零实现的 adapter，smoke 2/2 ok，9 个测试 |
| CALF | arXiv 2403.07300 | ✅ `--arch calf`（`cdf0344`） | 跨模态双分支（DUAL-BRANCH）前向：一个跨模态分支把 patch 重新编程进 LLM 的词嵌入空间（复用 timellm 的 ReprogrammingLayer）+ 一个时间分支，两者都经过同一个共享的冻结 GPT-2，再融合。Additive 身份注入。从零实现的 adapter；特征/输出/梯度对齐损失（ALIGNMENT LOSSES）是任务层的扩展（不在 forward 里）。已端到端验证（smoke 2/2 ok，两个分支都有贡献，7 个测试）。 |

每个模型都会在适用的地方跑相同的 Level-A 模式 → 用来检验身份效应是 Time-LLM
特有的，还是对 LLM-TS 模型普遍成立。已完成的这三个模型都是纯 ADDITIVE（没有
prompt 通路），所以把它们的身份效应拿来和 Time-LLM 的 prompt 通路对比，是一个
干净的对照：身份信息是否既能通过数值型的 additive 通道起作用，也能通过 LLM
prompt 起作用？

---

## 3.1 Epoch / 早停策略（为什么不用固定 epoch 数）

不要把"正确的 epoch 数"写死。训练时给一个宽松的上限 + 基于验证集的**早停
（early stopping）**，让验证集自己挑出最佳 epoch —— 这也是 Time-LLM 论文和
本项目 harness 的做法：**train_epochs=30，patience=10**（`timellm_config.yaml`
的默认值）。

- `--phase dev`（train_epochs=5）只用于 PIPELINE 验证，不是科学配置，也不对齐
  任何论文。证据是它太少了：在 dev 版的 Tier-0 上，`best_epoch=4` 卡在了
  5-epoch 的上限 ⟹ 模型当时还在变好；早停从未真正触发过。
- 与论文/harness 对齐的 BASELINE 跑法用的是 YAML 里的 30 epoch + patience 10：
  `run_matrix.py --phase dev --train-epochs 30`（dev = 无 HPO、无 quick_test；
  `--train-epochs` 会覆盖 5 的上限；patience 10 来自 YAML）。由早停挑出最佳
  epoch。
- 记录逐 epoch 的验证曲线是好习惯（pipeline 会记录并报告
  `best_epoch`/`best_val_score`）；早停本身就已经编码了"最合适的 epoch"这件事。
- `--phase full` 会在 30-epoch 训练之上额外跑 Ray Tune HPO（50 次试验）——这是
  给最终论文级别数字用的，开销约为前者的 50 倍。

§3 中上面那些 dev-5 的 Tier-0 数字仅用于验证，已被 30-epoch 的跑法取代。

## 3.2 调试真实入口（`run_matrix.py`）

调试真正的 matrix 入口，而不是自己写一个脚本（自定义 driver 会偏离真实
pipeline —— 比如它构建的 val/test loader 就不一样 —— 所以它上面打的断点什么也
证明不了）。`run_matrix.py` 是在同一进程内（IN-PROCESS）执行每个 cell 的
（`_run_in_process`），所以 PyCharm 断点能命中 driver 以及 HPO 之后的
rebuild/retrain（主进程）。

- **快速调试配置：** `experiments/swiss_river/debug.yaml` —— 与
  `timellm_config.yaml` 对齐，但做了缩小（64 个训练窗口，2 个 epoch）。通过
  真实入口的 `--config` 透传参数加载它（由 `9b68db0` 添加）：

  ```
  python experiments/hydro_llm/run_matrix.py --config experiments/swiss_river/debug.yaml \
      --phase full --arch timellm --datasets swiss-river-1990 --modes none \
      --seeds 2026 --hpo-num-samples 2
  ```
  已验证：能加载 debug.yaml，应用其上限（max_train_samples 163968→64），并
  进入真实的 Ray Tune HPO（`Starting HPO via RayOptimizer`，samples=2）。
  `--config` 默认值是 `timellm_config.yaml`，所以真实跑法不受影响。
- **HPO 编排相关断点**（`build_optimizer`、`resolve_search_space`、ASHA、
  best-config、rebuild/retrain）：用 `--phase full`。Ray 2.x 会在 worker 进程
  里跑每个 trial 的 trainable，所以打在 trial 内部的断点不会命中 —— 但一个
  trial 跑的和 HPO 之后 retrain（主进程）用的是同一套
  `build_model`/`timellm.forecast`/`trainer.fit`，所以把断点打在那里。
- **立刻命中模型/训练相关断点**（不用等 HPO）：用 `--phase dev` —— 真实
  pipeline，直接在主进程训练，`build_model`/`forecast`/`fit` 里的断点会立刻
  命中。和 HPO trial 用的是同一套模型代码，只是少了 HPO 的外壳。
- 用 `--modes` / `--a2` 切换被测分支（例如 `--modes numeric_embedding --a2
  coordinates`）。

## 4. 执行顺序（按依赖排序）—— 截至 2026-08-04 的状态

1. ✅ task 3：把 `hydro_llm/run_matrix.py` 重新接到 pipeline + HPO 空间上。
   **（基础性工作）**
2. ✅ task 4：Level A 模式（`soft_prompt`/`text_embedding`/重命名为
   `numeric_embedding`）、A2 子变体（含**coordinates**，2026-08-04 接通）、A1
   丰富度（default/minimal/stats；coords 的文本格式化还没做）、A1.1 LoRA +
   `ln_only`、多 backbone（GPT2/BERT/**LLAMA**）。
3. ✅ task 5：其他 SOTA（GPT4TS/TEMPO/AutoTimes/CALF）作为换了 backbone 的
   adapter。
4. ✅ task 2：harness `run_experiment.py` 已标记为弃用（横幅 + 运行期
   DeprecationWarning）。
5. ✅ CHECKPOINT：已通知用户调试（task 6 的关卡）；用户正在本地调试 `none`
   这个 cell。
6. 🟡 task 7：完善文档（即本文件）—— 本轮进行中。
7. 🟡 task 6：集群 —— **Tier 0 运行中**（job 11557210，7 个 cell，`--phase
   full` HPO）；接下来是 Tier 1。
8. ⚪ 最终完整写作（等 Tier 0/1 的结果出来之后）。

剩余的收尾工作（不阻塞，按优先级排列）：(a) Tier-0 结果 → 旗舰基线表；
(b) Tier-1 剩余的 Level A + 含 coordinates 的 A2 梯度；(c) A1 `coords` 的文本
格式化；(d) BERT 权重同步 + LLAMA/BERT backbone sweep；(e) Tier-2 消融
（调优梯度、backbone 敏感性、A1 丰富度）；(f) Tier-2.4 身份×可训练性交互
（优先级最低）。

---

## 5. 弃用说明

- `experiments/swiss_river/run_experiment.py`（harness）以及
  `experiments/hydro_llm` 中旧的、驱动 harness 的代码路径：一旦 task 3 落地
  即视为**已弃用**。harness 仍会保留在代码树中，但只作为官方 Time-LLM 复现的
  参考（task 2 中加了弃用横幅）。任何实验入口都不得再调用它。
