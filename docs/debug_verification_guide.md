# 实验逐个验证指南（PyCharm 手动 Debug）

> **用途**：在 PyCharm 里手动 debug，逐个验证**所有 entity 方案 × 所有模型 × 所有数据**的
> 程序跳转流程与数据变化。本文是"从哪里下断点、看什么、应该看到什么"的完整地图。
>
> **最后更新** 2026-07-16 · 配套断点文件 `.debug/breakpoints.entity-verification.json`
>
> **本文合并了**：[`architecture.md`](architecture.md) · [`forecasting_pipeline.md`](forecasting_pipeline.md) ·
> [`entity_identifiers.md`](entity_identifiers.md) · [`data_loaders.md`](data_loaders.md) ·
> [`breakpoint_bridge.md`](breakpoint_bridge.md) · [`results_json.md`](results_json.md) ·
> [`research/entity-id-deep/`](research/entity-id-deep/)（20+ 篇逐模型深读）

---

## 0. 术语速查（先扫一眼）

| 术语 | 含义 |
|---|---|
| **entity / 实体** | 一条独立序列所属的对象：河流站点、路段传感器、用电客户 |
| **identifier_mode** | 身份编码方式：`none` / `embedding` / `onehot` / `numeric_id` / `sinusoidal` / `random` / `coordinates` / `descriptors` |
| **transparent 模式** | 非学习型身份（onehot/numeric_id/sinusoidal/random/coordinates）——固定向量，**0 可学习参数** |
| **per_entity（每实体）** | 一个共享模型，每个样本自带身份；样本 = 一个站点的一段序列 |
| **multi_channel（多通道）** | 所有实体当作通道一次性喂入；样本 = 全站点同一时间窗 |
| **注入位置** | 身份加在**每通道归一化之前**（`concat_to_x`）还是**之后**（`add_after_patch`） |
| **x_enc / x_mark_enc** | 编码器输入序列 `(B, L, C)` / 其时间戳+身份列 `(B, L, M)` |
| **B / L / C / N / D** | batch / 回看长度 / 特征通道数 / 实体数 / 身份向量维度 |

---

## 1. 从哪里开始跑——四个入口

按"想验证什么"选入口。**验证单元格逻辑请用入口 A**（最容易下断点）。

| # | 入口 | 文件 | 适合验证 |
|---|---|---|---|
| **A** | 单次实验（推荐起点） | [`liulian/pipeline.py`](../liulian/pipeline.py) → `run_experiment()` (L939) | 一个 (数据,模型,身份模式) 单元格的完整链路 |
| **B** | 矩阵批跑 | [`experiments/entity_identifier/run.py`](../experiments/entity_identifier/run.py) → `run_matrix()` (L453) | 单元格枚举、跳过逻辑、manifest 记录 |
| **C** | 集群作业封装 | [`experiments/entity_identifier/run_job.py`](../experiments/entity_identifier/run_job.py) | sbatch 参数、resume、run-tag |
| **D** | Time-LLM 独立 harness | [`experiments/swiss_river/run_experiment.py`](../experiments/swiss_river/run_experiment.py) | Time-LLM 的文本/嵌入身份注入（**独立链路，见 §6**）|

### 入口 A 的 PyCharm Run Configuration

```
Module name (不是 script):  liulian.cli
Parameters:                 run --config experiments/swiss_river/configs/lstm_1990.yaml
Working directory:          <repo root>
Python interpreter:         <repo>/.venv/bin/python
Environment variables:      PYTHONUNBUFFERED=1
```

> **先把规模调小**再 debug，否则单步要等很久。在 config 里临时设
> `train_epochs: 1`、`hpo_num_samples: 1`（或直接用 `--phase dry`）。
> ⚠ 调小后的结果**不可用于论文**——`_assert_no_dev_caps_in_full()`
> ([run.py:407](../experiments/entity_identifier/run.py)) 就是防止这种配置混入正式跑批的守卫。

---

## 2. 端到端调用链（Debug 时的"地图"）

```
liulian/cli.py  run
      │
      ▼
pipeline.run_experiment()                         L939   ← ★入口断点
      │
      ├─ build_dataset(config)                    L163   ← 数据构建（§4）
      │     └─ liulian/data/data_factory.py → 具体 loader
      │           · swiss_river.py / csv_dataset.py / pems_dataset.py
      │           └─ data/ts/timeseriesdataset.py
      │                 make_entity_features()      L80   ← ★transparent 身份"烘焙进数据"
      │
      ├─ auto_detect_enc_in(dataset)              L392   ← ★enc_in 自动推断（易错点！）
      │     ├─ _transparent_injection_kind()      L339
      │     └─ _transparent_feature_dim()         L373   ← 身份维度 D 从这里来
      │
      ├─ build_model(config, dataset)             L419   ← ★三种 wrapper 在此挂载（§3）
      │     └─ liulian/models/torch/entity_mixin.py
      │           · EntityWrapper                  L50   （embedding, per_entity）
      │           · ChannelEntityWrapper           L153  （embedding, multi_channel）
      │           · EntityTransparentWrapper       L372  （transparent, per_entity）
      │           · EntityAwareMixin               L599  （适配器混入）
      │
      ├─ build_loaders(dataset, config)           L654
      ├─ build_optimizer(config)                  L693   ← Ray Tune / 网格
      ├─ build_experiment(...)                    L804
      │
      └─ Experiment 状态机  INIT→TRAIN→EVAL→INFER→COMPLETED
            └─ liulian/runtime/trainer.py
                  loss.backward()                 L722   ← ★确认真的在反传
                  optimizer.step()                L723
                  evaluate()                      L368   ← 指标计算
                        │
                        ▼
                  results.json  （见 docs/results_json.md）
```

---

## 3. 身份注入机制——**验证的核心**

三条互斥的注入路径。**先确认当前单元格走的是哪一条**，否则断点会打空。

| 路径 | 触发条件 | 实现 | 可学习参数 | 注入时机 |
|---|---|---|---|---|
| **P1 数据层烘焙** | transparent 模式 + 数据层支持 | [`make_entity_features()`](../liulian/data/ts/timeseriesdataset.py) L80 | 无 | 建 Dataset 时就写进 `x_enc` |
| **P2 模型层 transparent** | transparent 模式 + per_entity | `EntityTransparentWrapper` L372 | **0** | forward 里查表→拼接 |
| **P3 学习型 embedding** | `identifier_mode='embedding'` | `EntityWrapper` L50 / `ChannelEntityWrapper` L153 | `nn.Embedding` | forward 里查表→拼接→**投影回原宽度** |

> **P1 与 P2 按设计是 bitwise 相同的**——P2 的表就是逐站点调 P1 生成的。
> 回归测试：[`tests/models/torch/test_transparent_injection_equivalence.py`](../tests/models/torch/test_transparent_injection_equivalence.py)。
> Debug 时若发现两者不等，**那就是真 bug**。

### 3.1 身份特征表 `_build_channel_features()` — [entity_mixin.py:274](../liulian/models/torch/entity_mixin.py)

返回 `(N, D)`，每行是一个站点的身份向量。**逐模式的期望值**（断点必查）：

| mode | D（维度） | 生成方式 | 断点应看到 |
|---|---|---|---|
| `onehot` | **N** | `torch.eye(N)` (L308) | 每行恰好一个 1，其余 0；行和 = 1 |
| `numeric_id` | **1** | `i/(N-1)` (L310-313) | 递增等距，首行 0.0、末行 1.0 |
| `sinusoidal` | 16 | 前半 sin、后半 cos (L315-326) | 值域 `[-1,1]`；**行与行不同**；idx=0 行 sin 部分全 0 |
| `random` | 16 | sha256(`seed:id`)→rng→标准正态→**L2 归一化** (L328-343) | 每行 L2 范数 = 1.0；同 seed 可复现 |
| `coordinates` | **2** | 站点 (lat,lon)，**逐维 min-max 归一化** (L345-367) | 值域 `[0,1]`；**绝不能全 0** |

> **⚠ 历史 bug 守卫（重点验证）**：`coordinates` 模式若有任何站点缺坐标，
> 代码**直接抛 ValueError**（L354），**不做静默补零**。
> 补零会让"所有通道身份相同"却仍然跑成功——这正是 2026-06-11 之前
> multi_channel swiss 坐标格全部作废的原因
> （见 [`research/2026-06-13-swiss3dt-results.md`](research/2026-06-13-swiss3dt-results.md)）。
> **验证方法**：故意删掉一个站点坐标，确认它 raise 而不是继续跑。

### 3.2 `EntityTransparentWrapper.forward()` — [entity_mixin.py:459](../liulian/models/torch/entity_mixin.py)

数据变化（**逐行核对形状**）：

```
输入   x_enc      (B, L, C_base)         例：swiss per_entity → (32, 90, 2)
       x_mark_enc (B, L, M)              最后一列 = 每样本 entity 索引
  1. ids   = x_mark_enc[:, 0, id_col].long()      → (B,)
  2. feat  = table[ids]                           → (B, D)      查固定表
  3. feat  = feat.unsqueeze(1).expand(B, L, D)    → (B, L, D)   沿时间铺开
  4. x_enc = cat([x_enc, feat], dim=-1)           → (B, L, C_base + D)   ★拼接
输出   直接喂内层模型；**无投影**、**无可学习参数**
```

**因此内层模型必须以 `enc_in = C_base + D` 构建**——这是最常见的 shape 崩溃点，
由 `auto_detect_enc_in()` ([pipeline.py:392](../liulian/pipeline.py)) 负责。

### 3.3 `EntityWrapper.forward()`（embedding）— [entity_mixin.py:95](../liulian/models/torch/entity_mixin.py)

```
  1. ids  = x_mark_enc[:, :, entity_id_col]            → (B, L)
  2. emb  = nn.Embedding(N, E)(ids)                    → (B, L, E)   ★可学习
  3. cat   → (B, L, C_base + E)
  4. Linear 投影回 (B, L, C_base)                      ★与 transparent 的关键区别
输出   内层模型看到**原始宽度**，架构无需改动
```

> **容量对照的意义**：`random`（0 可学习）若 ≈ `embedding`（有可学习参数），
> 说明增益来自**每实体可区分性**而非容量。这是论文 C5 的核心论证。

---

## 4. 数据层：三类数据的不同链路

| 数据 | loader | 切分方式 | 实体来源 | 注意 |
|---|---|---|---|---|
| **swiss-river** (1990/2010/zurich) | [`data/swiss_river.py`](../liulian/data/swiss_river.py) | per_entity（ConcatDataset，每站一个子集） | 28 个 FOEN 站点，有坐标 | 唯一有真实站点元数据的数据 |
| **traffic / electricity / ETT / weather...** | [`data/csv_dataset.py`](../liulian/data/csv_dataset.py) | multi_channel（通道=实体） | 通道索引 | ETT/weather 是**弱实体**（同一物体的不同变量）|
| **PEMS03/04/07/08** | [`data/pems_dataset.py`](../liulian/data/pems_dataset.py) | multi_channel | 传感器 | 自带邻接图 |

**断点必查的数据不变量**：

1. `x_enc` 无 NaN（swiss 2010/zurich 曾因 NaN 训练崩溃，已用逐站 dropna 修复）
2. 归一化后 `x_enc` 均值≈0、标准差≈1（**每通道**，不是全局）
3. `x_mark_enc` 最后一列 = 实体索引，且 `0 ≤ id < N`
4. train/val/test **时间不重叠**（无泄漏）——在 `build_dataset` 后查各 split 的时间范围

---

## 5. 断点清单（直接导入 PyCharm）

配套文件：`.debug/breakpoints.entity-verification.json`
导入命令（**导入前先关闭 PyCharm**，见 [`breakpoint_bridge.md`](breakpoint_bridge.md)）：

```bash
python tools/breakpoint_bridge.py import-pycharm --bridge .debug/breakpoints.entity-verification.json
```

| # | 文件:行 | 停下来看什么 | 期望 |
|---|---|---|---|
| 1 | `pipeline.py:939` `run_experiment` | `config` 全貌 | `identifier_mode`/`train_epochs`/`seed` 与预期一致 |
| 2 | `pipeline.py:163` `build_dataset` | 返回的 dataset 类型 | swiss=ConcatDataset，其余=单 Dataset |
| 3 | `pipeline.py:392` `auto_detect_enc_in` | 返回值 | = `C_base + D`（transparent）或 `C_base`（none/embedding）|
| 4 | `entity_mixin.py:274` `_build_channel_features` | 返回表 `(N,D)` | 按 §3.1 逐模式核对 |
| 5 | `entity_mixin.py:459` `EntityTransparentWrapper.forward` | `x_enc` 拼接前后形状 | 末维 `C_base → C_base+D` |
| 6 | `entity_mixin.py:95` `EntityWrapper.forward` | emb 形状 + 投影后形状 | 投影后回到 `C_base` |
| 7 | `trainer.py:722` `loss.backward()` | `loss` 是否有限、是否下降 | 非 NaN；跨 epoch 下降 |
| 8 | `trainer.py:723` `optimizer.step()` | **权重是否真的变了** | step 前后取一个参数对比，必须不同 |
| 9 | `trainer.py:368` `evaluate` | 指标字典 | `rmse`/`denorm_rmse` 均有限 |

> **断点 8 是防"假跑"的关键**：只要 `loss.backward()` 漏了或梯度被截断，
> 训练照样"跑完"但权重没动。务必亲眼确认参数变化。

---

## 6. Time-LLM 是**独立链路**（最大的坑）

Time-LLM 不走上面的 pipeline，走 [`experiments/swiss_river/run_experiment.py`](../experiments/swiss_river/run_experiment.py) +
`refer_projects/Time-LLM-Revised/` 的 data_provider。

**关键事实（验证时必须知道）**：该 harness 对**所有数据集**都是
**channel-independent**，模型 `forecast()` 里永远 `N=1`——每个样本是**一个通道/站点**
的单变量切片。所以：

- ETT 的 `__len__ = 窗口数 × enc_in`（ETTh1 训练集 56231 = 7 × 8033 可验证）
- 每样本的实体 id 在数据层**算出来后被丢弃**了
- 因此 multi_channel 的 `b % N` 取身份**在这里完全无效**（`b % 1 = 0`）

正确机制是把实体 id 作为 **`x_mark` 最后一列**透传，模型侧用
`self.entity_id_mark_col` 读取。断点看 `timellm.py` 的
`_resolve_entity_descs()`（文本身份）与 `forecast()` 里
`enc_out = enc_out + self.entity_embedding(ids).unsqueeze(1)`（数值身份，**patch 之后=归一化之后**）。

详见 [`research/2026-06-24-timellm-verification-and-id-plan.md`](research/2026-06-24-timellm-verification-and-id-plan.md)。

---

## 7. 逐维度验证矩阵（按这个顺序跑）

### 7.1 按 identifier_mode（8 种）

| mode | 建议验证数据 | 重点确认 |
|---|---|---|
| `none` | 任意 | 基线：**无任何身份列**进入模型 |
| `embedding` | swiss-1990 lstm | `nn.Embedding` 梯度非零（真的在学）|
| `onehot` | swiss-1990 lstm | 表 = 单位阵；`enc_in` 增加 N |
| `sinusoidal` | swiss-1990 lstm | 行间不同；D=16 |
| `random` | swiss-1990 lstm | 每行 L2=1；换 seed 表变、结果可复现 |
| `coordinates` | swiss-1990 | **非全 0**；缺坐标时 raise |
| `numeric_id` | ⚠ **尚未纳入实验矩阵** | 代码支持（L310），可作为新增模式 |
| `descriptors` | ⚠ **尚未纳入实验矩阵** | 代码声明支持，需确认实现完整性 |

### 7.2 按模型

先验证 **LSTM**（最简单、身份增益最大 −24~−35%），再 **DLinear**（身份无感，作阴性对照），
再 **PatchTST**（有 instance-norm，注入位置敏感），最后 **Time-LLM**（独立链路）。
逐模型细节见 [`research/entity-id-deep/`](research/entity-id-deep/) 下 20+ 篇。

### 7.3 按数据

`swiss-1990`（实体丰富，主战场）→ `swiss-2010`/`zurich`（复现性）→
`electricity`/`traffic`（多通道、冗余）→ `ETTh1`（**弱实体阴性对照**）。

---

## 8. 已知陷阱清单（debug 时优先怀疑这些）

1. **instance-norm 抹除**：PatchTST 等带每通道归一化的模型，身份注在归一化**前**会被均值相减抹掉
   （swiss 12/12 格回归 +32~85%）。见 Figure 1 与 [`research/2026-06-16-channel-as-identity-ablation-design.md`](research/2026-06-16-channel-as-identity-ablation-design.md)。
2. **坐标补零**（已修，务必回归）：见 §3.1。
3. **Time-LLM 的 N=1**：见 §6。
4. **`enc_in` 不匹配**：transparent 模式忘了加 D → shape 崩溃。
5. **dev cap 混入正式跑**：`train_epochs=1` / `max_train_samples` 残留 →
   `_assert_no_dev_caps_in_full()` 会拦，但本地 debug 时要自己记得改回来。
6. **旧 tag 跨代码时代混用**：不同时期跑的格子**不可同表比较**（2026-07 已因此拦下一次并表）。
7. **`docs/entity_identifiers.md` L185 的 caveat 可能已过期**——它说 transparent 只在
   swiss+lstm 生效，但 multi_channel transparent wrapper 后来已接通（task #33）。**需实地验证并更新该文档**。

---

## 9. 验证完成的判定标准

一个单元格算"验证通过"，需要同时满足：

- [ ] 身份表 `(N,D)` 数值符合 §3.1 的逐模式期望
- [ ] `x_enc` 在注入前后的形状变化符合 §3.2/§3.3
- [ ] `optimizer.step()` 前后权重确实改变
- [ ] loss 有限且随 epoch 下降
- [ ] `results.json` 的 metrics 有限，且与已提交表格中的该格数值一致
- [ ] 无 NaN、无时间泄漏、实体 id 在合法范围

---

## 10. 相关文档索引

| 文档 | 内容 |
|---|---|
| [`architecture.md`](architecture.md) | 分层架构与边界 |
| [`forecasting_pipeline.md`](forecasting_pipeline.md) | 预测流水线 |
| [`entity_identifiers.md`](entity_identifiers.md) | 身份模式 + **数据集实体丰富度判定表** |
| [`data_loaders.md`](data_loaders.md) · [`datasets.md`](datasets.md) | 数据加载 |
| [`results_json.md`](results_json.md) | 结果文件结构 |
| [`search_spaces.md`](search_spaces.md) | HPO 搜索空间 |
| [`research/entity-id-deep/`](research/entity-id-deep/) | 20+ 篇逐模型深读 |
| [`research/STATUS.md`](research/STATUS.md) | 实验现状总表 |
| [`research/paper-draft.md`](research/paper-draft.md) | 论文初稿 |
