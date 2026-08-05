> **语言：** [English](DELIVERY.md) | 中文

# Hydro-LLM levels — 交付与调试交接文档 (2026-08-04)

一页说明：已完成什么、还剩什么、如何调试、如何运行。完整设计见
[00-MASTER-SPEC.md](00-MASTER-SPEC.md)。

## 已完成的部分(代码,已在本地验证——build + forward + 与 `none` 有差异)

| task | status |
|---|---|
| 1. ln_only explained | ✅ MASTER-SPEC §2.1 |
| 2. harness deprecated | ✅ 在 `experiments/swiss_river/run_experiment.py` 中加了 banner + 运行时 DeprecationWarning |
| 3. entry = hydro_llm/run_matrix.py → pipeline (run_with_config) + Ray Tune HPO | ✅ 端到端 smoke 验证 |
| 3. HPO space `timellm_swiss`(config 外部化) | ✅ 已提交 `0b929c3`;d_model{16,32,64} d_ff{32,128,256} lr{1e-3,1e-2} llm_layers{3,6},以 canonical 为中心,配了 dead-knob guard + 6 个测试 |
| 3. config 对齐到 AUTHORITATIVE 上游(而非 deprecated harness) | ✅ `215b057`;d_model 32/d_ff 128/batch 32/patch 16(Time-LLM ETTh1 + LSTM/PatchTST 共享) |
| 3. epoch 策略由诊断结果决定 | ✅ 30 + early stop 已足够(两个 lr 都在约第 8 个 epoch 收敛);`figs/epoch_diagnostic_1990_none.png`;在 swiss 上 lr 1e-3 优于 1e-2 |
| 3. single-loader contract 已文档化 | ✅ `3112476`;run_with_config→load_config 是唯一的入口点 |
| 4. Level A(5 种模式) | ✅ none / entity_description / numeric_embedding / soft_prompt / text_embedding |
| 4. A2(learnable/random/onehot/sinusoidal) | ✅ |
| 4. llm_tuning frozen/ln_only | ✅ |
| 4. multi-backbone GPT2/BERT | ✅ |
| 4. llm_tuning lora(A1.1) | ✅ peft 已安装并验证(trainable 50.9M/132.8M) |
| 5. GPT4TS(负对照,仅 additive) | ✅ 基于同一 entry+pipeline 构建,`--arch gpt4ts` |
| 6. entity_description 可用性 guardrail | ✅ `ca38e89`;Tier-0 = 7 个 cell(2010/zurich 的 entity_description 自动跳过)+ 5 个测试 |
| 4. A2 coordinates | ✅ 已接通 `8b58f83` —— 来自 `graph_*.pth` 的真实 CH1903 坐标,28 个不同值,e2e 1/1 ok(此前的 "BLOCKED / no data" 是一次错误的排查结论) |
| 4. multi-backbone LLAMA | ✅ `huggyllama/llama-7b` 权重已下载到集群(13G),加载正常(hidden 4096, vocab 32000) |
| 6. cluster Tier-0(真实、对齐后的 config) | 🟡 RUNNING —— job **11557210** `--phase full` HPO,7 个 cell,cell 1 正在探索 `timellm_swiss` |
| debug entry | ✅ `run_matrix.py --config debug.yaml`(`9b68db0`)—— 真实的 matrix entry 加载快速 debug config;已验证可进入 Ray HPO |

回归测试:`tests/runtime/test_entity_identifier_pipeline.py` 33 passed / 1 skipped
(coords 新增 2 个)。2×2(表示方式 × 注入位置)组合已全部完成。

### 本轮构建中发现并修复的 bug(6 个)

1. `results.json` 缺少 rmse → 导致 cell 被图表构建器静默跳过。
2. harness YAML 静默覆盖了所有 CLI override(`--train_epochs 1` 实际跑了 30)。
3. matrix 回归:timellm 在每个 dataset 上都被枚举 → KeyError(已限制为仅 swiss)。
4. entity_description 在 pipeline 中被静默降级为 baseline(描述文本未被加载)。
5. "text = zero effect" 的告警其实是一个损坏的 LOCAL tokenizer(vocab 为 1);模型本身是正确的。
6. pipeline 把模型 CONSTRUCTION 错误误标记为 "Unknown model"。
另外还有两个由首次真实集群运行暴露出的集群环境问题:`cache_dir` 指向了一个空的项目缓存(→ 改为默认 HF 缓存),以及 "Too many open files"(→ 改为 file_system sharing + `ulimit -n`)。

## 还剩什么(按顺序)

**task 4 的收尾部分(Tier-1/2 消融实验,首批集群运行不需要)。**
按当前是否可立即执行,或被上游资源阻塞进行分类(测量时间 2026-08-03):

*本轮已完成:*
1. A1 prompt richness `default` / `minimal` / `stats` —— 已完成。`default` = 人工撰写的富文本,
   `minimal` = 纯位置 id(`adab88e`),`stats` = id + 每个站点仅基于 TRAIN 数据的温度
   均值/标准差/最小值/最大值,不存在泄漏(`_compute_station_train_stats` 只读取 train 部分数据;
   `309cc15`)。run_matrix `--a1` 驱动该功能;端到端 smoke 已验证。只有 `coords` richness
   仍被阻塞(阻塞于 #28,坐标数据流问题)。
2. lora(A1.1)—— 已完成(peft 已安装,trainable 50.9M 已验证);剩余部分只差集群上的 lora sweep。

*此前标记为 "BLOCKED" 的两项——现在均已完成(此前的阻塞结论有误):*
3. A2 `coordinates` —— ✅ 已完成 `8b58f83`。此前 "no coord data" 的结论是一次错误的排查:
   坐标数据其实一直都在 `dataset/swiss_river/graph_*.pth` 中(x 的第 0-1 列是 CH1903 坐标,
   第 2 列是站点 id)。此前的排查使用的是 `identifier_mode='none'`,该模式从不加载 graph。
   现在 `_load_topology` 会在 `coordinates_embedding` 时触发,pipeline 会暴露
   `config['coordinates']`,timellm 通过 `_build_channel_features` 构建该特征
   (no-fake-zero guard 通过:28 行互不相同的非零值)。e2e smoke 1/1 ok。
4. LLAMA backbone —— ✅ 已完成。`huggyllama/llama-7b`(公开重传版本,无 gated license)
   已于 2026-08-04 下载到集群 HF 缓存(13G,2 个 safetensors shard);加载正常
   (hidden 4096, vocab 32000)。剩余部分只差集群上的 LLAMA backbone sweep。
   (A1 的 `coords` prompt-richness 是唯一剩下的与坐标相关的项目,只涉及文本格式化。)

**task 5 —— 其他 SOTA LLM-TS 模型(设计;沿用同一 entry + pipeline):**
每个模型都以 `liulian/models/torch/<name>.py` 的形式实现,与 timellm 保持相同的接口约定
(`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None)`),在
`matrix`/`BASE_CONFIG_BY_PAIR` 中注册,并复用 identity 相关的管线逻辑。优先级与角色如下:

| model | ref | role | status |
|---|---|---|---|
| GPT4TS (OneFitsAll) | arXiv 2302.11939 | 🧪 负对照 | ✅ 已完成 —— patch → frozen GPT-2(LN+pos 可训练)→ 线性头,不含 prompt/reprogramming;仅 additive identity(`--arch gpt4ts`)。 |
| TEMPO | arXiv 2310.04948 | 分解 + frozen LLM | ✅ 已完成(`974c658`,`--arch tempo`)—— series_decomp 趋势+季节性分解,每个分量都经过同一个共享的 frozen GPT-2,再求和。从零实现的适配器(2 分量,非完整 STL);仅 additive identity,soft-prompt 是计划中的扩展。已端到端验证(smoke 2/2 ok,8 个单元测试)。 |
| AutoTimes | arXiv 2402.02370 | 自回归 | ✅ 已完成(`8ab418f`,`--arch autotimes`)—— 切分为 time token(token_len=pred_len),因果 frozen GPT-2,从最后一个 token 解码出下一段。从零实现的适配器(单步解码,无 timestamp token);仅 additive identity。已端到端验证(smoke 2/2 ok,9 个单元测试)。 |
| CALF | arXiv 2403.07300 | 跨模态对齐 | ✅ 已完成(`cdf0344`,`--arch calf`)—— 双分支 forward(跨模态 reprogramming + 时序分支),共享同一个 frozen GPT-2,再融合。Additive identity。从零实现的适配器;特征/输出/梯度层面的 ALIGNMENT LOSS 属于 task 层的扩展内容(不在 forward 中,因为 loss 由 task 所有)。已端到端验证(smoke 2/2 ok,两个分支均有贡献,7 个测试)。 |

identity 轴(Level A / A2)适用于每一个存在 prompt 或 embedding 注入点的模型,
GPT4TS(无 prompt)只支持 embedding/additive 模式 —— 这恰好可以用来检验 identity 效应
究竟是 Time-LLM 特有的,还是对 LLM-TS 模型普遍存在的。

**task 6 —— 集群(debug 之后进行):** 先跑 Tier 0 —— 共 **7 个 cell**(none / numeric_embedding.learnable
分别对应 swiss-1990/2010/zurich,以及 entity_description 仅对应 swiss-1990 —— 2010/zurich 没有
站点文本,会被 guardrail 自动跳过)。随后是 Tier 1(soft_prompt / text_embedding / A2 阶梯)。
全部为 gratis 资源、single seed 2026,通过 `experiments/hydro_llm/run_matrix.py --phase full`(开启
Ray Tune HPO,使用 `timellm_swiss` 空间)运行。决策(autorun 模式下做出):带 HPO 运行
(phase full),num_samples 取较小值,因为 HPO 是明确的需求;epoch 诊断已经把 epoch 预算
定在了 30 + early stop,所以每个 trial 的时长是有界的。

## 如何调试(PyCharm)—— 真实入口,核心部分现已就绪

调试 `run_matrix.py` 本身(自定义的 driver 会与真实 pipeline 产生偏差)。它以 IN-PROCESS 方式
运行每个 cell,因此断点可以命中 driver 和 post-HPO 的 retrain(主进程)。

```
Script:  experiments/hydro_llm/run_matrix.py
Workdir: <repo root>              Python: <repo>/.venv/bin/python
Env:     HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1
```

根据你要调试的内容,挑选对应的 Params 行:

- **真实 HPO 路径(使用快速 debug config):**
  `--config experiments/swiss_river/debug.yaml --phase full --arch timellm --datasets swiss-river-1990 --modes none --seeds 2026 --hpo-num-samples 2`
  —— 加载 `debug.yaml`(64 个 train window,2 个 epoch),进入真实的 Ray Tune HPO。断点可以命中
  `build_optimizer`/`resolve_search_space`/best-config/retrain。Ray 2.x 会在 worker 进程中运行
  每个 trial(trial 内部的断点不会命中——但 retrain 运行的是同一份代码)。
- **立即进入模型/训练代码(不用等待 HPO):**
  `--config experiments/swiss_river/debug.yaml --phase dev --arch timellm --datasets swiss-river-1990 --modes none`
  —— 真实 pipeline,主进程直接训练;断点可以在 `build_model`/`forecast`/`fit` 中立即命中。
- 用 `--modes` / `--a2` 切换被测分支(例如 `--modes numeric_embedding --a2 coordinates`)。
- 值得设置断点的位置:`liulian/models/torch/timellm.py` 的 `forecast()` —— `self._station_ids`
  的解析、identity 的注入点(entity_embedding / soft_prompt / text_proj /
  transparent_proj),以及 `_compose_prompt`。
- ⚠️ **本地 tokenizer 必须是完整的**,否则 prompt 路径会被静默地失效。目前的 guard 会在
  `len(tokenizer) < 1000` 时抛出异常;修复方法是运行
  `python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('openai-community/gpt2')"`
  (BERT 同理)。集群上的 gpt2 已经是完整的。

## 本轮的修正说明

- "text prompt = zero effect" 这一告警的真实原因是一个**损坏的本地 tokenizer**(vocab 为 1 →
  空 prompt),而不是模型本身的 bug。在使用完整 tokenizer 的情况下,text 确实会改变输出
  (diff 为 0.4958)。集群上的 tokenizer 是正常的(vocab 50257),因此已发布的结果以及 text
  这条路径都是有效的。详见 [[project_timellm_text_zero_effect]]。
