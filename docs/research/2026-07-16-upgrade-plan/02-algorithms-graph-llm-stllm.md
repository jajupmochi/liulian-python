# 02 — Algorithms: graph / TS-LLM / ST-LLM

> Part of the **2026-07-16 upgrade plan** (target: NeurIPS / TPAMI). Master index:
> [`00-INDEX.md`](00-INDEX.md). Covers goal items **(c) graph as implicit entity identifier**,
> **(d) a second SOTA TS-LLM**, **(e) spatio-temporal LLM**.
>
> **Provenance**: delegated research pass, 2026-07-16. Every arXiv ID fetched and title-checked;
> mechanism columns taken from **official source code** where available. UNVERIFIED items marked.

---

## ⚠ Bottom line first

The real pressure on our novelty comes from the **graph literature**, not the LLM one. STGNNs have
carried learnable per-node embeddings since 2019, and **two papers have already theorized
"node embedding = sequence identity"** — neither is in our current draft:

- **Taming Local Effects in Graph-based Spatiotemporal Forecasting**, NeurIPS 2023
  ([2302.04071](https://arxiv.org/abs/2302.04071))
- **On the Regularization of Learnable Embeddings for Time Series Forecasting**, TMLR 2025
  ([2410.14630](https://arxiv.org/abs/2410.14630)) — its phrase *"may end up acting as mere
  sequence identifiers"* is almost our terminology.

**But the graph literature almost never uses per-window instance normalization** (STID and
STAEformer have *no* in-model normalization; they use dataset-level z-score). Therefore the
interaction we study — *injection position relative to per-channel normalization* — is
**structurally unobservable** in that literature. **C1 is clean.**

---

## FAMILY 1 — Graph / STGNN: is a graph an implicit entity identifier?

| Method | arXiv | Venue | Code | Identity mechanism | Equivalent to an entity identifier? | Position vs normalization |
|---|---|---|---|---|---|---|
| STGCN | [1709.04875](https://arxiv.org/abs/1709.04875) | IJCAI 2018 | [repo](https://github.com/VeritasYin/STGCN_IJCAI-18) | Fixed adjacency (distance kernel) | ❌ topology only, no per-node params | N/A |
| DCRNN | [1707.01926](https://arxiv.org/abs/1707.01926) | ICLR 2018 | [repo](https://github.com/liyaguang/DCRNN) | Bidirectional random-walk diffusion on a fixed graph | ❌ same | N/A |
| Graph WaveNet | [1906.00121](https://arxiv.org/abs/1906.00121) | IJCAI 2019 | [repo](https://github.com/nnzhan/Graph-WaveNet) | `adp = softmax(ReLU(nodevec1 @ nodevec2))`, `nodevec1/2` are `nn.Parameter(N×10)` | ✅ **YES** — free per-node vectors, merely constrained into a low-rank adjacency | after dataset-level z-score (no instance-norm) |
| ASTGCN | no arXiv (**UNVERIFIED**) | AAAI 2019, [ojs](https://ojs.aaai.org/index.php/AAAI/article/view/3881) | [repo](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch) | Fixed adjacency + ST attention | ⚠️ attention computed from input, not free params | N/A |
| MTGNN | [2005.11650](https://arxiv.org/abs/2005.11650) | KDD 2020 | [repo](https://github.com/nnzhan/MTGNN) | Graph-learning layer from learnable node embeddings M1/M2 | ✅ yes, as GWNet | same |
| **AGCRN** | [2007.02842](https://arxiv.org/abs/2007.02842) | NeurIPS 2020 | [repo](https://github.com/LeiBAI/AGCRN) | **Dual use**: (1) adjacency `softmax(ReLU(E@Eᵀ))`; (2) **NAPL** — `einsum('nd,dkio->nkio', E, W_pool)` + `E @ bias_pool`, i.e. **a dedicated convolution weight and bias per node** | ✅✅ **strongest evidence** — beyond an identifier, this is an identity→parameter hypernetwork | after dataset-level normalization |
| StemGNN | [2103.07719](https://arxiv.org/abs/2103.07719) | NeurIPS 2020 | [repo](https://github.com/microsoft/StemGNN) | Self-attention latent graph + graph Fourier | ⚠️ graph from input, no free per-node params | — |
| GTS | [2101.06861](https://arxiv.org/abs/2101.06861) | ICLR 2021 | [repo](https://github.com/chaoshangcs/GTS) | Discrete graph structure learning | ⚠️ graph generated from series features | — |
| **STID** | [2208.05233](https://arxiv.org/abs/2208.05233) | CIKM 2022 | [repo](https://github.com/GestaltCogTeam/STID) | Explicit: `node_emb = nn.Parameter(N × node_dim)`; forward is `cat([time_series_emb] + node_emb + tem_emb, dim=1)` | ✅✅✅ **literally identity** (the title says IDentity); proves a plain MLP + identity matches STGNNs | **no normalization anywhere in the model**; identity concatenated alongside already-z-scored embeddings ⟹ effectively post-norm |
| STAEformer | [2308.10425](https://arxiv.org/abs/2308.10425) | CIKM 2023 | [repo](https://github.com/XDZhelheim/STAEformer) | `node_emb (N×d)` + `adaptive_embedding (T×N×d)` = free params per (node, timeslot) | ✅✅ yes, extended to joint spatio-temporal identity | no input normalization; only LayerNorm inside attention |
| SimST | [2301.12603](https://arxiv.org/abs/2301.12603) | preprint | — | Drops message passing for local/global spatial learning | ⚠️ node identity not stated (**UNVERIFIED**) | — |
| **Taming Local Effects** | [2302.04071](https://arxiv.org/abs/2302.04071) | **NeurIPS 2023** | [repo](https://github.com/Graph-Machine-Learning-Group/taming-local-effects-stgnns) | Formalizes trainable node embeddings as **amortized node-specific components**; global/local trade-off framework | ✅✅✅ **theorizes it** | framework level |
| **Learnable-Emb Regularization** | [2410.14630](https://arxiv.org/abs/2410.14630) | **TMLR 2025** | see paper | First large-scale empirical study: end-to-end local embeddings *"may end up acting as mere sequence identifiers"*, harming transfer; fixed by perturbation / periodic reset | ✅✅✅ **studies identity itself** | cross-architecture |
| Montero-Manso & Hyndman | [2008.00444](https://arxiv.org/abs/2008.00444) | IJF 2021 | — | Global-vs-local theory | foundational premise | — |
| STGNN survey | [2303.14483](https://arxiv.org/abs/2303.14483) | TKDE 2023 | — | Urban-computing STGNN survey | — | — |
| BasicTS+ | [2310.06119](https://arxiv.org/abs/2310.06119) | TKDE 2024 | [repo](https://github.com/GestaltCogTeam/BasicTS) | 30+ methods, one benchmark; stresses dataset heterogeneity | fair-comparison protocol source | — |

### Verdict (Family 1)

The graph literature **has** implicitly answered "identity helps", and more deeply than our draft admits:

1. **Empirically answered**: GWNet/MTGNN adaptive adjacency *is* a per-node embedding under a
   low-rank constraint; AGCRN's NAPL maps node embeddings to per-node convolution weights; STID
   strips the graph and shows **identity itself**, not graph structure, is the main gain. So
   "adding identity improves accuracy" is **not** claimable as new. (Our draft's ❌ list already
   correctly excludes it.)
2. **Theoretically answered**: Taming Local Effects (NeurIPS 2023) explains node embeddings as
   amortized node-specific models; TMLR 2025's wording is nearly our terminology. **These two are
   the biggest missing-citation risk in the draft.**
3. **Three things the graph literature did NOT do — our defensible ground**:
   - It **structurally cannot observe** the injection-position × per-window-normalization
     interaction, because it does not use per-window instance normalization → **C1 clean**.
   - It reports only aggregate accuracy and never characterizes **when** identity helps
     (regime / entity-richness) → **C2/C3 clean**.
   - It is entirely numeric identity, with **no text-identity contrast** → **C5 clean**.

**Action**: split §2's "Channel identity" into two, adding a *Graph-based identity* paragraph that
explicitly concedes "identity helps" to this literature, and claims only the
**position/mechanism-dependence** characterization.

### The nearest neighbour: Channel Normalization (ICML 2025)

CN ([2506.00432](https://arxiv.org/abs/2506.00432), [code](https://github.com/seunghan96/CN)) —
PDF body extracted. It formalizes CID (*"for identical inputs xᵢ = xⱼ, a non-CID model φ always
produces identical outputs"*), and **its Table 1 is titled "Necessity of CID. Simply adding
different constant vectors to each channel token improves the performance."** — very close to our
post-norm arm.

**Good news**: the full text mentions RevIN only once in passing while surveying RMLP. It does
**not** argue that normalization erases identity, and does not study *when* identity is useful.
**Differentiating wording**: CN proves *whether channels can be distinguished* (an architectural
property) and offers a method; we study *where an identity signal must be injected to survive
per-window normalization*, and *when* injecting is worth it. Related: PCD/channel masks
([2410.23222](https://arxiv.org/abs/2410.23222), ICASSP 2026).

---

## FAMILY 2 — TS-LLM / foundation models beyond Time-LLM

| Model | arXiv | Venue | Class | Channels | Identity/covariate support | Normalization ↔ injection | Single-GPU |
|---|---|---|---|---|---|---|---|
| GPT4TS / One-Fits-All | [2302.11939](https://arxiv.org/abs/2302.11939) | NeurIPS 2023 Spotlight | frozen LLM + PEFT (GPT-2 first 6 layers) | CI | **none** | RevIN **before** patching → a constant added to x is erased | 4090 easily |
| TEST | [2308.08241](https://arxiv.org/abs/2308.08241) | ICLR 2024 | frozen-LLM reprogramming | mixed at input | task-level soft prompt only | not documented (**UNVERIFIED**) | needs multi-GPU |
| S2IP-LLM | [2403.05798](https://arxiv.org/abs/2403.05798) | ICML 2024 | frozen LLM + PEFT | CI | retrieval semantic anchors (not identity) | **RevIN first, anchors prepended after → post-norm survives** | 4090 ok |
| AutoTimes | [2402.02370](https://arxiv.org/abs/2402.02370) | NeurIPS 2024 | frozen LLM, 0.1% adapter | CI | **textual timestamps** encoded by the frozen LLM, added | **no instance norm** → the normalization axis collapses | 4090 ok |
| Time-MoE | [2409.16040](https://arxiv.org/abs/2409.16040) | ICLR 2025 Spotlight | from-scratch MoE | CI | none (repo TODO still lists covariates) | RMSNorm only | base/large ok, ultra needs H100 |
| Chronos | [2403.07815](https://arxiv.org/abs/2403.07815) | TMLR 2024 | from-scratch (T5) | strictly univariate | **none at all** | mean-scaling then quantization, **no post-norm injection point** | fine-tune single GPU |
| Moirai | [2402.02592](https://arxiv.org/abs/2402.02592) | ICML 2024 Oral | from-scratch | mixed | any-variate attention is **permutation-invariant in the variate index** = identity-blind by design | instance norm; variate bias inside attention → post-norm | fine-tune single GPU |
| UniTS | [2403.00131](https://arxiv.org/abs/2403.00131) | NeurIPS 2024 | from-scratch, multi-task | mixed | **per-dataset prompt token** (a de-facto dataset ID) | not documented (**UNVERIFIED**) | 3.4M params, very light |
| TimesFM | [2310.10688](https://arxiv.org/abs/2310.10688) | ICML 2024 | from-scratch decoder | univariate | paper says no covariates; **code disagrees**: 1.0/2.0 have a `freq` categorical, 2.5 removes it, 2025-10 adds XReg | first-patch statistics scaling; `freq` embedding added to patch embedding → post-norm survives | single GPU + LoRA |
| Lag-Llama | [2310.08278](https://arxiv.org/abs/2310.08278) | preprint (venue **UNVERIFIED**) | from-scratch | univariate | no identity, **but re-injects the mean/var that normalization removed, as covariates** | robust standardization first, scale **explicitly re-injected** → a direct precedent for our thesis | original work used a P100 12GB |
| TEMPO | [2310.04948](https://arxiv.org/abs/2310.04948) | ICLR 2024 | frozen LLM + LoRA | CI | **prompt pool** (M=30,K=3) + optional **real text** (TETS) | RevIN applied per decomposed component **first**, prompt prepended **after** → post-norm survives | single GPU |
| **UniTime** | [2310.09751](https://arxiv.org/abs/2310.09751) | WWW 2024 | frozen LLM + PEFT (GPT-2, 6 layers) | **strictly CI** | **domain instruction = a human-written natural-language identity string**, prepended | stationarization before patching, instruction at transformer input → **post-norm survives** | author used 1×A100; 4090 with smaller batch |
| CALF (was LLaTA) | [2403.07300](https://arxiv.org/abs/2403.07300) | AAAI 2025 | frozen LLM + LoRA, dual branch | CI | the "text" branch is really a PCA of vocabulary embeddings, **not real text** | not documented (**UNVERIFIED**) | single GPU |

### Recommended second TS-LLM: **UniTime**

The only model that writes *identity text* into its mechanism. Swap the domain instruction for an
**entity name** → text identity; swap it for a learned **ID embedding** → numeric identity.
**Changing one field gives a clean 2×2.** Stationarization comes first and the instruction goes in
after, so the published configuration *is* the post-norm arm natively; the pre-norm arm only
requires putting the same identity into `x` as a constant channel — exactly isomorphic to the
`concat_to_x` vs `add_after_patch` contrast already validated on PatchTST. Strict channel
independence guarantees the premise that the model *cannot* otherwise know which series it is.
GPT-2 (6 layers) runs on one 4090, and the code follows Time-Series-Library conventions, so it
shares a harness with Time-LLM.

**Runner-up TEMPO** (the only model with both modality paths natively: prompt pool = numeric,
TETS = real text; unambiguous RevIN placement) — cost: the STL three-branch split introduces a
"which component should identity attach to" confound. **Third GPT4TS**: the cleanest control, no
prompt mechanism at all, so injection position becomes the sole independent variable; cheapest.

**Explicitly not recommended**: **AutoTimes** (no instance norm → the normalization axis collapses,
half the experiment becomes unmeasurable); **Chronos** (strictly univariate, zero covariates,
scaling before quantization ⟹ structurally **no post-norm injection point** — but for exactly that
reason it is a perfect *negative control*, runnable zero-shot at near-zero cost for the discussion);
**Moirai** (permutation-invariant by design — a paragraph in the discussion, not an experiment).

---

## FAMILY 3 — Spatio-temporal LLMs: the ready-made text-vs-numeric battleground

| Method | arXiv | Venue | Code | How spatial identity enters the LLM | vs normalization | LLM frozen? | Single-GPU |
|---|---|---|---|---|---|---|---|
| **ST-LLM** | [2401.10134](https://arxiv.org/abs/2401.10134) | MDM 2024 | [repo](https://github.com/ChenxiLiu-HNU/ST-LLM) | **(b) learned per-node numeric embedding**: `node_emb = nn.Parameter(N, gpt_channel)`, `cat([input_data]+[tem_emb]+node_emb, dim=1)` → 1×1 Conv → GPT | **after normalization, but normalization is a single global scalar** (`StandardScaler` over the whole training tensor) → **per-node offsets survive anyway**; no RevIN | partial (PFA), last U=1–2 layers + LayerNorm trainable | GPT-2 first 6 layers → ample on 4090 |
| **UrbanGPT** | [2403.00813](https://arxiv.org/abs/2403.00813) | KDD 2024 | [repo](https://github.com/HKUDS/UrbanGPT) | **(a) natural-language identity + (b′) projected numeric ST tokens**. The instruction literally names the borough and POI categories ("located within the Staten Island borough district…"); numerics go through a dilated TCN → alignment `W_p∈R^{d×4096}` → inserted at `<ST_start>…<ST_end>`. **No coordinates** | **structurally immune**: identity lives entirely in the text channel, so numeric normalization cannot erase it | instruction tuning + trainable ST-MLP adapter; whether the base moves is **unclear** | Vicuna-7B; repo trains on 8 GPUs w/ DeepSpeed; fp16 inference ≈14GB → 4090 can infer |
| TPLLM | [2403.02221](https://arxiv.org/abs/2403.02221) | preprint | not found | **(c) graph encoder, no per-node ID vector**: GCN + parallel 1D-CNN; identity purely topological | global scaler; **plus `LN(ReLU(GE+SE))` after fusion — LayerNorm is a second potential erasure point**. No code ⟹ unverifiable | frozen GPT-2 + LoRA(Q,K), ~0.95% trainable | GPT-2 small, 1×4090 |
| GATGPT | [2311.14332](https://arxiv.org/abs/2311.14332) | preprint | not found | **(c) graph attention, no explicit node ID**: Gaussian-kernel distance adjacency + multi-head GAT; two nodes with identical neighbourhoods are indistinguishable | input normalization not documented | partial: self-attention + positional embeddings frozen; add-and-norm + linear head tuned | GPT-2 small→XL, 1×4090 (est.) |
| STG-LLM | [2401.14192](https://arxiv.org/abs/2401.14192) | no venue listed | not found | (c/partial) STG-Tokenizer turns graph data into tokens; whether it holds per-node learnable IDs is **UNVERIFIED** | **unknown, no code** | adapter-only | backbone size **UNVERIFIED** |
| ST-LINK | [2509.13753](https://arxiv.org/abs/2509.13753) | CIKM 2025 | [repo](https://github.com/HyoTaek98/ST_LINK) | **new mechanism**: spatial correlation as a RoPE-style **rotary transform** (SE-Attention) — neither a text name nor an additive ID | unclear, needs code reading | **UNVERIFIED** | **UNVERIFIED** |
| TimeCMA | [2406.01638](https://arxiv.org/abs/2406.01638) | AAAI 2025 Oral | [repo](https://github.com/ChenxiLiu-HNU/TimeCMA) | **(e) essentially none**: the prompt holds only `<time>`/`<color>` (timestamps + values); **no variable or station name in the template** | **RevIN before embedding, nothing re-injected afterwards** → precisely the "identity destroyed" configuration our theory predicts | **fully frozen** GPT-2 | 4090 (est.) |
| UrbanCLIP | [2310.18340](https://arxiv.org/abs/2310.18340) | WWW 2024 | [repo](https://github.com/siruzhong/WWW24-UrbanCLIP) | **(a) text identity** (image-generated descriptions); region profiling, not forecasting | N/A | contrastive training | 4090 (est.) |
| CityGPT | [2406.13948](https://arxiv.org/abs/2406.13948) | KDD 2025 | [repo](https://github.com/tsinghua-fib-lab/CityGPT) | **(a) pure text identity** (street/POI/address knowledge in instructions); not a numeric forecaster | N/A | full fine-tune (SWFT) | 6–8B full → 4090 insufficient; H100+LoRA ok |

> **Citation trap (record in the .bib)**: three different papers are called "ST-LLM". The traffic
> one is [2401.10134](https://arxiv.org/abs/2401.10134); [2404.00308](https://arxiv.org/abs/2404.00308)
> is video understanding; [2507.05258](https://arxiv.org/abs/2507.05258) is embodied 3D reasoning.
> **ST-LLM+** (TKDE 2025, [Xplore 11005661](https://ieeexplore.ieee.org/document/11005661/)) has
> **no arXiv version** — cite the journal.

Surveys, all verified: [2310.10196](https://arxiv.org/abs/2310.10196) (CSUR) ·
[2402.01749](https://arxiv.org/abs/2402.01749) · [2504.02009](https://arxiv.org/abs/2504.02009) (TIST 2025) ·
[2503.13502](https://arxiv.org/abs/2503.13502).

### Verdict (Family 3)

The ST-LLM literature has **split into two identity modalities that have never been compared**:
a *text* branch (UrbanGPT writes district names and POI categories into the instruction; CityGPT and
UrbanCLIP go further, the whole representation being a textual description of place) and a *numeric*
branch (ST-LLM's free `nn.Parameter` lookup; TPLLM/GATGPT's adjacency-driven encoders; ST-LINK's
rotary spatial transform). Both report gains over identity-free baselines, but **every identity
mechanism is bundled with a new encoder, a new graph, or a new freezing strategy — no published
number isolates the modality itself.** That is exactly the gap our C5 fills.

Injection position relative to normalization is an entirely undiscussed variable here, and the two
data points verified from code point in opposite directions: **ST-LLM** uses a single global scalar
mean/std, so per-node offsets survive regardless; **TimeCMA** uses RevIN *and* has no series name in
the prompt — the identity-destroyed-and-never-restored configuration. Three of the eight
(TPLLM/GATGPT/STG-LLM) document no normalization and ship no code, so it cannot be reconstructed.

---

## Priorities: what the paper should add

| # | Item | Type | Single-GPU | Est. GPU-h | Top-venue required? |
|---|---|---|---|---|---|
| **1** | Add a *Graph-based identity* paragraph to §2: GWNet/MTGNN adaptive adjacency, **AGCRN NAPL**, STID, STAEformer, **Taming Local Effects**, **Learnable-Emb Regularization**, **Montero-Manso & Hyndman**; explicitly concede "identity helps" to this literature | citation only | — | **0** | ✅ **REQUIRED** — the largest missing-citation risk |
| **2** | Sharpen the differentiation from **CN (ICML 2025)** (its Table 1 "adding different constant vectors to each channel token improves performance" = our post-norm arm); confirm it does **not** claim normalization-erasure, and hold C1 on that basis | writing only | — | **0** | ✅ **REQUIRED** |
| **3** | **Second TS-LLM = UniTime** (4 datasets × {none, text ID, numeric ID} × {pre-norm, post-norm} × 3 seeds) | experiment | ✅ 1×RTX4090 (smaller batch) | **~30–70** | ✅ **REQUIRED** — the draft itself lists "GPT-2 only" as a limitation; UniTime turns text identity from our bolt-on probe into a model-native mechanism |
| **4** | **Generalize C1 to a second instance-norm backbone**: iTransformer + RevIN on/off toggle | experiment | ✅ 1×RTX4090 | **~10–20** | ✅ **REQUIRED** — draft §9 already concedes this gap, and CN specifically attacks iTransformer as non-CID |
| **5** | **Graph control arm**: STID or STAEformer as a third architecture — it has **no per-window normalization**, so it is a natural counterexample where injection position should *not* matter | experiment | ✅ 1×RTX4090 (MLP, very light) | **~3–8** | 🟡 strongly recommended — upgrades C1 from "a property of PatchTST" to "a property of normalization" |
| **6** | **Chronos zero-shot negative control** — structurally no post-norm injection point ⟹ the "identity cannot be injected" endpoint | experiment (inference only) | ✅ 1×RTX4090 | **~1–2** | 🟡 nice-to-have, very cheap, strong discussion material |
| **7** | Add a *text vs numeric identity in ST-LLMs* paragraph: UrbanGPT (text) vs ST-LLM (numeric) vs TimeCMA (RevIN + no identity); note nobody has isolated the modality | citation only | — | **0** | 🟡 strongly recommended — lifts C5 from "an observation on GPT-2" to "filling a stated gap in the ST-LLM literature" |
| **8** | TEMPO as a third LLM arm (native prompt pool + real text) | experiment | ✅ 1×RTX4090 | **~20–40** | ⚪ nice-to-have; diminishing returns after #3 |
| **9** | Moirai's any-variate permutation invariance in the discussion ("identity-blind by design") | writing only | — | **0** | ⚪ nice-to-have, one paragraph |

> **All GPU-h are estimates**, extrapolated to a 4090 from each paper's self-reported hardware
> (UniTime 1×A100, GPT4TS 1×V100 32GB, STID 1×2080Ti). Not measured.

### Must stay flagged as UNVERIFIED

ASTGCN has no arXiv version · Lag-Llama's venue and parameter count · whether SimST uses node
identity · normalization in CALF / UniTS / TEST · STG-LLM's backbone size and normalization ·
ST-LINK's freezing strategy and scale · TPLLM/GATGPT/STG-LLM all lack usable code, so their
normalization placement cannot be reconstructed · TimesFM's paper and code contradict each other on
the `freq` input (if used as an identity hook, cite the model card, not the paper).
