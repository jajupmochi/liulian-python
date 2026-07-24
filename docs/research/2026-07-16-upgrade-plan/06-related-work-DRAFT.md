# 06 — Related Work: publication-standard draft prose (§2)

> Part of the **2026-07-16 upgrade plan**. Master index: [`00-INDEX.md`](00-INDEX.md).
> This is **paper-ready prose** for goal item **(f)**, organised methodologically per the
> skeleton in [`01-related-work-survey.md`](01-related-work-survey.md). Every citation here
> appears in [`refs/refs.bib`](refs/refs.bib) (150 entries, programmatically fetched).
>
> Citation keys below are written as `[Author Year]` placeholders; swap for the `.bib` keys at
> LaTeX time. **UNVERIFIED items from the surveys are deliberately excluded from this draft.**

---

## 2 Related Work

Identity in a forecasting model is the answer to one question: does the model know *which*
series it is looking at? We organise prior work by **where that answer physically enters the
model**, because that turns out to be the axis along which results disagree.

### 2.1 Identity as a modelling choice, not an architecture

The question long predates deep learning. In panel econometrics a per-unit intercept (fixed
effects) or a shrunk per-unit draw (random effects) is exactly an entity identifier, and the
decision to include one is the decision of how much to pool. Modern forecasting inherited this
as the **global-versus-local** axis, which [Januschowski et al. 2020] elevate to a primary
classification criterion, displacing the older statistical-versus-machine-learning split.

The theoretical anchor is [Montero-Manso & Hyndman 2021]: a single global model can reproduce
any set of local forecasts, with **no assumption that the series are similar**, because local
complexity grows with the number of series while global complexity stays constant. Globality is
therefore a *complexity* argument rather than a *similarity* argument — which reframes identity
as a capacity dial rather than a domain assumption. The empirical counterpart is
[Hewamalage et al. 2022], who characterise when global models win as a function of homogeneity,
series count and series length; [Pesaran et al. 2024] give the panel-forecasting version, where
the pooled/per-unit/shrinkage choice depends jointly on the degree of heterogeneity, its
correlation with the regressors, and the ratio N/T.

We take from this literature the right to ask *when*, rather than *whether*, identity helps —
a question the deep-learning work that follows almost never poses.

### 2.2 A taxonomy of identity encodings

Prior mechanisms fall into six families. The families matter more than the individual papers,
because the same family recurs independently in machine learning and in hydrology, and the two
communities have never compared notes.

**(i) Implicit identity — one model per entity.** The oldest and strongest form: fit separate
parameters per entity and identity is total. In hydrology this is the norm, from the per-station
logistic air–water regression of [Mohseni et al. 1998] through the per-site calibrated ODE of
air2stream [Toffolon & Piccolroaz 2015] to the per-lake process-guided networks of
[Read et al. 2019]. Its machine-learning echo is the `individual=True` switch in the
DLinear/FITS family [Zeng et al. 2023; Xu et al. 2024], which is **off by default in every
published table** — a fact worth stating, since it means the headline numbers of the most-cited
linear baselines are identity-free.

**(ii) A learned index embedding.** The dominant deep form, originating with DeepAR
[Salinas et al. 2020], which embeds the series' categorical id and concatenates it to the RNN
input at every step. STID [Shao et al. 2022] makes the mechanism the entire contribution,
showing that a plain MLP with spatial and temporal identity embeddings matches spatio-temporal
graph networks; LightSAE [Ren & Yu 2025] decomposes the same table into a shared base plus a
per-channel low-rank term. Hydrology reached the identical construct independently:
[Shalev et al. 2019] show a learned per-site embedding **replaces curated catchment attributes
at equal accuracy**.

**(iii) A static attribute vector.** Instead of an opaque index, feed real covariates: the
static covariate encoders of TFT [Lim et al. 2021], the per-series static features of TiDE
[Das et al. 2023], and, at continental scale, the catchment attributes of
[Kratzert et al. 2019a] and the global flood model of [Nearing et al. 2024]. The striking result
here is [Li et al. 2022], who find that **a vector of random values matches real physical
descriptors** — identity's value lies in distinctness, not semantics.

**(iv) Per-entity parameter generation.** Rather than feeding identity as input, use it to
*produce* parameters. DeepState [Rangapuram et al. 2018] emits a per-series state-space
parameterisation; GPVar [Salinas et al. 2019] a per-series low-rank latent for the copula;
HimNet [Dong et al. 2024] indexes meta-parameter pools; C-LoRA [Nie et al. 2024] attaches
per-channel low-rank adapters. AGCRN [Bai et al. 2020] is the purest case: node embeddings are
mapped to **a dedicated convolution weight and bias for every node**.

**(v) Identity carried by normalization.** Channel Normalization [Lee et al. 2025] gives each
channel distinct learnable affine parameters and formalises *channel identifiability*: for
identical inputs, a non-identifiable model must produce identical outputs.

**(vi) Relational or structural identity.** A row of an adjacency matrix is a near-unique
fingerprint, so graph models deliver identity implicitly — the recurrent graph networks of
[Jia et al. 2021] and [Topp et al. 2023] for river networks, the variate-indexed positional
embeddings of Crossformer [Zhang & Yan 2023], and the cluster-level identity of CCM
[Chen et al. 2024], which replaces individual codes with cluster codes and thereby buys
zero-shot transfer to unseen channels.

### 2.3 Where identity enters, and what destroys it

Cutting across the taxonomy is a second, largely unexamined axis: the **point in the
computational graph** at which identity is injected. Existing mechanisms sit at input concat
[Shao et al. 2022; Das et al. 2023], a gated side branch [Kratzert et al. 2019b; Lim et al.
2021], a post-patch token [Wang et al. 2024; Zhang & Yan 2023], a post-normalization residual
[Lin et al. 2024], the affine parameters of the normalizer itself [Lee et al. 2025], or a
hypernetwork [Bai et al. 2020; Dong et al. 2024].

This axis interacts with a now-standard component: reversible per-instance normalization
[Kim et al. 2022], which subtracts each window's own mean before the backbone and restores it
after. An additive constant identity code injected *before* that subtraction is removed by it —
the code is, definitionally, constant within the window. The interaction is easy to state and,
as far as we can determine, has never been ablated: of the recent architectures we surveyed,
none reports a pre- versus post-normalization comparison of an identity signal.

Two observations sharpen the point. First, CycleNet [Lin et al. 2024] applies its per-channel
learnable cycle *after* reversible normalization and *before* de-normalization — the arrangement
our analysis predicts is necessary — but presents this as a design detail rather than a finding.
Second, the spatio-temporal graph literature, where identity embeddings are most developed, does
**not** use per-window instance normalization at all: STID and STAEformer [Liu et al. 2023]
contain no in-model normalization, relying on dataset-level standardization. The interaction is
therefore structurally invisible in precisely the literature that uses identity most heavily.

### 2.4 The identity-free counter-tradition

A strong line of work succeeds with no identity at all, and it must be taken seriously rather
than dismissed. N-BEATS [Oreshkin et al. 2020] advertises the absence of time-series-specific
components as a result. PatchTST [Nie et al. 2023] makes channel independence explicit: one
shared embedding and one shared set of weights for every series. iTransformer [Liu et al. 2024]
is frequently misread as an identity method because it tokenizes per variate, but its projection
is shared and carries no variate-index embedding, and the paper demonstrates generalization to
**variates unseen during training**. Foundation models go further: Moirai's any-variate attention
[Woo et al. 2024] is permutation-invariant in the variate index, encoding *sameness* but never
*which*, and Timer-XL [Liu et al. 2025] enforces variable equivalence by design. For these
models identity is not an oversight but a prerequisite, since a fixed per-series parameter is
incompatible with arbitrary-variate generalization.

[Han et al. 2023] supply the mechanism that reconciles the two traditions: channel-dependent
models have more capacity, channel-independent models more robustness under distribution shift.
Identity is capacity, and capacity is not free.

### 2.5 Evidence on when identity helps, and when it hurts

The scattered evidence that identity has costs has never been consolidated. [Butera et al. 2025]
study learnable local embeddings at scale and find they may degenerate into **mere sequence
identifiers**, damaging transfer, which perturbation and periodic reset partially repair.
[Cini et al. 2023] formalise trainable node embeddings as amortised node-specific components and
show they can be fitted for new nodes more effectively than fine-tuning. In hydrology,
[Rahmani et al. 2021b] report *attribute overfitting* across more than four hundred basins,
requiring an input-selection ensemble to damp it, and the ungauged-basin literature
[Kratzert et al. 2019a] finds that only attribute-grounded identity transfers to catchments
absent from training. CCM's [Chen et al. 2024] retreat from individual to cluster identity is
motivated by the same failure.

Against this, [Nematirad et al. 2025] ablate embedding layers across fifteen models and find
that removing them often *improves* accuracy. The result is narrower than it appears: the
ablated components are value, temporal, positional, inverted and patch embeddings — **no
channel-identity embedding is isolated** — and the evaluation uses four ETT datasets of seven
channels each, a regime in which the channels are heterogeneous variables of a single
transformer rather than distinct entities.

### 2.6 Positioning

What the literature establishes is that identity often helps, that it is implemented in at least
six ways, and that it sometimes hurts. What it does not establish is *which encoding*, injected
*where*, helps under *what data conditions*. Every existing mechanism is bundled with an
architecture, so the encoding, its position and the backbone vary together. Our contribution is
the controlled factorization of those axes on a fixed backbone, together with the observation
that placement relative to per-window normalization determines whether an additive identity code
survives at all — an interaction the graph literature cannot observe because it does not
normalize per window, and the channel-identity literature has not tested because it varies only
the encoding.

---

## Coverage check against goal item (f)

| Required | Where covered |
|---|---|
| 最新时序模型 (2024–2026) | §2.4 + the 21-model table in [02](02-algorithms-graph-llm-stllm.md) |
| 水温预测模型研究 | §2.2(i), §2.2(iii), §2.5 — Mohseni, air2stream, Read, Rahmani, Kratzert, Nearing, Jia, Topp |
| 历史上任何提到 entity 区分的片段 | §2.1 (panel fixed effects, global-vs-local), §2.2 (all six families), §2.5 |
| 图方法 | §2.2(vi) + §2.3 (why the graph literature cannot see the interaction) |
| 组织方式 | Methodological, by where identity lives — not paper-by-paper |
