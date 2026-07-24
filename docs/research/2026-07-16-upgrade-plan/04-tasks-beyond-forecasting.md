# 04 — Tasks beyond long-horizon forecasting

> Part of the **2026-07-16 upgrade plan** (target: NeurIPS / TPAMI). Master index:
> [`00-INDEX.md`](00-INDEX.md). Covers goal item **(g)** — tasks. Metrics are in
> [`05-metrics.md`](05-metrics.md).
>
> **Provenance**: delegated web-research pass, 2026-07-16. arXiv titles/authors/years from
> the arXiv API; venue strings confirmed on-page where marked. **UNVERIFIED items are listed
> explicitly at the bottom — confirm before camera-ready.**

---

## ⚠ The headline: our current evaluation may be measuring memorization, not mechanism

Under a **same-entities split** (train and test share the same stations/channels — which is
what every cell in our current matrix does), an identifier is *nearly guaranteed* to help,
because the model can simply memorize a per-entity offset. A reviewer can therefore read our
entire long-horizon table as **measuring memorization capacity, not a mechanism**.

A **leave-entities-out** split turns the central claim into a falsifiable dichotomy:

- a **pure lookup key** (one-hot, learned embedding indexed by position) **collapses** on unseen
  stations — it has no row for them;
- an **attribute-grounded identifier** (coordinates, station descriptors, text) **degrades
  gracefully**, because the attributes still carry meaning for a new entity.

This is the only task on the list that can **falsify** the paper's claim rather than confirm it.
It is therefore the highest-value addition in the whole upgrade plan.

---

## Task table

| Task | Standard benchmark | Standard metric + protocol | Canonical citation | Verified URL | Entity identity interesting? |
|---|---|---|---|---|---|
| **1. Short-term forecasting / nowcasting** | ETT, Electricity, Traffic, Weather (Informer suite); M4 (100k series) | MSE/MAE, z-normalized, lookback 96, horizons {96,192,336,720}; M4 uses sMAPE/MASE/OWA, fixed origin, no retraining | Zhou et al. AAAI 2021; Wu et al. 2021 (Autoformer); Wu et al. 2022 (TimesNet); Makridakis et al. IJF 36(1):54–74, 2020 | [2012.07436](https://arxiv.org/abs/2012.07436) · [2106.13008](https://arxiv.org/abs/2106.13008) · [2210.02186](https://arxiv.org/abs/2210.02186) · [M4 doi](https://doi.org/10.1016/j.ijforecast.2019.04.014) | **Weak yes.** Same entity set at train and test, so an identifier is trivially learnable and largely re-derives what a long lookback gives. Adds little beyond our LTSF table. |
| **2. Imputation** | TimesNet imputation suite (ETT, Electricity, Weather); AQI/AQI-36; METR-LA, PEMS-BAY | MSE/MAE on masked points only; point-missing vs block-missing masks; GRIN additionally reports **out-of-sample (unseen sensor)** imputation | Cao et al. 2018 (BRITS); Du et al. 2022 (SAITS); **Cini, Marisca & Alippi, ICLR 2022 (GRIN)** | [1805.10572](https://arxiv.org/abs/1805.10572) · [2202.08516](https://arxiv.org/abs/2202.08516) · [2108.00298](https://arxiv.org/abs/2108.00298) | **Strong yes.** Reconstructing a sensor's values is exactly where "who is this sensor" carries information the model cannot read off the gap. GRIN is explicitly multi-sensor. |
| **3. Anomaly detection** | SMD (28 machines), MSL, SMAP, SWaT, PSM | F1 after **point adjustment** is the de-facto protocol and the one reviewers attack. Report raw point-wise F1, PA%K, or AUC | Xu et al. 2021 (Anomaly Transformer); Su et al. KDD 2019 (SMD); Abdulaal et al. KDD 2021 (PSM); Mathur & Tippenhauer 2016 (SWaT). **Critiques:** Kim et al. AAAI 2022; Wu & Keogh TKDE [10.1109/TKDE.2021.3112126](https://doi.org/10.1109/TKDE.2021.3112126) | [2110.02642](https://arxiv.org/abs/2110.02642) · [KDD19](https://dl.acm.org/doi/10.1145/3292500.3330672) · [KDD21](https://dl.acm.org/doi/10.1145/3447548.3467174) · [AAAI 20680](https://ojs.aaai.org/index.php/AAAI/article/view/20680) · [2009.13807](https://arxiv.org/abs/2009.13807) | Yes in principle (per-machine normal behaviour differs), but entering this task means inheriting a **contested protocol**. Only worth it with corrected metrics. |
| **4. Classification** | UCR (112+ univariate), UEA (30 multivariate) | Accuracy over 30 resamples per dataset, aggregated by average rank; Friedman + post-hoc, drawn as a critical-difference diagram | Dau et al. 2018 (UCR); Bagnall et al. 2018 (UEA); Demšar JMLR 7:1–30, 2006; Middlehurst et al. DMKD 2024 | [1810.07758](https://arxiv.org/abs/1810.07758) · [1811.00075](https://arxiv.org/abs/1811.00075) · [2304.13029](https://arxiv.org/abs/2304.13029) | **NO — avoid.** Instances are i.i.d. series with no entity persisting across the split. An identifier is meaningless or straight **leakage**. |
| **5. Few-shot / zero-shot transfer to UNSEEN entities** ★ | Kriging / virtual sensors on METR-LA, PEMS-BAY, USHCN; **hydrology PUB on CAMELS (531 basins, k-fold over basins)**; Monash; GIFT-Eval | Hold out whole sensors/basins from training, evaluate only on them (MAE/RMSE; NSE in hydrology). Foundation models: strictly held-out datasets, MASE/CRPS-family, no target training | Wu et al. AAAI 2021 (IGNNK); Wu et al. 2021 (SATCN); Li et al. 2017 (DCRNN); **Kratzert et al. WRR 2019 (PUB)**; Kratzert et al. HESS 23:5089, 2019 (EA-LSTM); Godahewa et al. 2021 (Monash); Aksu et al. 2024 (GIFT-Eval); Ansari et al. 2024 (Chronos); Woo et al. 2024 (Moirai); Das et al. 2023 (TimesFM); Rasul et al. 2023 (Lag-Llama) | [2006.07527](https://arxiv.org/abs/2006.07527) · [AAAI 16575](https://ojs.aaai.org/index.php/AAAI/article/view/16575) · [2109.12144](https://arxiv.org/abs/2109.12144) · [1707.01926](https://arxiv.org/abs/1707.01926) · [WRR PUB](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019WR026065) · [HESS EA-LSTM](https://hess.copernicus.org/articles/23/5089/2019/) · [2105.06643](https://arxiv.org/abs/2105.06643) · [2410.10393](https://arxiv.org/abs/2410.10393) · [2403.07815](https://arxiv.org/abs/2403.07815) · [2402.02592](https://arxiv.org/abs/2402.02592) · [2310.10688](https://arxiv.org/abs/2310.10688) · [2310.08278](https://arxiv.org/abs/2310.08278) | **THE DECISIVE TEST.** A learned code has no value for an entity absent from training, so this is where the mechanism either generalizes through shared attributes or is exposed as a lookup table. |
| **6. Cold-start (new entity, few obs.)** | **No canonical TS benchmark — UNVERIFIED that one exists.** Anchors: recommender cold-start; e-commerce new-item demand; the k-shot slice of any multi-entity dataset | Recommender side: rank/error on items with zero or k ratings, plus CROC. TS side: error as a function of observations available for the new entity (a **k-shot curve**) | Schein et al. SIGIR 2002; Salinas et al. IJF 36(3):1181–1191, 2020 (DeepAR); Chauhan et al. WWW 2020 Companion | [SIGIR02](https://dl.acm.org/doi/10.1145/564376.564421) · [1704.04110](https://arxiv.org/abs/1704.04110) · [WWW20](https://dl.acm.org/doi/abs/10.1145/3366424.3382728) | **Yes — the continuous version of Task 5.** Sweep k instead of a binary seen/unseen and show where an identifier starts paying for itself. |
| **7. Cross-domain transfer** | UCI HAR, HHAR, WISDM, Opportunity, Sleep-EDF (as assembled by AdaTime) | Source-trained, target-unlabelled; accuracy/macro-F1 on target, averaged over source→target pairs and seeds, fixed model-selection rule | Wilson, Doppa & Cook KDD 2020 (CoDATS); Ragab et al. 2022 (AdaTime) | [KDD20](https://dl.acm.org/doi/10.1145/3394486.3403228) · [2203.08321](https://arxiv.org/abs/2203.08321) | Yes, indirectly. Identifier vocabularies do not align across networks, so the defensible claim concerns the **attribute-conditioned** variant, not the codebook. |

---

## Cite regardless of which tasks we add

Two papers study our exact object and pre-empt the obvious objection:

- **Cini, Marisca, Zambon, Alippi — *Taming Local Effects in Graph-based Spatiotemporal
  Forecasting*, NeurIPS 2023** ([2302.04071](https://arxiv.org/abs/2302.04071)): node embeddings
  amortize local components, and **can be fitted for new nodes better than fine-tuning** — i.e. a
  constructive answer to the unseen-entity problem.
- **Butera, De Felice, Cini & Alippi, TMLR 2025** ([2410.14630](https://arxiv.org/abs/2410.14630)):
  embeddings degenerate into mere sequence identifiers, and **that is what limits transfer**.

---

## Ranking (what to add, in order)

**★ Task 5 first — leave-entities-out.** The only task that can falsify rather than confirm the
central claim. Hydrology gives the strongest precedent: **Kratzert's PUB protocol (k-fold over
CAMELS basins) is reviewer-recognized, matches our river-station data directly, and carries real
stakes** (predicting ungauged basins is a live problem). IGNNK and SATCN supply the traffic
equivalent (kriging / virtual sensors). Concretely for us: hold out whole swiss stations, retrain,
and compare `onehot`/`embedding` (pure lookup — expected to collapse) against
`coordinates`/`descriptors`/text (attribute-grounded — expected to degrade gracefully). **This
single experiment converts C3 from an observation into a mechanism claim.**

**Task 6 second — a k-shot sweep** over the same held-out entities, not a separate dataset. Almost
no extra infrastructure once Task 5 exists, and it converts a binary result into a curve: *how
many observations does a new station need before a fitted identifier beats the identifier-free
model?* That is the question a practitioner actually asks when a sensor is installed.

**Task 2 third — imputation**, as the breadth axis. Different objective, same multi-entity data,
protocol already established by GRIN at ICLR 2022, and it tests whether identity aids
**reconstruction** rather than only extrapolation.

**Avoid classification** (no persistent entities; an identifier is leakage). **Treat anomaly
detection as optional**, since entering it obliges us to defend against the point-adjustment
critiques.

---

## Verification notes

All URLs above were retrieved. Venue strings **confirmed on-page**: GRIN → ICLR 2022; Taming Local
Effects → NeurIPS 2023; Butera → TMLR 2025; Kim → AAAI 2022; Su → KDD 2019; Abdulaal → KDD 2021;
Wilson → KDD 2020; Schein → SIGIR 2002; Kratzert → WRR/HESS 2019.

**UNVERIFIED venue labels** (arXiv metadata verified, venue string not seen — confirm before
camera-ready): TimesNet → ICLR 2023; Autoformer → NeurIPS 2021; Informer → AAAI 2021; BRITS →
NeurIPS 2018; SAITS → ESWA 2023; Anomaly Transformer → ICLR 2022; DCRNN → ICLR 2018; Monash →
NeurIPS 2021 D&B; Chronos → TMLR 2024; Moirai, TimesFM → ICML 2024; AdaTime → TKDD 2023.
Also UNVERIFIED: full author list of the WWW 2020 cold-start paper (only Ayush Chauhan confirmed).
**Task 6 has no field-standard benchmark; none was invented.**
