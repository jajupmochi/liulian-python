# 03 — Datasets: the standard suite, beyond it, and graph benchmarks

> Part of the **2026-07-16 upgrade plan** (target: NeurIPS / TPAMI). Master index:
> [`00-INDEX.md`](00-INDEX.md). Covers goal items **(a)** standard/TSL datasets with
> entity-meaningful vs meaningless separation, and **(b)** beyond-standard data + whether graph
> data is worth adding.
>
> **Provenance**: delegated research pass, 2026-07-16. Variate counts and split sizes taken
> verbatim from [iTransformer Table 4](https://arxiv.org/html/2310.06625v4) and
> [TimesNet Table 6](https://ar5iv.labs.arxiv.org/html/2210.02186); descriptions from
> [Autoformer §Datasets](https://ar5iv.labs.arxiv.org/html/2106.13008) and
> [LSTNet §4.3](https://ar5iv.labs.arxiv.org/html/1703.07015).
>
> **Rubric (fixed a priori, not derived from results)** — **ENTITY-RICH**: the C channels are C
> separately instantiated real-world objects. **WEAK-ENTITY**: the C channels are heterogeneous
> sensor-variables of ONE object. **AMBIGUOUS**: channels are distinct objects but share one
> dominant driver, so identity may carry little signal.

---

## 🔴 TWO STRUCTURAL FINDINGS THAT MUST SHAPE THE PAPER

### (1) Entity-richness is **perfectly confounded with channel count** in the standard suite

Every ENTITY-RICH set has **C ≥ 137**; every WEAK-ENTITY set has **C ≤ 21**. On the standard suite
alone, **any measured effect of identity injection is indistinguishable from an effect of
dimensionality**. This is the **single largest methodological risk in the paper** — our own
swiss (28 stations) vs ETTh1 (7 channels) contrast inherits it too.

**It must be broken by design.** Two ways, both in the priority list below:

- **Channel-count control** — downsample Traffic to C ∈ {7, 21, 137} and compare against ETT /
  Weather *at matched C*.
- **SMD (28 machines × 38 metrics)** — the cleanest two-factor design available: the 38 metric
  types stay fixed while entity identity switches across 28 machines, so entity-richness can be
  flipped **with C held constant at 38**. The strongest single control to break the confound.

### (2) The benchmark files **strip entity metadata**

The LSTNet release is raw numeric matrices with no headers, IDs, names or coordinates; the thuml
CSVs use integer column names. **Text-name and coordinate identity injection are therefore
impossible on the standard suite** — with two exceptions: Solar-Energy (NREL encodes lat/lon in
the filenames, e.g. `Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv`) and Traffic (PeMS Clearinghouse
metadata exists, but the mapping to the 862 benchmark columns is **unpublished**). This is itself
the motivation for Part B and for LargeST.

---

## Part A — the canonical TSL / PatchTST / iTransformer / TimesNet suite

| Dataset | C | What a channel physically IS | Rate / length | Source | Introduced by | **Verdict** |
|---|---|---|---|---|---|---|
| ETTh1, ETTh2 | 7 | 6 load features + oil temperature of **one** transformer (h1/h2 = two stations) | 1 h / 17,420 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | [Informer, AAAI'21](https://arxiv.org/abs/2012.07436) | **WEAK** — entity varies at file level, not channel level |
| ETTm1, ETTm2 | 7 | same, 15-min | 15 min / 69,680 | same | same | **WEAK** |
| Electricity (ECL) | 321 | one anonymized client's hourly kWh | 1 h / 26,304 | [UCI 321](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · [processed](https://github.com/laiguokun/multivariate-time-series-data) | [LSTNet, SIGIR'18](https://arxiv.org/abs/1703.07015) | **RICH** — 321 separate metering points |
| Traffic | 862 | occupancy at **one** freeway loop detector, SF Bay | 1 h / 17,544 | [PeMS](https://pems.dot.ca.gov) | LSTNet | **RICH** — strongest identity signal in the suite; per-sensor mean occupancy is near-uniquely identifying |
| Weather | 21 | one meteorological quantity at **one** Jena station | 10 min / 52,696 | [MPI-BGC](https://www.bgc-jena.mpg.de/wetter/) | [Autoformer, NeurIPS'21](https://arxiv.org/abs/2106.13008) | **WEAK** — 21 different quantities, different units, one site |
| Exchange-Rate | 8 | one country's daily FX vs USD | 1 d / 7,588 | [laiguokun](https://github.com/laiguokun/multivariate-time-series-data) | LSTNet | **AMBIGUOUS** — 8 real entities but near-random-walk co-moving against one numeraire |
| ILI / Illness | 7 | a national ILI ratio or count (age bands, totals) | 1 wk / 966 | [CDC FluView](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) | Autoformer | **WEAK** — nested aggregations of the *same* national stream, not 7 regions |
| Solar-Energy | 137 | one **simulated** PV plant in Alabama | 10 min / 52,560 | [NREL](https://www.nrel.gov/grid/solar-power-data) | LSTNet | **AMBIGUOUS** — distinct sites, but simulation outputs sharing one irradiance driver |
| PEMS03 | 358 | one loop detector's flow | 5 min / 26,208 (91 d) | [STSGCN](https://github.com/Davidham3/STSGCN) | [STSGCN, AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5438) | **RICH** |
| PEMS04 | 307 | same | 5 min / 16,992 (59 d) | same | same | **RICH** |
| PEMS07 | 883 | same | 5 min / 28,224 (98 d) | same | same | **RICH** |
| PEMS08 | 170 | same | 5 min / 17,856 (62 d) | same | same | **RICH** |
| M4 | 100,000 series | one univariate business/economic series | 6 freqs | [M4-methods](https://github.com/Mcompetitions/M4-methods) | Makridakis et al. | **RICH but univariate** — identity = per-series conditioning, a different regime |

Bundles: [TSLib on HuggingFace](https://huggingface.co/datasets/thuml/Time-Series-Library) ·
[Autoformer Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

---

## Part B — beyond the benchmark suite (escaping benchmark overfitting)

| Dataset | #entities | Variables / rate | Entity metadata | Licence | Source |
|---|---|---|---|---|---|
| **CAMELS-CH-Chem** 🇨🇭 | 115 catchments; **86 stations with water temp** | water temp, DO, pH, EC — **hourly + daily**; 1981–2020 | names, WGS84 lon/lat **and** LV95 E/N, area | CC-BY-4.0 | [10.5281/zenodo.16158375](https://doi.org/10.5281/zenodo.16158375) · [Sci Data 2025](https://www.nature.com/articles/s41597-025-05625-1) |
| **CAMELS-CH** 🇨🇭 | 331 (298 river, 33 lake) | daily Q, level, P, air temp; **no water temp** | gauge_id, names, coords, shapefiles, 9 attribute groups | CC-BY-4.0 | [10.5281/zenodo.7784632](https://doi.org/10.5281/zenodo.7784632) · [ESSD 15, 5755](https://essd.copernicus.org/articles/15/5755/2023/) |
| **USGS Delaware River Basin** | **456 reaches + 2 reservoirs** | daily water temp + flow | reach shapefiles, **network adjacency** | US public domain | [ScienceBase](https://www.sciencebase.gov/catalog/item/5f6a26af82ce38aaa2449100) · [Jia et al.](https://arxiv.org/pdf/2009.12575) |
| **NorWeST** | **>20,000 stream sites**, >200M records | **hourly** stream temperature | observation-point shapefiles, NHDPlus IDs | US Gov | [USFS RMRS](https://research.fs.usda.gov/rmrs/projects/norwest) |
| **CAMELS-Chem** | 516 US catchments | 18 constituents **incl. water temp**, grab samples | inherits CAMELS attributes | CC-BY-4.0 | [HydroShare](https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/) · [HESS 28, 611](https://hess.copernicus.org/articles/28/611/2024/) |
| **Caravan** | **6,830** catchments | daily ERA5-Land + streamflow; **no water temp** | HydroATLAS attributes, coords, polygons | CC-BY-4.0 | [10.5281/zenodo.7540792](https://doi.org/10.5281/zenodo.7540792) · [Sci Data 10, 61](https://www.nature.com/articles/s41597-023-01975-w) |
| **CAMELS (US)** | 671 basins | daily forcings + flow; no water temp | 6 attribute classes, gauge IDs + coords | UCAR ToU | [NCAR RAL](https://ral.ucar.edu/solutions/products/camels) · [Addor et al.](https://hess.copernicus.org/articles/21/5293/2017/) |
| **LamaH-CE** | 859 basins | **hourly + daily** Q/met | intermediate-catchment topology, attributes | CC-BY-SA-4.0 | [Zenodo](https://zenodo.org/records/5153305) · [ESSD 13, 4529](https://essd.copernicus.org/articles/13/4529/2021/) |
| **FOEN/BAFU** 🇨🇭 | ~80 river gauges with water temp | **10-min**, 1970s– | station names + coords | free, cite source | [hydrodaten.admin.ch](https://www.hydrodaten.admin.ch/en/seen-und-fluesse/messstationen-temperatur) |
| **USGS NWIS** (param 00010) | 213 daily-value stream sites in **PA alone** | water temp, 15-min + daily | `station_nm`, `dec_lat_va`, `dec_long_va`, `alt_va`, HUC | US public domain | [waterservices](https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=pa&parameterCd=00010&hasDataTypeCd=dv&siteType=ST) |
| **GHCN-Daily** | **>80,000** stations | Tmax/Tmin/precip, daily | IDs, names, lat/lon, elevation | US public domain | [NCEI](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) |
| **ISMN** | 2,842 stations / 71 networks | soil moisture, sub-daily | names, coords, depth, Köppen, soil, landcover | free registration | [ismn.earth](https://ismn.earth/en/data/) · [HESS 25, 5749](https://hess.copernicus.org/articles/25/5749/2021/) |
| **SDWPF** | 134 wind turbines, one farm | 10 min; 2020-01→2021-12 | **relative x/y in metres + elevation** | CC-BY-4.0 | [figshare](https://doi.org/10.6084/m9.figshare.24798654) · [arXiv:2208.04360](https://arxiv.org/abs/2208.04360) |
| **London Smart Meters** | 5,567 households | 30 min; 2011-11→2014-02 | LCLid, ACORN group, affluence band, tariff | CC-BY-4.0 | [data.london.gov.uk](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/) |
| **Buildings-900K** | 900,000 simulated + >1,900 real | hourly | building type, lat/lon, per-building ID | CC-BY-4.0 / BSD-3 | [OpenEI](https://data.openei.org/submissions/5859) · [NeurIPS'23 D&B](https://arxiv.org/abs/2307.00142) |
| **M5 (Walmart)** | **30,490** item-store series | daily; 2011→2016 | item/dept/cat/store/state, price, SNAP, calendar | Kaggle terms | [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/overview) |
| **Rossmann** | 1,115 stores | daily; 2013→2015 | StoreType, Assortment, CompetitionDistance, Promo2 | Kaggle terms | [Kaggle](https://www.kaggle.com/c/rossmann-store-sales) |
| **SMD** ★ | **28 machines × 38 metrics** | 1 min; 5 weeks | machine index only | see repo | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |
| **Beijing Multi-Site AQ** | 12 sites | hourly; 2013-03→2017-02 | site **names**; lat/lon not included | CC-BY-4.0 | [UCI 501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) |

> ⚠ **Correction worth flagging**: CAMELS US / GB / BR / CH / DE / AUS **and Caravan contain NO
> water temperature**. Only **CAMELS-Chem** and **CAMELS-CH-Chem** do. This is routinely miscited.

---

## Part C — graph / spatio-temporal benchmarks

| Dataset | #nodes | Node = | Graph shipped? how built | Node metadata | Source |
|---|---|---|---|---|---|
| **METR-LA** | 207 | LA loop detector | Yes. Paper: `W_ij = exp(−dist²/σ²)` if `dist ≤ κ`. **Released code instead thresholds the kernel value** (`normalized_k=0.1`) | sensor IDs, **lat/lon**, distance matrix | [DCRNN](https://github.com/liyaguang/DCRNN) · [1707.01926](https://arxiv.org/abs/1707.01926) |
| **PEMS-BAY** | 325 | Bay Area PeMS sensor | same kernel/threshold | **lat/lon** | same |
| **PEMS03/04/07/08** | 358/307/883/170 | Caltrans detector | Yes but **binary same-road connectivity**; the loader sets `A[i,j]=1` and discards the distance column | row index only | [STSGCN](https://github.com/Davidham3/STSGCN) · [ASTGNN data](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| **PeMSD7(M)/(L)** | 228 / 1,026 | District-7 station | Yes, precomputed `exp(−d²/σ²)` if `≥ ε`, **σ²=10, ε=0.5** | station lists, no coords | [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| **LargeST** ★ | **8,600** (GLA 3,834 / GBA 2,352 / SD 716) | CA mainline sensor | Yes, `exp(−d²/σ²)` if `≥ 0.01` | **richest available**: lat/lon, county, district, freeway, direction, lanes, sensor type | [GitHub](https://github.com/liuxu77/LargeST) · [Kaggle](https://www.kaggle.com/datasets/liuxu77/largest) · [NeurIPS'23 D&B](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html) |
| **KnowAir** | 184 cities | Chinese city | Computed at runtime, not shipped: edge if distance < 3° and no >1,200 m terrain between | city name, lon/lat, altitude | [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN) · [2002.12898](https://arxiv.org/abs/2002.12898) |
| **NYCBike1/2, NYCTaxi** | 128 / 200 / 200 | **grid cell**, not a named station | none shipped; adjacency over neighbouring cells | grid index only | [ST-SSL_Dataset](https://github.com/Echo-Ji/ST-SSL_Dataset) · [2212.04475](https://arxiv.org/abs/2212.04475) |
| **Chickenpox Hungary** | 20 counties | county | **static graph shipped**, edges = shared border | county identity | [PyG-Temporal](https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json) · [UCI 580](https://archive.ics.uci.edu/dataset/580/hungarian+chickenpox+cases) |
| **Wiki-Math** | 1,068 pages | Wikipedia page | **static graph shipped**, directed, weighted by link count | page title | [PyG-Temporal](https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json) |
| Solar / Electricity / Traffic | 137 / 321 / 862 | — | **none** — the canonical learned-graph pair | none | [laiguokun](https://github.com/laiguokun/multivariate-time-series-data) |

### Is adding graph data meaningful here? **Yes — as a metadata source and an identity upper bound, not as a rival architecture.**

**For.** A row of the adjacency matrix is a near-unique fingerprint of node *i*, so any model
consuming **A** already receives a distinguishing per-node code — **a graph is identity delivered
implicitly**. The strongest evidence is published: [STID](https://arxiv.org/abs/2208.05233) shows an
explicit spatial-and-temporal identity embedding on a plain MLP matches or exceeds STGNNs, arguing
the real bottleneck is "the indistinguishability of samples in both spatial and temporal
dimensions" — our own hypothesis, demonstrated on graph data. Graph sets are also **the only large
benchmarks that ship node coordinates**, which the entire standard suite lacks.

**Against.** A graph is not *only* identity: it simultaneously imposes a relational inductive bias,
so message passing changes both who-am-I and what-information-reaches-me. Adding a graph is a
**confounded treatment**; attributing gain to identity needs a matched control such as a
**degree-preserving permuted adjacency**. The graphs are also not one kind of object — METR-LA,
PEMS-BAY, PeMSD7 and LargeST use a weighted thresholded Gaussian kernel over road-network distance,
whereas PEMS03/04/07/08 use unweighted same-road connectivity, so a claim spanning both compares
weighted geometry against bare topology. Finally, entering the STGNN arena imports a separate SOTA
we would then be expected to beat.

**Practical resolution.** Keep a **fixed non-graph backbone** and run one ablation ladder on
METR-LA / PEMS-BAY / LargeST:
`no id → random permuted id → learned id → coordinate id → adjacency-row id`.
Report **identity mechanisms**, not STGNN rankings.

---

## Prioritized recommendation

### MUST-TEST — required for a top-venue claim

| # | Dataset / experiment | Why |
|---|---|---|
| 1 | **Traffic-862 + Electricity-321** | The two strongest entity-rich cells in the standard suite; reviewers will check them first |
| 2 | **PEMS04 + PEMS08** | Entity-rich, 5-min sampling, different domain — shows the result is not one dataset's artifact |
| 3 | **LargeST (at least the SD 716 subset)** | The **only** large recognized benchmark shipping full entity metadata (lat/lon, county, freeway, lanes) — the only place text/coordinate identity can be injected on a *standard* benchmark |
| 4 | **CAMELS-CH-Chem (86 Swiss water-temp stations)** | The key cell for escaping benchmark overfitting: hourly water temp, station names + WGS84 + LV95 coords, CC-BY-4.0, and continuous with the project's existing swiss-river line |
| **5** ★ | **Channel-count control** — downsample Traffic to C ∈ {7, 21, 137}, compare against ETT / Weather at matched C | **Without this cell, "entity-richness" and "channel count" cannot be separated. The single largest methodological risk in the paper.** |

### EXTENDED-VALIDATION — weak-entity controls

| # | Dataset | Why |
|---|---|---|
| 6 | ETTh1/h2/m1/m2 | The standard null cell (7 variables of one transformer) |
| 7 | Weather-21 | Second null cell, higher C |
| 8 | ILI-7 | Nested aggregations of one national signal — third null cell |
| **9** ★ | **SMD (28 machines × 38 metrics)** | **The cleanest two-factor design in the whole table**: 38 metric types fixed, entity identity switches across 28 machines ⟹ flips entity-richness with **C held constant at 38**. The strongest single control against the confound in #5 |
| 10 | Exchange-8 | Real entities, near-random-walk process — tests the "distinguishable but uninformative" boundary |

### OPTIONAL

11. **Solar-Energy-137** — nominally rich but simulated with one shared irradiance driver; a good "distinct but low-distinguishability" probe (NREL filenames carry lat/lon).
12. **M4 (100k)** — univariate per-series identity; a different paradigm, suits a foundation-model framing.
13. **SDWPF** (134 turbines with x/y) · **London Smart Meters** (5,567 households with ACORN attributes) · **M5 / Rossmann** (hierarchical categorical identity) — three different shapes of entity metadata.
14. **METR-LA / PEMS-BAY** — only when running the graph-vs-identity ablation ladder.
15. **Caravan** (6,830 catchments + HydroATLAS attributes) — the upper bound on entity count and richest static attributes, but **no water temperature**; a second domain only.

---

## UNVERIFIED — must not enter the paper without re-checking

NWIS national water-temp station total · Oliver et al. 2024 site/observation totals · GRQA station
count · NDBC station count (two sources conflict) · DCRNN and PEMS0X release licences (no LICENSE
file in repo) · all Kaggle dataset licences (JS-rendered pages) · Ausgrid (official link 404) ·
Chickenpox time span (UCI and PyG-Temporal docs conflict).
Also note: **LSTNet's README claim of "48 months (2015-2016)" for Traffic is an error in the
original paper.**
