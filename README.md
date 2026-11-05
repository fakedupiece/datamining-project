# Beyond Engagement

Reproducible Stage 0–2 research workspace for a Data Mining master's final project.

## Current evidence-based status

- **Stage 0 — PIVOT:** no audited candidate supports real influencer campaign sales/ROI. The selected YouTube Trending US data are traceable real API observations but contain content-performance outcomes only.
- **Stage 1 / H1 — GO for engineering readiness under the pivot:** 40,949 daily snapshots become 6,347 unique-video analysis rows. Seven valid engineered features and seven EDA figures show that normalized engagement intensity produces rankings distinct from raw views.
- **Stage 2 / H2 — REVIEW:** 240 Hopkins trials are consistently high, but the independent marginal-permutation nearest-neighbour diagnostic is not significant for the redundancy-free normalized representation (`p=0.161–0.290`). H2 is not conclusively supported.

The active research question is:

> Do scale-normalized interaction and timing features reveal non-random behavioral structure among US trending YouTube videos beyond raw view volume?

There are no sales, revenue, campaign cost, attribution, or ROI fields. The repository makes no conversion, ROI, causal, or population-wide YouTube claim. Stages 3–8 are intentionally absent.

## Setup

Python 3.12 is the tested runtime.

```bash
cd /workspace/datamining-project
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
```

The selected public archive is about 201 MiB compressed. Raw and processed data are ignored by Git. Downloading verifies pinned SHA-256 values and extracts only the US CSV and category JSON:

```bash
.venv/bin/python -m src.download_data --config configs/experiment.yaml
```

## Reproduce Stage 0–2

Run the complete, tested workflow:

```bash
./scripts/run_stage_0_2.sh
```

Equivalent individual commands are:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"

.venv/bin/python -m src.data_audit --config configs/experiment.yaml
.venv/bin/python -m src.preprocess --config configs/experiment.yaml
.venv/bin/python -m src.eda --config configs/experiment.yaml
.venv/bin/python -m src.clustering_tendency --config configs/experiment.yaml
.venv/bin/python -m pytest -q
```

## Evidence map

- Dataset provenance, candidate matrix, hard vetoes, selected scope, file hashes: [`reports/dataset_decision.md`](reports/dataset_decision.md)
- Stage 1 cleaning, engineered features, rank evidence, leakage boundary: [`reports/stage_01_eda.md`](reports/stage_01_eda.md)
- Stage 2 Hopkins convention, seed/sample sensitivity, null-distance diagnostic, PCA limits: [`reports/stage_02_clusterability.md`](reports/stage_02_clusterability.md)
- Consolidated decisions and success criteria: [`reports/stage_gates.md`](reports/stage_gates.md)
- Machine-readable metrics and data dictionary: [`reports/tables/`](reports/tables/)
- Generated plots: [`reports/figures/`](reports/figures/)

The immutable raw archive has SHA-256 `9f59324bd23ebd95aa16f854de1e0a6299bacf988b30c5e7db5f1e4216d8deda`; the selected `USvideos.csv` has SHA-256 `09b4eb71295752705e472ebefeac9d2afab4177b7a818af795dea62744a48eb2`.

## Method boundaries

The processed table uses one row per unique video, represented by its latest retained US trending snapshot. Post-publication views, interaction rates, trending duration, and time-to-trending are descriptive outcomes and leakage risks for prediction. The metadata sensitivity set uses title/tag/description values captured at the latest snapshot; because those fields may have been edited, they are not labeled pre-publication predictors. Any future predictive preprocessing must be fit on training data only and should use creator/time-aware splits.

PCA plots are exploratory. Hopkins uses an axis-aligned bounding-box null; the second diagnostic independently permutes empirical feature marginals. Both can detect constrained dependence or manifolds as well as discrete clusters. The current disagreement requires a support-aware null or another justified low-redundancy sensitivity analysis before Stage 3. A later Stage 3 would still need algorithm comparison, stability, cluster-size, and interpretation checks before claiming archetypes.
