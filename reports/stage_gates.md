# Stage 0–2 Gate Register

| Stage | Status | Hypothesis assessment | Primary evidence | Next allowed action |
|---|---|---|---|---|
| Stage 0 / H0 | **PIVOT** | The verified YouTube data are real, traceable content-performance observations, but no candidate supports the original commercial influencer/sales/ROI question. | [`dataset_decision.md`](dataset_decision.md), [`raw_data_audit.json`](tables/raw_data_audit.json) | Keep all claims within the documented content-performance scope, or source a separately permissioned commercial attribution dataset. |
| Stage 1 / H1 | **GO** under the pivot | H1 has preliminary engineering support: normalized features are valid, variable, interpretable, and produce rankings distinct from raw view volume. This is not final incremental-value proof. | [`stage_01_eda.md`](stage_01_eda.md), [`data_dictionary.csv`](tables/data_dictionary.csv), seven Stage 1 figures | Preserve descriptive/outcome and captured-metadata boundaries in any later experiment. |
| Stage 2 / H2 | **REVIEW** | Hopkins is consistently high, but the independent NND null loses significance for the redundancy-free normalized representation. H2 is not conclusively supported. | [`stage_02_clusterability.md`](stage_02_clusterability.md), [`hopkins_sensitivity.csv`](tables/hopkins_sensitivity.csv), [`null_distance_comparison.csv`](tables/null_distance_comparison.csv) | Refine the null/support model before any Stage 3 work. |

## Stage 0 / H0 — Dataset validation

### Criteria and evidence

- **Creator/collection path:** GO for YouTube Trending. The Kaggle description links the public collection repository, whose scraper calls the YouTube Data API `videos` endpoint with `chart=mostPopular`, region, statistics, and snippet fields.
- **Actual file inspected:** GO. `USvideos.csv` was downloaded from the official public archive, measured at 62,756,152 bytes, and verified as SHA-256 `09b4eb71295752705e472ebefeac9d2afab4177b7a818af795dea62744a48eb2`.
- **Reality boundary:** GO. YouTube and UCI are documented observations; SAGraph mixes observed interactions with derived/simulated attributes; GA4 is obfuscated/emulated; the Influencer ROI file has no observational provenance and strong generation signatures.
- **Quantity/quality:** GO for the pivot. The raw US file has 40,949 snapshots, 6,351 videos, 2,207 channels, 16 categories, and 205 dates. Duplicate, missing, disabled, error, zero, and date checks were executed.
- **Claim/field alignment:** **PIVOT.** The selected file has views and interactions but no sales, revenue, cost, customer, attribution, conversion, or ROI field.
- **Access/ethics:** GO with limitations. Kaggle labels the selected dataset CC0; it contains public video/channel metadata rather than user profiles. Underlying content rights and YouTube API Services Terms still apply. Trending selection bias and retained metadata from later-removed public content are documented.

### Gate rationale

The original question cannot pass a commercial-outcome hard veto. Stage 0 is **PIVOT**, not a failed data download. The verified data support the narrower question recorded in the dataset decision report, so Stages 1–2 legitimately proceed only under that scope.

## Stage 1 / H1 — EDA and feature engineering

### Criteria and evidence

- **Granularity:** one unique `video_id`, represented by its latest retained US daily trending snapshot. Result: 6,347 analysis rows and zero duplicate video IDs.
- **Feature validity:** seven engineered features meet the predeclared ≥80% non-missing and ≥10-value variation checks: engagement-per-view, reaction rate, comment rate, like share, days-to-trending, trending days, and views-per-active-day.
- **Denominator/missingness policy:** non-positive denominators return missing; 105 disabled-comment videos and 30 disabled-rating videos have affected raw and derived fields set missing rather than observed zero; one impossible negative timing interval was set missing.
- **Quality handling:** 48 exact rows and two remaining duplicate video/day rows were removed before aggregation; four latest error/removed videos were excluded. Thirty-five source view-decrease events are retained as a recorded source diagnostic.
- **EDA:** seven figures plus category, correlation, rank-shift, feature-summary, cleaning, and schema tables were generated.
- **Raw versus engineered evidence:** views versus engagement-per-view Spearman rho is 0.066; their top-decile Jaccard overlap is 0.043; median absolute percentile-rank shift is 0.275. Normalized intensity therefore surfaces a materially different ranking from raw scale.
- **Leakage control:** raw/derived interaction metrics and trending timing are marked post-publication outcomes. Title/tag/description values are labeled captured metadata because the latest snapshot does not prove they are unedited publication-time predictors.

### Gate rationale

Stage 1 is **GO for engineering readiness under the pivot**. H1 receives preliminary support because the normalized features are measurable and reveal heterogeneity not reducible to the raw view ranking. H1 is not finally accepted: later raw-only versus engineered ablation would be needed for an incremental predictive or decision-value claim.

## Stage 2 / H2 — Clustering tendency

### Criteria and evidence

- **Implementation convention:** tested formula `sum(u) / [sum(u) + sum(w)]`, where approximately 0.5 is random under the axis-aligned bounding-box null and values toward 1 indicate clustering tendency. Toy uniform and two-cluster tests verify the orientation.
- **Repeated trials:** each feature-set/scaler/duplicate condition uses seeds 11, 23, 42, 73, and 101 at requested sample sizes 100, 250, and 500 (15 Hopkins trials per condition; 240 total).
- **Representations:** raw volume; a redundancy-free normalized set (`engagement_per_view`, `days_to_trending`, `trending_days`); a combined set with log views; and captured content metadata. Standard/robust scaling and exact-duplicate keep/drop sensitivity are included. Algebraically redundant rate sums and deterministic sine/cosine pairs were removed from the primary sets.
- **Primary duplicate-dropped results:**

| Feature set | Scaler | Hopkins mean ± SD | Trials ≥ 0.65 | Null/observed median NND | One-sided null p |
|---|---|---:|---:|---:|---:|
| Raw volume | standard | 0.935 ± 0.005 | 15/15 | 2.176 | 0.032 |
| Normalized behavior | robust | 0.982 ± 0.035 | 15/15 | 1.028 | 0.290 |
| Combined behavior | standard | 0.979 ± 0.002 | 15/15 | 1.071 | 0.032 |
| Captured content metadata | standard | 0.964 ± 0.003 | 15/15 | 1.150 | 0.032 |

The p-value resolution is limited by 30 null repetitions: the minimum possible corrected value is `1 / 31 = 0.032`. The normalized set's independent distance result is not significant after exact duplicates are dropped under either scaler (robust `p=0.290`, standard `p=0.161`), despite very high Hopkins values. Duplicate retention raises the NND ratio slightly, so duplicate sensitivity also matters for that representation.

### Gate rationale and limits

Stage 2 is **REVIEW**. H2 is not conclusively supported because the two diagnostics disagree on the most direct, redundancy-free normalized behavior representation. Bounding-box Hopkins can be inflated by skewed, correlated, or constrained supports; marginal permutation detects dependence and manifolds as well as separated groups. The raw/combined and captured-metadata results are signals worth investigating, but they do not override the failed independent check for normalized behavior. A support-aware null or an explicitly justified low-redundancy representation must resolve this before Stage 3. PCA figures remain exploratory only.

## Scope enforcement

No Stage 3–8 algorithm, association rule, anomaly model, classifier, or ROI calculation has been implemented. No sales, purchase conversion, causal, or population-wide YouTube claim follows from these gates.
