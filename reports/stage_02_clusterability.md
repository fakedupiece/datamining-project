# Stage 2 — Clustering Tendency

**Suggested gate: REVIEW.** Only 6/8 primary sensitivity configurations pass both diagnostics; representation sensitivity requires review before Stage 3.

This is a diagnostic gate, not a clustering result. No Stage 3 algorithm was fitted.

## Design

- Hopkins uses `sum(u) / (sum(u) + sum(w))`: about 0.5 is random under the observed bounding-box null and values toward 1 indicate clustering tendency.
- The independent distance diagnostic permutes each standardized feature separately, preserving empirical marginals while destroying dependence.
- Primary normalized/combined sets remove exact algebraic redundancy, and captured metadata excludes deterministic sine/cosine pairs; this is a sensitivity guard against null-model artifacts.
- Exact-duplicate-dropped configurations are primary; duplicate-kept runs expose duplicate concentration sensitivity.
- PCA figures are exploratory only and are not treated as statistical evidence.

## Configuration summary

| Feature set | Scaler | Duplicates | Hopkins mean | Hopkins SD | Fraction ≥ 0.65 | Null/observed median NND | Null p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| captured_content_metadata | robust | drop | 0.879 | 0.007 | 100.0% | 1.112 | 0.032 |
| captured_content_metadata | robust | keep | 0.884 | 0.009 | 100.0% | 1.073 | 0.032 |
| captured_content_metadata | standard | drop | 0.964 | 0.003 | 100.0% | 1.150 | 0.032 |
| captured_content_metadata | standard | keep | 0.965 | 0.003 | 100.0% | 1.120 | 0.032 |
| combined_behavior | robust | drop | 0.988 | 0.004 | 100.0% | 1.188 | 0.032 |
| combined_behavior | robust | keep | 0.988 | 0.004 | 100.0% | 1.188 | 0.032 |
| combined_behavior | standard | drop | 0.979 | 0.002 | 100.0% | 1.071 | 0.032 |
| combined_behavior | standard | keep | 0.979 | 0.002 | 100.0% | 1.071 | 0.032 |
| normalized_behavior | robust | drop | 0.982 | 0.035 | 100.0% | 1.028 | 0.290 |
| normalized_behavior | robust | keep | 0.993 | 0.004 | 100.0% | 1.055 | 0.097 |
| normalized_behavior | standard | drop | 0.995 | 0.002 | 100.0% | 1.028 | 0.161 |
| normalized_behavior | standard | keep | 0.996 | 0.001 | 100.0% | 1.061 | 0.065 |
| raw_volume | robust | drop | 0.934 | 0.005 | 100.0% | 2.172 | 0.032 |
| raw_volume | robust | keep | 0.934 | 0.005 | 100.0% | 2.172 | 0.032 |
| raw_volume | standard | drop | 0.935 | 0.005 | 100.0% | 2.176 | 0.032 |
| raw_volume | standard | keep | 0.935 | 0.005 | 100.0% | 2.176 | 0.032 |

## Interpretation limits

- A bounding-box uniform null can exaggerate structure for correlated, constrained, or non-convex supports.
- The marginal-permutation null tests dependence beyond empirical univariate marginals; it is not a universal no-cluster null.
- A disagreement after redundancy removal is evidence for REVIEW because dependency/manifold structure can mimic separated groups.
- Exact duplicates can inflate clusterability, so duplicate-dropped runs are primary and duplicate-kept runs are sensitivity checks.
- A GO signal permits Stage 3 comparison; it does not prove that useful or stable clusters exist.

Detailed trial-level evidence is in `reports/tables/hopkins_sensitivity.csv` and `reports/tables/null_distance_comparison.csv`.
