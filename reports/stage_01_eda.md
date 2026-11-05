# Stage 1 — EDA and Feature Engineering

## Scope and analysis unit

This analysis uses one unique US trending `video_id`, represented by its latest retained daily snapshot. Daily snapshots are collapsed only after exact and video-date duplicate handling. The dataset contains content-performance outcomes, not sales, revenue, campaign cost, attribution, or ROI. All rate names therefore describe observed platform interactions and never purchase conversion.

## Evidence

- Analysis rows: **6,347** unique videos; duplicate video IDs: **0**.
- Engineered features meeting the predeclared completeness/variation check: **7** (engagement_per_view, reaction_rate, comment_rate, like_share, days_to_trending, trending_days, views_per_active_day).
- Seven EDA figures were generated, covering raw volume, normalized features, correlations, volume-versus-rate ranking, category variation, missingness, and timing.
- Views versus engagement-per-view Spearman rho: **0.066** (descriptive association, not causation).
- Top-decile views versus top-decile engagement-per-view Jaccard overlap: **0.043**.
- Median absolute percentile-rank shift between those measures: **0.275**.

The rank comparison demonstrates that normalized interaction intensity changes which videos appear exceptional relative to a raw view ranking. It does not establish incremental predictive value; that requires the later raw-only versus engineered ablation defined in the project brief.

## Cleaning and denominator policies

- Negative numeric values are set missing before aggregation.
- A duplicate video/day retains the most complete, highest-view snapshot.
- Error/removed videos and non-positive view denominators are excluded.
- Disabled comment/rating counts and every feature derived from them are missing rather than observed zero; feature-set preprocessing then reports its imputation policy.
- Impossible negative publication-to-trending intervals are set missing.
- Heavy-tailed counts receive `log1p` representations; raw values remain available for audit.

## Leakage boundary

`views`, interactions, normalized interaction rates, trending duration, and time-to-trending are post-publication outcomes. They may be used for descriptive behavior-space diagnostics, but not as predictors of the same outcomes. The captured-content-metadata sensitivity set uses title, tag, and description measurements from the latest trending snapshot; these values may have been edited after publication and are not labeled pre-publication predictors. Category is nominal and intentionally excluded from the initial numeric distance diagnostic unless explicitly encoded.

## Gate decision

**GO (engineering readiness under the content-performance pivot).** H1 has preliminary evidence that normalized features provide distinct, interpretable rankings beyond raw volume. This is an engineering-readiness decision, not final scientific acceptance of H1. Formal incremental-value evidence remains outside Stage 1.
