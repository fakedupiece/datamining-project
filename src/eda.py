"""Stage 1 exploratory analysis and evidence report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .utils import ensure_directory, load_config, resolve_project_path


RAW_VOLUME = ["views", "likes", "dislikes", "comment_count", "engagements"]
ENGINEERED = [
    "engagement_per_view",
    "reaction_rate",
    "comment_rate",
    "like_share",
    "days_to_trending",
    "trending_days",
    "views_per_active_day",
]


def _category_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels: dict[int, str] = {}
    for item in payload.get("items", []):
        try:
            labels[int(item["id"])] = item["snippet"]["title"]
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_raw_distributions(dataframe: pd.DataFrame, output: Path) -> None:
    columns = [column for column in RAW_VOLUME if column in dataframe]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, column in zip(axes.flat, columns, strict=False):
        values = np.log1p(dataframe[column].dropna().clip(lower=0))
        axis.hist(values, bins=50, color="#3b82f6", alpha=0.85)
        axis.set_title(f"log1p({column})")
        axis.set_xlabel("log-transformed value")
        axis.set_ylabel("videos")
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    figure.suptitle("Raw content-performance distributions (log scale)")
    _save(figure, output)


def _plot_engineered_distributions(dataframe: pd.DataFrame, output: Path) -> None:
    columns = [column for column in ENGINEERED if column in dataframe]
    figure, axes = plt.subplots(3, 3, figsize=(13, 10))
    for axis, column in zip(axes.flat, columns, strict=False):
        values = dataframe[column].dropna()
        if len(values):
            upper = values.quantile(0.99)
            lower = values.quantile(0.01)
            axis.hist(values.clip(lower=lower, upper=upper), bins=45, color="#14b8a6", alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("videos")
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    figure.suptitle("Engineered feature distributions (1st-99th percentile display)")
    _save(figure, output)


def _plot_correlation(dataframe: pd.DataFrame, output: Path, table_path: Path) -> pd.DataFrame:
    columns = [
        "log1p_views",
        "log1p_likes",
        "log1p_comment_count",
        "engagement_per_view",
        "reaction_rate",
        "comment_rate",
        "like_share",
        "days_to_trending",
        "trending_days",
        "log1p_views_per_active_day",
    ]
    columns = [column for column in columns if column in dataframe]
    correlation = dataframe[columns].corr(method="spearman")
    correlation.to_csv(table_path)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    axis.set_yticks(range(len(columns)), columns)
    for row in range(len(columns)):
        for column in range(len(columns)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    axis.set_title("Raw-volume and engineered-feature rank correlations")
    _save(figure, output)
    return correlation


def _plot_scatter(dataframe: pd.DataFrame, output: Path, seed: int) -> None:
    valid = dataframe.dropna(subset=["views", "engagement_per_view", "trending_days"])
    sample = valid.sample(n=min(6000, len(valid)), random_state=seed)
    figure, axis = plt.subplots(figsize=(9, 6))
    points = axis.scatter(
        np.log10(sample["views"].clip(lower=1)),
        sample["engagement_per_view"],
        c=sample["trending_days"],
        cmap="viridis",
        s=11,
        alpha=0.45,
    )
    axis.set_ylim(0, sample["engagement_per_view"].quantile(0.99))
    axis.set_xlabel("log10(latest views)")
    axis.set_ylabel("observed engagements / views")
    axis.set_title("Volume and normalized engagement capture different rankings")
    figure.colorbar(points, ax=axis, label="distinct trending days")
    _save(figure, output)


def _plot_category_boxplot(
    dataframe: pd.DataFrame, output: Path, category_labels: dict[int, str]
) -> pd.DataFrame:
    valid = dataframe.dropna(subset=["category_id", "engagement_per_view"]).copy()
    counts = valid["category_id"].value_counts().head(10)
    category_ids = counts.index.tolist()
    groups = [
        valid.loc[valid["category_id"] == category, "engagement_per_view"].clip(
            upper=valid["engagement_per_view"].quantile(0.99)
        )
        for category in category_ids
    ]
    labels = [category_labels.get(int(category), str(category)) for category in category_ids]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(groups, tick_labels=labels, showfliers=False)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("observed engagements / views")
    axis.set_title("Normalized engagement differs across high-support categories")
    _save(figure, output)

    summary = (
        valid.groupby("category_id")
        .agg(
            videos=("video_id", "size"),
            median_views=("views", "median"),
            median_engagement_per_view=("engagement_per_view", "median"),
            median_days_to_trending=("days_to_trending", "median"),
            median_trending_days=("trending_days", "median"),
        )
        .reset_index()
        .sort_values("videos", ascending=False)
    )
    summary.insert(
        1,
        "category_name",
        summary["category_id"].map(lambda value: category_labels.get(int(value), str(value))),
    )
    return summary


def _plot_missingness(dataframe: pd.DataFrame, output: Path) -> None:
    missing = dataframe.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(missing.index, missing.values, color="#f59e0b")
    axis.tick_params(axis="x", rotation=55)
    axis.set_ylabel("missing fraction")
    axis.set_ylim(0, max(0.05, float(missing.max()) * 1.1) if len(missing) else 0.05)
    axis.set_title("Feature missingness after structural-disabled policies")
    _save(figure, output)


def _plot_timing(dataframe: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, column, color in [
        (axes[0], "days_to_trending", "#8b5cf6"),
        (axes[1], "trending_days", "#ec4899"),
    ]:
        values = dataframe[column].dropna()
        axis.hist(values.clip(upper=values.quantile(0.99)), bins=45, color=color, alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("videos")
    figure.suptitle("Trending timing and persistence")
    _save(figure, output)


def _rank_comparison(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = dataframe.dropna(subset=["video_id", "views", "engagement_per_view"]).copy()
    cutoff = max(1, int(np.ceil(len(valid) * 0.10)))
    top_views = set(valid.nlargest(cutoff, "views")["video_id"])
    top_rate = set(valid.nlargest(cutoff, "engagement_per_view")["video_id"])
    intersection = len(top_views & top_rate)
    union = len(top_views | top_rate)
    rho, pvalue = spearmanr(valid["views"], valid["engagement_per_view"], nan_policy="omit")

    valid["views_rank_pct"] = valid["views"].rank(pct=True, method="average")
    valid["engagement_rate_rank_pct"] = valid["engagement_per_view"].rank(
        pct=True, method="average"
    )
    valid["absolute_rank_shift"] = (
        valid["views_rank_pct"] - valid["engagement_rate_rank_pct"]
    ).abs()
    examples = valid.nlargest(25, "absolute_rank_shift")[
        [
            "video_id",
            "channel_title",
            "category_id",
            "views",
            "engagement_per_view",
            "views_rank_pct",
            "engagement_rate_rank_pct",
            "absolute_rank_shift",
        ]
    ]
    metrics = {
        "eligible_videos": int(len(valid)),
        "top_decile_size": cutoff,
        "top_decile_intersection": intersection,
        "top_decile_jaccard": intersection / union if union else np.nan,
        "spearman_views_vs_engagement_per_view": float(rho),
        "spearman_pvalue": float(pvalue),
        "median_absolute_percentile_rank_shift": float(valid["absolute_rank_shift"].median()),
    }
    return examples, metrics


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = config["outputs"]
    seed = int(config["project"]["random_seed"])
    processed_path = resolve_project_path(output["processed_file"])
    dataframe = pd.read_csv(processed_path, parse_dates=["publish_time", "first_trending_date", "last_trending_date"])
    figures = ensure_directory(output["figures_dir"])
    tables = ensure_directory(output["tables_dir"])
    category_labels = _category_labels(resolve_project_path(config["dataset"]["category_file"]))

    _plot_raw_distributions(dataframe, figures / "stage_01_raw_volume_distributions.png")
    _plot_engineered_distributions(dataframe, figures / "stage_01_engineered_distributions.png")
    correlation = _plot_correlation(
        dataframe,
        figures / "stage_01_correlation_heatmap.png",
        tables / "stage_01_spearman_correlations.csv",
    )
    _plot_scatter(dataframe, figures / "stage_01_volume_vs_normalized_scatter.png", seed)
    category_summary = _plot_category_boxplot(
        dataframe, figures / "stage_01_category_boxplots.png", category_labels
    )
    category_summary.to_csv(tables / "stage_01_category_summary.csv", index=False)
    _plot_missingness(dataframe, figures / "stage_01_missingness.png")
    _plot_timing(dataframe, figures / "stage_01_timing_distributions.png")

    rank_examples, rank_metrics = _rank_comparison(dataframe)
    rank_examples.to_csv(tables / "stage_01_rank_shift_examples.csv", index=False)

    valid_engineered = [
        column
        for column in ENGINEERED
        if column in dataframe
        and dataframe[column].notna().mean() >= 0.80
        and dataframe[column].nunique(dropna=True) >= 10
    ]
    evidence = {
        "analysis_rows": int(len(dataframe)),
        "duplicate_video_ids": int(dataframe["video_id"].duplicated().sum()),
        "valid_engineered_features": valid_engineered,
        "valid_engineered_feature_count": len(valid_engineered),
        "generated_figure_count": 7,
        "rank_comparison": rank_metrics,
        "strongest_absolute_cross_family_spearman": float(
            correlation.loc[
                [column for column in correlation.index if column.startswith("log1p_")],
                [column for column in correlation.columns if column in ENGINEERED],
            ]
            .abs()
            .max()
            .max()
        ),
    }

    decision = (
        "GO (engineering readiness under the content-performance pivot)"
        if len(valid_engineered) >= 4
        and evidence["duplicate_video_ids"] == 0
        and evidence["generated_figure_count"] >= 5
        else "REVIEW"
    )
    report = f"""# Stage 1 — EDA and Feature Engineering

## Scope and analysis unit

This analysis uses one unique US trending `video_id`, represented by its latest retained daily snapshot. Daily snapshots are collapsed only after exact and video-date duplicate handling. The dataset contains content-performance outcomes, not sales, revenue, campaign cost, attribution, or ROI. All rate names therefore describe observed platform interactions and never purchase conversion.

## Evidence

- Analysis rows: **{len(dataframe):,}** unique videos; duplicate video IDs: **{evidence['duplicate_video_ids']}**.
- Engineered features meeting the predeclared completeness/variation check: **{len(valid_engineered)}** ({', '.join(valid_engineered)}).
- Seven EDA figures were generated, covering raw volume, normalized features, correlations, volume-versus-rate ranking, category variation, missingness, and timing.
- Views versus engagement-per-view Spearman rho: **{rank_metrics['spearman_views_vs_engagement_per_view']:.3f}** (descriptive association, not causation).
- Top-decile views versus top-decile engagement-per-view Jaccard overlap: **{rank_metrics['top_decile_jaccard']:.3f}**.
- Median absolute percentile-rank shift between those measures: **{rank_metrics['median_absolute_percentile_rank_shift']:.3f}**.

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

**{decision}.** H1 has preliminary evidence that normalized features provide distinct, interpretable rankings beyond raw volume. This is an engineering-readiness decision, not final scientific acceptance of H1. Formal incremental-value evidence remains outside Stage 1.
"""
    report_path = resolve_project_path("reports/stage_01_eda.md")
    report_path.write_text(report, encoding="utf-8")
    return {"decision": decision, **evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(f"Stage 1 decision: {result['decision']}")


if __name__ == "__main__":
    main()
