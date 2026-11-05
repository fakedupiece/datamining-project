"""Stage 1 cleaning and feature engineering for YouTube Trending snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_audit import NUMERIC_METRICS, parse_trending_dates
from .utils import load_config, resolve_project_path, safe_divide, write_json


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def _tag_count(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    empty = text.eq("") | text.str.lower().eq("[none]")
    return (text.str.count(r"\|") + 1).where(~empty, 0).astype(int)


def _dictionary() -> pd.DataFrame:
    rows = [
        ("video_id", "ID", "YouTube video identifier", "Do not use as a numeric feature."),
        ("channel_title", "ID / metadata", "Channel label", "Repeated channels can create group leakage."),
        ("category_id", "predictor / metadata", "YouTube category identifier", "Nominal; encode rather than treat as ordered."),
        ("publish_time", "metadata", "UTC video publication timestamp", "Use only to derive pre-outcome timing features."),
        ("first_trending_date", "post-publication metadata", "First observed US trending date", "Unavailable before outcome observation."),
        ("last_trending_date", "post-publication metadata", "Last observed US trending date", "Unavailable before outcome observation."),
        ("views", "outcome", "Cumulative views at the latest trending snapshot", "Post-publication outcome."),
        ("likes", "outcome", "Cumulative likes at the latest trending snapshot", "Post-publication outcome."),
        ("dislikes", "outcome", "Cumulative dislikes at the latest trending snapshot", "Post-publication outcome."),
        ("comment_count", "outcome", "Cumulative comments at the latest trending snapshot", "Post-publication outcome."),
        ("engagements", "derived outcome", "likes + dislikes + comment_count", "Circular for engagement prediction."),
        ("engagement_per_view", "derived outcome / leakage-risk", "Observed engagements divided by views", "Not a sales conversion rate; invalid when ratings/comments are disabled."),
        ("reaction_rate", "derived outcome / leakage-risk", "(likes + dislikes) divided by views", "Invalid when ratings are disabled."),
        ("comment_rate", "derived outcome / leakage-risk", "comment_count divided by views", "Invalid when comments are disabled."),
        ("like_share", "derived outcome / leakage-risk", "likes divided by likes + dislikes", "Invalid when ratings are disabled or denominator is zero."),
        ("comments_per_engagement", "derived outcome / leakage-risk", "comments divided by observed engagements", "Invalid when comments/ratings are disabled."),
        ("days_to_trending", "derived outcome", "Calendar days from publication to first trending observation", "Post-publication timing outcome."),
        ("trending_days", "derived outcome", "Number of distinct dates observed as trending", "Selection-window and right-censoring risk."),
        ("snapshot_count", "metadata", "Number of retained daily snapshots for the video", "Repeated-measure diagnostic; not a pre-campaign feature."),
        ("active_days_at_last_snapshot", "derived outcome", "Calendar age in days at the latest trending snapshot", "Post-publication exposure denominator."),
        ("views_per_active_day", "derived outcome / leakage-risk", "Latest views divided by calendar age", "Content velocity, not sales or ROI."),
        ("title_length", "captured metadata", "Title length in characters", "Captured at a trending snapshot and may have been edited after publication."),
        ("title_word_count", "captured metadata", "Whitespace-tokenized title length", "Captured at a trending snapshot; language-dependent and not verified as publication-time metadata."),
        ("tag_count", "captured metadata", "Pipe-delimited tag count", "[none] is treated as zero; tags may have been edited after publication."),
        ("description_length", "captured metadata", "Description length in characters", "Missing description is represented as length zero together with description_missing."),
        ("description_missing", "metadata / missingness", "Whether the source description is missing", "Separates missing descriptions from observed empty descriptions."),
        ("publish_hour", "predictor", "UTC publication hour", "Cyclic, so use sine/cosine for distance methods."),
        ("publish_weekday", "predictor", "UTC weekday, Monday=0", "Cyclic, so use sine/cosine for distance methods."),
        ("publish_hour_sin", "engineered predictor", "Sine encoding of UTC publication hour", "Pre-outcome feature."),
        ("publish_hour_cos", "engineered predictor", "Cosine encoding of UTC publication hour", "Pre-outcome feature."),
        ("publish_weekday_sin", "engineered predictor", "Sine encoding of publication weekday", "Pre-outcome feature."),
        ("publish_weekday_cos", "engineered predictor", "Cosine encoding of publication weekday", "Pre-outcome feature."),
        ("comments_disabled", "metadata / missingness", "Whether comments were disabled", "Structural missingness; zero comments is not observed zero."),
        ("ratings_disabled", "metadata / missingness", "Whether ratings were disabled", "Structural missingness; zero ratings is not observed zero."),
        ("log1p_views", "transformed outcome", "Natural log(1 + views)", "Post-publication outcome."),
        ("log1p_likes", "transformed outcome", "Natural log(1 + likes)", "Post-publication outcome."),
        ("log1p_dislikes", "transformed outcome", "Natural log(1 + dislikes)", "Post-publication outcome."),
        ("log1p_comment_count", "transformed outcome", "Natural log(1 + comment_count)", "Post-publication outcome."),
        ("log1p_engagements", "transformed outcome", "Natural log(1 + observed engagements)", "Post-publication outcome."),
        ("log1p_views_per_active_day", "transformed outcome", "Natural log(1 + views per active day)", "Post-publication outcome."),
    ]
    return pd.DataFrame(rows, columns=["column", "role", "definition", "risk_or_usage_note"])


def build_feature_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse daily snapshots to a transparent video-level analysis table."""
    required = {
        "video_id",
        "trending_date",
        "publish_time",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = raw.copy()
    source_rows = len(working)
    exact_duplicates = int(working.duplicated().sum())
    working = working.drop_duplicates().copy()

    working["trending_dt"] = parse_trending_dates(working["trending_date"])
    working["publish_dt"] = pd.to_datetime(working["publish_time"], errors="coerce", utc=True)
    for column in NUMERIC_METRICS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working.loc[working[column] < 0, column] = np.nan
    for column in ["comments_disabled", "ratings_disabled", "video_error_or_removed"]:
        working[column] = _as_boolean(working[column])

    invalid_core = (
        working["video_id"].isna()
        | working["trending_dt"].isna()
        | working["publish_dt"].isna()
        | working["views"].isna()
    )
    invalid_core_rows = int(invalid_core.sum())
    working = working.loc[~invalid_core].copy()

    # For duplicate video/day observations, retain the most complete/highest-view snapshot.
    working["non_null_fields"] = working.notna().sum(axis=1)
    working = working.sort_values(
        ["video_id", "trending_dt", "views", "non_null_fields"], kind="stable"
    )
    duplicate_video_day_rows = int(working.duplicated(["video_id", "trending_dt"]).sum())
    working = working.drop_duplicates(["video_id", "trending_dt"], keep="last").copy()

    working = working.sort_values(["video_id", "trending_dt"], kind="stable")
    view_decrease_events = int((working.groupby("video_id")["views"].diff() < 0).sum())
    metadata = working.groupby("video_id", sort=False).agg(
        first_trending_date=("trending_dt", "min"),
        last_trending_date=("trending_dt", "max"),
        trending_days=("trending_dt", "nunique"),
        snapshot_count=("trending_dt", "size"),
        observed_category_count=("category_id", "nunique"),
        observed_channel_count=("channel_title", "nunique"),
    )
    latest = working.drop_duplicates("video_id", keep="last").set_index("video_id")
    features = latest.join(metadata, how="left").reset_index()

    error_or_removed = int(features["video_error_or_removed"].sum())
    nonpositive_views = int((features["views"] <= 0).sum())
    features = features.loc[
        ~features["video_error_or_removed"] & (features["views"] > 0)
    ].copy()

    ratings_invalid = features["ratings_disabled"]
    comments_invalid = features["comments_disabled"]
    structurally_missing_ratings = int(ratings_invalid.sum())
    structurally_missing_comments = int(comments_invalid.sum())
    features.loc[ratings_invalid, ["likes", "dislikes"]] = np.nan
    features.loc[comments_invalid, "comment_count"] = np.nan

    features["engagements"] = features[["likes", "dislikes", "comment_count"]].sum(
        axis=1, min_count=3
    )
    features["engagement_per_view"] = safe_divide(features["engagements"], features["views"])
    features["reaction_rate"] = safe_divide(
        features["likes"] + features["dislikes"], features["views"]
    )
    features["comment_rate"] = safe_divide(features["comment_count"], features["views"])
    features["like_share"] = safe_divide(
        features["likes"], features["likes"] + features["dislikes"]
    )
    features["comments_per_engagement"] = safe_divide(
        features["comment_count"], features["engagements"]
    )

    features.loc[ratings_invalid, ["reaction_rate", "like_share"]] = np.nan
    features.loc[comments_invalid, "comment_rate"] = np.nan
    features.loc[ratings_invalid | comments_invalid, ["engagement_per_view", "comments_per_engagement"]] = np.nan

    publish_day = features["publish_dt"].dt.floor("D")
    first_day = features["first_trending_date"].dt.floor("D")
    last_day = features["last_trending_date"].dt.floor("D")
    features["days_to_trending"] = (first_day - publish_day).dt.days.astype(float)
    features["active_days_at_last_snapshot"] = ((last_day - publish_day).dt.days + 1).astype(float)
    negative_time_rows = int(
        ((features["days_to_trending"] < 0) | (features["active_days_at_last_snapshot"] <= 0)).sum()
    )
    features.loc[features["days_to_trending"] < 0, "days_to_trending"] = np.nan
    features.loc[features["active_days_at_last_snapshot"] <= 0, "active_days_at_last_snapshot"] = np.nan
    features["views_per_active_day"] = safe_divide(
        features["views"], features["active_days_at_last_snapshot"]
    )

    title = features.get("title", pd.Series("", index=features.index)).fillna("").astype(str)
    source_description = features.get("description", pd.Series(np.nan, index=features.index))
    features["description_missing"] = source_description.isna()
    description = source_description.fillna("").astype(str)
    tags = features.get("tags", pd.Series("", index=features.index))
    features["title_length"] = title.str.len()
    features["title_word_count"] = title.str.split().str.len()
    features["tag_count"] = _tag_count(tags)
    features["description_length"] = description.str.len()
    features["publish_hour"] = features["publish_dt"].dt.hour
    features["publish_weekday"] = features["publish_dt"].dt.weekday
    features["publish_hour_sin"] = np.sin(2 * np.pi * features["publish_hour"] / 24)
    features["publish_hour_cos"] = np.cos(2 * np.pi * features["publish_hour"] / 24)
    features["publish_weekday_sin"] = np.sin(2 * np.pi * features["publish_weekday"] / 7)
    features["publish_weekday_cos"] = np.cos(2 * np.pi * features["publish_weekday"] / 7)

    log_sources = {
        "views": "log1p_views",
        "likes": "log1p_likes",
        "dislikes": "log1p_dislikes",
        "comment_count": "log1p_comment_count",
        "engagements": "log1p_engagements",
        "views_per_active_day": "log1p_views_per_active_day",
    }
    for source, target in log_sources.items():
        features[target] = np.log1p(features[source].clip(lower=0))

    output_columns = [
        "video_id",
        "channel_title",
        "category_id",
        "publish_time",
        "first_trending_date",
        "last_trending_date",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "engagements",
        "engagement_per_view",
        "reaction_rate",
        "comment_rate",
        "like_share",
        "comments_per_engagement",
        "days_to_trending",
        "trending_days",
        "snapshot_count",
        "active_days_at_last_snapshot",
        "views_per_active_day",
        "title_length",
        "title_word_count",
        "tag_count",
        "description_length",
        "description_missing",
        "publish_hour",
        "publish_weekday",
        "publish_hour_sin",
        "publish_hour_cos",
        "publish_weekday_sin",
        "publish_weekday_cos",
        "comments_disabled",
        "ratings_disabled",
        "log1p_views",
        "log1p_likes",
        "log1p_dislikes",
        "log1p_comment_count",
        "log1p_engagements",
        "log1p_views_per_active_day",
    ]
    features = features[output_columns].sort_values("video_id", kind="stable").reset_index(drop=True)

    summary = {
        "source_rows": int(source_rows),
        "exact_duplicate_rows_removed": exact_duplicates,
        "invalid_core_rows_removed": invalid_core_rows,
        "duplicate_video_day_rows_removed": duplicate_video_day_rows,
        "view_decrease_events_retained_as_source_diagnostic": view_decrease_events,
        "unique_videos_before_eligibility_filter": int(len(latest)),
        "video_error_or_removed_excluded": error_or_removed,
        "nonpositive_view_videos_excluded": nonpositive_views,
        "negative_or_impossible_timing_rows_set_missing": negative_time_rows,
        "videos_with_structurally_missing_ratings": structurally_missing_ratings,
        "videos_with_structurally_missing_comments": structurally_missing_comments,
        "analysis_rows": int(len(features)),
        "analysis_columns": int(features.shape[1]),
        "analysis_unit": "one unique video_id, latest retained US trending snapshot",
        "sales_revenue_cost_roi_columns": [],
    }
    return features, summary


def run(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config(config_path)
    raw_path = resolve_project_path(config["dataset"]["raw_file"])
    output_path = resolve_project_path(config["outputs"]["processed_file"])
    raw = pd.read_csv(raw_path, low_memory=False)
    features, summary = build_feature_table(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    write_json(summary, config["outputs"]["cleaning_json"])

    dictionary_path = resolve_project_path(config["outputs"]["data_dictionary_csv"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    _dictionary().to_csv(dictionary_path, index=False)

    numeric = features.select_dtypes(include=[np.number, "bool"])
    feature_summary = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    feature_summary.insert(0, "missing_count", numeric.isna().sum())
    feature_summary.insert(1, "unique_count", numeric.nunique(dropna=True))
    summary_path = resolve_project_path(config["outputs"]["feature_summary_csv"])
    feature_summary.reset_index(names="column").to_csv(summary_path, index=False)
    return features, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    features, summary = run(args.config)
    print(
        f"Created {len(features):,} video-level rows after removing "
        f"{summary['exact_duplicate_rows_removed']:,} exact duplicate snapshots."
    )


if __name__ == "__main__":
    main()
