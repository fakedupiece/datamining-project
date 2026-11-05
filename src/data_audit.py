"""Stage 0 file-level audit for the selected YouTube Trending dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_config, resolve_project_path, sha256_file, write_json


EXPECTED_COLUMNS = [
    "video_id",
    "trending_date",
    "title",
    "channel_title",
    "category_id",
    "publish_time",
    "tags",
    "views",
    "likes",
    "dislikes",
    "comment_count",
    "thumbnail_link",
    "comments_disabled",
    "ratings_disabled",
    "video_error_or_removed",
    "description",
]

NUMERIC_METRICS = ["views", "likes", "dislikes", "comment_count"]


def parse_trending_dates(series: pd.Series) -> pd.Series:
    """Parse the source's documented YY.DD.MM date format."""
    return pd.to_datetime(series, format="%y.%d.%m", errors="coerce", utc=True)


def audit_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return observable quality and schema evidence without changing the data."""
    missing_expected = sorted(set(EXPECTED_COLUMNS) - set(dataframe.columns))
    unexpected = sorted(set(dataframe.columns) - set(EXPECTED_COLUMNS))

    trending_dates = parse_trending_dates(dataframe.get("trending_date", pd.Series(dtype="object")))
    publish_times = pd.to_datetime(
        dataframe.get("publish_time", pd.Series(dtype="object")), errors="coerce", utc=True
    )

    negative_counts: dict[str, int] = {}
    zero_counts: dict[str, int] = {}
    for column in NUMERIC_METRICS:
        if column in dataframe:
            values = pd.to_numeric(dataframe[column], errors="coerce")
            negative_counts[column] = int((values < 0).sum())
            zero_counts[column] = int((values == 0).sum())

    video_date_duplicates = 0
    repeated_video_rows = 0
    if {"video_id", "trending_date"}.issubset(dataframe.columns):
        video_date_duplicates = int(dataframe.duplicated(["video_id", "trending_date"]).sum())
    if "video_id" in dataframe:
        repeated_video_rows = int(dataframe["video_id"].duplicated(keep=False).sum())

    categorical_cardinality = {
        column: int(dataframe[column].nunique(dropna=True))
        for column in ["video_id", "channel_title", "category_id"]
        if column in dataframe
    }

    boolean_true_counts = {
        column: int(dataframe[column].fillna(False).astype(bool).sum())
        for column in ["comments_disabled", "ratings_disabled", "video_error_or_removed"]
        if column in dataframe
    }

    return {
        "shape": {"rows": int(len(dataframe)), "columns": int(dataframe.shape[1])},
        "columns_match_expected_schema": not missing_expected and not unexpected,
        "missing_expected_columns": missing_expected,
        "unexpected_columns": unexpected,
        "exact_duplicate_rows": int(dataframe.duplicated().sum()),
        "duplicate_video_date_rows": video_date_duplicates,
        "rows_belonging_to_repeated_videos": repeated_video_rows,
        "missing_values": {column: int(value) for column, value in dataframe.isna().sum().items()},
        "negative_numeric_values": negative_counts,
        "zero_numeric_values": zero_counts,
        "cardinality": categorical_cardinality,
        "boolean_true_counts": boolean_true_counts,
        "trending_date": {
            "valid": int(trending_dates.notna().sum()),
            "invalid": int(trending_dates.isna().sum()),
            "minimum": trending_dates.min().isoformat() if trending_dates.notna().any() else None,
            "maximum": trending_dates.max().isoformat() if trending_dates.notna().any() else None,
            "unique_days": int(trending_dates.dt.date.nunique()),
        },
        "publish_time": {
            "valid": int(publish_times.notna().sum()),
            "invalid": int(publish_times.isna().sum()),
            "minimum": publish_times.min().isoformat() if publish_times.notna().any() else None,
            "maximum": publish_times.max().isoformat() if publish_times.notna().any() else None,
        },
    }


def run(config_path: str | Path) -> dict[str, Any]:
    """Audit the configured raw file and persist machine-readable evidence."""
    config = load_config(config_path)
    dataset = config["dataset"]
    outputs = config["outputs"]
    raw_path = resolve_project_path(dataset["raw_file"])
    archive_path = resolve_project_path(dataset["raw_archive"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file is missing: {raw_path}. Download and extract the configured Kaggle archive first."
        )

    dataframe = pd.read_csv(raw_path, low_memory=False)
    audit = audit_dataframe(dataframe)
    audit.update(
        {
            "dataset_name": dataset["name"],
            "source_url": dataset["source_url"],
            "analysis_unit_candidate": dataset["analysis_unit"],
            "commercial_outcomes_available": bool(dataset["commercial_outcomes_available"]),
            "raw_file": str(raw_path.relative_to(resolve_project_path("."))),
            "raw_file_bytes": raw_path.stat().st_size,
            "raw_file_sha256": sha256_file(raw_path),
            "archive_file": str(archive_path.relative_to(resolve_project_path(".")))
            if archive_path.exists()
            else None,
            "archive_bytes": archive_path.stat().st_size if archive_path.exists() else None,
            "archive_sha256": sha256_file(archive_path) if archive_path.exists() else None,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    schema = pd.DataFrame(
        {
            "column": dataframe.columns,
            "pandas_dtype": [str(dataframe[column].dtype) for column in dataframe.columns],
            "non_null_count": [int(dataframe[column].notna().sum()) for column in dataframe.columns],
            "missing_count": [int(dataframe[column].isna().sum()) for column in dataframe.columns],
            "unique_count": [int(dataframe[column].nunique(dropna=True)) for column in dataframe.columns],
        }
    )
    schema_path = resolve_project_path(outputs["schema_csv"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema.to_csv(schema_path, index=False)
    write_json(audit, outputs["audit_json"])
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    audit = run(args.config)
    print(
        f"Audited {audit['shape']['rows']:,} rows; "
        f"SHA-256={audit['raw_file_sha256']}"
    )


if __name__ == "__main__":
    main()
