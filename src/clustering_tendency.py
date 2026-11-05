"""Reproducible Stage 2 diagnostics for clustering tendency.

This module deliberately stops before fitting any clustering algorithm.  It
combines a bounding-box Hopkins statistic with a second, marginal-permutation
nearest-neighbour null model and reports sensitivity to feature choice,
scaling, sampling, random seed, and exact duplicate rows.

Hopkins orientation used here is ``sum(u) / (sum(u) + sum(w))`` where ``u``
are distances from uniform reference points to the observed sample and ``w``
are leave-one-out observed nearest-neighbour distances.  Therefore values
near 0.5 are compatible with spatial randomness under the bounding-box null,
and values approaching 1 indicate clustering tendency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


HOPKINS_SIGNAL_LEVEL = 0.65


@dataclass(frozen=True)
class PreparedMatrix:
    """A numeric analysis matrix plus preprocessing diagnostics."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    rows_before: int
    rows_after: int
    exact_duplicates_removed: int
    all_missing_features_removed: tuple[str, ...]
    constant_features_removed: tuple[str, ...]


def _as_finite_2d(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")
    if matrix.shape[0] < 3:
        raise ValueError("At least three observations are required.")
    if matrix.shape[1] < 1:
        raise ValueError("At least one non-constant feature is required.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return matrix


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    scaler_name: str,
    drop_exact_duplicates: bool,
) -> PreparedMatrix:
    """Coerce, median-impute, remove constants, scale, and optionally dedupe.

    This is descriptive unsupervised preprocessing.  A future predictive task
    must fit its imputer/scaler on its training partition instead of reusing
    this whole-data fit.
    """

    requested = tuple(feature_names)
    if not requested:
        raise ValueError("A feature set cannot be empty.")
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing configured features: {missing}")

    numeric = frame.loc[:, requested].apply(pd.to_numeric, errors="coerce")
    all_missing = tuple(name for name in requested if numeric[name].isna().all())
    if all_missing:
        numeric = numeric.drop(columns=list(all_missing))
    if numeric.shape[1] == 0:
        raise ValueError("Every configured feature is entirely missing.")

    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    variances = np.var(imputed, axis=0)
    keep_mask = np.isfinite(variances) & (variances > 0.0)
    constants = tuple(numeric.columns[~keep_mask].tolist())
    imputed = imputed[:, keep_mask]
    kept_names = tuple(numeric.columns[keep_mask].tolist())
    if imputed.shape[1] == 0:
        raise ValueError("Every configured feature is constant after imputation.")

    if scaler_name == "standard":
        scaled = StandardScaler().fit_transform(imputed)
    elif scaler_name == "robust":
        scaled = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(imputed)
    else:
        raise ValueError(f"Unknown scaler: {scaler_name!r}")
    scaled = _as_finite_2d(scaled)

    rows_before = int(scaled.shape[0])
    duplicates_removed = 0
    if drop_exact_duplicates:
        # Scaling is deterministic; rounding prevents insignificant floating
        # representation differences from hiding mathematically equal rows.
        _, unique_indices = np.unique(
            np.round(scaled, decimals=12), axis=0, return_index=True
        )
        unique_indices = np.sort(unique_indices)
        scaled = scaled[unique_indices]
        duplicates_removed = rows_before - int(scaled.shape[0])
        scaled = _as_finite_2d(scaled)

    return PreparedMatrix(
        values=scaled,
        feature_names=kept_names,
        rows_before=rows_before,
        rows_after=int(scaled.shape[0]),
        exact_duplicates_removed=duplicates_removed,
        all_missing_features_removed=all_missing,
        constant_features_removed=constants,
    )


def hopkins_statistic(
    values: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> float:
    """Compute Hopkins with the conventional high-means-clustered orientation.

    The reference distribution is uniform over the observed axis-aligned
    bounding box.  Distances are Euclidean and are not raised to the feature
    dimension.  This common definition is kept explicit because alternative
    implementations sometimes reverse the statistic or exponentiate distance.
    """

    matrix = _as_finite_2d(values)
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")
    effective_size = min(int(sample_size), matrix.shape[0] - 1)
    if effective_size < 2:
        raise ValueError("Not enough rows for the requested Hopkins sample.")

    rng = np.random.default_rng(random_state)
    observed_indices = rng.choice(matrix.shape[0], size=effective_size, replace=False)

    observed_nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=1)
    observed_nn.fit(matrix)
    observed_distances = observed_nn.kneighbors(
        matrix[observed_indices], return_distance=True
    )[0][:, 1]

    lower = matrix.min(axis=0)
    upper = matrix.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError("Hopkins requires non-constant feature dimensions.")
    reference = rng.uniform(lower, upper, size=(effective_size, matrix.shape[1]))
    reference_distances = observed_nn.kneighbors(
        reference, n_neighbors=1, return_distance=True
    )[0][:, 0]

    denominator = float(reference_distances.sum() + observed_distances.sum())
    if denominator <= 0.0:
        raise ValueError("Hopkins is undefined because all neighbour distances are zero.")
    return float(reference_distances.sum() / denominator)


def nearest_neighbor_distances(values: np.ndarray) -> np.ndarray:
    """Return leave-one-out nearest-neighbour distance for every row."""

    matrix = _as_finite_2d(values)
    model = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=1)
    model.fit(matrix)
    return model.kneighbors(matrix, return_distance=True)[0][:, 1]


def marginal_permutation_null_comparison(
    values: np.ndarray,
    *,
    repetitions: int,
    random_state: int,
    max_rows: int | None = 3000,
) -> dict[str, Any]:
    """Compare observed NNDs with a marginal-preserving permutation null.

    Every null replicate independently permutes each column.  This preserves
    each feature's empirical marginal distribution while destroying
    multivariate dependence.  A small one-sided permutation p-value and a null
    to observed median ratio above one indicate that observed rows are more
    locally concentrated than expected under this particular null model.
    """

    matrix = _as_finite_2d(values)
    if repetitions < 9:
        raise ValueError("Use at least 9 null repetitions for an interpretable p-value.")
    rng = np.random.default_rng(random_state)
    if max_rows is not None and matrix.shape[0] > max_rows:
        indices = rng.choice(matrix.shape[0], size=max_rows, replace=False)
        matrix = matrix[indices]

    observed = nearest_neighbor_distances(matrix)
    observed_median = float(np.median(observed))
    observed_mean = float(np.mean(observed))
    null_medians: list[float] = []
    null_means: list[float] = []

    for _ in range(repetitions):
        permuted = np.empty_like(matrix)
        for column_index in range(matrix.shape[1]):
            permuted[:, column_index] = matrix[
                rng.permutation(matrix.shape[0]), column_index
            ]
        distances = nearest_neighbor_distances(permuted)
        null_medians.append(float(np.median(distances)))
        null_means.append(float(np.mean(distances)))

    null_median_center = float(np.median(null_medians))
    null_median_std = float(np.std(null_medians, ddof=1))
    p_value = float(
        (1 + sum(value <= observed_median for value in null_medians))
        / (repetitions + 1)
    )
    ratio = (
        float(null_median_center / observed_median)
        if observed_median > 0.0
        else None
    )
    z_effect = (
        float((null_median_center - observed_median) / null_median_std)
        if null_median_std > 0.0
        else None
    )

    return {
        "rows_used": int(matrix.shape[0]),
        "features_used": int(matrix.shape[1]),
        "repetitions": int(repetitions),
        "observed_nnd_median": observed_median,
        "observed_nnd_mean": observed_mean,
        "observed_zero_nnd_fraction": float(np.mean(observed == 0.0)),
        "null_nnd_median_center": null_median_center,
        "null_nnd_median_std": null_median_std,
        "null_nnd_mean_center": float(np.mean(null_means)),
        "null_to_observed_median_ratio": ratio,
        "one_sided_permutation_p_value": p_value,
        "median_difference_standardized_by_null": z_effect,
        "null_medians": null_medians,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _save_pca_plot(
    values: np.ndarray,
    *,
    title: str,
    destination: Path,
    random_state: int,
    max_points: int = 5000,
) -> dict[str, Any]:
    matrix = _as_finite_2d(values)
    if matrix.shape[1] < 2:
        return {
            "generated": False,
            "reason": "PCA 2D requires at least two retained features.",
        }
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(matrix)
    rng = np.random.default_rng(random_state)
    if coordinates.shape[0] > max_points:
        plot_indices = rng.choice(coordinates.shape[0], size=max_points, replace=False)
        coordinates = coordinates[plot_indices]

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=7, alpha=0.3, linewidths=0)
    axis.set_title(title + " (exploratory only)")
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return {
        "generated": True,
        "path": str(destination),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_cumulative_variance_ratio": float(
            pca.explained_variance_ratio_[:2].sum()
        ),
        "visualization_role": "exploratory_only_not_clusterability_evidence",
    }


def _assess_gate(
    hopkins_frame: pd.DataFrame, null_frame: pd.DataFrame
) -> tuple[str, str, list[dict[str, Any]]]:
    grouped = (
        hopkins_frame.groupby(["feature_set", "scaler", "duplicate_mode"], as_index=False)
        .agg(
            hopkins_mean=("hopkins", "mean"),
            hopkins_std=("hopkins", "std"),
            fraction_at_or_above_0_65=(
                "hopkins",
                lambda values: float(np.mean(values >= HOPKINS_SIGNAL_LEVEL)),
            ),
            trials=("hopkins", "size"),
        )
        .merge(
            null_frame[
                [
                    "feature_set",
                    "scaler",
                    "duplicate_mode",
                    "null_to_observed_median_ratio",
                    "one_sided_permutation_p_value",
                ]
            ],
            on=["feature_set", "scaler", "duplicate_mode"],
            how="left",
        )
    )
    records = grouped.where(pd.notna(grouped), None).to_dict(orient="records")

    # Duplicate-dropped runs are primary.  Duplicate-kept runs exist to expose
    # artificial clusterability caused by repeated rows.
    primary = grouped[grouped["duplicate_mode"] == "drop"]
    if primary.empty:
        primary = grouped
    ratios = pd.to_numeric(primary["null_to_observed_median_ratio"], errors="coerce")
    strong = (
        (primary["fraction_at_or_above_0_65"] >= 0.80)
        & (primary["one_sided_permutation_p_value"] <= 0.05)
        & (ratios >= 1.05)
    )
    strong_fraction = float(strong.mean()) if len(strong) else 0.0

    if len(primary) >= 2 and strong_fraction >= 0.80:
        return (
            "GO",
            f"{int(strong.sum())}/{len(primary)} primary sensitivity configurations agree on both "
            "Hopkins and the marginal-permutation distance diagnostic.",
            records,
        )
    no_hopkins_signal = bool((primary["hopkins_mean"] < 0.60).all())
    no_distance_signal = bool(
        (
            (primary["one_sided_permutation_p_value"] > 0.05)
            | ratios.isna()
            | (ratios < 1.05)
        ).all()
    )
    if no_hopkins_signal and no_distance_signal:
        return (
            "PIVOT",
            "Primary configurations show neither stable Hopkins evidence nor a "
            "material nearest-neighbour distance departure from the null.",
            records,
        )
    return (
        "REVIEW",
        f"Only {int(strong.sum())}/{len(primary)} primary sensitivity configurations pass both diagnostics; "
        "representation sensitivity requires review before Stage 3.",
        records,
    )


def _write_summary_plots(
    hopkins_frame: pd.DataFrame,
    null_frame: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    paths: list[str] = []
    summary = (
        hopkins_frame.groupby(
            ["feature_set", "scaler", "duplicate_mode", "requested_sample_size"],
            as_index=False,
        )["hopkins"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    for keys, group in summary.groupby(["feature_set", "scaler", "duplicate_mode"]):
        group = group.sort_values("requested_sample_size")
        label = " / ".join(keys)
        axis.plot(group["requested_sample_size"], group["mean"], marker="o", label=label)
    axis.axhline(HOPKINS_SIGNAL_LEVEL, color="tab:red", linestyle="--", linewidth=1, label="0.65 signal guide")
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1, label="0.50 random guide")
    axis.set_xlabel("Requested sample size")
    axis.set_ylabel("Mean Hopkins statistic")
    axis.set_title("Hopkins sensitivity across seeds and representations")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    hopkins_path = figures_dir / "hopkins_sensitivity.png"
    fig.savefig(hopkins_path, dpi=160)
    plt.close(fig)
    paths.append(str(hopkins_path))

    labels = (
        null_frame["feature_set"]
        + " / "
        + null_frame["scaler"]
        + " / "
        + null_frame["duplicate_mode"]
    )
    ratios = pd.to_numeric(
        null_frame["null_to_observed_median_ratio"], errors="coerce"
    )
    fig_height = max(4.8, 0.42 * len(null_frame))
    fig, axis = plt.subplots(figsize=(9.0, fig_height))
    y_positions = np.arange(len(null_frame))
    axis.barh(y_positions, ratios.fillna(0.0), color="tab:blue", alpha=0.75)
    axis.axvline(1.0, color="black", linestyle=":", linewidth=1)
    axis.axvline(1.05, color="tab:red", linestyle="--", linewidth=1)
    axis.set_yticks(y_positions, labels=labels, fontsize=7)
    axis.set_xlabel("Median null NND / median observed NND")
    axis.set_title("Marginal-permutation nearest-neighbour diagnostic")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    null_path = figures_dir / "null_distance_comparison.png"
    fig.savefig(null_path, dpi=160)
    plt.close(fig)
    paths.append(str(null_path))
    return paths


def run_clustering_tendency(
    *,
    input_path: Path,
    feature_sets: Mapping[str, Sequence[str]],
    output_dir: Path,
    scalers: Sequence[str] = ("standard", "robust"),
    duplicate_modes: Sequence[str] = ("keep", "drop"),
    seeds: Sequence[int] = (11, 23, 42, 73, 101),
    sample_sizes: Sequence[int] = (100, 250, 500),
    null_repetitions: int = 49,
    max_null_rows: int = 3000,
) -> dict[str, Any]:
    """Run and persist every Stage 2 diagnostic."""

    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not feature_sets:
        raise ValueError("At least one feature set is required.")
    invalid_duplicate_modes = set(duplicate_modes) - {"keep", "drop"}
    if invalid_duplicate_modes:
        raise ValueError(f"Unknown duplicate modes: {sorted(invalid_duplicate_modes)}")

    frame = pd.read_csv(input_path, low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hopkins_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []

    for feature_set_name, feature_names in feature_sets.items():
        for scaler_name in scalers:
            for duplicate_mode in duplicate_modes:
                prepared = prepare_feature_matrix(
                    frame,
                    feature_names,
                    scaler_name=scaler_name,
                    drop_exact_duplicates=duplicate_mode == "drop",
                )
                common = {
                    "feature_set": feature_set_name,
                    "scaler": scaler_name,
                    "duplicate_mode": duplicate_mode,
                }
                preprocessing_rows.append(
                    {
                        **common,
                        "configured_features": list(feature_names),
                        "retained_features": list(prepared.feature_names),
                        "rows_before": prepared.rows_before,
                        "rows_after": prepared.rows_after,
                        "exact_duplicates_removed": prepared.exact_duplicates_removed,
                        "all_missing_features_removed": list(
                            prepared.all_missing_features_removed
                        ),
                        "constant_features_removed": list(
                            prepared.constant_features_removed
                        ),
                    }
                )
                for requested_size in sample_sizes:
                    effective_size = min(int(requested_size), prepared.rows_after - 1)
                    for seed in seeds:
                        hopkins_rows.append(
                            {
                                **common,
                                "seed": int(seed),
                                "requested_sample_size": int(requested_size),
                                "effective_sample_size": effective_size,
                                "rows": prepared.rows_after,
                                "features": len(prepared.feature_names),
                                "hopkins": hopkins_statistic(
                                    prepared.values,
                                    sample_size=requested_size,
                                    random_state=seed,
                                ),
                            }
                        )

                null_result = marginal_permutation_null_comparison(
                    prepared.values,
                    repetitions=null_repetitions,
                    random_state=42,
                    max_rows=max_null_rows,
                )
                null_rows.append(
                    {
                        **common,
                        **{key: value for key, value in null_result.items() if key != "null_medians"},
                        "null_medians": json.dumps(null_result["null_medians"]),
                    }
                )

                pca_path = figures_dir / (
                    "pca_"
                    + _slug(feature_set_name)
                    + "_"
                    + _slug(scaler_name)
                    + "_"
                    + duplicate_mode
                    + ".png"
                )
                pca_rows.append(
                    {
                        **common,
                        **_save_pca_plot(
                            prepared.values,
                            title=" / ".join(
                                [feature_set_name, scaler_name, duplicate_mode]
                            ),
                            destination=pca_path,
                            random_state=42,
                        ),
                    }
                )

    hopkins_frame = pd.DataFrame(hopkins_rows)
    null_frame = pd.DataFrame(null_rows)
    preprocessing_frame = pd.DataFrame(preprocessing_rows)
    pca_frame = pd.DataFrame(pca_rows)

    hopkins_path = tables_dir / "hopkins_sensitivity.csv"
    null_path = tables_dir / "null_distance_comparison.csv"
    preprocessing_path = tables_dir / "clusterability_preprocessing.csv"
    pca_path = tables_dir / "pca_exploratory_variance.csv"
    hopkins_frame.to_csv(hopkins_path, index=False)
    null_frame.to_csv(null_path, index=False)
    preprocessing_frame.to_csv(preprocessing_path, index=False)
    pca_frame.to_csv(pca_path, index=False)
    figure_paths = _write_summary_plots(hopkins_frame, null_frame, figures_dir)

    gate, gate_reason, config_summary = _assess_gate(hopkins_frame, null_frame)
    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "design": {
            "hopkins_formula": "sum(reference_nnd) / (sum(reference_nnd) + sum(observed_leave_one_out_nnd))",
            "hopkins_orientation": "approximately 0.5 random; toward 1 clustered",
            "hopkins_reference": "uniform_axis_aligned_observed_bounding_box",
            "distance_null": "independent_within-column_permutation_preserving_empirical_marginals",
            "distance_alternative": "observed median NND is smaller than null replicate medians",
            "pca_role": "exploratory visualization only",
            "outcome_warning": "Outcome-derived descriptive features cannot later serve as independent outcome validation.",
            "redundancy_policy": "Primary normalized and combined sets exclude algebraically redundant rate/count features; captured metadata excludes deterministic sine/cosine pairs.",
        },
        "parameters": {
            "feature_sets": {key: list(value) for key, value in feature_sets.items()},
            "scalers": list(scalers),
            "duplicate_modes": list(duplicate_modes),
            "seeds": [int(seed) for seed in seeds],
            "sample_sizes": [int(size) for size in sample_sizes],
            "null_repetitions": int(null_repetitions),
            "max_null_rows": int(max_null_rows),
        },
        "suggested_gate": gate,
        "suggested_gate_reason": gate_reason,
        "configuration_summary": config_summary,
        "artifacts": {
            "hopkins_table": str(hopkins_path),
            "null_distance_table": str(null_path),
            "preprocessing_table": str(preprocessing_path),
            "pca_table": str(pca_path),
            "summary_figures": figure_paths,
        },
        "limitations": [
            "A bounding-box uniform null can exaggerate structure for correlated, constrained, or non-convex supports.",
            "The marginal-permutation null tests dependence beyond empirical univariate marginals; it is not a universal no-cluster null.",
            "A disagreement after redundancy removal is evidence for REVIEW because dependency/manifold structure can mimic separated groups.",
            "Exact duplicates can inflate clusterability, so duplicate-dropped runs are primary and duplicate-kept runs are sensitivity checks.",
            "A GO signal permits Stage 3 comparison; it does not prove that useful or stable clusters exist.",
        ],
    }
    metrics_path = output_dir / "stage_02_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [
        "# Stage 2 — Clustering Tendency",
        "",
        f"**Suggested gate: {gate}.** {gate_reason}",
        "",
        "This is a diagnostic gate, not a clustering result. No Stage 3 algorithm was fitted.",
        "",
        "## Design",
        "",
        "- Hopkins uses `sum(u) / (sum(u) + sum(w))`: about 0.5 is random under the observed bounding-box null and values toward 1 indicate clustering tendency.",
        "- The independent distance diagnostic permutes each standardized feature separately, preserving empirical marginals while destroying dependence.",
        "- Primary normalized/combined sets remove exact algebraic redundancy, and captured metadata excludes deterministic sine/cosine pairs; this is a sensitivity guard against null-model artifacts.",
        "- Exact-duplicate-dropped configurations are primary; duplicate-kept runs expose duplicate concentration sensitivity.",
        "- PCA figures are exploratory only and are not treated as statistical evidence.",
        "",
        "## Configuration summary",
        "",
        "| Feature set | Scaler | Duplicates | Hopkins mean | Hopkins SD | Fraction ≥ 0.65 | Null/observed median NND | Null p-value |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config_summary:
        ratio_value = row.get("null_to_observed_median_ratio")
        ratio_text = "undefined (observed median is zero)" if ratio_value is None else f"{ratio_value:.3f}"
        summary_lines.append(
            "| {feature_set} | {scaler} | {duplicate_mode} | {hopkins_mean:.3f} | {hopkins_std:.3f} | {fraction_at_or_above_0_65:.1%} | {ratio} | {p:.3f} |".format(
                **row,
                ratio=ratio_text,
                p=row["one_sided_permutation_p_value"],
            )
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
            "Detailed trial-level evidence is in `reports/tables/hopkins_sensitivity.csv` and `reports/tables/null_distance_comparison.csv`.",
        ]
    )
    report_path = output_dir / "stage_02_clusterability.md"
    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    result["artifacts"]["metrics_json"] = str(metrics_path)
    result["artifacts"]["report"] = str(report_path)
    return result


def _parse_feature_set(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Feature sets use NAME=column_a,column_b syntax."
        )
    name, raw_columns = value.split("=", 1)
    columns = [column.strip() for column in raw_columns.split(",") if column.strip()]
    if not name.strip() or not columns:
        raise argparse.ArgumentTypeError(
            "Feature sets require a non-empty name and at least one column."
        )
    return name.strip(), columns


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - only relevant to CLI environments
        raise RuntimeError("PyYAML is required when --config is used.") from error
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def _configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    config_root: dict[str, Any] = {}
    if args.config:
        loaded = _load_yaml_config(args.config)
        config_root = loaded.get("stage2", loaded.get("clustering_tendency", {}))
        if not isinstance(config_root, dict):
            raise ValueError("stage2 config must be a mapping.")

    configured_input = config_root.get(
        "input", loaded.get("processed_data", "data/processed/us_video_features.csv")
    )
    configured_output = config_root.get("output_dir", loaded.get("reports_dir", "reports"))
    feature_sets = config_root.get("feature_sets", {})
    if args.feature_set:
        feature_sets = dict(args.feature_set)
    if not isinstance(feature_sets, dict) or not feature_sets:
        raise ValueError("Provide feature sets via config or --feature-set.")

    return {
        "input_path": args.input or Path(configured_input),
        "feature_sets": feature_sets,
        "output_dir": args.output_dir or Path(configured_output),
        "scalers": args.scalers or config_root.get("scalers", ["standard", "robust"]),
        "duplicate_modes": args.duplicate_modes
        or config_root.get("duplicate_modes", ["keep", "drop"]),
        "seeds": args.seeds or config_root.get("seeds", [11, 23, 42, 73, 101]),
        "sample_sizes": args.sample_sizes
        or config_root.get("sample_sizes", [100, 250, 500]),
        "null_repetitions": args.null_repetitions
        if args.null_repetitions is not None
        else config_root.get("null_repetitions", 49),
        "max_null_rows": args.max_null_rows
        if args.max_null_rows is not None
        else config_root.get("max_null_rows", 3000),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 clustering-tendency diagnostics without fitting clusters."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--feature-set",
        action="append",
        type=_parse_feature_set,
        help="Repeatable NAME=column_a,column_b definition; overrides config sets.",
    )
    parser.add_argument("--scalers", nargs="+", choices=["standard", "robust"])
    parser.add_argument("--duplicate-modes", nargs="+", choices=["keep", "drop"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--null-repetitions", type=int)
    parser.add_argument("--max-null-rows", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configuration = _configuration_from_args(arguments)
    result = run_clustering_tendency(**configuration)
    print(
        json.dumps(
            {
                "suggested_gate": result["suggested_gate"],
                "suggested_gate_reason": result["suggested_gate_reason"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
