"""Tests for the Stage 2 clusterability diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.clustering_tendency import (
    hopkins_statistic,
    marginal_permutation_null_comparison,
    prepare_feature_matrix,
)


class HopkinsStatisticTests(unittest.TestCase):
    def test_uniform_data_is_close_to_half(self) -> None:
        rng = np.random.default_rng(1234)
        uniform = rng.uniform(0.0, 1.0, size=(1800, 2))

        value = hopkins_statistic(uniform, sample_size=350, random_state=42)

        self.assertGreater(value, 0.40)
        self.assertLess(value, 0.60)

    def test_tight_clusters_use_high_is_clustered_orientation(self) -> None:
        rng = np.random.default_rng(1234)
        first = rng.normal(loc=(-2.0, -2.0), scale=0.12, size=(600, 2))
        second = rng.normal(loc=(2.0, 2.0), scale=0.12, size=(600, 2))
        clustered = np.vstack([first, second])

        value = hopkins_statistic(clustered, sample_size=300, random_state=42)

        self.assertGreater(value, 0.80)

    def test_same_seed_is_reproducible(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(500, 4))

        first = hopkins_statistic(values, sample_size=100, random_state=73)
        second = hopkins_statistic(values, sample_size=100, random_state=73)

        self.assertEqual(first, second)


class MarginalPermutationTests(unittest.TestCase):
    def test_correlated_clustered_data_differs_from_permuted_null(self) -> None:
        rng = np.random.default_rng(99)
        centers = rng.choice([-2.5, 2.5], size=1000)
        x = centers + rng.normal(scale=0.16, size=1000)
        y = centers + rng.normal(scale=0.16, size=1000)
        values = np.column_stack([x, y])

        result = marginal_permutation_null_comparison(
            values, repetitions=19, random_state=42, max_rows=None
        )

        self.assertLessEqual(result["one_sided_permutation_p_value"], 0.10)
        self.assertIsNotNone(result["null_to_observed_median_ratio"])
        self.assertGreater(result["null_to_observed_median_ratio"], 1.20)


class PreprocessingTests(unittest.TestCase):
    def test_imputation_constant_filter_and_duplicate_sensitivity(self) -> None:
        frame = pd.DataFrame(
            {
                "varying": [1.0, 1.0, 2.0, np.nan, 2.0],
                "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
                "second": [3.0, 3.0, 4.0, 5.0, 4.0],
            }
        )

        prepared = prepare_feature_matrix(
            frame,
            ["varying", "constant", "second"],
            scaler_name="standard",
            drop_exact_duplicates=True,
        )

        self.assertEqual(prepared.feature_names, ("varying", "second"))
        self.assertEqual(prepared.constant_features_removed, ("constant",))
        self.assertEqual(prepared.exact_duplicates_removed, 2)
        self.assertTrue(np.isfinite(prepared.values).all())


if __name__ == "__main__":
    unittest.main()
