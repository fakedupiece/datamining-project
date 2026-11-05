"""Unit tests for Stage 0/1 cleaning and feature policies."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_audit import audit_dataframe
from src.preprocess import build_feature_table
from src.utils import safe_divide


def _row(video_id: str, trending_date: str, views: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "video_id": video_id,
        "trending_date": trending_date,
        "title": f"Title {video_id}",
        "channel_title": f"Channel {video_id}",
        "category_id": 10,
        "publish_time": "2017-11-10T12:00:00.000Z",
        "tags": "one|two",
        "views": views,
        "likes": 10,
        "dislikes": 2,
        "comment_count": 3,
        "thumbnail_link": "https://example.invalid/image.jpg",
        "comments_disabled": False,
        "ratings_disabled": False,
        "video_error_or_removed": False,
        "description": "A description",
    }
    row.update(overrides)
    return row


class SafeDivideTests(unittest.TestCase):
    def test_nonpositive_denominators_become_missing(self) -> None:
        actual = safe_divide([4, 4, np.nan], [2, 0, 2])
        np.testing.assert_allclose(actual[0], 2.0)
        self.assertTrue(np.isnan(actual[1]))
        self.assertTrue(np.isnan(actual[2]))


class FeatureTableTests(unittest.TestCase):
    def test_latest_snapshot_and_structural_missingness_policies(self) -> None:
        first = _row("a", "17.14.11", 100)
        latest = _row("a", "17.15.11", 200, likes=20, dislikes=4, comment_count=6)
        disabled = _row(
            "b",
            "17.14.11",
            80,
            comments_disabled=True,
            ratings_disabled=True,
            likes=0,
            dislikes=0,
            comment_count=0,
            description=None,
        )
        raw = pd.DataFrame([first, first.copy(), latest, disabled])
        features, summary = build_feature_table(raw)

        self.assertEqual(len(features), 2)
        self.assertEqual(summary["exact_duplicate_rows_removed"], 1)
        video_a = features.set_index("video_id").loc["a"]
        self.assertEqual(video_a["views"], 200)
        self.assertEqual(video_a["trending_days"], 2)
        self.assertAlmostEqual(video_a["engagement_per_view"], 30 / 200)
        self.assertEqual(video_a["tag_count"], 2)

        video_b = features.set_index("video_id").loc["b"]
        self.assertTrue(np.isnan(video_b["likes"]))
        self.assertTrue(np.isnan(video_b["dislikes"]))
        self.assertTrue(np.isnan(video_b["comment_count"]))
        self.assertTrue(np.isnan(video_b["log1p_engagements"]))
        self.assertTrue(np.isnan(video_b["engagement_per_view"]))
        self.assertTrue(np.isnan(video_b["reaction_rate"]))
        self.assertTrue(np.isnan(video_b["comment_rate"]))
        self.assertTrue(video_b["description_missing"])
        self.assertEqual(video_b["description_length"], 0)

    def test_audit_detects_snapshot_duplicates_and_date_coverage(self) -> None:
        raw = pd.DataFrame(
            [
                _row("a", "17.14.11", 100),
                _row("a", "17.14.11", 100),
                _row("a", "17.15.11", 120),
            ]
        )
        audit = audit_dataframe(raw)
        self.assertEqual(audit["exact_duplicate_rows"], 1)
        self.assertEqual(audit["duplicate_video_date_rows"], 1)
        self.assertEqual(audit["trending_date"]["unique_days"], 2)
        self.assertTrue(audit["columns_match_expected_schema"])


if __name__ == "__main__":
    unittest.main()
