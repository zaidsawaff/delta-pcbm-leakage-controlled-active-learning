"""Synthetic tests for the public selector reference implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_pcbm import select_global_margin, select_pcbm  # noqa: E402


def token(index: int) -> str:
    return f"{index:024x}"


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "opaque_candidate_token": [token(index) for index in range(12)],
                "predicted_label": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6],
                "margin": [0.01, 0.02, 0.30, 0.31, 0.10, 0.11, 0.20, 0.21, 0.40, 0.41, 0.50, 0.60],
            }
        )

    def test_pcbm_represents_all_seven_classes(self) -> None:
        selected = select_pcbm(self.frame, 7)
        labels = self.frame.set_index("opaque_candidate_token").loc[selected, "predicted_label"]
        self.assertEqual(set(labels.tolist()), set(range(7)))

    def test_global_margin_uses_smallest_margins(self) -> None:
        observed = select_global_margin(self.frame, 7)
        expected = (
            self.frame.sort_values(["margin", "opaque_candidate_token"], kind="mergesort")
            .head(7)["opaque_candidate_token"]
            .tolist()
        )
        self.assertEqual(observed, expected)

    def test_pcbm_is_deterministic_under_row_permutation(self) -> None:
        first = select_pcbm(self.frame, 7)
        second = select_pcbm(self.frame.sample(frac=1.0, random_state=42), 7)
        self.assertEqual(first, second)

    def test_pcbm_fills_when_fewer_classes_are_represented(self) -> None:
        frame = self.frame.assign(predicted_label=[0] * 6 + [1] * 6)
        selected = select_pcbm(frame, 7)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(set(selected)), 7)
        self.assertIn(token(0), selected)
        self.assertIn(token(6), selected)

    def test_extra_selector_column_is_rejected(self) -> None:
        invalid = self.frame.assign(true_label=0)
        with self.assertRaises(ValueError):
            select_pcbm(invalid, 7)

    def test_duplicate_tokens_are_rejected(self) -> None:
        invalid = self.frame.copy()
        invalid.loc[1, "opaque_candidate_token"] = invalid.loc[0, "opaque_candidate_token"]
        with self.assertRaises(ValueError):
            select_pcbm(invalid, 7)

    def test_nonfinite_margin_is_rejected(self) -> None:
        invalid = self.frame.copy()
        invalid.loc[0, "margin"] = np.nan
        with self.assertRaises(ValueError):
            select_pcbm(invalid, 7)


if __name__ == "__main__":
    unittest.main()

