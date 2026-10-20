"""Tests for the chance baselines and the constrained-decoding rescoring.

Both scripts produce numbers the README and report quote, so the tests pin
those numbers to the tracked inputs and check the baseline arithmetic on a
small frame where the answer is known by hand.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import build_chance_baselines as cb  # noqa: E402
import build_schema_parity as sp  # noqa: E402
import task_registry  # noqa: E402

TASKS = {t["name"]: t for t in task_registry.load_task_definitions()}


class ChanceBaselineTests(unittest.TestCase):
    def test_binary_majority_negative_scores_zero_and_stratified_is_prevalence(self):
        task = TASKS["ballard_incivility"]
        gold = pd.DataFrame({"item_id": range(10), "gt_uncivil": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
        row = cb.baselines(task, gold)
        self.assertEqual(row["majority_f1"], 0.0)
        self.assertAlmostEqual(row["majority_accuracy"], 0.8)
        self.assertAlmostEqual(row["stratified_f1"], 0.2)

    def test_binary_majority_positive_scores_its_f1(self):
        task = TASKS["ballard_incivility"]
        gold = pd.DataFrame({"item_id": range(4), "gt_uncivil": [1, 1, 1, 0]})
        row = cb.baselines(task, gold)
        self.assertAlmostEqual(row["majority_f1"], 2 * 0.75 / 1.75)

    def test_categorical_uses_only_labels_present_in_gold(self):
        task = TASKS["gilardi_stance"]
        gold = pd.DataFrame({"item_id": range(4), "gt_stance": ["neutral"] * 3 + ["pro"]})
        row = cb.baselines(task, gold)
        # Two scored classes: neutral F1 = 2*0.75/1.75, pro F1 = 0.
        self.assertAlmostEqual(row["majority_f1"], (2 * 0.75 / 1.75) / 2)
        self.assertAlmostEqual(row["stratified_f1"], (0.75 + 0.25) / 2)

    def test_tracked_table_covers_the_release_tasks(self):
        table = pd.read_csv(REPO / "output/chance_baselines.csv")
        summary = pd.read_csv(REPO / "output/summary.csv")
        self.assertEqual(set(table["task"]), set(summary["task"]))
        self.assertAlmostEqual(table["majority_f1"].mean(), 0.154, places=3)
        self.assertAlmostEqual(table["stratified_f1"].mean(), 0.253, places=3)


class SchemaParityTests(unittest.TestCase):
    def test_tracked_predictions_reproduce_the_reported_difference(self):
        predictions = pd.read_csv(sp.DIR / "predictions.csv.gz", low_memory=False)
        cells = sp.score_cells(predictions, TASKS)
        self.assertEqual(len(cells), 2 * 4 * 34)
        self.assertEqual(cells.groupby("arm")["n"].sum().to_dict(), {"guided": 65700, "plain": 65700})
        result = sp.paired_difference(cells)
        self.assertEqual(result["cells"], 136)
        self.assertAlmostEqual(result["mean_diff"], -0.0014, places=4)
        self.assertEqual(result["guided_better"], 53)
        self.assertLess(result["ci_low"], 0)
        self.assertLessEqual(result["ci_high"], 0.0001)


if __name__ == "__main__":
    unittest.main()
