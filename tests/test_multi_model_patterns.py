import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analyze_multi_model_patterns import (  # noqa: E402
    binary_metrics,
    bootstrap_weight_matrices,
    build_analysis,
    categorical_metrics,
    multi_binary_metrics,
    task_score_draw_pair,
    task_score_draws,
)
from build_compact_frontier import bootstrap_indices  # noqa: E402
from panel_manifest import load_panel_manifest  # noqa: E402


class MetricCalculationTests(unittest.TestCase):
    def test_binary_absent_positive_predictions(self):
        result = binary_metrics([1, 0, 0], [0, 0, 0], valid=[True, True, True])
        self.assertEqual(result["predicted_positive_rate"], 0)
        self.assertAlmostEqual(result["gold_positive_rate"], 1 / 3)
        self.assertAlmostEqual(result["positive_rate_bias"], -1 / 3)
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["positive_recall"], 0)
        self.assertEqual(result["negative_recall"], 1)
        self.assertEqual(result["positive_f1"], 0)
        self.assertAlmostEqual(result["negative_f1"], .8)
        self.assertAlmostEqual(result["symmetric_macro_f1"], .4)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_binary_malformed_outputs_are_wrong_not_negative(self):
        result = binary_metrics([1, 0], [np.nan, np.nan], valid=[False, False])
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["positive_recall"], 0)
        self.assertEqual(result["negative_recall"], 0)
        self.assertEqual(result["positive_f1"], 0)
        self.assertEqual(result["negative_f1"], 0)
        self.assertEqual(result["symmetric_macro_f1"], 0)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["valid_prediction_rate"], 0)
        self.assertTrue(np.isnan(result["predicted_positive_rate"]))
        self.assertTrue(np.isnan(result["gold_positive_rate"]))
        self.assertTrue(np.isnan(result["positive_rate_bias"]))

    def test_categorical_absent_classes_and_malformed_output(self):
        result = categorical_metrics(
            ["a", "b", "c", "c"],
            ["a", "a", "a", "invalid"],
            ["a", "b", "c"],
            valid=[True, True, True, False],
        )
        self.assertAlmostEqual(result["accuracy"], .25)
        self.assertAlmostEqual(result["symmetric_macro_f1"], 1 / 6)
        self.assertAlmostEqual(result["label_coverage"], 1 / 3)
        self.assertEqual(result["prediction_entropy"], 0)
        self.assertAlmostEqual(result["rare_class_recall"], .5)
        self.assertAlmostEqual(result["valid_prediction_rate"], .75)

    def test_multi_binary_row_failure_is_wrong_for_every_atomic_decision(self):
        frame = pd.DataFrame(
            {
                "gt_x": [1, 0],
                "pred_x": [1, 0],
                "gt_y": [0, 1],
                "pred_y": [0, 1],
            }
        )
        result = multi_binary_metrics(
            frame, {"labels": ["x", "y"]}, np.array([False, True])
        )
        self.assertAlmostEqual(result["positive_f1"], 1 / 3)
        self.assertAlmostEqual(result["symmetric_macro_f1"], 1 / 3)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["valid_prediction_rate"], .5)

    def test_shared_bootstrap_and_rescoring_are_deterministic(self):
        benchmark = load_panel_manifest(ROOT / "experiments" / "frontier_broad_18.yaml")
        left = bootstrap_indices(benchmark, iterations=100, seed=20260820)
        right = bootstrap_indices(benchmark, iterations=100, seed=20260820)
        self.assertEqual(set(left), set(right))
        for task in left:
            np.testing.assert_array_equal(left[task], right[task])

        scored = pd.DataFrame({"gt_flag": [1, 0, 1], "pred_flag": [1, 1, 0]})
        task = {"label_kind": "binary", "label_key": "flag", "labels": ["flag"]}
        indices = np.array([[0, 1, 2], [2, 2, 1], [0, 0, 0]])
        headline_one = task_score_draws(scored, task, indices, symmetric=False)
        headline_two = task_score_draws(scored, task, indices, symmetric=False)
        symmetric_one = task_score_draws(scored, task, indices, symmetric=True)
        symmetric_two = task_score_draws(scored, task, indices, symmetric=True)
        weights = bootstrap_weight_matrices({"toy": indices})["toy"]
        combined = task_score_draw_pair(scored, task, weights)
        np.testing.assert_array_equal(headline_one, headline_two)
        np.testing.assert_array_equal(symmetric_one, symmetric_two)
        np.testing.assert_allclose(combined[:, 0], headline_one, atol=0)
        np.testing.assert_allclose(combined[:, 1], symmetric_one, atol=0)
        self.assertFalse(np.array_equal(headline_one, symmetric_one))


class FrozenPanelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.results = build_analysis(
            output_dir=cls.output,
            iterations=100,
            seed=20260820,
            make_figures=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_audited_panel_and_headline_reproduction(self):
        metrics = self.results["model_task_metrics"]
        self.assertEqual(len(metrics), 378)
        self.assertEqual(metrics["checkpoint_id"].nunique(), 21)
        self.assertEqual(metrics["task"].nunique(), 18)
        self.assertFalse(metrics.duplicated(["checkpoint_id", "task"]).any())
        expected = pd.read_csv(
            ROOT
            / "output"
            / "sidecar"
            / "frontier_2026"
            / "broad18"
            / "model_by_task.csv"
        )[["checkpoint_id", "task", "headline_f1"]]
        checked = metrics.merge(
            expected,
            on=["checkpoint_id", "task"],
            suffixes=("_new", "_existing"),
            validate="one_to_one",
        )
        np.testing.assert_allclose(
            checked["headline_f1_new"], checked["headline_f1_existing"], atol=1e-12
        )
        audit = self.results["pattern_summary"].query("section == 'audit'").set_index(
            "result"
        )
        self.assertEqual(audit.loc["frozen_task_item_keys_per_checkpoint", "estimate"], 3600)
        self.assertEqual(audit.loc["reproduced_existing_headline_f1_values", "estimate"], 378)
        self.assertEqual(audit.loc["malformed_outputs_scored_as_gold_class_misses", "estimate"], 4)
        decision = self.results["pattern_summary"].query(
            "section == 'decision_space' and "
            "result == 'effective_labels_vs_headline_f1_trend' and "
            "scope == 'all_21:13_source_families'"
        ).iloc[0]
        self.assertAlmostEqual(decision["p_value"], 0.0560943905609439)
        self.assertEqual(
            decision["p_value_type"],
            "locked_h1_one_sided_positive_permutation_post_outcome",
        )

    def test_required_metrics_lineages_and_sensitivity_scopes(self):
        metrics = self.results["model_task_metrics"]
        required = {
            "predicted_positive_rate",
            "precision",
            "positive_recall",
            "negative_recall",
            "positive_f1",
            "symmetric_macro_f1",
            "label_coverage",
            "prediction_entropy",
            "accuracy",
            "rare_class_recall",
        }
        self.assertTrue(required.issubset(metrics.columns))
        self.assertEqual(len(self.results["lineage_changes"]), 4)
        self.assertEqual(len(self.results["pairwise_profiles"]), 210)
        trends = self.results["task_trends"]
        expected_scopes = {
            "all_21",
            "Qwen dense 27–32B",
            "Qwen dense 72B",
            "Llama dense 70B",
        }
        self.assertEqual(set(trends["scope"]), expected_scopes)
        headline_counts = trends.loc[trends["metric"].eq("headline_f1")].groupby(
            "scope"
        ).size()
        self.assertTrue((headline_counts == 18).all())

    def test_existing_compute_staircase_is_reused(self):
        summary = self.results["model_summary"]
        self.assertEqual(int(summary["class_frontier_setter"].sum()), 9)
        self.assertTrue(
            summary.groupby("compute_class")["class_frontier_f1"].apply(
                lambda values: values.is_monotonic_increasing
            ).all()
        )
        self.assertEqual(
            set(summary["compute_staircase_source"]),
            {"output/sidecar/frontier_2026/twitter_claims/05_compute_class_staircase.csv"},
        )

    def test_outputs_and_figures_are_nontrivial(self):
        for name in self.results:
            path = self.output / f"{name}.csv"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)
        for stem in ("01_binary_selectivity", "02_task_profiles"):
            self.assertGreater((self.output / f"{stem}.png").stat().st_size, 20_000)
            self.assertGreater((self.output / f"{stem}.pdf").stat().st_size, 5_000)


if __name__ == "__main__":
    unittest.main()
