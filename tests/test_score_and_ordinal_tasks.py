"""Tests for the `score` task type and the `ordinal: true` categorical flag.

`score` tasks ask for a number in a fixed range and are scored by correlation
and mean absolute error with a numeric gold value (first used for the Le Mens
and Gallego 2025 tweet ratings). `ordinal: true` adds weighted kappa and rank
correlation to categorical tasks whose labels are listed in scale order.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import batch_benchmark  # noqa: E402
import benchmark  # noqa: E402
import build_summary  # noqa: E402
import scoring  # noqa: E402
import task_registry  # noqa: E402


SCORE_TASK = {
    "label_kind": "score",
    "label_key": "score",
    "labels": ["score"],
    "score_range": [0.0, 100.0],
}

ORDINAL_TASK = {
    "label_kind": "categorical",
    "label_key": "position",
    "labels": ["left", "center", "right"],
    "ordinal": True,
}


class ScoreParsingTest(unittest.TestCase):
    def test_json_number(self):
        self.assertEqual(benchmark.parse_content('{"score": 72}', SCORE_TASK), ({"score": 72.0}, None))

    def test_numeric_string_and_fence(self):
        pred, err = benchmark.parse_content('```json\n{"score": "35.5"}\n```', SCORE_TASK)
        self.assertEqual((pred, err), ({"score": 35.5}, None))

    def test_bare_number_fallback(self):
        self.assertEqual(benchmark.parse_content("64", SCORE_TASK), ({"score": 64.0}, None))

    def test_not_applicable_is_flagged_not_failed(self):
        for content in ['{"score": "NA"}', '{"score": null}', "NA"]:
            pred, err = benchmark.parse_content(content, SCORE_TASK)
            self.assertEqual(pred, {"score": None})
            self.assertEqual(err, "not_applicable", content)

    def test_out_of_range_and_garbage_fail(self):
        for content in ['{"score": 140}', '{"score": -3}', "left wing", '{"score": true}']:
            pred, err = benchmark.parse_content(content, SCORE_TASK)
            self.assertEqual(pred, {"score": None})
            self.assertTrue(err.startswith("parse_fail"), (content, err))

    def test_batch_parser_object(self):
        self.assertEqual(batch_benchmark._pred_from_obj({"score": 10}, SCORE_TASK), ({"score": 10.0}, None))
        self.assertEqual(batch_benchmark._pred_from_obj({"score": "NA"}, SCORE_TASK), ({"score": None}, "not_applicable"))

    def test_example_payload_is_in_range(self):
        self.assertEqual(benchmark.example_payload_for_task(SCORE_TASK), {"score": 50})


class ScoreMetricsTest(unittest.TestCase):
    def frame(self, preds):
        gt = [10.0, 30.0, 50.0, 70.0, 90.0]
        return pd.DataFrame({"gt_score": gt, "pred_score": preds})

    def test_perfect_prediction(self):
        m = scoring.score_metrics(SCORE_TASK, self.frame([10, 30, 50, 70, 90]))
        self.assertAlmostEqual(m["pearson_r"], 1.0)
        self.assertAlmostEqual(m["spearman_rho"], 1.0)
        self.assertEqual(m["mae"], 0.0)
        self.assertEqual(m["n_scored"], 5)

    def test_missing_predictions_are_dropped(self):
        m = scoring.score_metrics(SCORE_TASK, self.frame([20, None, 50, None, 80]))
        self.assertEqual(m["n_scored"], 3)
        self.assertAlmostEqual(m["mae"], (10 + 0 + 10) / 3)

    def test_constant_prediction_has_no_correlation(self):
        m = scoring.score_metrics(SCORE_TASK, self.frame([50] * 5))
        self.assertTrue(np.isnan(m["pearson_r"]))
        self.assertEqual(m["mae"], 24.0)

    def test_no_f1_for_score_tasks(self):
        self.assertEqual(scoring.scored_labels(SCORE_TASK, self.frame([1] * 5)), [])
        self.assertTrue(np.isnan(scoring.headline_f1(SCORE_TASK, self.frame([1] * 5))))

    def test_not_applicable_scored_at_midpoint(self):
        g = self.frame([10, None, 50, 70, None])
        g["parse_error"] = ["not_applicable", "not_applicable", None, None, "malformed"]
        g.loc[0, "pred_score"] = None
        m = scoring.score_metrics(SCORE_TASK, g[g.parse_error.ne("malformed")])
        # Rows 0 and 1 become 50; the malformed row is dropped.
        self.assertEqual(m["n_scored"], 4)
        self.assertAlmostEqual(m["mae"], (40 + 20 + 0 + 0) / 4)
        self.assertAlmostEqual(m["mae_answered"], 0.0)

    def test_summary_row(self):
        g = self.frame([12, 28, None, 75, 85])
        g["parse_error"] = [None, None, "not_applicable", None, None]
        g["latency_s"] = 1.0
        row = build_summary._metrics_for_group(SCORE_TASK, g)
        self.assertTrue(np.isnan(row["headline_f1"]))
        self.assertEqual(row["not_applicable_rate"], 0.2)
        self.assertEqual(row["n_scored"], 5)
        self.assertAlmostEqual(row["mae"], (2 + 2 + 0 + 5 + 5) / 5)
        self.assertGreater(row["pearson_r_answered"], 0.99)


class OrdinalMetricsTest(unittest.TestCase):
    def test_near_miss_beats_far_miss(self):
        gt = ["left", "left", "center", "right", "right"]
        near = pd.DataFrame({"gt_position": gt, "pred_position": ["center", "left", "center", "right", "center"]})
        far = pd.DataFrame({"gt_position": gt, "pred_position": ["right", "left", "center", "right", "left"]})
        m_near = scoring.ordinal_metrics(ORDINAL_TASK, near)
        m_far = scoring.ordinal_metrics(ORDINAL_TASK, far)
        # Both have 3 of 5 correct, so accuracy cannot separate them.
        self.assertGreater(m_near["weighted_kappa"], m_far["weighted_kappa"])
        self.assertLess(m_near["mae_steps"], m_far["mae_steps"])

    def test_summary_adds_ordinal_columns_only_when_flagged(self):
        g = pd.DataFrame({
            "gt_position": ["left", "center", "right"],
            "pred_position": ["left", "center", "right"],
            "parse_error": [None] * 3,
            "latency_s": [1.0] * 3,
        })
        flagged = build_summary._metrics_for_group(ORDINAL_TASK, g)
        self.assertAlmostEqual(flagged["weighted_kappa"], 1.0)
        plain = build_summary._metrics_for_group({**ORDINAL_TASK, "ordinal": False}, g)
        self.assertNotIn("weighted_kappa", plain)


class ManifestTest(unittest.TestCase):
    def test_extension_manifests_load(self):
        for name, kind, ordinal in [
            ("lemens_gallego_tweet_position", "score", False),
            ("lemens_gallego_tweet_position3", "categorical", True),
            ("benoit_manifesto_economic_position", "categorical", True),
            ("benoit_manifesto_social_position", "categorical", True),
        ]:
            task = task_registry.load_task_definition(REPO / "tasks_ext" / f"{name}.yaml")
            self.assertEqual((task["label_kind"], task["ordinal"]), (kind, ordinal), name)
            items = task["loader"]()
            self.assertEqual(len(items), len({i["item_id"] for i in items}), name)

    def test_score_gold_is_numeric_and_in_range(self):
        task = task_registry.load_task_definition(REPO / "tasks_ext" / "lemens_gallego_tweet_position.yaml")
        golds = [i["gt"]["score"] for i in task["loader"]()]
        self.assertTrue(all(isinstance(v, float) and 0 <= v <= 100 for v in golds))

    def test_score_task_requires_range(self):
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(yaml.safe_dump({
                "name": "bad", "label_kind": "score", "label_key": "score",
                "text": {"template": "{text}"}, "ground_truth": {"column": "y"},
            }))
            with self.assertRaises(ValueError):
                task_registry.load_task_definition(path)


if __name__ == "__main__":
    unittest.main()
