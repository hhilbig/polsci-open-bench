"""Tests for the supervised learning-curve baseline.

The design depends on one thing being exactly true: the classifier must train on
rows the benchmark did not sample and be tested on the rows it did, with text and
gold reconstructed byte-identically to what the LLM runner produced. If that
breaks, the supervised and LLM columns stop being comparable and every number in
the section is wrong in a way no test of the model itself would catch. Most of
what follows checks that reconstruction rather than the classifier.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import build_supervised_baseline as sb  # noqa: E402
import task_registry  # noqa: E402


ACTIVE = {t["name"]: t for t in task_registry.load_task_definitions(active_only=True)}
NO_REMAINDER = {
    "brandt_political_relevance",
    "plover_cameo_event",
    "burnham_covid_threat_minimization",
}


class SplitReconstructionTests(unittest.TestCase):
    def test_every_task_reconstructs_the_benchmark_sample_exactly(self):
        for name, task in ACTIVE.items():
            with self.subTest(task=name):
                frame, texts, test_idx, pool_idx, gt_spec = sb.task_split(task)
                items = task["loader"]()

                self.assertEqual(len(test_idx), len(items))
                self.assertEqual(set(test_idx) & set(pool_idx), set())
                self.assertEqual(len(test_idx) + len(pool_idx), len(frame))

                # Text parity on a sample of rows: this is the property that makes
                # the comparison fair.
                for i in range(min(20, len(items))):
                    self.assertEqual(texts[test_idx[i]], items[i]["user_content"])

    def test_gold_labels_match_the_loader(self):
        for name, task in ACTIVE.items():
            with self.subTest(task=name):
                frame, _, test_idx, _, gt_spec = sb.task_split(task)
                gold = sb.gold_matrix(task, frame, gt_spec, test_idx)
                items = task["loader"]()
                kind = task["label_kind"]
                for i in range(min(20, len(items))):
                    if kind == "multi_binary":
                        expected = np.array([items[i]["gt"][l] for l in task["labels"]])
                        np.testing.assert_array_equal(gold[i], expected)
                    elif kind == "binary":
                        self.assertEqual(int(gold[i]), items[i]["gt"][task["label_key"]])
                    else:
                        self.assertEqual(str(gold[i]), str(items[i]["gt"][task["label_key"]]))

    def test_tasks_that_consumed_their_frame_have_no_training_pool(self):
        """These three are excluded from the curve; cross-validating them instead
        would be a different estimator and must not be mixed in."""
        for name in NO_REMAINDER:
            with self.subTest(task=name):
                _, _, _, pool_idx, _ = sb.task_split(ACTIVE[name])
                self.assertEqual(len(pool_idx), 0)

    def test_v2_label_map_is_applied_before_indices_are_drawn(self):
        """A task whose manifest renames gold values must still line up with the
        loader, or the recovered test set silently diverges from the LLM's."""
        v2 = {t["name"]: t for t in
              task_registry.load_task_definitions(tasks_dir=REPO / "tasks_v2")}
        task = v2["cap_crs_policy_topic"]
        frame, _, test_idx, _, gt_spec = sb.task_split(task)
        gold = sb.gold_matrix(task, frame, gt_spec, test_idx)
        self.assertNotIn("Labor and Immigration", set(gold))
        self.assertIn("Labor", set(gold))


class ScoringFrameTests(unittest.TestCase):
    def test_binary_frame_shape(self):
        task = {"label_kind": "binary", "label_key": "relevant", "labels": ["relevant"]}
        f = sb.scoring_frame(task, np.array([1, 0]), np.array([1, 1]))
        self.assertEqual(list(f.columns), ["gt_relevant", "pred_relevant"])

    def test_multi_binary_frame_has_a_column_pair_per_label(self):
        task = {"label_kind": "multi_binary", "label_key": None, "labels": ["A", "B"]}
        gold = np.array([[1, 0], [0, 1]])
        f = sb.scoring_frame(task, gold, gold)
        self.assertEqual(set(f.columns), {"gt_A", "pred_A", "gt_B", "pred_B"})


class DrawTests(unittest.TestCase):
    def test_draw_keeps_one_row_per_observed_class(self):
        """Without this a small draw can miss a class entirely and the fit
        silently cannot predict it."""
        rng = np.random.default_rng(0)
        gold = np.array(["a"] * 95 + ["b"] * 4 + ["c"])
        sel = sb.draw_indices(np.arange(100), gold, "categorical", 10, rng)
        self.assertEqual(len(sel), 10)
        self.assertEqual(set(gold[sel]), {"a", "b", "c"})

    def test_draw_falls_back_when_classes_outnumber_the_budget(self):
        rng = np.random.default_rng(0)
        gold = np.array([str(i) for i in range(40)])
        sel = sb.draw_indices(np.arange(40), gold, "categorical", 10, rng)
        self.assertEqual(len(sel), 10)


class FitTests(unittest.TestCase):
    def test_class_weighting_prevents_all_negative_prediction(self):
        """Several tasks sit near a 6 percent positive rate. An unweighted fit
        predicts all-negative there and scores zero, which would look like a
        finding rather than a bug."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (190, 8)), rng.normal(2.5, 1, (10, 8))])
        y = np.array([0] * 190 + [1] * 10)
        pred = sb.fit_predict("binary", X, y, X)
        self.assertGreater(pred.sum(), 0)

    def test_multi_binary_uses_one_vs_rest_and_returns_a_matrix(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (60, 5))
        Y = (rng.random((60, 3)) > 0.5).astype(int)
        pred = sb.fit_predict("multi_binary", X, Y, X)
        self.assertEqual(pred.shape, (60, 3))


class CurveTests(unittest.TestCase):
    def test_size_beyond_the_remainder_yields_nan_not_a_smaller_sample(self):
        """Silently shrinking the draw would make the x-axis a lie."""
        task = ACTIVE["gilardi_stance"]          # 288 rows of remainder
        rows = sb.run_task(task, "tfidf")
        df = pd.DataFrame(rows)
        too_big = df[df.n_train == 2000]
        self.assertTrue(len(too_big) >= 1)
        self.assertTrue(too_big.headline_f1.isna().all())

    def test_small_task_curve_is_scored_and_improves_with_more_labels(self):
        task = ACTIVE["ornstein_scotus_sentiment"]
        df = pd.DataFrame(sb.run_task(task, "tfidf"))
        scored = df[df.headline_f1.notna()]
        self.assertGreater(len(scored), 0)
        by_n = scored.groupby("n_train").headline_f1.mean()
        self.assertGreater(by_n.loc[250], by_n.loc[50])


if __name__ == "__main__":
    unittest.main()
