from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from analyze_heterogeneity_drivers import build, load_spec  # noqa: E402


class HeterogeneityDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.results = build(output_dir=cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_spec_is_locked_and_inputs_are_verified(self):
        spec = load_spec()
        self.assertEqual(spec["status"], "exploratory_locked_after_outcome_inspection")
        statuses = {row["id"]: row["status"] for row in spec["hypotheses"]}
        self.assertEqual(statuses["H1_decision_space"], "testable")
        self.assertEqual(statuses["H3_gold_determinacy"], "not_testable")
        self.assertEqual(statuses["H4_inferential_depth"], "not_testable")

    def test_longitudinal_tests_have_complete_scope_and_source_inference(self):
        slopes = self.results["task_progress_slopes"]
        tests = self.results["longitudinal_moderator_tests"]
        self.assertEqual(slopes["scope"].nunique(), 4)
        self.assertTrue((slopes.groupby("scope").size() == 18).all())
        self.assertEqual(set(tests["unit"]), {"task", "source_family"})
        source = tests.loc[tests["unit"].eq("source_family")]
        self.assertTrue((source["units"] == 13).all())
        self.assertTrue(source["p_positive_permutation"].between(0, 1).all())

    def test_endpoint_replication_and_calibration_outputs(self):
        endpoint = self.results["endpoint_moderator_tests"].set_index("scope")
        self.assertGreater(endpoint.loc["broad18", "spearman_rho"], 0)
        self.assertGreater(endpoint.loc["held-out_16", "spearman_rho"], 0)
        detail = self.results["calibration_task_metrics"]
        summary = self.results["calibration_summary"]
        summary = summary.loc[summary["scope"].eq("all_34")].set_index("result")
        self.assertEqual(len(detail), 34)
        self.assertEqual(int(detail["binary_output_format"].sum()), 14)
        self.assertGreater(
            summary.loc["nonbinary_headline_mean_delta", "estimate"],
            summary.loc["binary_headline_mean_delta", "estimate"],
        )
        self.assertLess(
            abs(summary.loc["binary_symmetric_mean_delta", "estimate"]),
            abs(summary.loc["binary_headline_mean_delta", "estimate"]),
        )
        self.assertLess(
            summary.loc["binary_tasks_lower_positive_rate", "p_value"], .01
        )
        for name in self.results:
            path = self.output / f"{name}.csv"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)
            self.assertGreater(len(pd.read_csv(path)), 0)


if __name__ == "__main__":
    unittest.main()
