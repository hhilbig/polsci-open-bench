import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from plot_broad18_holdout import holdout_data, render, task_deltas  # noqa: E402


class Broad18HoldoutFigureTests(unittest.TestCase):
    def test_holdout_coverage_and_headline_contrasts(self):
        means, metrics, _ = holdout_data()
        self.assertEqual(metrics["task"].nunique(), 16)
        self.assertEqual(means["model"].nunique(), 7)
        self.assertTrue((metrics.groupby("model")["task"].nunique() == 16).all())

        deltas, aggregate = task_deltas()
        self.assertEqual(len(deltas), 34)
        self.assertEqual(set(deltas["set"]), {"Broad18", "Held-out 16"})
        values = aggregate.set_index("set")
        self.assertAlmostEqual(values.loc["Broad18", "mean_delta_f1"], 0.0053977353)
        self.assertAlmostEqual(values.loc["Held-out 16", "mean_delta_f1"], 0.0029345598)
        self.assertLess(values.loc["Held-out 16", "ci_low"], 0)
        self.assertGreater(values.loc["Held-out 16", "ci_high"], 0)

    def test_rendered_outputs_are_nontrivial_and_csvs_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            png, pdf = render(output)
            self.assertGreater(png.stat().st_size, 100_000)
            self.assertGreater(pdf.stat().st_size, 10_000)
            deltas = pd.read_csv(output / "holdout_task_deltas.csv")
            means = pd.read_csv(output / "holdout_checkpoint_means.csv")
            self.assertEqual(len(deltas), 34)
            self.assertEqual(set(means["task_set"]), {"Broad18", "Held-out 16"})


if __name__ == "__main__":
    unittest.main()
