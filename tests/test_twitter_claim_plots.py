import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from build_twitter_claim_plots import (  # noqa: E402
    category_plot,
    external_task_deltas,
    progress_plot,
    task_set_plot,
)


class TwitterClaimPlotTests(unittest.TestCase):
    def test_all_three_claim_plots(self):
        base = ROOT / "output" / "sidecar" / "frontier_2026"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            progress = progress_plot(pd.read_csv(base / "broad18" / "checkpoint_summary.csv"), out / "one")
            categories = category_plot(pd.read_csv(base / "twitter_figures" / "full34_category_summary.csv"), out / "two")
            external = external_task_deltas(base)
            tasks = task_set_plot(
                pd.read_csv(base / "validity_checks" / "task_set_sensitivity.csv"),
                external,
                out / "three",
            )
            self.assertEqual(len(progress), 21)
            self.assertEqual(progress.architecture_group.value_counts().to_dict(),
                             {"dense": 15, "sparse_or_moe": 6})
            self.assertAlmostEqual(progress.frontier_f1.iloc[-1] - progress.loc[progress.checkpoint_id.eq("llama3_70b_instruct_fp8_hive"), "mean_f1"].iloc[0], .0049636198)
            self.assertEqual(int(categories.task_count.sum()), 34)
            self.assertEqual(len(external), 4)
            self.assertAlmostEqual(external.delta_f1_points.mean(), -2.5679363048)
            self.assertEqual(tasks.set_type.value_counts().to_dict(),
                             {"external": 4, "repository": 2})
            for stem in ("one", "two", "three"):
                self.assertGreater((out / f"{stem}.png").stat().st_size, 20_000)
                self.assertGreater((out / f"{stem}.pdf").stat().st_size, 5_000)
            for stem in ("two", "three"):
                self.assertEqual(mpimg.imread(out / f"{stem}.png").shape[:2], (2160, 3420))


if __name__ == "__main__":
    unittest.main()
