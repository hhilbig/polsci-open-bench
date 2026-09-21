import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from build_architecture_adjusted_plots import (  # noqa: E402
    architecture_data,
    compute_staircase_plot,
    lineage_plot,
)


class ArchitectureAdjustedPlotTests(unittest.TestCase):
    def test_frozen_groups_and_plots(self):
        source = pd.read_csv(
            ROOT / "output" / "sidecar" / "frontier_2026" / "broad18" / "checkpoint_summary.csv"
        )
        data = architecture_data(source)
        self.assertEqual(len(data), 21)
        self.assertEqual(data.architecture.value_counts().to_dict(), {"dense": 15, "moe": 6})
        self.assertEqual(data.compute_class.value_counts().to_dict(),
                         {"Large (>40B active)": 7, "Medium (10–40B active)": 8,
                          "Low (≤10B active)": 6})
        self.assertTrue(data.parameter_source.str.startswith("https://").all())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            lineages = lineage_plot(data, out / "lineages")
            mapped = compute_staircase_plot(data, out / "staircase")
            self.assertEqual(lineages.lineage.value_counts().to_dict(),
                             {"Qwen dense 27–32B": 4, "Qwen dense 72B": 3,
                              "Llama dense 70B": 3})
            self.assertEqual(len(mapped), 21)
            self.assertTrue(mapped.groupby("compute_class").class_frontier_f1.apply(
                lambda values: values.is_monotonic_increasing
            ).all())
            self.assertEqual(int(mapped.class_frontier_setter.sum()), 9)
            for stem in ("lineages", "staircase"):
                self.assertGreater((out / f"{stem}.png").stat().st_size, 20_000)
                self.assertGreater((out / f"{stem}.pdf").stat().st_size, 5_000)
                self.assertEqual(mpimg.imread(out / f"{stem}.png").shape[:2], (2160, 3420))


if __name__ == "__main__":
    unittest.main()
