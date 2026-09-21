import sys, unittest
from pathlib import Path
import matplotlib.image as mpimg
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
from analyze_event_progress import analyze, external_confirmation, longitudinal


class EventProgressTests(unittest.TestCase):
    def test_event_pattern_replicates_but_omnibus_does_not(self):
        sensitivity, omnibus = analyze(draws=10000)
        event = sensitivity[sensitivity.category.eq("Events and protest")].set_index("set")
        self.assertEqual(event.loc["Broad18", "tasks_improved"], 3)
        self.assertEqual(event.loc["Held-out 16", "tasks_improved"], 4)
        self.assertGreater(event.ci_low.min(), 0)
        self.assertGreater(event.loo_min.min(), 0)
        self.assertAlmostEqual(omnibus.estimate.iloc[0], .1330697811, places=9)
        self.assertGreater(omnibus.permutation_p.iloc[0], .25)

    def test_longitudinal_source_and_model_sensitivity(self):
        tasks, trends = longitudinal(draws=10000)
        self.assertEqual(len(tasks), 18)
        self.assertEqual(tasks.source_cluster_id.nunique(), 13)
        self.assertEqual(int(tasks.is_event.sum()), 3)
        overall = trends[trends.scope.eq("all_21_source_equal")].iloc[0]
        self.assertEqual(overall.models, 21)
        self.assertGreater(overall.event_minus_other_points_per_year, 0)
        self.assertGreater(overall.ci_low, 0)
        self.assertGreater(overall.ci_high, 0)
        self.assertEqual(sum(trends.scope.str.startswith("compute:")), 3)
        self.assertEqual(sum(trends.scope.str.startswith("lineage:")), 3)

    def test_external_confirmation_is_complete_and_disconfirming(self):
        detail, summary = external_confirmation(draws=1000)
        self.assertEqual(len(detail),4); self.assertEqual(detail["items"].sum(),2000)
        self.assertEqual(detail.source_family.nunique(),3)
        self.assertEqual(int((detail.delta_f1_points>0).sum()),1)
        self.assertLess(summary.set_index("aggregation").loc["equal_source_family","delta_f1_points"],0)
        base=ROOT/"output/sidecar/frontier_2026/event_progress/external_event_confirmation"
        self.assertGreater(base.with_suffix(".png").stat().st_size,20_000)
        self.assertGreater(base.with_suffix(".pdf").stat().st_size,5_000)
        self.assertEqual(mpimg.imread(base.with_suffix(".png")).shape[:2],(2160,3420))


if __name__ == "__main__": unittest.main()
