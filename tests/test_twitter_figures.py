import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from build_twitter_figures import PARAMETERS, disagreement, overall, task_changes  # noqa: E402


class TestTwitterFigures(unittest.TestCase):
    def test_real_outputs_are_complete_and_nontrivial(self):
        broad=ROOT/"output/sidecar/frontier_2026/broad18"
        disagreement_path=ROOT/"output/sidecar/frontier_2026/human_votes/external_analysis/item_disagreement_effects.csv"
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            models=overall(pd.read_csv(broad/"checkpoint_summary.csv"),out/"one")
            tasks=task_changes(pd.read_csv(broad/"model_by_task.csv"),out/"two")
            human=disagreement(pd.read_csv(disagreement_path),out/"three")
            self.assertEqual(len(models), len(PARAMETERS))
            self.assertEqual(len(tasks),5)
            self.assertEqual(set(tasks.group),{"claims","issues","events","position","relevance"})
            self.assertEqual(len(human),12)
            self.assertEqual(tasks.task_count.sum(),18)
            self.assertEqual(tasks.tasks_better.sum(),11)
            self.assertEqual(human.dataset.nunique(),6)
            self.assertEqual(human.agreement_group.nunique(),2)
            for stem in ["one","two","three"]:
                self.assertGreater((out/f"{stem}.png").stat().st_size,10_000)
                self.assertGreater((out/f"{stem}.pdf").stat().st_size,5_000)


if __name__=="__main__": unittest.main()
