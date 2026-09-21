import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from task_registry import load_task_definitions  # noqa: E402


class ExternalHumanVoteTaskTests(unittest.TestCase):
    def test_frozen_tasks_have_exact_coverage(self):
        tasks = load_task_definitions(tasks_dir=ROOT / "experiments" / "external_human_votes_tasks")
        self.assertEqual(len(tasks), 3)
        total = 0
        ids = set()
        for task in tasks:
            items = task["loader"]()
            frame = pd.DataFrame(items)
            self.assertEqual(len(frame), 1000)
            item_ids = set(frame["item_id"].astype(str))
            self.assertEqual(len(item_ids), 1000)
            self.assertTrue(ids.isdisjoint(item_ids))
            ids |= item_ids; total += len(frame)
            source = pd.read_csv(task["data_path"])
            self.assertTrue(source["vote_share_majority"].between(0.5, 1).all())
        self.assertEqual(total, 3000)

    def test_build_outputs_are_deterministic(self):
        original = {p.name: pd.read_csv(p) for p in (ROOT / "data" / "human_votes" / "external").glob("*.csv")}
        self.assertEqual(set(original), {"hatexplain_votes.csv", "measuring_hate_speech_votes.csv", "mfrc_votes.csv"})
        for frame in original.values():
            self.assertEqual(len(frame), 1000)
            self.assertFalse(frame["source_id"].duplicated().any())


if __name__ == "__main__": unittest.main()
