import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from analyze_regression_mechanisms import build  # noqa: E402


class RegressionMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        build(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mechanical_outputs_have_complete_coverage(self):
        confusion = pd.read_csv(self.root / "confusion_decomposition.csv")
        prompt = pd.read_csv(self.root / "prompt_sensitivity.csv")
        trajectories = pd.read_csv(self.root / "task_trajectories.csv")
        timing = pd.read_csv(self.root / "source_timing.csv")
        self.assertEqual(len(confusion), 34)
        self.assertEqual(confusion["task"].nunique(), 34)
        self.assertEqual(len(prompt), 18)
        self.assertEqual(len(trajectories), 18)
        self.assertEqual(len(timing), 34)
        self.assertEqual(set(timing["publication_group"]), {"2025–2026", "through 2024"})

    def test_blinded_audit_is_balanced_and_reproducible(self):
        audit = pd.read_csv(self.root / "blinded_disagreement_audit.csv")
        key = json.loads((self.root / "blinding_key.json").read_text())
        self.assertEqual(len(audit), 120)
        self.assertEqual(audit["task"].nunique(), 6)
        self.assertTrue((audit.groupby("task").size() == 20).all())
        self.assertNotIn("llama", " ".join(audit.columns).lower())
        self.assertNotIn("qwen", " ".join(audit.columns).lower())
        self.assertEqual(key["seed"], 20260820)

    def test_key_threshold_and_prompt_results(self):
        confusion = pd.read_csv(self.root / "confusion_decomposition.csv").set_index("task")
        ballard = confusion.loc["ballard_incivility"]
        self.assertLess(ballard["qwen_positive_rate"], ballard["llama_positive_rate"])
        self.assertLess(ballard["qwen_recall"], ballard["llama_recall"])
        self.assertGreater(ballard["qwen_only_correct"], ballard["llama_only_correct"])
        prompt = pd.read_csv(self.root / "prompt_sensitivity.csv")
        self.assertAlmostEqual(prompt["single_user_minus_baseline"].mean(), 0.00772835, places=6)


if __name__ == "__main__":
    unittest.main()
