from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from analyze_human_vote_baseline import build  # noqa: E402
from build_human_vote_tasks import build_alia, build_semeval  # noqa: E402


class HumanVoteBaselineTests(unittest.TestCase):
    def test_source_builders_preserve_votes(self):
        source_root = Path("/tmp/polsci-human-votes")
        if not source_root.exists():
            self.skipTest("separately downloaded source archives are unavailable")
        semeval = build_semeval(
            REPO / "data" / "semeval_stance.csv",
            source_root / "semeval-raw" / "SemEval2016-Task6-raw-annotations-stance.csv",
        )
        alia = build_alia(source_root / "alia.jsonl")
        self.assertEqual(semeval["n_votes"].notna().sum(), 1690)
        self.assertEqual(len(alia), 2850)
        self.assertTrue((alia["n_votes"] == 3).all())

    def test_frozen_sample_and_model_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root)
            items = pd.read_csv(root / "semeval_item_human_agreement.csv")
            summary = pd.read_csv(root / "semeval_human_model_summary.csv")
            alia = pd.read_csv(root / "alia_human_baseline.csv")
            alia_models = pd.read_csv(root / "alia_human_model_summary.csv")
            gaps = pd.read_csv(root / "human_agreement_accuracy_gaps.csv")
            self.assertEqual(len(items), 500)
            self.assertFalse(items[["llama_prediction", "qwen_prediction"]].isna().any().any())
            self.assertEqual(set(summary["model"]), {"llama", "qwen", "individual human vote"})
            self.assertEqual(int(alia.loc[0, "n_items"]), 500)
            self.assertEqual(int(alia.loc[0, "unanimous_items"] + alia.loc[0, "two_of_three_items"]), 500)
            self.assertEqual(set(alia_models["model"]), {"llama", "qwen"})
            alia_all = alia_models.loc[alia_models["group"] == "all"].set_index("model")
            self.assertAlmostEqual(alia_all.loc["llama", "accuracy_vs_human_majority"], 0.794, places=6)
            self.assertAlmostEqual(alia_all.loc["qwen", "accuracy_vs_human_majority"], 0.790, places=6)
            self.assertIn("SemEval stance", set(gaps["dataset"]))
            semeval_all = gaps.loc[(gaps["dataset"] == "SemEval stance") & (gaps["agreement_group"] == "all")].iloc[0]
            self.assertAlmostEqual(semeval_all["llama_minus_qwen_accuracy"], 0.038, places=6)


if __name__ == "__main__":
    unittest.main()
