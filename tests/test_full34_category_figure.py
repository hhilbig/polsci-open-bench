import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from build_full34_category_figure import CATEGORIES, category_frame, render  # noqa: E402


class Full34CategoryFigureTests(unittest.TestCase):
    def test_exact_partition_and_real_evidence(self):
        self.assertEqual(len(set().union(*CATEGORIES.values())), 34)
        self.assertEqual(sum(map(len, CATEGORIES.values())), 34)
        detail, summary = category_frame(
            REPO / "output/sidecar/frontier_2026/broad18/holdout16/llama3_1_70b_instruct_fp8_dynamic_full34/predictions.csv",
            REPO / "output/sidecar/hive_model_bakeoff_extension_cuda130_native_cutlass_20260805/qwen3_6_27b_fp8/predictions.csv",
            REPO / "tasks",
        )
        self.assertEqual(len(detail), 34); self.assertEqual(len(summary), 5)
        self.assertEqual(int(summary.task_count.sum()), 34)
        self.assertFalse(summary[["llama3_1_f1", "qwen3_6_f1"]].isna().any().any())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "figure"; render(summary, base)
            for suffix in (".png", ".pdf"):
                self.assertGreater(base.with_suffix(suffix).stat().st_size, 5000)

if __name__ == "__main__": unittest.main()
