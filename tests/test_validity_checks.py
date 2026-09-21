import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from build_validity_checks import DEFAULT_ROOT, build_checks, render  # noqa: E402


class ValidityCheckTests(unittest.TestCase):
    def test_frozen_checks_and_rosters(self):
        task, leakage, size = build_checks(DEFAULT_ROOT)
        self.assertEqual(task.task_count.tolist(), [18, 16, 4])
        self.assertAlmostEqual(task.delta_f1_points.iloc[0], 0.5397735258)
        self.assertAlmostEqual(task.delta_f1_points.iloc[1], 0.2934559848)
        self.assertLess(task.delta_f1_points.iloc[2], 0)
        self.assertEqual(set(leakage.task_count), {8, 10})
        self.assertEqual(len(size), 7)
        self.assertEqual(size.groupby("family_series").size().to_dict(),
                         {"Llama dense 70B": 3, "Qwen dense 27–32B": 4})

    def test_outputs_are_nontrivial(self):
        task, _, size = build_checks(DEFAULT_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "figure"
            render(task, size, base)
            self.assertGreater(base.with_suffix(".png").stat().st_size, 20_000)
            self.assertGreater(base.with_suffix(".pdf").stat().st_size, 5_000)


if __name__ == "__main__":
    unittest.main()
