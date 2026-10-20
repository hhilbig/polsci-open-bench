"""Skip tests whose inputs are local run outputs that are not in the public repository.

Some tests check figures and tables built from per-request model outputs under
output/sidecar/. Those outputs are too large to track, so on a fresh clone
these tests are skipped with the missing path named, rather than failing.
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PANEL = "output/sidecar/refresh_20260910/panel.json"
BROAD18_SUMMARY = "output/sidecar/frontier_2026/broad18/checkpoint_summary.csv"
BAKEOFF_METRICS = "output/sidecar/hive_model_bakeoff_20260804/comparison/task_metrics.csv"
HOLDOUT_FULL34 = "output/sidecar/frontier_2026/broad18/holdout16/llama3_1_70b_instruct_fp8_dynamic_full34/predictions.csv"


def _missing(paths):
    return [p for p in paths if not (REPO / p).exists()]


def needs_local_outputs(*paths):
    """Decorator for a test function, method or class."""
    missing = _missing(paths)
    return unittest.skipIf(
        bool(missing),
        "needs local run outputs that are not in the public repository: " + ", ".join(missing),
    )


def skip_module_without(*paths):
    """Call at module level, before code that loads the missing inputs."""
    missing = _missing(paths)
    if missing:
        raise unittest.SkipTest(
            "needs local run outputs that are not in the public repository: " + ", ".join(missing)
        )
