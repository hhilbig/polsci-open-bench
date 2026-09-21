from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_broad_robustness import (  # noqa: E402
    BASELINE_IDS,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CANDIDATE_ID,
    DEFAULT_BROAD_DIR,
    DEFAULT_MANIFEST,
    PRIMARY_BASELINE_ID,
    SOURCE_CLUSTERS,
    _load_manifest,
    build_outputs,
    source_cluster_bootstrap,
    source_cluster_order,
)


class BroadRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temp_dir.name)
        cls.summary, cls.heterogeneity = build_outputs(output_dir=cls.output_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _aggregation(self, baseline: str) -> pd.DataFrame:
        return (
            self.summary.loc[
                (self.summary["record_type"] == "aggregation")
                & (self.summary["baseline_checkpoint_id"] == baseline)
            ]
            .set_index("aggregation_id", verify_integrity=True)
            .sort_index()
        )

    def test_frozen_task_source_and_timing_counts(self):
        _, metadata = _load_manifest(DEFAULT_MANIFEST)
        self.assertEqual(len(metadata), 18)
        self.assertEqual(metadata["task"].nunique(), 18)
        self.assertEqual(metadata["source_cluster_id"].nunique(), 13)
        self.assertEqual(
            metadata["publication_period"].value_counts().to_dict(),
            {"2025_2026": 10, "through_2024": 8},
        )
        self.assertEqual(
            source_cluster_order(metadata["task"].tolist()),
            tuple(sorted(SOURCE_CLUSTERS)),
        )

    def test_four_aggregation_estimates_against_both_baselines(self):
        expected = {
            BASELINE_IDS[0]: {
                "equal_task": 0.005397735257559013,
                "equal_annotation_family": 0.007332662976988691,
                "equal_source_family": 0.012952701768378757,
                "equal_complexity": 0.017963074180872615,
            },
            BASELINE_IDS[1]: {
                "equal_task": 0.01527368084904615,
                "equal_annotation_family": 0.018245596535870427,
                "equal_source_family": 0.026688262983487556,
                "equal_complexity": 0.028426047434676166,
            },
        }
        for baseline, values in expected.items():
            observed = self._aggregation(baseline)
            self.assertEqual(len(observed), 4)
            for aggregation, value in values.items():
                self.assertAlmostEqual(observed.at[aggregation, "delta_f1"], value, places=12)

        primary_points = self._aggregation(BASELINE_IDS[0])["delta_f1_points"]
        secondary_points = self._aggregation(BASELINE_IDS[1])["delta_f1_points"]
        self.assertAlmostEqual(primary_points.min(), 0.5397735257559013, places=10)
        self.assertAlmostEqual(primary_points.max(), 1.7963074180872614, places=10)
        self.assertAlmostEqual(secondary_points.min(), 1.527368084904615, places=10)
        self.assertAlmostEqual(secondary_points.max(), 2.8426047434676165, places=10)

    def test_source_cluster_bootstrap_is_deterministic_and_crosses_zero(self):
        task_rows = self.heterogeneity.loc[
            (self.heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
            & (self.heterogeneity["dimension"] == "source_family")
        ].sort_values("level_id")
        source_order = tuple(task_rows["level_id"])
        deltas = dict(zip(source_order, task_rows["delta_f1"]))
        first = source_cluster_bootstrap(deltas, source_order)
        second = source_cluster_bootstrap(deltas, source_order)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), BOOTSTRAP_ITERATIONS)
        self.assertEqual(BOOTSTRAP_SEED, 20260820)

        expected = {
            BASELINE_IDS[0]: (-0.01723158025849936, 0.04572572892928405, 0.7866),
            BASELINE_IDS[1]: (-0.0011151230314338095, 0.05382475275723, 0.9693),
        }
        for baseline, (low, high, probability) in expected.items():
            row = self._aggregation(baseline).loc["equal_source_family"]
            self.assertAlmostEqual(row["source_cluster_ci_low"], low, places=12)
            self.assertAlmostEqual(row["source_cluster_ci_high"], high, places=12)
            self.assertAlmostEqual(
                row["source_cluster_probability_positive"], probability, places=4
            )
            self.assertLess(row["source_cluster_ci_low"], 0)
            self.assertGreater(row["source_cluster_ci_high"], 0)

    def test_existing_item_and_task_intervals_are_preserved(self):
        row = self._aggregation(PRIMARY_BASELINE_ID).loc["equal_task"]
        self.assertAlmostEqual(row["paired_item_ci_low"], -0.015460, places=6)
        self.assertAlmostEqual(row["paired_item_ci_high"], 0.023139, places=6)
        self.assertAlmostEqual(row["paired_item_probability_positive"], 0.6527)
        self.assertAlmostEqual(row["task_bootstrap_ci_low"], -0.031108, places=6)
        self.assertAlmostEqual(row["task_bootstrap_ci_high"], 0.039605, places=6)
        self.assertAlmostEqual(row["task_bootstrap_probability_positive"], 0.6248)

    def test_task_and_annotation_family_directions(self):
        primary_tasks = self.heterogeneity.loc[
            (self.heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
            & (self.heterogeneity["dimension"] == "task")
        ]
        self.assertEqual(int(primary_tasks["tasks_better"].sum()), 11)
        self.assertEqual(int(primary_tasks["tasks_tied"].sum()), 0)
        self.assertEqual(int(primary_tasks["tasks_worse"].sum()), 7)

        families = self.heterogeneity.loc[
            (self.heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
            & (self.heterogeneity["dimension"] == "annotation_family")
        ].set_index("level_id", verify_integrity=True)
        self.assertEqual(len(families), 5)
        self.assertEqual(int((families["delta_f1"] > 0).sum()), 4)
        self.assertLess(families.at["claims", "delta_f1"], 0)
        for family in ("events", "issues", "position", "relevance"):
            self.assertGreater(families.at[family, "delta_f1"], 0)

    def test_complexity_and_publication_period_heterogeneity(self):
        primary = self.heterogeneity.loc[
            self.heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID
        ]
        complexity = primary.loc[primary["dimension"] == "complexity"].set_index(
            "level_id", verify_integrity=True
        )
        expected_complexity = {
            "low": (-0.01914401590530391, 9),
            "medium": (0.01678522081334405, 6),
            "high": (0.05624801763457771, 3),
        }
        for level, (delta, count) in expected_complexity.items():
            self.assertAlmostEqual(complexity.at[level, "delta_f1"], delta, places=12)
            self.assertEqual(complexity.at[level, "task_count"], count)

        timing = primary.loc[primary["dimension"] == "publication_period"].set_index(
            "level_id", verify_integrity=True
        )
        self.assertEqual(timing.at["2025_2026", "task_count"], 10)
        self.assertEqual(timing.at["through_2024", "task_count"], 8)
        self.assertAlmostEqual(
            timing.at["2025_2026", "delta_f1"], -0.006563347294536298, places=12
        )
        self.assertAlmostEqual(
            timing.at["through_2024", "delta_f1"], 0.020349088447678153, places=12
        )

    def test_protocol_audit_records_matches_and_caveats(self):
        audit = self.summary.loc[
            (self.summary["record_type"] == "protocol_audit")
            & (self.summary["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
        ].set_index("audit_field", verify_integrity=True)
        for field in (
            "generation.temperature",
            "generation.max_tokens",
            "generation.structured_outputs",
            "generation.enable_thinking",
            "runtime.vllm_version",
            "runtime.cuda_toolkit_meta_version",
            "runtime.runtime_lock_sha256",
        ):
            self.assertEqual(audit.at[field, "audit_status"], "matched_exactly")
            self.assertEqual(audit.at[field, "baseline_value"], audit.at[field, "candidate_value"])

        self.assertEqual(
            audit.at["immutable_model_identity", "audit_status"],
            "verified_independently",
        )
        self.assertNotEqual(
            audit.at["immutable_model_identity", "baseline_value"],
            audit.at["immutable_model_identity", "candidate_value"],
        )
        self.assertEqual(audit.at["evidence_protocol", "candidate_value"], "legacy_full34")
        self.assertEqual(audit.at["evidence_protocol", "audit_status"], "differing_caveat")
        self.assertIn("16,425", audit.at["source_evidence_scope", "candidate_value"])
        self.assertEqual(audit.at["generation.seed", "baseline_value"], "20260820")
        self.assertEqual(audit.at["generation.seed", "candidate_value"], "20260804")
        self.assertIn("parity is not claimed", audit.at["generation.seed", "audit_note"])

    def test_csv_and_figure_outputs_exist_and_are_nontrivial(self):
        expected_minimum_sizes = {
            "robustness_summary.csv": 5_000,
            "source_heterogeneity.csv": 10_000,
            "robustness_diagnostics.png": 25_000,
            "robustness_diagnostics.pdf": 5_000,
        }
        for filename, minimum_size in expected_minimum_sizes.items():
            path = self.output_dir / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, minimum_size, filename)

        summary_from_disk = pd.read_csv(self.output_dir / "robustness_summary.csv")
        heterogeneity_from_disk = pd.read_csv(
            self.output_dir / "source_heterogeneity.csv"
        )
        self.assertEqual(len(summary_from_disk), 36)
        self.assertEqual(len(heterogeneity_from_disk), 82)
        self.assertEqual(
            set(summary_from_disk["candidate_checkpoint_id"]), {CANDIDATE_ID}
        )
        self.assertEqual(
            set(heterogeneity_from_disk["baseline_checkpoint_id"]), set(BASELINE_IDS)
        )
        self.assertTrue((DEFAULT_BROAD_DIR / "model_by_task.csv").is_file())


if __name__ == "__main__":
    unittest.main()
