import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from build_frontier_2026 import (  # noqa: E402
    _base_row,
    _score_malformed_as_incorrect,
    _validate_hive_metadata,
    apply_promotion_and_frontiers,
    build_outputs,
    build_lag_table,
    calibrated_panel_ci,
    file_sha256,
    passes_reliability_gate,
)
from panel_manifest import load_panel_manifest  # noqa: E402


PANEL = load_panel_manifest(REPO / "experiments" / "frontier_panel_18.yaml")


class MalformedScoringTests(unittest.TestCase):
    def test_reliability_gate_boundaries(self):
        self.assertTrue(passes_reliability_gate(0.0, PANEL))
        self.assertTrue(passes_reliability_gate(0.05, PANEL))
        self.assertFalse(passes_reliability_gate(0.0500001, PANEL))

    def test_malformed_binary_row_is_retained_and_forced_wrong(self):
        task = {
            "name": "binary_task",
            "label_kind": "binary",
            "label_key": "relevant",
            "labels": [0, 1],
        }
        frame = pd.DataFrame(
            {
                "gt_relevant": [1] * 20,
                "pred_relevant": [1] * 20,
                "parse_error": ["malformed"] + [np.nan] * 19,
                "latency_s": [0.1] * 20,
            }
        )
        metrics = _score_malformed_as_incorrect(frame, task)
        self.assertEqual(metrics["n"], 20)
        self.assertEqual(metrics["parse_ok"], 19)
        self.assertAlmostEqual(metrics["parse_err_rate"], 0.05)
        self.assertLess(metrics["headline_f1"], 1.0)

    def test_malformed_multibinary_row_is_wrong_for_every_decision(self):
        task = {
            "name": "multi_task",
            "label_kind": "multi_binary",
            "label_key": None,
            "labels": ["a", "b"],
        }
        frame = pd.DataFrame(
            {
                "gt_a": [1, 1],
                "pred_a": [1, 1],
                "gt_b": [0, 1],
                "pred_b": [0, 1],
                "parse_error": ["malformed", np.nan],
                "latency_s": [0.1, 0.1],
            }
        )
        metrics = _score_malformed_as_incorrect(frame, task)
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["parse_ok"], 1)
        self.assertLess(metrics["headline_f1"], 1.0)


class CalibrationAndPromotionTests(unittest.TestCase):
    def test_frozen_calibration_reproduces_from_ten_canonical_models(self):
        # Reads the LEGACY summary on purpose. The calibration constants frozen in
        # experiments/frontier_panel_18.yaml were fitted on headline F1 computed by
        # averaging macro F1 over every manifest label, including labels with no
        # gold support. code/scoring.py corrected that on 2026-09-18 and
        # output/summary.csv now carries the corrected metric, which shifts every
        # model mean by roughly +0.009 and moves the fitted intercept from
        # 0.05447 to 0.05167.
        #
        # Refitting the constants here would silently change a frozen calibration
        # that downstream frontier results depend on, so the panel stays pinned to
        # the artifact it was derived from. Recalibrating the 18-task panel against
        # the corrected metric is a deliberate analysis change, not a side effect
        # of a scoring fix.
        summary = pd.read_csv(REPO / "output" / "summary_legacy_all_labels.csv")
        panel_names = set(PANEL.task_names)
        pairs = []
        for _, group in summary.groupby("model"):
            pairs.append(
                (
                    group.loc[group["task"].isin(panel_names), "headline_f1"].mean(),
                    group["headline_f1"].mean(),
                )
            )
        self.assertEqual(len(pairs), 10)
        design = np.column_stack([np.ones(10), [pair[0] for pair in pairs]])
        outcome = np.asarray([pair[1] for pair in pairs])
        intercept, slope = np.linalg.lstsq(design, outcome, rcond=None)[0]
        self.assertAlmostEqual(intercept, PANEL.calibration["intercept"], places=14)
        self.assertAlmostEqual(slope, PANEL.calibration["slope"], places=14)
        self.assertAlmostEqual(min(pair[0] for pair in pairs), 0.599246018214532)
        self.assertAlmostEqual(max(pair[0] for pair in pairs), 0.6835794338371117)

    def test_six_held_out_residuals_and_validation_statistics_reproduce(self):
        registry = yaml.safe_load(
            (REPO / "experiments" / "frontier_checkpoints_2026.yaml").read_text()
        )
        panel_names = set(PANEL.task_names)
        residuals = []
        for checkpoint in registry["checkpoints"]:
            result = checkpoint.get("result", {})
            if not (
                checkpoint.get("family") == "open_weight"
                and checkpoint.get("hardware_tier") == "hive_98gb"
                and checkpoint.get("result_status") == "full34_confirmed"
            ):
                continue
            metrics = pd.read_csv(REPO / result["task_metrics_path"])
            metrics = metrics.loc[metrics["model"].astype(str) == str(result["model_key"])]
            panel_f1 = metrics.loc[metrics["task"].isin(panel_names), "headline_f1"].mean()
            full_f1 = metrics["headline_f1"].mean()
            predicted = PANEL.calibration["intercept"] + PANEL.calibration["slope"] * panel_f1
            residuals.append(full_f1 - predicted)
        frozen = PANEL.calibration["held_out_validation"]
        self.assertEqual(len(residuals), 6)
        np.testing.assert_allclose(residuals, frozen["residuals"], atol=5e-15)
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.square(residuals)))), frozen["rmse"], places=4)
        self.assertAlmostEqual(float(np.max(np.abs(residuals))), frozen["max_absolute_error"], places=4)

    def test_empirical_residual_bootstrap_is_deterministic(self):
        values = np.linspace(0.58, 0.70, 18)
        first = calibrated_panel_ci(values, PANEL.calibration, 1000, 20260820)
        second = calibrated_panel_ci(values, PANEL.calibration, 1000, 20260820)
        self.assertEqual(first, second)
        self.assertLess(first[1], first[0])
        self.assertGreater(first[2], first[0])

    def test_upper_difference_boundary_triggers_promotion(self):
        rows = [
            {
                "checkpoint_id": "confirmed",
                "hardware_tier": "hive_98gb",
                "plot_date": "2026-01-01",
                "result_status": "full34_confirmed",
                "raw_panel_f1": 0.62,
                "calibrated_full34_f1": 0.60,
                "calibrated_ci_low": 0.59,
                "calibrated_ci_high": 0.61,
                "calibration_supported": True,
                "hardware_qualification": "confirmed",
                "reliability_pass": True,
                "full34_coverage_audit": "passed_exact_task_item_gold_keys",
                "immutable": True,
                "historical_lag_eligible": True,
                "model_identity_verified": True,
                "observed_peak_memory_mib": 90000,
                "observed_gpu_capacity_mib": 97887,
                "full34_f1": 0.65,
            },
            {
                "checkpoint_id": "candidate",
                "hardware_tier": "hive_98gb",
                "plot_date": "2026-02-01",
                "result_status": "panel_only",
                "raw_panel_f1": 0.62,
                "calibrated_full34_f1": 0.63,
                "calibrated_ci_low": 0.62,
                "calibrated_ci_high": 0.64,
                "calibration_supported": True,
                "hardware_qualification": "confirmed",
                "reliability_pass": True,
                "full34_coverage_audit": "not_run",
                "immutable": True,
                "historical_lag_eligible": True,
                "model_identity_verified": True,
                "observed_peak_memory_mib": 90000,
                "observed_gpu_capacity_mib": 97887,
                "full34_f1": np.nan,
            },
        ]
        result = apply_promotion_and_frontiers(pd.DataFrame(rows), PANEL)
        candidate = result.loc[result["checkpoint_id"] == "candidate"].iloc[0]
        self.assertTrue(candidate["promotion_required"])
        self.assertIn("upper_difference_bound_at_least_minus_0.01", candidate["promotion_reasons"])

    def test_hive_checkpoint_above_capacity_is_ineligible(self):
        row = {
            "checkpoint_id": "over_capacity",
            "hardware_tier": "hive_98gb",
            "plot_date": "2026-01-01",
            "result_status": "full34_confirmed",
            "raw_panel_f1": 0.62,
            "calibrated_full34_f1": 0.62,
            "calibrated_ci_low": 0.61,
            "calibrated_ci_high": 0.63,
            "calibration_supported": True,
            "hardware_qualification": "confirmed",
            "reliability_pass": True,
            "full34_coverage_audit": "passed_exact_task_item_gold_keys",
            "immutable": True,
            "historical_lag_eligible": True,
            "model_identity_verified": True,
            "observed_peak_memory_mib": 97888,
            "observed_gpu_capacity_mib": 97887,
            "full34_f1": 0.65,
            "frontier_setter": False,
        }
        result = apply_promotion_and_frontiers(pd.DataFrame([row]), PANEL)
        self.assertFalse(bool(result.iloc[0]["frontier_eligible"]))
        self.assertFalse(bool(result.iloc[0]["frontier_setter"]))


class DatesAndLagTests(unittest.TestCase):
    def _checkpoint(self, **updates):
        checkpoint = {
            "checkpoint_id": "x",
            "display_name": "X",
            "family": "open_weight",
            "provider": "Provider",
            "model_id": "provider/x",
            "revision": "a" * 40,
            "release_date": "2026-01-01",
            "release_source": "https://example.com/x",
            "quantization": "Q4",
            "hardware_tier": "hive_98gb",
            "hardware_qualification": "confirmed",
            "immutable": True,
            "historical_lag_eligible": True,
            "reasoning": "disabled",
            "runtime_profile": "runtime",
            "observed_peak_memory_mib": 20000,
            "hardware_capacity_mib": 97887,
            "result_status": "pending",
        }
        checkpoint.update(updates)
        return checkpoint

    def test_open_plot_date_uses_exact_artifact_date(self):
        row = _base_row(
            self._checkpoint(
                base_model_release_date="2026-01-01",
                artifact_release_date="2026-01-08",
            ),
            PANEL,
        )
        self.assertEqual(row["plot_date"], "2026-01-08")
        self.assertEqual(row["date_basis"], "artifact_release_date")

    def test_mutable_api_uses_evaluation_date(self):
        checkpoint = self._checkpoint(
            family="api",
            immutable=False,
            historical_lag_eligible=False,
            hardware_tier="api",
            result={"evaluated_at": "2026-02-03"},
        )
        row = _base_row(checkpoint, PANEL)
        self.assertEqual(row["plot_date"], "2026-02-03")
        self.assertEqual(row["date_basis"], "evaluation_date_mutable_endpoint")

    def test_unpinned_open_artifact_uses_base_date_only_as_unqualified_marker(self):
        row = _base_row(
            self._checkpoint(immutable=False, historical_lag_eligible=False, revision=""),
            PANEL,
        )
        self.assertEqual(row["plot_date"], "2026-01-01")
        self.assertEqual(row["artifact_release_date"], "")
        self.assertEqual(row["date_basis"], "base_model_release_date_unqualified_artifact")

    def test_no_prior_api_coverage_has_no_numeric_lag(self):
        scores = pd.DataFrame(
            [
                {
                    "checkpoint_id": "open",
                    "display_name": "Open",
                    "hardware_tier": "hive_98gb",
                    "frontier_setter": True,
                    "plot_date": "2025-07-28",
                    "release_date": "2025-07-28",
                    "full34_f1": 0.60,
                    "immutable": True,
                    "historical_lag_eligible": True,
                }
            ]
        )
        lags = build_lag_table(scores, {}, 1000, 1)
        self.assertEqual(lags.iloc[0]["lag_status"], "no_prior_api_coverage")
        self.assertTrue(pd.isna(lags.iloc[0]["lag_months"]))


class TwoSeriesRegistryTests(unittest.TestCase):
    def test_hive_metadata_accepts_exact_panel_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / "predictions.csv"
            predictions_path.write_text("task,model,item_id\n")
            metadata_path = root / "run_metadata.json"
            task_names = set(PANEL.task_names)
            metadata = {
                "status": "completed",
                "benchmark_commit": PANEL.benchmark_commit,
                "model_key": "candidate",
                "model_id": "provider/candidate",
                "revision": "a" * 40,
                "total_tasks": PANEL.expected_tasks,
                "total_items": PANEL.expected_items,
                "rows_written": PANEL.expected_items,
                "panel_id": PANEL.panel_id,
                "panel_sha256": PANEL.panel_sha256,
                "task_counts": dict(PANEL.task_counts),
                "predictions_sha256": file_sha256(predictions_path),
                "config_sha256": "b" * 64,
                "generation": {},
                "runtime": {},
                "runtime_versions": {},
                "task_results": {
                    task_name: {
                        "task_fingerprint": "frozen",
                        "panel_sha256": PANEL.panel_sha256,
                        "panel_task_fingerprint": PANEL.task_fingerprints[task_name],
                    }
                    for task_name in task_names
                },
            }
            metadata_path.write_text(json.dumps(metadata))
            checkpoint = {
                "checkpoint_id": "candidate_hive",
                "model_id": "provider/candidate",
                "revision": "a" * 40,
                "result_status": "panel_only",
            }
            result = {
                "model_key": "candidate",
                "metadata_path": str(metadata_path),
            }
            tasks = {
                task_name: {"name": task_name, "loader": lambda: []}
                for task_name in task_names
            }
            with patch(
                "build_frontier_2026.task_fingerprint", return_value="frozen"
            ) as fingerprint:
                observed, _ = _validate_hive_metadata(
                    checkpoint,
                    result,
                    predictions_path,
                    tasks,
                    dict(PANEL.task_counts),
                    PANEL,
                )
            self.assertEqual(observed["total_tasks"], 18)
            self.assertEqual(observed["total_items"], 8793)
            self.assertEqual(fingerprint.call_count, 18)
            for call in fingerprint.call_args_list:
                self.assertEqual(call.kwargs["panel_sha256"], PANEL.panel_sha256)
                task_name = call.args[0]["name"]
                self.assertEqual(
                    call.kwargs["panel_task_fingerprint"],
                    PANEL.task_fingerprints[task_name],
                )

    def test_hive_panel_metadata_rejects_task_ledger_panel_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / "predictions.csv"
            predictions_path.write_text("task,model,item_id\n")
            metadata_path = root / "run_metadata.json"
            task_names = set(PANEL.task_names)
            metadata = {
                "status": "completed",
                "benchmark_commit": PANEL.benchmark_commit,
                "model_key": "candidate",
                "model_id": "provider/candidate",
                "revision": "a" * 40,
                "total_tasks": PANEL.expected_tasks,
                "total_items": PANEL.expected_items,
                "rows_written": PANEL.expected_items,
                "panel_id": PANEL.panel_id,
                "panel_sha256": PANEL.panel_sha256,
                "task_counts": dict(PANEL.task_counts),
                "predictions_sha256": file_sha256(predictions_path),
                "config_sha256": "b" * 64,
                "generation": {},
                "runtime": {},
                "runtime_versions": {},
                "task_results": {
                    task_name: {
                        "task_fingerprint": "frozen",
                        "panel_sha256": "0" * 64,
                        "panel_task_fingerprint": PANEL.task_fingerprints[task_name],
                    }
                    for task_name in task_names
                },
            }
            metadata_path.write_text(json.dumps(metadata))
            checkpoint = {
                "checkpoint_id": "candidate_hive",
                "model_id": "provider/candidate",
                "revision": "a" * 40,
                "result_status": "panel_only",
            }
            result = {
                "model_key": "candidate",
                "metadata_path": str(metadata_path),
            }
            tasks = {
                task_name: {"name": task_name, "loader": lambda: []}
                for task_name in task_names
            }
            with patch("build_frontier_2026.task_fingerprint", return_value="frozen"):
                with self.assertRaisesRegex(
                    ValueError, "task ledger panel hash mismatch"
                ):
                    _validate_hive_metadata(
                        checkpoint,
                        result,
                        predictions_path,
                        tasks,
                        dict(PANEL.task_counts),
                        PANEL,
                    )

    def test_registry_has_only_hive_and_api_and_exact_new_pins(self):
        registry = yaml.safe_load(
            (REPO / "experiments" / "frontier_checkpoints_2026.yaml").read_text()
        )
        self.assertEqual(set(registry["hardware_tiers"]), {"hive_98gb", "api"})
        self.assertEqual(len(registry["checkpoints"]), 27)
        self.assertFalse(
            any(
                row["hardware_tier"] == "laptop_32gb"
                or row["checkpoint_id"].endswith("_laptop")
                for row in registry["checkpoints"]
            )
        )
        expected = {
            "gpt_oss_20b_mxfp4_hive": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "gpt_oss_120b_mxfp4_hive": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "qwen3_next_80b_a3b_fp8_hive": "c5f5f263bdd5cc134092897864e8905d8fe7b928",
            "ministral_3_14b_bf16_hive": "29439f81c2be264d8d393273f99e7db9c0961120",
            "qwen3_5_35b_a3b_fp8_hive": "9d1823d2dee688a6b25e77009dc727688c44936e",
        }
        observed = {
            row["checkpoint_id"]: row.get("revision")
            for row in registry["checkpoints"]
            if row["checkpoint_id"] in expected
        }
        self.assertEqual(observed, expected)
        ministral = next(
            row
            for row in registry["checkpoints"]
            if row["checkpoint_id"] == "ministral_3_14b_bf16_hive"
        )
        self.assertEqual(ministral["result_status"], "failed_provenance_mismatch")
        self.assertFalse(ministral["immutable"])
        gpt_oss_120b = next(
            row
            for row in registry["checkpoints"]
            if row["checkpoint_id"] == "gpt_oss_120b_mxfp4_hive"
        )
        self.assertEqual(gpt_oss_120b["result_status"], "failed_parser_pilot")
        self.assertEqual(gpt_oss_120b["pilot_malformed_rate"], 1.0)
        self.assertFalse(gpt_oss_120b["historical_lag_eligible"])
        gpt_oss_20b = next(
            row
            for row in registry["checkpoints"]
            if row["checkpoint_id"] == "gpt_oss_20b_mxfp4_hive"
        )
        self.assertEqual(gpt_oss_20b["result_status"], "failed_parser_pilot")
        self.assertEqual(gpt_oss_20b["pilot_malformed_rate"], 1.0)
        qwen3_5 = next(
            row
            for row in registry["checkpoints"]
            if row["checkpoint_id"] == "qwen3_5_35b_a3b_fp8_hive"
        )
        self.assertEqual(qwen3_5["result_status"], "panel_only")
        self.assertEqual(qwen3_5["hardware_qualification"], "confirmed")

    def test_public_score_output_has_no_laptop_fields_or_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scores, _ = build_outputs(
                output_dir=Path(temp_dir),
                iterations=100,
                render_plot=False,
            )
        self.assertEqual(set(scores["hardware_tier"]), {"hive_98gb", "api"})
        self.assertNotIn("peak_swap_increase_mib", scores.columns)
        self.assertNotIn("oom_observed", scores.columns)
        gpt_oss_120b = scores.loc[
            scores["checkpoint_id"] == "gpt_oss_120b_mxfp4_hive"
        ].iloc[0]
        self.assertEqual(gpt_oss_120b["result_status"], "failed_parser_pilot")
        self.assertEqual(gpt_oss_120b["parse_error_rate"], 1.0)
        self.assertFalse(gpt_oss_120b["reliability_pass"])


if __name__ == "__main__":
    unittest.main()
