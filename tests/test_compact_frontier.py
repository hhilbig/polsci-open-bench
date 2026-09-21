from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_compact_frontier import (  # noqa: E402
    CompactFrontierError,
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    PUBLIC_CAVEAT,
    PUBLIC_LABEL,
    SERIES_ORDER,
    _measured_hive_gpu,
    _score_task_with_malformed_predictions,
    _validate_api_evidence,
    _validate_compact_hive_qualification,
    _validate_compact_rows,
    _validate_hive_evidence,
    bootstrap_indices,
    build_outputs,
    file_sha256,
    load_compact_registry,
    passes_reliability_gate,
    resolve_repo_path,
)
from panel_manifest import load_panel_manifest  # noqa: E402


BROAD_MANIFEST = REPO / "experiments" / "frontier_broad_18.yaml"
BROAD_REGISTRY = REPO / "experiments" / "frontier_broad_checkpoints_2026.yaml"


class CompactFrontierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_panel_manifest(DEFAULT_MANIFEST)
        cls.registry = load_compact_registry(DEFAULT_REGISTRY, cls.benchmark)

    def test_registry_has_frozen_seventeen_open_and_seven_api_checkpoints(self):
        checkpoints = self.registry["checkpoints"]
        self.assertEqual(len(checkpoints), 24)
        self.assertEqual(
            Counter(row["series"] for row in checkpoints),
            {"open_hive": 17, "api": 7},
        )
        self.assertEqual(tuple(SERIES_ORDER), ("open_hive", "api"))
        self.assertEqual(self.registry["public_label"], PUBLIC_LABEL)
        self.assertEqual(self.registry["public_caveat"], PUBLIC_CAVEAT)
        for series in SERIES_ORDER:
            dates = [row["plot_date"] for row in checkpoints if row["series"] == series]
            self.assertEqual(dates, sorted(dates))
        self.assertTrue(all(row["immutable"] for row in checkpoints))
        self.assertTrue(
            all(
                len(row["identity"]) == 40
                for row in checkpoints
                if row["series"] == "open_hive"
            )
        )

    def test_exact_rosters(self):
        observed_open = [
            row["model_id"]
            for row in self.registry["checkpoints"]
            if row["series"] == "open_hive"
        ]
        observed_api = [
            row["model_id"]
            for row in self.registry["checkpoints"]
            if row["series"] == "api"
        ]
        self.assertEqual(
            observed_open,
            [
                "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8-dynamic",
                "Qwen/Qwen2.5-32B-Instruct",
                "RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic",
                "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "RedHatAI/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "Qwen/Qwen3-32B",
                "Qwen/Qwen3-30B-A3B",
                "RedHatAI/gemma-3-27b-it-FP8-dynamic",
                "openai/gpt-oss-120b",
                "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
                "zai-org/GLM-4.7-Flash",
                "Qwen/Qwen3.5-35B-A3B-FP8",
                "mistralai/Mistral-Small-4-119B-2603-NVFP4",
                "Qwen/Qwen3.6-27B-FP8",
                "google/gemma-4-31B-it-qat-w4a16-ct",
            ],
        )
        self.assertEqual(
            observed_api,
            [
                "gpt-4o-2024-08-06",
                "gpt-4o-2024-11-20",
                "gpt-4.1-2025-04-14",
                "gpt-5-2025-08-07",
                "gpt-5.2-2025-12-11",
                "gpt-5.4-2026-03-05",
                "claude-sonnet-5",
            ],
        )
        selected = [
            row
            for row in self.registry["checkpoints"]
            if row.get("selection_basis") == "external_historical_frontier_candidate"
        ]
        self.assertEqual(
            [row["model_id"] for row in selected],
            [
                "Qwen/Qwen2.5-32B-Instruct",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "Qwen/Qwen3-32B",
                "Qwen/Qwen3.5-35B-A3B-FP8",
            ],
        )
        self.assertTrue(all(row.get("selection_source") for row in selected))
        targeted = [
            row
            for row in self.registry["checkpoints"]
            if row.get("selection_basis") == "targeted_historical_contender"
        ]
        self.assertEqual(len(targeted), 6)
        self.assertTrue(all(row.get("selection_source") for row in targeted))

    def test_malformed_threshold_includes_exactly_five_percent(self):
        self.assertTrue(passes_reliability_gate(0.0, self.benchmark))
        self.assertTrue(passes_reliability_gate(0.05, self.benchmark))
        self.assertFalse(passes_reliability_gate(np.nextafter(0.05, 1.0), self.benchmark))

    def test_bootstrap_indices_are_deterministic_and_within_task(self):
        first = bootstrap_indices(self.benchmark, iterations=100, seed=20260820)
        second = bootstrap_indices(self.benchmark, iterations=100, seed=20260820)
        self.assertEqual(set(first), set(self.benchmark.task_names))
        for task_name in self.benchmark.task_names:
            np.testing.assert_array_equal(first[task_name], second[task_name])
            self.assertEqual(first[task_name].shape, (100, 500))
            self.assertGreaterEqual(int(first[task_name].min()), 0)
            self.assertLess(int(first[task_name].max()), 500)

    def test_prediction_rows_are_canonicalized_before_shared_bootstrap(self):
        checkpoint = next(
            row
            for row in self.registry["checkpoints"]
            if row["result_status"] == "completed"
        )
        predictions = pd.read_csv(
            resolve_repo_path(checkpoint["result"]["predictions_path"]),
            low_memory=False,
        )
        shuffled = predictions.sample(frac=1, random_state=20260820).reset_index(drop=True)
        canonical = _validate_compact_rows(shuffled, self.benchmark, checkpoint)
        expected_keys = [
            (str(task["name"]), str(item["item_id"]))
            for task in self.benchmark.tasks
            for item in task["loader"]()
        ]
        observed_keys = list(zip(canonical["task"], canonical["item_id"]))
        self.assertEqual(observed_keys, expected_keys)

    def test_invalid_prediction_is_counted_malformed_and_scored_wrong(self):
        task = next(task for task in self.benchmark.tasks if task["label_kind"] == "binary")
        key = str(task["label_key"])
        group = pd.DataFrame(
            {
                "parse_error": [None, None],
                f"gt_{key}": [1, 0],
                f"pred_{key}": [1, 7],
            }
        )
        metrics, scored, malformed = _score_task_with_malformed_predictions(group, task)
        self.assertEqual(malformed.tolist(), [False, True])
        self.assertEqual(metrics["parse_ok"], 1)
        self.assertEqual(metrics["parse_err_rate"], 0.5)
        self.assertEqual(int(scored.loc[1, f"pred_{key}"]), 1)
        self.assertAlmostEqual(metrics["headline_f1"], 2 / 3)

    def test_hive_hardware_uses_measured_records_not_registry_fallback(self):
        checkpoint = next(
            row
            for row in self.registry["checkpoints"]
            if row["series"] == "open_hive" and row["result_status"] == "completed"
        )
        result = dict(checkpoint["result"])
        predictions_path = resolve_repo_path(result["predictions_path"])
        metadata = json.loads(resolve_repo_path(result["metadata_path"]).read_text())
        self.assertEqual(
            _measured_hive_gpu(metadata, str(checkpoint["checkpoint_id"])),
            (int(metadata["observed_peak_gpu_memory_used_mib"]), 97_887),
        )
        metadata.pop("gpu_before_load")
        metadata["observed_gpu_capacity_mib"] = 97_887
        result["observed_gpu_capacity_mib"] = 97_887
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata))
            result["metadata_path"] = str(metadata_path)
            with self.assertRaisesRegex(CompactFrontierError, "measured gpu_before_load"):
                _validate_hive_evidence(
                    checkpoint,
                    result,
                    predictions_path,
                    self.benchmark,
                )

    def test_compact_hive_requires_offline_oom_and_qualification_metadata(self):
        revision = "a" * 40
        checks = {
            field: True
            for field in {
                "completed",
                "exact_model_id",
                "exact_revision",
                "exact_row_count",
                "one_typed_gpu",
                "exact_capacity",
                "peak_within_capacity",
                "oom_free",
                "malformed_rate_within_limit",
            }
        }
        metadata = {
            "execution_protocol": "compact_direct",
            "provenance_status": "qualified",
            "model_snapshot_path": f"/cache/models--provider--model/snapshots/{revision}",
            "evidence_stage": "terminal_qualified",
            "pilot": {"passed": True, "checks": checks},
            "qualification": {
                "passed": True,
                "reliability": {"passed": True},
                "exact_model_identity": True,
                "exact_coverage": True,
                "within_gpu_capacity": True,
                "oom_free": True,
            },
        }
        _validate_compact_hive_qualification(metadata, "candidate", revision)
        missing_offline = copy.deepcopy(metadata)
        missing_offline.pop("provenance_status")
        with self.assertRaisesRegex(CompactFrontierError, "offline snapshot"):
            _validate_compact_hive_qualification(missing_offline, "candidate", revision)
        oom = copy.deepcopy(metadata)
        oom["qualification"]["oom_free"] = False
        with self.assertRaisesRegex(CompactFrontierError, "final qualification"):
            _validate_compact_hive_qualification(oom, "candidate", revision)

    def test_api_evidence_requires_finalizer_and_passed_pilot_gate(self):
        checkpoint = next(
            row for row in self.registry["checkpoints"] if row["series"] == "api"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / "predictions.csv"
            predictions_path.write_text("placeholder\n")
            manifest_rows = []
            for task in self.benchmark.tasks:
                task_name = str(task["name"])
                for item in task["loader"]():
                    manifest_rows.append(
                        {
                            "task": task_name,
                            "item_id": str(item["item_id"]),
                            "panel_id": self.benchmark.panel_id,
                            "panel_sha256": self.benchmark.panel_sha256,
                            "panel_task_fingerprint": self.benchmark.task_fingerprints[
                                task_name
                            ],
                            "requested_model": checkpoint["model_id"],
                            "expected_response_model": checkpoint["identity"],
                            "request_sha256": "a" * 64,
                        }
                    )
            manifest_path = root / "request_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            metadata = {
                "evidence_type": "compact8_api_pilot_remainder_final",
                "status": "completed",
                "benchmark_commit": self.benchmark.benchmark_commit,
                "panel_id": self.benchmark.panel_id,
                "panel_sha256": self.benchmark.panel_sha256,
                "requested_model": checkpoint["model_id"],
                "returned_model_identity": checkpoint["identity"],
                "expected_items": 4000,
                "rows_written": 4000,
                "pilot_items": 16,
                "remainder_items": 3984,
                "pilot_malformed_count": 0,
                "pilot_malformed_rate": 0.0,
                "pilot_malformed_limit": 0.05,
                "pilot_reliability_pass": True,
                "reliability_pass": True,
                "source_stages": {
                    "pilot": {"request_count": 16},
                    "remainder": {"request_count": 3984},
                },
                "predictions_sha256": file_sha256(predictions_path),
                "requests_manifest_sha256": file_sha256(manifest_path),
            }
            metadata_path = root / "metadata.json"
            result = {
                "metadata_path": str(metadata_path),
                "requests_manifest_path": str(manifest_path),
            }
            metadata_path.write_text(json.dumps(metadata))
            _validate_api_evidence(
                checkpoint, result, predictions_path, self.benchmark
            )

            rejected = {
                "missing_finalizer_type": {**metadata, "evidence_type": ""},
                "failed_pilot": {**metadata, "pilot_reliability_pass": False},
                "one_of_sixteen_malformed": {
                    **metadata,
                    "pilot_malformed_count": 1,
                    "pilot_malformed_rate": 1 / 16,
                },
                "missing_stage_counts": {**metadata, "source_stages": {}},
            }
            for label, invalid_metadata in rejected.items():
                with self.subTest(label=label):
                    metadata_path.write_text(json.dumps(invalid_metadata))
                    with self.assertRaises(CompactFrontierError):
                        _validate_api_evidence(
                            checkpoint, result, predictions_path, self.benchmark
                        )

    def test_outputs_are_compact_audited_and_consistent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            summary, by_task = build_outputs(
                DEFAULT_MANIFEST,
                DEFAULT_REGISTRY,
                output,
                iterations=100,
                seed=20260820,
            )
            self.assertEqual(len(summary), 24)
            completed = summary.loc[summary["result_status"] == "completed"]
            self.assertEqual(len(by_task), len(completed) * self.benchmark.expected_tasks)
            self.assertEqual(
                set(by_task.groupby("checkpoint_id")["task"].nunique()),
                {self.benchmark.expected_tasks},
            )
            self.assertTrue((completed["observed_items"] == 4000).all())
            self.assertTrue((completed["observed_tasks"] == 8).all())
            self.assertTrue((completed["malformed_rate"] <= 0.05).all())
            gpt_oss = completed.loc[
                completed["checkpoint_id"] == "gpt_oss_120b_mxfp4_hive"
            ].iloc[0]
            self.assertEqual(gpt_oss["malformed_count"], 4)
            self.assertAlmostEqual(gpt_oss["malformed_rate"], 0.001)
            self.assertTrue(completed["frontier_eligible"].all())
            self.assertFalse(summary.loc[summary["series"] == "api", "frontier_eligible"].any())
            forbidden = ("calibrat", "promotion", "confirmation", "full34")
            for column in summary.columns:
                self.assertFalse(any(token in column.lower() for token in forbidden), column)
            ordered = summary.sort_values(["plot_date", "series", "checkpoint_id"])
            pd.testing.assert_frame_equal(summary, ordered.reset_index(drop=True))
            for filename, minimum_size in {
                "checkpoint_summary.csv": 1000,
                "model_by_task.csv": 1000,
                "task_sensitivity.csv": 500,
                "frontier_staircase.png": 10_000,
                "frontier_staircase.pdf": 5_000,
            }.items():
                path = output / filename
                self.assertTrue(path.exists(), filename)
                self.assertGreater(path.stat().st_size, minimum_size, filename)
            disk_summary = pd.read_csv(output / "checkpoint_summary.csv")
            self.assertEqual(
                disk_summary.loc[disk_summary["frontier_setter"], "checkpoint_id"].tolist(),
                summary.loc[summary["frontier_setter"], "checkpoint_id"].tolist(),
            )


class BroadFrontierTest(unittest.TestCase):
    def test_audited_legacy_evidence_subsets_to_frozen_broad_keys(self):
        benchmark = load_panel_manifest(BROAD_MANIFEST)
        registry = load_compact_registry(BROAD_REGISTRY, benchmark)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            summary, by_task = build_outputs(
                BROAD_MANIFEST,
                BROAD_REGISTRY,
                output,
                iterations=100,
                seed=20260820,
            )
            completed = summary.loc[summary["result_status"] == "completed"]
            completed_specs = [
                checkpoint
                for checkpoint in registry["checkpoints"]
                if checkpoint["result_status"] == "completed"
            ]
            legacy_specs = [
                checkpoint
                for checkpoint in completed_specs
                if checkpoint["result"]["source_protocol"].startswith("legacy_")
            ]
            direct_specs = [
                checkpoint
                for checkpoint in completed_specs
                if checkpoint["result"]["source_protocol"] == "broad18"
            ]
            self.assertEqual(len(completed), len(completed_specs))
            self.assertEqual(len(legacy_specs), 6)
            self.assertGreaterEqual(len(direct_specs), 9)
            self.assertTrue(completed["frontier_eligible"].all())
            self.assertEqual(len(by_task), len(completed_specs) * 18)
            self.assertTrue((completed["observed_items"] == 3600).all())
            sensitivity = pd.read_csv(output / "task_sensitivity.csv")
            self.assertEqual(len(sensitivity), len(completed_specs) + 1)
            self.assertEqual(
                len(sensitivity.loc[
                    (sensitivity["checkpoint_id"] == "qwen3_6_27b_fp8_hive")
                    & (sensitivity["baseline_checkpoint_id"] == "llama3_1_70b_instruct_fp8_dynamic_hive")
                ]),
                1,
            )
            self.assertTrue((sensitivity["task_count"] == 18).all())
            self.assertTrue(
                sensitivity["leave_one_task_out_min_delta"].notna().all()
            )
            self.assertEqual(
                registry["public_label"],
                "Best tested checkpoints on a frozen 18-task political-science classification benchmark.",
            )
            self.assertEqual(
                registry["public_caveat"],
                "Broader task coverage, but non-exhaustive; not a claim about the global model frontier.",
            )


if __name__ == "__main__":
    unittest.main()
