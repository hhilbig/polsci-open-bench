import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from panel_manifest import (  # noqa: E402
    PanelManifestError,
    assert_panel_checkout,
    load_panel_manifest,
    task_input_fingerprint,
    validate_panel_checkpoint_frame,
)
import batch_benchmark  # noqa: E402
import benchmark  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402


PANEL_PATH = REPO / "experiments" / "frontier_panel_18.yaml"


class FrontierPanelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel_manifest(PANEL_PATH)

    def test_frozen_panel_resolves_exact_scope_and_calibration(self):
        panel = self.panel
        self.assertEqual(panel.panel_id, "frontier_panel_18")
        self.assertEqual(
            panel.benchmark_commit,
            "3c7ad0756d447b1d57ed4daf26bbb85f5f296042",
        )
        self.assertEqual(panel.expected_tasks, 18)
        self.assertEqual(panel.expected_items, 8793)
        self.assertEqual(len(panel.tasks), 18)
        self.assertEqual(sum(panel.task_counts.values()), 8793)
        self.assertEqual(panel.scorer_version, "headline_f1_v1")
        self.assertEqual(panel.calibration["intercept"], 0.054470073651511)
        self.assertEqual(panel.calibration["slope"], 0.897971355245587)
        self.assertEqual(
            panel.calibration["supported_panel_f1"],
            {"min": 0.599246, "max": 0.683579},
        )
        self.assertEqual(len(panel.panel_sha256), 64)
        self.assertEqual(len(assert_panel_checkout(panel, REPO)), 40)

    def test_task_order_and_item_keys_are_frozen(self):
        expected_names = (
            "burnham_covid_threat_minimization",
            "burnham_polnli_entailment",
            "burnham_polnli_event_entailment",
            "burnham_trump_stance",
            "cap_crs_policy_topic",
            "cap_party_platform_policy_topic",
            "chae_semeval_stance",
            "dicocco_manifesto_populism",
            "douglass_icbe_sentence_event_type",
            "erlich_ati_topics",
            "gilardi_relevance",
            "gilardi_stance",
            "halterman_keith_bfrs",
            "haunss_papea_fgz_forms",
            "muller_fujimura_campaign_policy_area",
            "rheault_line_of_fire_incivility",
            "theocharis_dynamics_incivility",
            "twitcivility_impoliteness",
        )
        self.assertEqual(self.panel.task_names, expected_names)
        for task in self.panel.tasks:
            with self.subTest(task=task["name"]):
                items = task["loader"]()
                self.assertEqual(len(items), self.panel.task_counts[task["name"]])
                self.assertEqual(len(items), len({str(item["item_id"]) for item in items}))
                self.assertEqual(
                    task_input_fingerprint(task),
                    self.panel.task_fingerprints[task["name"]],
                )

    def test_count_and_fingerprint_drift_fail_closed(self):
        raw = yaml.safe_load(PANEL_PATH.read_text())
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "panel.yaml"

            wrong_count = copy.deepcopy(raw)
            wrong_count["tasks"][0]["expected_items"] += 1
            wrong_count["expected_items"] += 1
            temp_path.write_text(yaml.safe_dump(wrong_count, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "observed 293 items"):
                load_panel_manifest(temp_path)

            wrong_fingerprint = copy.deepcopy(raw)
            wrong_fingerprint["tasks"][0]["fingerprint_sha256"] = "0" * 64
            temp_path.write_text(yaml.safe_dump(wrong_fingerprint, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "input fingerprint drift"):
                load_panel_manifest(temp_path)

    def test_resume_checkpoint_requires_exact_panel_identity_and_coverage(self):
        task = self.panel.tasks[0]
        metadata = self.panel.row_metadata(task["name"])
        rows = [
            {
                "task": task["name"],
                "model": "model-a",
                "batch_size": 1,
                "item_id": str(item["item_id"]),
                **metadata,
            }
            for item in task["loader"]()
        ]
        frame = pd.DataFrame(rows)
        validate_panel_checkpoint_frame(
            frame,
            self.panel,
            cell_columns=("model", "batch_size"),
        )

        missing = frame.iloc[:-1].copy()
        with self.assertRaisesRegex(PanelManifestError, "exact item coverage"):
            validate_panel_checkpoint_frame(
                missing,
                self.panel,
                cell_columns=("model", "batch_size"),
            )

        wrong_hash = frame.copy()
        wrong_hash["panel_sha256"] = "0" * 64
        with self.assertRaisesRegex(PanelManifestError, "different panel_sha256"):
            validate_panel_checkpoint_frame(
                wrong_hash,
                self.panel,
                cell_columns=("model", "batch_size"),
            )

        extra_task = frame.copy()
        extra_task.loc[0, "task"] = "not_in_panel"
        with self.assertRaisesRegex(PanelManifestError, "extra tasks"):
            validate_panel_checkpoint_frame(
                extra_task,
                self.panel,
                cell_columns=("model", "batch_size"),
            )


class PanelRunnerMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task = load_task_definitions(
            task_dir=REPO / "examples" / "minimal_custom_task"
        )[0]
        cls.task = {**task, "loader": lambda: task["loader"]()[:1]}
        cls.model = {"name": "test-model", "backend": "ollama"}
        cls.metadata = {
            "panel_id": "test-panel",
            "panel_sha256": "a" * 64,
            "panel_task_fingerprint": "b" * 64,
        }

    def test_serial_checkpoint_rows_include_panel_metadata(self):
        response = {"content": '{"relevant": 1}', "latency_s": 0.1, "eval_count": 1}
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "benchmark.warmup_ollama"
        ), mock.patch("benchmark.classify_ollama", return_value=response):
            rows = benchmark.run_task(
                self.task,
                [self.model],
                {},
                Path(temp_dir) / "checkpoint.csv",
                checkpoint_metadata=self.metadata,
            )
        self.assertEqual(len(rows), 1)
        for key, value in self.metadata.items():
            self.assertEqual(rows[0][key], value)

    def test_batched_checkpoint_rows_include_panel_metadata(self):
        with mock.patch(
            "batch_benchmark.classify_ollama_batched",
            return_value=('{"relevant": 1}', 0.1, 1),
        ):
            rows = batch_benchmark.run_cell(
                self.task,
                self.model,
                {},
                1,
                self.task["loader"](),
                checkpoint_metadata=self.metadata,
            )
        self.assertEqual(len(rows), 1)
        for key, value in self.metadata.items():
            self.assertEqual(rows[0][key], value)


if __name__ == "__main__":
    unittest.main()
