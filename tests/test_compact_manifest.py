from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import anthropic_batch_submit  # noqa: E402
import hive_vllm_benchmark  # noqa: E402
import openai_batch_submit  # noqa: E402
from api_batch_common import load_task_selection  # noqa: E402
from panel_manifest import (  # noqa: E402
    PanelManifestError,
    assert_panel_checkout,
    benchmark_requests,
    load_panel_manifest,
    split_pilot_remainder,
    task_identity_hashes,
)
from task_registry import load_task_definitions  # noqa: E402


COMPACT_PATH = REPO / "experiments" / "frontier_compact_8.yaml"
BROAD_PATH = REPO / "experiments" / "frontier_broad_18.yaml"
EXPECTED_TASKS = (
    "burnham_polnli_entailment",
    "dicocco_manifesto_populism",
    "cap_crs_policy_topic",
    "erlich_ati_topics",
    "gilardi_relevance",
    "douglass_icbe_sentence_event_type",
    "halterman_keith_bfrs",
    "chae_semeval_stance",
)


def request_key(request) -> tuple[str, str]:
    return request.task_name, request.item_id


class CompactManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compact = load_panel_manifest(COMPACT_PATH)
        cls.requests = benchmark_requests(cls.compact)

    def test_exact_frozen_scope_and_direct_scoring_design(self):
        compact = self.compact
        self.assertEqual(compact.schema_version, 2)
        self.assertEqual(compact.panel_id, "frontier_compact_8")
        self.assertEqual(compact.task_names, EXPECTED_TASKS)
        self.assertEqual(compact.expected_tasks, 8)
        self.assertEqual(compact.expected_items, 4000)
        self.assertEqual(compact.task_counts, {name: 500 for name in EXPECTED_TASKS})
        self.assertEqual(len(self.requests), 4000)
        self.assertEqual(len({request_key(row) for row in self.requests}), 4000)
        self.assertEqual(
            compact.task_item_keys_sha256,
            "087537ca643cdd3522f8e36e63d986dba5ac3c19fc28d3926903489026ad7a0c",
        )
        self.assertEqual(compact.scorer["task_weighting"], "equal")
        self.assertEqual(compact.calibration, {})
        self.assertEqual(compact.promotion, {})
        self.assertEqual(compact.bootstrap["iterations"], 10000)
        self.assertEqual(compact.bootstrap["seed"], 20260820)
        self.assertEqual(compact.bootstrap["unit"], "within_task_items")
        self.assertTrue(compact.bootstrap["paired"])
        self.assertEqual(compact.reliability_gate["max_malformed_rate"], 0.05)
        self.assertTrue(compact.reliability_gate["exactly_at_max_passes"])
        self.assertFalse(compact.reliability_gate["selective_retry_malformed"])
        self.assertEqual(len(compact.compact_manifest_sha256), 64)
        self.assertEqual(len(assert_panel_checkout(compact, REPO)), 40)

    def test_every_input_component_hash_is_frozen_and_reproducible(self):
        for task in self.compact.tasks:
            with self.subTest(task=task["name"]):
                observed = task_identity_hashes(task)
                self.assertEqual(
                    observed["fingerprint_sha256"],
                    self.compact.task_fingerprints[task["name"]],
                )
                self.assertEqual(
                    {key: observed[key] for key in self.compact.task_identity_hashes[task["name"]]},
                    self.compact.task_identity_hashes[task["name"]],
                )

    def test_fixed_pilot_and_remainder_are_an_exact_partition(self):
        pilot, remainder = split_pilot_remainder(self.compact)
        second_pilot, second_remainder = split_pilot_remainder(self.compact)
        all_keys = [request_key(row) for row in self.requests]
        pilot_keys = [request_key(row) for row in pilot]
        remainder_keys = [request_key(row) for row in remainder]

        self.assertEqual(len(pilot), 16)
        self.assertEqual(len(remainder), 3984)
        self.assertEqual(self.compact.pilot_items, 16)
        self.assertEqual(self.compact.remainder_items, 3984)
        self.assertEqual(pilot_keys + remainder_keys, all_keys)
        self.assertFalse(set(pilot_keys) & set(remainder_keys))
        self.assertEqual(pilot_keys, [request_key(row) for row in second_pilot])
        self.assertEqual(remainder_keys, [request_key(row) for row in second_remainder])
        self.assertEqual({row.task_name for row in pilot}, {EXPECTED_TASKS[0]})
        self.assertEqual(remainder[0].ordinal, 16)

    def test_local_hive_openai_and_anthropic_resolve_identical_inputs(self):
        definitions = load_task_definitions(tasks_dir=REPO / "tasks")
        api_selection = load_task_selection(definitions, COMPACT_PATH)
        hive_tasks = hive_vllm_benchmark.selected_tasks(
            {"task_scope": {"tasks_dir": "tasks"}},
            panel=self.compact,
        )
        expected = [request_key(row) for row in self.requests]

        def keys_from_tasks(tasks):
            return [
                (str(task["name"]), str(item["item_id"]))
                for task in tasks
                for item in task["loader"]()
            ]

        self.assertEqual(keys_from_tasks(self.compact.tasks), expected)
        self.assertEqual(keys_from_tasks(hive_tasks), expected)
        self.assertEqual(keys_from_tasks(api_selection.tasks), expected)
        self.assertEqual(
            api_selection.task_item_keys_sha256,
            self.compact.task_item_keys_sha256,
        )

        openai_rows = list(
            openai_batch_submit.iter_requests(
                list(api_selection.tasks),
                {
                    "name": "gpt-test",
                    "backend": "openai",
                    "response_format_type": "json_schema",
                    "max_output_tokens": 128,
                },
            )
        )
        anthropic_rows = list(
            anthropic_batch_submit.iter_requests(
                list(api_selection.tasks),
                {
                    "name": "claude-test",
                    "backend": "anthropic",
                    "max_output_tokens": 128,
                    "thinking_mode": "disabled",
                },
            )
        )
        self.assertEqual(
            [
                (str(task["name"]), str(item["item_id"]))
                for _, task, item, _ in openai_rows
            ],
            expected,
        )
        self.assertEqual(
            [
                (str(task["name"]), str(item["item_id"]))
                for _, task, item, _ in anthropic_rows
            ],
            expected,
        )
        self.assertEqual(
            [item["user_content"] for _, _, item, _ in openai_rows],
            [item["user_content"] for _, _, item, _ in anthropic_rows],
        )
        self.assertEqual(
            [item["gt"] for _, _, item, _ in openai_rows],
            [item["gt"] for _, _, item, _ in anthropic_rows],
        )

    def test_compact_manifest_fails_closed_on_component_or_design_drift(self):
        raw = yaml.safe_load(COMPACT_PATH.read_text())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "compact.yaml"

            wrong_prompt_hash = copy.deepcopy(raw)
            wrong_prompt_hash["tasks"][0]["prompt_sha256"] = "0" * 64
            path.write_text(yaml.safe_dump(wrong_prompt_hash, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "prompt_sha256 drift"):
                load_panel_manifest(path)

            missing_component = copy.deepcopy(raw)
            del missing_component["tasks"][0]["gold_labels_sha256"]
            path.write_text(yaml.safe_dump(missing_component, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "missing component hashes"):
                load_panel_manifest(path)

            wrong_global_keys = copy.deepcopy(raw)
            wrong_global_keys["task_item_keys_sha256"] = "0" * 64
            path.write_text(yaml.safe_dump(wrong_global_keys, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "task_item_keys_sha256 drift"):
                load_panel_manifest(path)

            legacy_promotion = copy.deepcopy(raw)
            legacy_promotion["promotion"] = {"distance_f1": 0.01}
            path.write_text(yaml.safe_dump(legacy_promotion, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "cannot define calibration or promotion"):
                load_panel_manifest(path)

            wrong_total = copy.deepcopy(raw)
            wrong_total["expected_items"] = 3999
            path.write_text(yaml.safe_dump(wrong_total, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "per-task expected_items"):
                load_panel_manifest(path)


class BroadManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.broad = load_panel_manifest(BROAD_PATH)
        cls.requests = benchmark_requests(cls.broad)

    def test_frozen_label_blind_18_by_200_scope(self):
        self.assertEqual(self.broad.panel_id, "frontier_broad_18")
        self.assertEqual((self.broad.expected_tasks, self.broad.expected_items), (18, 3600))
        self.assertEqual(set(self.broad.task_counts.values()), {200})
        self.assertEqual(len({request_key(row) for row in self.requests}), 3600)
        self.assertEqual(
            self.broad.task_item_keys_sha256,
            "d7317d91c9ba05736c8f90920513a9c7ed2f8d667d608250d099b91003f88c30",
        )
        self.assertEqual(
            self.broad.bootstrap["task_sensitivity"],
            {"task_bootstrap_iterations": 10000, "leave_one_task_out": True},
        )
        for spec in self.broad.task_specs:
            self.assertEqual(spec.selection_method, "hash_rank_sha256_v1")
            self.assertEqual(spec.selection_seed, 20260821)
            self.assertEqual(len(spec.source_fingerprint_sha256), 64)

    def test_broad_pilot_and_remainder_partition(self):
        pilot, remainder = split_pilot_remainder(self.broad)
        self.assertEqual((len(pilot), len(remainder)), (16, 3584))
        self.assertFalse(
            {request_key(row) for row in pilot}
            & {request_key(row) for row in remainder}
        )

    def test_sampling_seed_and_source_fingerprint_fail_closed(self):
        raw = yaml.safe_load(BROAD_PATH.read_text())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broad.yaml"
            wrong_seed = copy.deepcopy(raw)
            wrong_seed["tasks"][0]["selection"]["seed"] += 1
            path.write_text(yaml.safe_dump(wrong_seed, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "input fingerprint drift"):
                load_panel_manifest(path)

            wrong_source = copy.deepcopy(raw)
            wrong_source["tasks"][0]["source_fingerprint_sha256"] = "0" * 64
            path.write_text(yaml.safe_dump(wrong_source, sort_keys=False))
            with self.assertRaisesRegex(PanelManifestError, "source task fingerprint drift"):
                load_panel_manifest(path)


if __name__ == "__main__":
    unittest.main()
