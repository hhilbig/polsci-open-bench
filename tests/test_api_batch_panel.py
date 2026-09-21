from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import anthropic_batch_collect  # noqa: E402
import anthropic_batch_submit  # noqa: E402
import api_batch_preflight  # noqa: E402
import anthropic_token_count  # noqa: E402
import api_pilot_qualification  # noqa: E402
import api_prediction_validation  # noqa: E402
import build_api_cost_report  # noqa: E402
import openai_batch_collect  # noqa: E402
import openai_batch_submit  # noqa: E402
from api_batch_common import (  # noqa: E402
    BatchIntegrityError,
    BatchTaskSelection,
    RequestManifest,
    build_model_identity,
    file_sha256,
    load_task_selection,
    load_request_manifest,
    validate_cost_approval,
    validate_aggregate_approval,
    validate_prepared_batch,
)
from model_registry import load_model_definition  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402


def fake_task(prompt_path: Path, n: int = 2) -> dict:
    items = [
        {
            "item_id": f"item_{index}",
            "user_content": f"Text {index}",
            "gt": {"relevant": index % 2},
        }
        for index in range(n)
    ]
    return {
        "name": "fake_relevance",
        "prompt_path": prompt_path,
        "label_kind": "binary",
        "label_key": "relevant",
        "labels": ["relevant"],
        "json_schema": {
            "type": "object",
            "properties": {"relevant": {"type": "integer", "enum": [0, 1]}},
            "required": ["relevant"],
            "additionalProperties": False,
        },
        "loader": lambda: [dict(item) for item in items],
    }


def fake_selection(task: dict) -> BatchTaskSelection:
    return BatchTaskSelection(
        tasks=(task,),
        panel_id="test_panel",
        panel_sha256="a" * 64,
        benchmark_commit="b" * 40,
        task_item_keys_sha256="c" * 64,
        expected_tasks=1,
        expected_items=len(task["loader"]()),
        task_fingerprints={task["name"]: "d" * 64},
    )


def request_manifest_for(
    task: dict,
    *,
    provider: str,
    model: str,
    expected_response_model: str = "",
) -> RequestManifest:
    rows = []
    lookup = {}
    row_by_id = {}
    for index, item in enumerate(task["loader"](), start=1):
        custom_id = f"req_{index:06d}"
        row = {
            "custom_id": custom_id,
            "task": task["name"],
            "model": model,
            "item_id": str(item["item_id"]),
            "provider": provider,
            "requested_model": model,
            "expected_response_model": expected_response_model,
            "model_identity_sha256": "e" * 64,
            "panel_id": "test_panel",
            "panel_sha256": "a" * 64,
            "task_item_keys_sha256": "c" * 64,
        }
        rows.append(row)
        lookup[custom_id] = (task, item)
        row_by_id[custom_id] = row
    return RequestManifest(
        rows=tuple(rows),
        row_by_custom_id=row_by_id,
        lookup=lookup,
        provider=provider,
        requested_model=model,
        expected_response_model=expected_response_model,
        model_identity_sha256="e" * 64,
        panel_id="test_panel",
        panel_sha256="a" * 64,
        task_item_keys_sha256="c" * 64,
    )


class BatchPreparationTests(unittest.TestCase):
    def test_openai_request_manifest_carries_panel_and_model_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt)
            model = {
                "name": "gpt-test",
                "backend": "openai",
                "provider": None,
                "reasoning_effort": "medium",
                "response_format_type": "json_schema",
            }
            identity = build_model_identity(model)
            selection = fake_selection(task)
            jsonl_path, manifest_path, meta = openai_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "prepared",
                model_identity=identity,
                selection=selection,
            )

            with manifest_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["panel_sha256"] == "a" * 64 for row in rows))
            self.assertTrue(
                all(
                    row["model_identity_sha256"]
                    == identity.model_identity_sha256
                    for row in rows
                )
            )
            self.assertTrue(all(row["request_sha256"] for row in rows))
            self.assertEqual(meta["task_item_keys_sha256"], "c" * 64)
            prepared = validate_prepared_batch(jsonl_path, manifest_path)
            self.assertEqual(len(prepared.requests), 2)
            self.assertEqual(prepared.panel_sha256, "a" * 64)

    def test_anthropic_preparation_uses_same_provenance_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt)
            model = {"name": "claude-test", "backend": "anthropic"}
            identity = build_model_identity(model)
            jsonl_path, manifest_path, meta = anthropic_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "prepared",
                model_identity=identity,
                selection=fake_selection(task),
            )
            prepared = validate_prepared_batch(jsonl_path, manifest_path)
            self.assertEqual(prepared.provider, "anthropic")
            self.assertEqual(prepared.requested_model, "claude-test")
            self.assertEqual(meta["panel_id"], "test_panel")

    def test_anthropic_request_pins_declared_thinking_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt)
            params = anthropic_batch_submit.build_message_params(
                {
                    "name": "claude-sonnet-5",
                    "backend": "anthropic",
                    "thinking_mode": "disabled",
                    "max_output_tokens": 128,
                },
                task,
                prompt.read_text(),
                "Text",
            )
            self.assertEqual(params["thinking"], {"type": "disabled"})
            self.assertEqual(params["max_tokens"], 128)

    def test_real_frontier_panel_resolves_exact_api_scope(self):
        tasks = load_task_definitions(tasks_dir=REPO / "tasks")
        selection = load_task_selection(
            tasks,
            REPO / "experiments" / "frontier_panel_18.yaml",
        )
        self.assertEqual(selection.expected_tasks, 18)
        self.assertEqual(selection.expected_items, 8793)
        self.assertEqual(len(selection.tasks), 18)
        self.assertEqual(
            selection.panel_sha256,
            "01c7ba09b7cc82ab0ec785d6d8ef5390a0508058a3fb232cb6b7923bfa2a0824",
        )

    def test_compact_api_roster_has_exact_seven_pinned_configs(self):
        roster_path = REPO / "experiments" / "frontier_compact8_api_20260820.yaml"
        roster = yaml.safe_load(roster_path.read_text())
        checkpoints = list(roster["checkpoints"].items())
        self.assertEqual(len(checkpoints), 7)
        self.assertEqual(roster["expected_request_count"], 4000)
        self.assertEqual(roster["pilot_size"], 16)
        self.assertEqual(roster["remainder_size"], 3984)
        self.assertEqual(roster["hard_budget_usd"], 50)
        self.assertEqual(
            roster["drop_policy"],
            ["gpt_5_4_2026_03_05_api", "gpt_4o_2024_11_20_api"],
        )
        dates = [state["snapshot_release_date"] for _key, state in checkpoints]
        self.assertEqual(dates, sorted(dates))
        for checkpoint_id, state in checkpoints:
            manifest_path = REPO / state["model_manifest"]
            model = load_model_definition(manifest_path)
            self.assertEqual(model["name"], state["model_id"], checkpoint_id)
            self.assertEqual(model["max_output_tokens"], 128)
            self.assertTrue(state["staged_preflight_required"])
            self.assertEqual(len(build_model_identity(model, manifest_path).model_identity_sha256), 64)
        self.assertEqual(
            load_model_definition(
                REPO
                / roster["checkpoints"]["gpt_5_2025_08_07_api"]["model_manifest"]
            )["reasoning_effort"],
            "minimal",
        )
        for checkpoint_id in [
            "gpt_5_2_2025_12_11_api",
            "gpt_5_4_2026_03_05_api",
        ]:
            model = load_model_definition(
                REPO / roster["checkpoints"][checkpoint_id]["model_manifest"]
            )
            self.assertEqual(model["reasoning_effort"], "none")
        claude = load_model_definition(
            REPO / roster["checkpoints"]["claude_sonnet_5_api"]["model_manifest"]
        )
        self.assertEqual(claude["thinking_mode"], "disabled")

    def test_fixed_pilot_and_remainder_partition_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt, n=20)
            selection = fake_selection(task)
            model = {
                "name": "gpt-test",
                "backend": "openai",
                "response_format_type": "json_schema",
                "max_output_tokens": 128,
            }
            identity = build_model_identity(model)
            pilot_jsonl, pilot_manifest, pilot_meta = (
                openai_batch_submit.write_batch_files(
                    [task],
                    model,
                    root,
                    "pilot",
                    model_identity=identity,
                    selection=selection,
                    request_stage="pilot",
                )
            )
            remainder_jsonl, remainder_manifest, remainder_meta = (
                openai_batch_submit.write_batch_files(
                    [task],
                    model,
                    root,
                    "remainder",
                    model_identity=identity,
                    selection=selection,
                    request_stage="remainder",
                )
            )
            self.assertEqual(pilot_meta["requests"], 16)
            self.assertEqual(remainder_meta["requests"], 4)
            pilot = validate_prepared_batch(pilot_jsonl, pilot_manifest)
            remainder = validate_prepared_batch(remainder_jsonl, remainder_manifest)
            self.assertEqual(pilot.request_stage, "pilot")
            self.assertEqual(remainder.request_stage, "remainder")
            pilot_ids = {request["custom_id"] for request in pilot.requests}
            remainder_ids = {request["custom_id"] for request in remainder.requests}
            self.assertFalse(pilot_ids & remainder_ids)
            self.assertEqual(len(pilot_ids | remainder_ids), 20)
            loaded = load_request_manifest(
                pilot_manifest,
                [task],
                provider="openai",
                selection=selection,
                require_enhanced=True,
                require_complete=True,
            )
            self.assertEqual(len(loaded.rows), 16)
            first_request = json.loads(pilot_jsonl.read_text().splitlines()[0])
            self.assertEqual(first_request["body"]["max_tokens"], 128)


class OfflinePreflightTests(unittest.TestCase):
    def _prepared_files(self, root: Path):
        prompt = root / "prompt.txt"
        prompt.write_text("Classify relevance.")
        task = fake_task(prompt)
        model = {
            "name": "gpt-test",
            "backend": "openai",
            "provider": None,
            "reasoning_effort": "medium",
            "response_format_type": "json_schema",
        }
        identity = build_model_identity(model)
        jsonl_path, manifest_path, _ = openai_batch_submit.write_batch_files(
            [task],
            model,
            root,
            "prepared",
            model_identity=identity,
            selection=fake_selection(task),
        )
        pricing_path = root / "pricing.yaml"
        pricing_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "currency": "USD",
                    "models": {
                        "gpt-test": {
                            "provider": "openai",
                            "model_identity_sha256": identity.model_identity_sha256,
                            "batch_input_usd_per_million_tokens": 1,
                            "batch_output_usd_per_million_tokens": 2,
                            "input_token_safety_multiplier": 1.1,
                            "estimated_output_tokens_per_request": 25,
                            "tokenizer": {
                                "type": "tiktoken",
                                "encoding": "o200k_base",
                            },
                            "source_url": "https://example.test/official-pricing",
                            "effective_date": "2026-08-20",
                        }
                    },
                },
                sort_keys=False,
            )
        )
        return jsonl_path, manifest_path, pricing_path

    def test_preflight_counts_exact_requests_and_never_constructs_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path, manifest_path, pricing_path = self._prepared_files(root)
            with mock.patch.object(openai_batch_submit, "OpenAI") as client:
                report = api_batch_preflight.build_preflight_report(
                    jsonl_path,
                    manifest_path,
                    pricing_path,
                )
            client.assert_not_called()
            self.assertTrue(report["offline_preflight"])
            self.assertEqual(
                report["token_estimator"]["method"],
                "exact_jsonl_request_line",
            )
            self.assertEqual(report["request_count"], 2)
            self.assertEqual(report["task_counts"], {"fake_relevance": 2})
            self.assertGreater(report["estimated_input_tokens"], 0)
            self.assertEqual(report["estimated_output_tokens"], 50)
            self.assertEqual(report["maximum_output_tokens"], 2048)
            self.assertEqual(report["maximum_output_cost_usd"], "0.004096")

    def test_cost_approval_rejects_low_cap_and_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path, manifest_path, pricing_path = self._prepared_files(root)
            report = api_batch_preflight.build_preflight_report(
                jsonl_path,
                manifest_path,
                pricing_path,
            )
            report_path = root / "preflight.json"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(BatchIntegrityError, "exceeds approved budget"):
                validate_cost_approval(
                    report_path,
                    jsonl_path,
                    manifest_path,
                    "0",
                )
            validated = validate_cost_approval(
                report_path,
                jsonl_path,
                manifest_path,
                report["maximum_cost_usd"],
            )
            self.assertEqual(validated["request_count"], 2)

            with jsonl_path.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(
                BatchIntegrityError, "request_jsonl_sha256 does not match"
            ):
                validate_cost_approval(
                    report_path,
                    jsonl_path,
                    manifest_path,
                    report["maximum_cost_usd"],
                )

    def test_anthropic_preflight_requires_exact_official_count_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt)
            model = {"name": "claude-test", "backend": "anthropic"}
            identity = build_model_identity(model)
            jsonl_path, manifest_path, _ = anthropic_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "prepared",
                model_identity=identity,
                selection=fake_selection(task),
            )
            with manifest_path.open(newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            pricing_path = root / "pricing.yaml"
            pricing_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "currency": "USD",
                        "models": {
                            "claude-test": {
                                "provider": "anthropic",
                                "model_identity_sha256": identity.model_identity_sha256,
                                "batch_input_usd_per_million_tokens": 1,
                                "batch_output_usd_per_million_tokens": 2,
                                "input_token_safety_multiplier": 1,
                                "estimated_output_tokens_per_request": 25,
                                "tokenizer": {
                                    "type": "anthropic_token_count",
                                    "endpoint": "messages.count_tokens",
                                    "sdk_version": "test",
                                },
                                "source_url": "https://example.test/official-pricing",
                                "effective_date": "2026-08-20",
                            }
                        },
                    },
                    sort_keys=False,
                )
            )
            cache = {
                "schema_version": 1,
                "provider": "anthropic",
                "requested_model": "claude-test",
                "request_jsonl_sha256": file_sha256(jsonl_path),
                "request_manifest_sha256": file_sha256(manifest_path),
                "endpoint": "messages.count_tokens",
                "completed_at": "2026-08-20T00:00:00Z",
                "sdk_version": "test",
                "counts": [
                    {
                        "custom_id": row["custom_id"],
                        "request_sha256": row["request_sha256"],
                        "input_tokens": count,
                    }
                    for row, count in zip(manifest_rows, [11, 13], strict=True)
                ],
            }
            cache_path = root / "counts.json"
            cache_path.write_text(json.dumps(cache))
            report = api_batch_preflight.build_preflight_report(
                jsonl_path,
                manifest_path,
                pricing_path,
                token_count_cache_path=cache_path,
            )
            self.assertFalse(report["offline_preflight"])
            self.assertEqual(report["estimated_input_tokens"], 24)
            self.assertEqual(
                report["token_estimator"]["method"],
                "official_messages_count_tokens_per_exact_request",
            )
            self.assertEqual(
                report["token_count_provenance"]["sha256"],
                file_sha256(cache_path),
            )

            cache["counts"][0]["request_sha256"] = "0" * 64
            cache_path.write_text(json.dumps(cache))
            with self.assertRaisesRegex(BatchIntegrityError, "request hash mismatch"):
                api_batch_preflight.build_preflight_report(
                    jsonl_path,
                    manifest_path,
                    pricing_path,
                    token_count_cache_path=cache_path,
                )

    def test_anthropic_count_projection_drops_only_output_cap(self):
        params = {
            "model": "claude-test",
            "max_tokens": 2000,
            "system": "Classify.",
            "messages": [{"role": "user", "content": "Text"}],
            "tools": [{"name": "classify", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "classify"},
        }
        counted = anthropic_token_count.token_count_params(params)
        self.assertNotIn("max_tokens", counted)
        self.assertEqual(
            set(counted),
            {"model", "system", "messages", "tools", "tool_choice"},
        )
        with self.assertRaisesRegex(BatchIntegrityError, "unknown fields"):
            anthropic_token_count.token_count_params({**params, "new_field": True})

    def test_aggregate_cost_arithmetic_and_exact_approval_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            availability = {
                "schema_version": 1,
                "checkpoints": {
                    "a": {
                        "model_id": "model-a",
                        "provider": "OpenAI",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "immutable_dated_snapshot",
                        "preflight_required": True,
                    },
                    "b": {
                        "model_id": "model-b",
                        "provider": "Anthropic",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "fixed_canonical_snapshot",
                        "preflight_required": True,
                    },
                    "mutable": {
                        "model_id": "alias",
                        "provider": "OpenAI",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "mutable_alias",
                        "preflight_required": False,
                    },
                },
            }
            availability_path = root / "availability.yaml"
            availability_path.write_text(yaml.safe_dump(availability, sort_keys=False))
            incomplete = build_api_cost_report.build_aggregate_report(
                availability_path,
                root / "missing_preflights",
                expected_request_count=2,
            )
            self.assertFalse(incomplete["approval_ready"])
            self.assertEqual(incomplete["approval_status"], "incomplete_preflight")
            self.assertEqual(incomplete["missing_preflight_reports"], ["a", "b"])
            for checkpoint_id, provider, model_id, estimated, maximum in [
                ("a", "openai", "model-a", "1.250000", "2.500000"),
                ("b", "anthropic", "model-b", "3.750000", "4.500000"),
            ]:
                directory = root / checkpoint_id
                directory.mkdir()
                (directory / "preflight.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "provider": provider,
                            "requested_model": model_id,
                            "request_count": 2,
                            "estimated_input_tokens": 10,
                            "maximum_input_tokens": 11,
                            "estimated_output_tokens": 4,
                            "maximum_output_tokens": 8,
                            "estimated_total_cost_usd": estimated,
                            "maximum_cost_usd": maximum,
                        }
                    )
                )
            aggregate = build_api_cost_report.build_aggregate_report(
                availability_path,
                root,
                expected_request_count=2,
            )
            self.assertTrue(aggregate["approval_ready"])
            self.assertEqual(
                aggregate["approval_status"], "pending_explicit_user_approval"
            )
            self.assertEqual(aggregate["overall_estimated_cost_usd"], "5.000000")
            self.assertEqual(aggregate["overall_maximum_cost_usd"], "7.000000")
            self.assertEqual(aggregate["mutable_exclusions"], ["mutable"])
            markdown_path = root / "aggregate.md"
            build_api_cost_report.write_markdown(markdown_path, aggregate)
            markdown = markdown_path.read_text()
            self.assertIn("## Provider subtotals", markdown)
            self.assertIn("| anthropic | 1 | 3.750000 | 4.500000 |", markdown)
            self.assertIn("Other inaccessible checkpoints: none", markdown)

            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text(json.dumps(aggregate))
            approval_path = root / "approval.json"
            approval_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approval_status": "approved",
                        "aggregate_report_sha256": file_sha256(aggregate_path),
                        "overall_maximum_cost_usd": "7.000000",
                        "approved_model_ids": ["model-a", "model-b"],
                    }
                )
            )
            preflight_path = root / "a" / "preflight.json"
            validated = validate_aggregate_approval(
                aggregate_path,
                approval_path,
                preflight_path,
                {
                    "_validated_cost_approval": True,
                    "requested_model": "model-a",
                    "maximum_cost_usd": "2.500000",
                },
            )
            self.assertTrue(validated["_validated_aggregate_approval"])

            approval = json.loads(approval_path.read_text())
            approval["overall_maximum_cost_usd"] = "6.999999"
            approval_path.write_text(json.dumps(approval))
            with self.assertRaisesRegex(BatchIntegrityError, "does not match"):
                validate_aggregate_approval(
                    aggregate_path,
                    approval_path,
                    preflight_path,
                    {
                        "_validated_cost_approval": True,
                        "requested_model": "model-a",
                        "maximum_cost_usd": "2.500000",
                    },
                )

    def test_staged_aggregate_applies_exact_drop_policy_and_hard_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            availability = {
                "schema_version": 1,
                "report_type": "frontier_compact8_api_cost_approval",
                "expected_request_count": 20,
                "pilot_size": 4,
                "hard_budget_usd": 50,
                "drop_policy": ["drop_first", "drop_second"],
                "checkpoints": {
                    "keep_a": {
                        "model_id": "model-a",
                        "provider": "OpenAI",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "immutable_dated_snapshot",
                        "preflight_required": True,
                        "staged_preflight_required": True,
                    },
                    "drop_first": {
                        "model_id": "model-b",
                        "provider": "OpenAI",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "immutable_dated_snapshot",
                        "preflight_required": True,
                        "staged_preflight_required": True,
                    },
                    "keep_c": {
                        "model_id": "model-c",
                        "provider": "Anthropic",
                        "access_status": "accessible",
                        "lifecycle_status": "active",
                        "identity_class": "fixed_canonical_snapshot",
                        "preflight_required": True,
                        "staged_preflight_required": True,
                    },
                },
            }
            availability_path = root / "availability.yaml"
            availability_path.write_text(yaml.safe_dump(availability, sort_keys=False))
            maxima = {"keep_a": 30, "drop_first": 25, "keep_c": 5}
            providers = {"keep_a": "openai", "drop_first": "openai", "keep_c": "anthropic"}
            model_ids = {"keep_a": "model-a", "drop_first": "model-b", "keep_c": "model-c"}
            for checkpoint_id in availability["checkpoints"]:
                for stage, count, share in [("pilot", 4, Decimal("0.2")), ("remainder", 16, Decimal("0.8"))]:
                    directory = root / checkpoint_id / stage
                    directory.mkdir(parents=True)
                    maximum = Decimal(maxima[checkpoint_id]) * share
                    report = {
                        "schema_version": 1,
                        "provider": providers[checkpoint_id],
                        "requested_model": model_ids[checkpoint_id],
                        "model_identity_sha256": checkpoint_id,
                        "panel_id": "frontier_compact_8",
                        "panel_sha256": "a" * 64,
                        "task_item_keys_sha256": "b" * 64,
                        "pricing_manifest_sha256": "c" * 64,
                        "request_stage": stage,
                        "planned_request_count": 20,
                        "pilot_size": 4,
                        "request_count": count,
                        "estimated_input_tokens": count * 10,
                        "maximum_input_tokens": count * 11,
                        "estimated_output_tokens": count * 2,
                        "maximum_output_tokens": count * 4,
                        "estimated_total_cost_usd": str(maximum / 2),
                        "maximum_cost_usd": str(maximum),
                    }
                    (directory / "preflight.json").write_text(json.dumps(report))
            aggregate = build_api_cost_report.build_aggregate_report(
                availability_path,
                root,
            )
            self.assertTrue(aggregate["approval_ready"])
            self.assertTrue(aggregate["budget_compliant"])
            self.assertEqual(aggregate["overall_maximum_cost_usd"], "35.000000")
            self.assertEqual(
                [row["checkpoint_id"] for row in aggregate["budget_exclusions"]],
                ["drop_first"],
            )
            self.assertEqual(
                [row["checkpoint_id"] for row in aggregate["models"]],
                ["keep_a", "keep_c"],
            )
            self.assertTrue(
                all(len(row["preflight_reports"]) == 2 for row in aggregate["models"])
            )

            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text(json.dumps(aggregate))
            approval_path = root / "approval.json"
            approval_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approval_status": "approved",
                        "aggregate_report_sha256": file_sha256(aggregate_path),
                        "overall_maximum_cost_usd": "35.000000",
                        "approved_model_ids": ["model-a", "model-c"],
                    }
                )
            )
            pilot_preflight = root / "keep_a" / "pilot" / "preflight.json"
            validated = validate_aggregate_approval(
                aggregate_path,
                approval_path,
                pilot_preflight,
                {
                    "_validated_cost_approval": True,
                    "requested_model": "model-a",
                    "maximum_cost_usd": "6.0",
                    "panel_id": "frontier_compact_8",
                },
            )
            self.assertTrue(validated["_validated_aggregate_approval"])

            aggregate["hard_budget_usd"] = "50.000001"
            aggregate["approval_ready"] = True
            aggregate_path.write_text(json.dumps(aggregate))
            approval = json.loads(approval_path.read_text())
            approval["aggregate_report_sha256"] = file_sha256(aggregate_path)
            approval_path.write_text(json.dumps(approval))
            with self.assertRaisesRegex(BatchIntegrityError, "exceeds \\$50"):
                validate_aggregate_approval(
                    aggregate_path,
                    approval_path,
                    pilot_preflight,
                    {
                        "_validated_cost_approval": True,
                        "requested_model": "model-a",
                        "maximum_cost_usd": "6.0",
                        "panel_id": "frontier_compact_8",
                    },
                )

    def test_pilot_qualification_gates_remainder_and_retains_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt, n=20)
            selection = fake_selection(task)
            model = {
                "name": "gpt-test",
                "backend": "openai",
                "response_format_type": "json_schema",
                "max_output_tokens": 128,
            }
            identity = build_model_identity(model)
            pilot_jsonl, pilot_manifest, _ = openai_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "pilot",
                model_identity=identity,
                selection=selection,
                request_stage="pilot",
            )
            remainder_jsonl, remainder_manifest, _ = openai_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "remainder",
                model_identity=identity,
                selection=selection,
                request_stage="remainder",
            )
            with pilot_manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            predictions_path = root / "pilot_predictions.csv"
            predictions = [
                {
                    "task": row["task"],
                    "item_id": row["item_id"],
                    "parse_error": None,
                    "pred_relevant": 0,
                    "model_identity_sha256": identity.model_identity_sha256,
                    "panel_sha256": selection.panel_sha256,
                }
                for row in rows
            ]
            with predictions_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            report = api_pilot_qualification.build_pilot_qualification(
                pilot_jsonl,
                pilot_manifest,
                predictions_path,
                task_definitions=[task],
            )
            self.assertEqual(report["qualification_status"], "passed")
            report_path = root / "pilot_qualification.json"
            report_path.write_text(json.dumps(report))
            validated = api_pilot_qualification.validate_pilot_qualification(
                report_path,
                validate_prepared_batch(remainder_jsonl, remainder_manifest),
                task_definitions=[task],
            )
            self.assertTrue(validated["_validated_pilot_qualification"])
            tampered = dict(report)
            tampered["malformed_count"] = 1
            report_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(BatchIntegrityError, "recomputed evidence"):
                api_pilot_qualification.validate_pilot_qualification(
                    report_path,
                    validate_prepared_batch(remainder_jsonl, remainder_manifest),
                    task_definitions=[task],
                )

            predictions[0]["parse_error"] = "malformed"
            with predictions_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            failed = api_pilot_qualification.build_pilot_qualification(
                pilot_jsonl,
                pilot_manifest,
                predictions_path,
                task_definitions=[task],
            )
            self.assertEqual(failed["malformed_count"], 1)
            self.assertEqual(failed["request_count"], 16)
            self.assertEqual(failed["qualification_status"], "failed")

    def test_schema_invalid_value_obeys_exact_five_percent_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify relevance.")
            task = fake_task(prompt, n=40)
            selection = fake_selection(task)
            model = {
                "name": "gpt-test",
                "backend": "openai",
                "response_format_type": "json_schema",
                "max_output_tokens": 128,
            }
            identity = build_model_identity(model)
            pilot_jsonl, pilot_manifest, _ = openai_batch_submit.write_batch_files(
                [task],
                model,
                root,
                "pilot20",
                model_identity=identity,
                selection=selection,
                request_stage="pilot",
                pilot_size=20,
            )
            with pilot_manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            predictions = pd.DataFrame(
                [
                    {
                        "task": row["task"],
                        "item_id": row["item_id"],
                        "parse_error": "",
                        "pred_relevant": 0,
                        "model_identity_sha256": identity.model_identity_sha256,
                        "panel_sha256": selection.panel_sha256,
                    }
                    for row in rows
                ]
            )
            predictions.loc[0, "pred_relevant"] = 2
            predictions_path = root / "pilot20_predictions.csv"
            predictions.to_csv(predictions_path, index=False)
            at_boundary = api_pilot_qualification.build_pilot_qualification(
                pilot_jsonl,
                pilot_manifest,
                predictions_path,
                task_definitions=[task],
            )
            self.assertEqual(at_boundary["schema_invalid_response_count"], 1)
            self.assertEqual(at_boundary["malformed_count"], 1)
            self.assertEqual(at_boundary["malformed_rate"], 0.05)
            self.assertEqual(at_boundary["qualification_status"], "passed")

            predictions.loc[1, "pred_relevant"] = -1
            predictions.to_csv(predictions_path, index=False)
            above_boundary = api_pilot_qualification.build_pilot_qualification(
                pilot_jsonl,
                pilot_manifest,
                predictions_path,
                task_definitions=[task],
            )
            self.assertEqual(above_boundary["schema_invalid_response_count"], 2)
            self.assertEqual(above_boundary["malformed_rate"], 0.1)
            self.assertEqual(above_boundary["qualification_status"], "failed")

    def test_one_invalid_multi_binary_label_marks_whole_response_malformed(self):
        frame = pd.DataFrame(
            [
                {
                    "task": "multi",
                    "parse_error": "",
                    "pred_a": 1,
                    "pred_b": 7,
                },
                {
                    "task": "multi",
                    "parse_error": "",
                    "pred_a": 0,
                    "pred_b": 1,
                },
            ]
        )
        malformed, invalid = api_prediction_validation.prediction_malformed_masks(
            frame,
            [
                {
                    "name": "multi",
                    "label_kind": "multi_binary",
                    "label_key": None,
                    "labels": ["a", "b"],
                }
            ],
        )
        self.assertEqual(malformed.tolist(), [True, False])
        self.assertEqual(invalid.tolist(), [True, False])

    def test_provider_submit_helpers_fail_before_client_use_without_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            path.write_text("")
            with mock.patch.object(openai_batch_submit, "OpenAI") as client:
                with self.assertRaisesRegex(
                    BatchIntegrityError, "validated cost approval"
                ):
                    openai_batch_submit.submit_batch(path, {})
            client.assert_not_called()

            anthropic_client = mock.Mock()
            with self.assertRaisesRegex(
                BatchIntegrityError, "validated cost approval"
            ):
                anthropic_batch_submit.submit_batch(anthropic_client, path)
            anthropic_client.messages.batches.create.assert_not_called()


class CollectorIntegrityTests(unittest.TestCase):
    def test_openai_rejects_wrong_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("Classify.")
            task = fake_task(prompt, n=1)
            manifest = request_manifest_for(
                task,
                provider="openai",
                model="gpt-test",
                expected_response_model="gpt-test-2026-08-01",
            )
            path.write_text(
                json.dumps(
                    {
                        "custom_id": "req_000001",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "model": "gpt-test-2026-08-02",
                                "choices": [
                                    {"message": {"content": '{"relevant": 1}'}}
                                ],
                                "usage": {"completion_tokens": 4},
                            },
                        },
                    }
                )
                + "\n"
            )
            with mock.patch.object(
                openai_batch_collect,
                "_load_collection_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(
                    BatchIntegrityError, "response snapshot mismatch"
                ):
                    openai_batch_collect.collect(path, "unused")

    def test_openai_rejects_duplicate_and_missing_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify.")
            task = fake_task(prompt, n=2)
            manifest = request_manifest_for(
                task,
                provider="openai",
                model="gpt-test",
            )
            result = {
                "custom_id": "req_000001",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "gpt-test",
                        "choices": [{"message": {"content": '{"relevant": 1}'}}],
                        "usage": {"completion_tokens": 4},
                    },
                },
            }
            duplicate_path = root / "duplicate.jsonl"
            duplicate_path.write_text(
                json.dumps(result) + "\n" + json.dumps(result) + "\n"
            )
            missing_path = root / "missing.jsonl"
            missing_path.write_text(json.dumps(result) + "\n")
            with mock.patch.object(
                openai_batch_collect,
                "_load_collection_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(BatchIntegrityError, "Duplicate custom_id"):
                    openai_batch_collect.collect(duplicate_path, "unused")
                with self.assertRaisesRegex(ValueError, "missing 1 custom_ids"):
                    openai_batch_collect.collect(missing_path, "unused")

    def test_anthropic_rejects_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify.")
            task = fake_task(prompt, n=1)
            manifest = request_manifest_for(
                task,
                provider="anthropic",
                model="claude-test",
            )
            result_path = root / "results.jsonl"
            result_path.write_text(
                json.dumps(
                    {
                        "custom_id": "req_000001",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "model": "claude-other",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "name": "classify",
                                        "input": {"relevant": 1},
                                    }
                                ],
                                "usage": {"output_tokens": 4},
                            },
                        },
                    }
                )
                + "\n"
            )
            with mock.patch.object(
                anthropic_batch_collect,
                "_load_collection_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(
                    BatchIntegrityError, "response model mismatch"
                ):
                    anthropic_batch_collect.collect(
                        result_path,
                        "unused",
                        None,
                    )

    def test_collector_rejects_extra_custom_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("Classify.")
            task = fake_task(prompt, n=1)
            manifest = request_manifest_for(
                task,
                provider="openai",
                model="gpt-test",
            )
            path = root / "extra.jsonl"
            path.write_text(json.dumps({"custom_id": "not_in_manifest"}) + "\n")
            with mock.patch.object(
                openai_batch_collect,
                "_load_collection_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(KeyError, "Unknown custom_id"):
                    openai_batch_collect.collect(path, "unused")


if __name__ == "__main__":
    unittest.main()
