from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import api_compact_finalize  # noqa: E402
import build_compact_frontier  # noqa: E402
import openai_batch_submit  # noqa: E402
from api_batch_common import (  # noqa: E402
    BatchIntegrityError,
    build_model_identity,
    file_sha256,
    load_task_selection,
)
from panel_manifest import load_panel_manifest  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402


PANEL = REPO / "experiments" / "frontier_compact_8.yaml"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(root: Path) -> dict[str, Path | str]:
    root.mkdir(parents=True, exist_ok=True)
    tasks = load_task_definitions(tasks_dir=REPO / "tasks")
    selection = load_task_selection(tasks, PANEL)
    tasks = list(selection.tasks)
    model = {
        "schema_version": 1,
        "name": "gpt-test-2026-08-20",
        "backend": "openai",
        "expected_response_model": "gpt-test-2026-08-20",
        "response_format_type": "json_schema",
        "max_output_tokens": 128,
    }
    model_manifest = root / "model.yaml"
    model_manifest.write_text(yaml.safe_dump(model, sort_keys=False))
    identity = build_model_identity(model, model_manifest)

    paths: dict[str, Path | str] = {
        "model": model["name"],
        "identity": identity.model_identity_sha256,
    }
    for stage in ("pilot", "remainder"):
        stage_dir = root / stage
        request_jsonl, request_manifest, _metadata = (
            openai_batch_submit.write_batch_files(
                tasks,
                model,
                stage_dir,
                stage,
                model_identity=identity,
                selection=selection,
                request_stage=stage,
                pilot_size=16,
            )
        )
        manifest_rows = _read_manifest(request_manifest)
        responses = stage_dir / "responses.jsonl"
        with responses.open("w") as handle:
            for row in manifest_rows:
                handle.write(
                    json.dumps(
                        {
                            "custom_id": row["custom_id"],
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "model": model["expected_response_model"],
                                    "choices": [
                                        {"message": {"content": "{}"}}
                                    ],
                                    "usage": {"completion_tokens": 1},
                                },
                            },
                        }
                    )
                    + "\n"
                )

        item_lookup: dict[tuple[str, str], dict] = {}
        for task in tasks:
            for item in task["loader"]():
                item_lookup[(str(task["name"]), str(item["item_id"]))] = item
        predictions: list[dict[str, str]] = []
        for row in manifest_rows:
            item = item_lookup[(row["task"], row["item_id"])]
            prediction = {
                "task": row["task"],
                "model": model["name"],
                "item_id": row["item_id"],
                "latency_s": "",
                "eval_count": "1",
                "parse_error": "",
                "raw_content_preview": "{}",
                "response_model": model["expected_response_model"],
                "model_identity_sha256": identity.model_identity_sha256,
                "panel_id": selection.panel_id,
                "panel_sha256": selection.panel_sha256,
                "request_stage": row["request_stage"],
                "request_index": row["request_index"],
            }
            for label, value in item["gt"].items():
                prediction[f"pred_{label}"] = str(value)
                prediction[f"gt_{label}"] = str(value)
            predictions.append(prediction)
        predictions_path = stage_dir / "predictions.csv"
        pd.DataFrame(predictions).to_csv(predictions_path, index=False)
        paths[f"{stage}_request_jsonl"] = request_jsonl
        paths[f"{stage}_request_manifest"] = request_manifest
        paths[f"{stage}_responses"] = responses
        paths[f"{stage}_predictions"] = predictions_path
    return paths


def _finalize(paths: dict[str, Path | str], output_dir: Path) -> dict:
    return api_compact_finalize.finalize_compact_api_evidence(
        panel_manifest_path=PANEL,
        pilot_request_jsonl_path=paths["pilot_request_jsonl"],
        pilot_request_manifest_path=paths["pilot_request_manifest"],
        pilot_responses_path=paths["pilot_responses"],
        pilot_predictions_path=paths["pilot_predictions"],
        remainder_request_jsonl_path=paths["remainder_request_jsonl"],
        remainder_request_manifest_path=paths["remainder_request_manifest"],
        remainder_responses_path=paths["remainder_responses"],
        remainder_predictions_path=paths["remainder_predictions"],
        output_dir=output_dir,
    )


class CompactApiFinalizeTests(unittest.TestCase):
    def test_finalizer_rejects_one_of_sixteen_malformed_pilot_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _build_fixture(root / "input")
            pilot_path = Path(paths["pilot_predictions"])
            pilot = pd.read_csv(pilot_path, dtype=str, keep_default_na=False)
            pilot.loc[0, "pred_entails"] = "2"
            pilot.to_csv(pilot_path, index=False)

            output = root / "final"
            with self.assertRaisesRegex(
                BatchIntegrityError, "pilot malformed rate exceeds 5%"
            ):
                _finalize(paths, output)
            self.assertFalse((output / "metadata.json").exists())

    def test_finalizer_marks_invalid_values_and_accepts_exactly_five_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _build_fixture(root / "input")
            remainder_path = Path(paths["remainder_predictions"])
            remainder = pd.read_csv(
                remainder_path, dtype=str, keep_default_na=False
            )
            self.assertTrue((remainder.loc[:199, "task"] == "burnham_polnli_entailment").all())
            remainder.loc[:199, "pred_entails"] = "2"
            remainder.to_csv(remainder_path, index=False)

            output = root / "final"
            metadata = _finalize(paths, output)
            self.assertEqual(metadata["pilot_malformed_count"], 0)
            self.assertEqual(metadata["pilot_malformed_rate"], 0.0)
            self.assertTrue(metadata["pilot_reliability_pass"])
            self.assertEqual(metadata["schema_invalid_response_count"], 200)
            self.assertEqual(metadata["malformed_count"], 200)
            self.assertEqual(metadata["malformed_rate"], 0.05)
            self.assertTrue(metadata["reliability_pass"])
            combined = pd.read_csv(
                output / "predictions.csv", dtype=str, keep_default_na=False
            )
            invalid = combined["parse_error"] == "invalid_schema_output"
            self.assertEqual(int(invalid.sum()), 200)
            self.assertEqual(
                set(combined.loc[invalid, "source_request_stage"]), {"remainder"}
            )

    def test_finalizer_writes_exact_builder_ready_evidence_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _build_fixture(root / "input")
            output = root / "final"
            metadata = _finalize(paths, output)

            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["pilot_items"], 16)
            self.assertEqual(metadata["remainder_items"], 3984)
            self.assertEqual(metadata["rows_written"], 4000)
            self.assertEqual(metadata["logical_responses"], 4000)
            self.assertEqual(
                metadata["returned_model_identity"], "gpt-test-2026-08-20"
            )
            self.assertEqual(
                metadata["predictions_sha256"], file_sha256(output / "predictions.csv")
            )
            self.assertEqual(
                metadata["requests_manifest_sha256"],
                file_sha256(output / "request_manifest.csv"),
            )

            manifest = pd.read_csv(
                output / "request_manifest.csv", dtype=str, keep_default_na=False
            )
            predictions = pd.read_csv(
                output / "predictions.csv", dtype=str, keep_default_na=False
            )
            self.assertEqual(len(manifest), 4000)
            self.assertEqual(len(predictions), 4000)
            self.assertEqual(
                manifest["request_index"].astype(int).tolist(), list(range(1, 4001))
            )
            self.assertEqual(set(manifest["request_stage"]), {"all"})
            self.assertEqual(
                manifest["source_request_stage"].tolist(),
                ["pilot"] * 16 + ["remainder"] * 3984,
            )
            self.assertEqual(
                list(zip(predictions["task"], predictions["item_id"])),
                list(zip(manifest["task"], manifest["item_id"])),
            )
            response_ids = [
                json.loads(line)["custom_id"]
                for line in (output / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual(response_ids, manifest["custom_id"].tolist())

            task_definitions = load_task_definitions(tasks_dir=REPO / "tasks")
            benchmark = load_panel_manifest(PANEL, task_definitions=task_definitions)
            checkpoint = {
                "checkpoint_id": "gpt_test_api",
                "model_id": paths["model"],
                "identity": "gpt-test-2026-08-20",
            }
            result = {
                "metadata_path": output / "metadata.json",
                "requests_manifest_path": output / "request_manifest.csv",
            }
            validated, metadata_hash = build_compact_frontier._validate_api_evidence(
                checkpoint,
                result,
                output / "predictions.csv",
                benchmark,
            )
            self.assertEqual(validated["rows_written"], 4000)
            self.assertEqual(metadata_hash, file_sha256(output / "metadata.json"))
            selected = build_compact_frontier._validate_compact_rows(
                predictions,
                benchmark,
                checkpoint,
            )
            self.assertEqual(len(selected), 4000)

    def test_finalizer_fails_closed_on_stage_identity_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            _build_fixture(baseline)

            def scenario(name: str) -> dict[str, Path | str]:
                target = root / name
                shutil.copytree(baseline, target)
                return {
                    "model": "gpt-test-2026-08-20",
                    "identity": "unused",
                    "pilot_request_jsonl": target / "pilot" / "pilot.jsonl",
                    "pilot_request_manifest": target
                    / "pilot"
                    / "pilot_manifest.csv",
                    "pilot_responses": target / "pilot" / "responses.jsonl",
                    "pilot_predictions": target / "pilot" / "predictions.csv",
                    "remainder_request_jsonl": target
                    / "remainder"
                    / "remainder.jsonl",
                    "remainder_request_manifest": target
                    / "remainder"
                    / "remainder_manifest.csv",
                    "remainder_responses": target / "remainder" / "responses.jsonl",
                    "remainder_predictions": target
                    / "remainder"
                    / "predictions.csv",
                }

            cases = []

            paths = scenario("wrong_stage")
            rows = _read_manifest(Path(paths["remainder_request_manifest"]))
            rows[0]["request_stage"] = "pilot"
            _write_manifest(Path(paths["remainder_request_manifest"]), rows)
            cases.append(("stage", paths))

            paths = scenario("wrong_model")
            frame = pd.read_csv(
                paths["remainder_predictions"], dtype=str, keep_default_na=False
            )
            frame.loc[0, "model"] = "wrong-model"
            frame.to_csv(paths["remainder_predictions"], index=False)
            cases.append(("model", paths))

            paths = scenario("wrong_panel")
            frame = pd.read_csv(
                paths["remainder_predictions"], dtype=str, keep_default_na=False
            )
            frame.loc[0, "panel_sha256"] = "0" * 64
            frame.to_csv(paths["remainder_predictions"], index=False)
            cases.append(("panel", paths))

            paths = scenario("wrong_request_hash")
            rows = _read_manifest(Path(paths["remainder_request_manifest"]))
            rows[0]["request_sha256"] = "0" * 64
            _write_manifest(Path(paths["remainder_request_manifest"]), rows)
            cases.append(("request hash", paths))

            paths = scenario("wrong_returned_identity")
            responses_path = Path(paths["remainder_responses"])
            response_lines = responses_path.read_text().splitlines()
            response = json.loads(response_lines[0])
            response["response"]["body"]["model"] = "gpt-wrong-2026-08-20"
            response_lines[0] = json.dumps(response)
            responses_path.write_text("\n".join(response_lines) + "\n")
            cases.append(("response snapshot", paths))

            paths = scenario("duplicate_prediction_key")
            pilot = pd.read_csv(
                paths["pilot_predictions"], dtype=str, keep_default_na=False
            )
            remainder = pd.read_csv(
                paths["remainder_predictions"], dtype=str, keep_default_na=False
            )
            remainder.loc[0, ["task", "item_id"]] = pilot.loc[
                0, ["task", "item_id"]
            ].values
            remainder.to_csv(paths["remainder_predictions"], index=False)
            cases.append(("prediction key", paths))

            for label, paths in cases:
                with self.subTest(label=label):
                    with self.assertRaises(BatchIntegrityError):
                        _finalize(paths, root / f"out_{label.replace(' ', '_')}")


if __name__ == "__main__":
    unittest.main()
