#!/usr/bin/env python3
"""Build and validate the compact API parser-pilot qualification record."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from api_batch_common import (
    BatchIntegrityError,
    PreparedBatch,
    file_sha256,
    load_task_selection,
    validate_prepared_batch,
)
from api_prediction_validation import prediction_malformed_masks
from task_registry import load_task_definitions


REPO = Path(__file__).resolve().parent.parent
DEFAULT_COMPACT_MANIFEST = REPO / "experiments" / "frontier_compact_8.yaml"
MALFORMED_LIMIT = 0.05


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resolve_task_schema(
    prepared: PreparedBatch,
    *,
    panel_manifest_path: str | Path | None,
    tasks_dir: str | Path,
    task_definitions: Sequence[Mapping[str, Any]] | None,
) -> Sequence[Mapping[str, Any]]:
    if task_definitions is not None:
        return task_definitions
    manifest_path = (
        Path(panel_manifest_path)
        if panel_manifest_path is not None
        else DEFAULT_COMPACT_MANIFEST
    )
    loaded_tasks = load_task_definitions(tasks_dir=tasks_dir)
    selection = load_task_selection(loaded_tasks, manifest_path)
    if (
        selection.panel_id != prepared.panel_id
        or selection.panel_sha256 != prepared.panel_sha256
        or selection.task_item_keys_sha256 != prepared.task_item_keys_sha256
        or selection.expected_items != prepared.planned_request_count
    ):
        raise BatchIntegrityError(
            "pilot task schema does not match the prepared frozen benchmark"
        )
    return selection.tasks


def build_pilot_qualification(
    request_jsonl_path: str | Path,
    request_manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    panel_manifest_path: str | Path | None = None,
    tasks_dir: str | Path = REPO / "tasks",
    task_definitions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared = validate_prepared_batch(request_jsonl_path, request_manifest_path)
    if prepared.request_stage != "pilot":
        raise BatchIntegrityError("pilot qualification requires a pilot-stage batch")
    if len(prepared.requests) != prepared.pilot_size:
        raise BatchIntegrityError("pilot request count does not match pilot_size")

    predictions_file = Path(predictions_path)
    predictions = pd.read_csv(predictions_file, low_memory=False, dtype={"item_id": str})
    required = {"task", "item_id", "parse_error", "model_identity_sha256", "panel_sha256"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise BatchIntegrityError(f"pilot predictions are missing columns: {missing}")
    prediction_keys = [
        (str(row.task), str(row.item_id)) for row in predictions.itertuples()
    ]
    expected_keys = [
        (str(row["task"]), str(row["item_id"])) for row in prepared.manifest_rows
    ]
    if len(set(prediction_keys)) != len(prediction_keys):
        raise BatchIntegrityError("pilot predictions have duplicate task/item keys")
    if set(prediction_keys) != set(expected_keys):
        raise BatchIntegrityError(
            "pilot predictions do not match the exact pilot request keys"
        )
    if predictions["model_identity_sha256"].isna().any() or set(
        predictions["model_identity_sha256"].astype(str)
    ) != {prepared.model_identity_sha256}:
        raise BatchIntegrityError("pilot prediction model identity mismatch")
    if predictions["panel_sha256"].isna().any() or set(
        predictions["panel_sha256"].astype(str)
    ) != {prepared.panel_sha256}:
        raise BatchIntegrityError("pilot prediction compact-manifest hash mismatch")

    task_definitions = _resolve_task_schema(
        prepared,
        panel_manifest_path=panel_manifest_path,
        tasks_dir=tasks_dir,
        task_definitions=task_definitions,
    )
    malformed, schema_invalid = prediction_malformed_masks(
        predictions, task_definitions
    )
    malformed_count = int(malformed.sum())
    schema_invalid_count = int(schema_invalid.sum())
    request_count = len(predictions)
    malformed_rate = malformed_count / request_count
    passed = malformed_rate <= MALFORMED_LIMIT
    return {
        "schema_version": 1,
        "report_type": "compact8_api_parser_pilot_qualification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": prepared.provider,
        "requested_model": prepared.requested_model,
        "model_identity_sha256": prepared.model_identity_sha256,
        "panel_id": prepared.panel_id,
        "panel_sha256": prepared.panel_sha256,
        "task_item_keys_sha256": prepared.task_item_keys_sha256,
        "request_stage": prepared.request_stage,
        "request_count": request_count,
        "planned_request_count": prepared.planned_request_count,
        "pilot_size": prepared.pilot_size,
        "request_jsonl_path": str(Path(request_jsonl_path).resolve()),
        "request_jsonl_sha256": prepared.request_jsonl_sha256,
        "request_manifest_path": str(Path(request_manifest_path).resolve()),
        "request_manifest_sha256": prepared.request_manifest_sha256,
        "predictions_path": str(predictions_file.resolve()),
        "predictions_sha256": file_sha256(predictions_file),
        "malformed_count": malformed_count,
        "schema_invalid_response_count": schema_invalid_count,
        "malformed_rate": malformed_rate,
        "malformed_limit": MALFORMED_LIMIT,
        "malformed_scoring_rule": (
            "parse_error_or_invalid_label_value_whole_response"
        ),
        "task_schema_validated": True,
        "qualification_status": "passed" if passed else "failed",
        "remainder_submission_eligible": passed,
    }


def validate_pilot_qualification(
    report_path: str | Path,
    remainder: PreparedBatch,
    *,
    panel_manifest_path: str | Path | None = None,
    tasks_dir: str | Path = REPO / "tasks",
    task_definitions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if remainder.request_stage != "remainder":
        raise BatchIntegrityError(
            "pilot qualification can authorize only a remainder-stage batch"
        )
    report = json.loads(Path(report_path).read_text())
    expected = {
        "schema_version": 1,
        "report_type": "compact8_api_parser_pilot_qualification",
        "provider": remainder.provider,
        "requested_model": remainder.requested_model,
        "model_identity_sha256": remainder.model_identity_sha256,
        "panel_id": remainder.panel_id,
        "panel_sha256": remainder.panel_sha256,
        "task_item_keys_sha256": remainder.task_item_keys_sha256,
        "request_stage": "pilot",
        "request_count": remainder.pilot_size,
        "planned_request_count": remainder.planned_request_count,
        "pilot_size": remainder.pilot_size,
        "qualification_status": "passed",
        "remainder_submission_eligible": True,
        "malformed_scoring_rule": (
            "parse_error_or_invalid_label_value_whole_response"
        ),
        "task_schema_validated": True,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise BatchIntegrityError(
                f"pilot qualification {field} does not match remainder batch"
            )
    try:
        malformed_rate = float(report["malformed_rate"])
        malformed_limit = float(report["malformed_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchIntegrityError("pilot qualification malformed rate is invalid") from exc
    if malformed_limit != MALFORMED_LIMIT or malformed_rate > MALFORMED_LIMIT:
        raise BatchIntegrityError("pilot malformed rate exceeds 5%")
    for path_field, hash_field in [
        ("request_jsonl_path", "request_jsonl_sha256"),
        ("request_manifest_path", "request_manifest_sha256"),
    ]:
        evidence_path = Path(str(report.get(path_field) or ""))
        if not evidence_path.is_file() or file_sha256(evidence_path) != report.get(
            hash_field
        ):
            raise BatchIntegrityError("pilot qualification request evidence is stale")
    predictions_path = Path(str(report.get("predictions_path") or ""))
    if not predictions_path.is_file() or file_sha256(predictions_path) != report.get(
        "predictions_sha256"
    ):
        raise BatchIntegrityError("pilot qualification prediction evidence is stale")
    resolved_tasks = _resolve_task_schema(
        remainder,
        panel_manifest_path=panel_manifest_path,
        tasks_dir=tasks_dir,
        task_definitions=task_definitions,
    )
    recomputed = build_pilot_qualification(
        report["request_jsonl_path"],
        report["request_manifest_path"],
        report["predictions_path"],
        task_definitions=resolved_tasks,
    )
    recomputed_fields = (
        "request_count",
        "malformed_count",
        "schema_invalid_response_count",
        "malformed_rate",
        "malformed_limit",
        "malformed_scoring_rule",
        "task_schema_validated",
        "qualification_status",
        "remainder_submission_eligible",
    )
    for field in recomputed_fields:
        if report.get(field) != recomputed.get(field):
            raise BatchIntegrityError(
                f"pilot qualification {field} does not match recomputed evidence"
            )
    validated = dict(report)
    validated["_validated_pilot_qualification"] = True
    validated["qualification_report_sha256"] = file_sha256(report_path)
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-jsonl", required=True)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--panel-manifest", default=str(DEFAULT_COMPACT_MANIFEST)
    )
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_pilot_qualification(
        args.request_jsonl,
        args.request_manifest,
        args.predictions,
        panel_manifest_path=args.panel_manifest,
        tasks_dir=args.tasks_dir,
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["remainder_submission_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
