#!/usr/bin/env python3
"""Finalize compact API pilot and remainder evidence without provider calls.

The provider collectors intentionally operate on one submitted batch at a time.
This module is the offline boundary that proves the fixed 16-request pilot and
3,984-request remainder form one exact 4,000-request benchmark before the
frontier builder can consume them.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from api_batch_common import (
    BatchIntegrityError,
    PreparedBatch,
    RequestManifest,
    file_sha256,
    load_request_manifest,
    load_task_selection,
    staged_manifest_fields,
    validate_prepared_batch,
    validate_response_model,
)
from api_prediction_validation import prediction_malformed_masks
from task_registry import load_task_definitions


REPO = Path(__file__).resolve().parent.parent
COMPACT_PANEL_ID = "frontier_compact_8"
EXPECTED_ITEMS = 4_000
PILOT_ITEMS = 16
REMAINDER_ITEMS = EXPECTED_ITEMS - PILOT_ITEMS
MALFORMED_LIMIT = 0.05


def _atomic_replace(source: Path, destination: Path) -> None:
    """Publish a fully written file; metadata is deliberately published last."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _single_manifest_value(
    rows: Sequence[Mapping[str, Any]], field: str, *, required: bool = True
) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    nonempty = values - {""}
    if required and ("" in values or not nonempty):
        raise BatchIntegrityError(f"staged request manifests have missing {field}")
    if len(nonempty) != 1:
        raise BatchIntegrityError(
            f"staged request manifests have inconsistent {field}: {sorted(nonempty)}"
        )
    return next(iter(nonempty), "")


def _assert_same_batch_identity(pilot: PreparedBatch, remainder: PreparedBatch) -> None:
    if pilot.request_stage != "pilot" or remainder.request_stage != "remainder":
        raise BatchIntegrityError(
            "compact finalization requires pilot and remainder request stages"
        )
    expected = {
        "planned_request_count": EXPECTED_ITEMS,
        "pilot_size": PILOT_ITEMS,
    }
    for name, prepared in (("pilot", pilot), ("remainder", remainder)):
        for field, value in expected.items():
            if getattr(prepared, field) != value:
                raise BatchIntegrityError(
                    f"{name} {field} mismatch: {getattr(prepared, field)} != {value}"
                )
    if len(pilot.requests) != PILOT_ITEMS:
        raise BatchIntegrityError(
            f"pilot request count mismatch: {len(pilot.requests)} != {PILOT_ITEMS}"
        )
    if len(remainder.requests) != REMAINDER_ITEMS:
        raise BatchIntegrityError(
            "remainder request count mismatch: "
            f"{len(remainder.requests)} != {REMAINDER_ITEMS}"
        )
    identity_fields = (
        "provider",
        "requested_model",
        "model_identity_sha256",
        "panel_id",
        "panel_sha256",
        "task_item_keys_sha256",
        "planned_request_count",
        "pilot_size",
    )
    for field in identity_fields:
        if getattr(pilot, field) != getattr(remainder, field):
            raise BatchIntegrityError(
                f"pilot/remainder {field} mismatch: "
                f"{getattr(pilot, field)!r} != {getattr(remainder, field)!r}"
            )
    if pilot.panel_id != COMPACT_PANEL_ID:
        raise BatchIntegrityError(
            f"compact panel id mismatch: {pilot.panel_id!r} != {COMPACT_PANEL_ID!r}"
        )


def _manifest_rows_by_index(
    pilot: PreparedBatch,
    remainder: PreparedBatch,
    *,
    benchmark_commit: str,
) -> list[dict[str, str]]:
    source_rows = [*pilot.manifest_rows, *remainder.manifest_rows]
    required_fields = set(staged_manifest_fields())
    for row in source_rows:
        missing = sorted(required_fields - set(row))
        if missing:
            raise BatchIntegrityError(
                f"staged request manifest is missing fields: {missing}"
            )
    if _single_manifest_value(source_rows, "benchmark_commit") != benchmark_commit:
        raise BatchIntegrityError("staged request manifest benchmark commit mismatch")
    _single_manifest_value(source_rows, "provider")
    _single_manifest_value(source_rows, "requested_model")
    _single_manifest_value(source_rows, "expected_response_model")
    _single_manifest_value(source_rows, "model_identity_sha256")
    _single_manifest_value(source_rows, "model_manifest_sha256")
    _single_manifest_value(source_rows, "panel_id")
    _single_manifest_value(source_rows, "panel_sha256")
    _single_manifest_value(source_rows, "task_item_keys_sha256")

    combined: list[dict[str, str]] = []
    seen_custom_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    seen_indices: set[int] = set()
    for original in source_rows:
        row = {str(key): str(value) for key, value in original.items()}
        custom_id = row["custom_id"]
        key = (row["task"], row["item_id"])
        try:
            request_index = int(row["request_index"])
        except ValueError as exc:
            raise BatchIntegrityError(
                f"{custom_id}: request_index is not an integer"
            ) from exc
        if custom_id in seen_custom_ids:
            raise BatchIntegrityError(
                f"pilot and remainder have duplicate custom_id: {custom_id}"
            )
        if key in seen_keys:
            raise BatchIntegrityError(
                f"pilot and remainder have duplicate task/item key: {key[0]} {key[1]}"
            )
        if request_index in seen_indices:
            raise BatchIntegrityError(
                f"pilot and remainder have duplicate request_index: {request_index}"
            )
        seen_custom_ids.add(custom_id)
        seen_keys.add(key)
        seen_indices.add(request_index)
        row["source_request_stage"] = row["request_stage"]
        row["request_stage"] = "all"
        combined.append(row)
    if seen_indices != set(range(1, EXPECTED_ITEMS + 1)):
        missing = sorted(set(range(1, EXPECTED_ITEMS + 1)) - seen_indices)
        extra = sorted(seen_indices - set(range(1, EXPECTED_ITEMS + 1)))
        raise BatchIntegrityError(
            "pilot and remainder do not form the exact request-index partition: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    combined.sort(key=lambda row: int(row["request_index"]))
    return combined


def _request_serializations_by_custom_id(
    prepared: PreparedBatch,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for request, serialization in zip(
        prepared.requests, prepared.request_serializations, strict=True
    ):
        custom_id = str(request["custom_id"])
        if custom_id in output:
            raise BatchIntegrityError(f"duplicate prepared custom_id: {custom_id}")
        output[custom_id] = serialization
    return output


def _reported_model(record: Mapping[str, Any], provider: str) -> str:
    if provider == "openai":
        response = record.get("response") or {}
        body = response.get("body") or {}
        return str(body.get("model") or "").strip()
    if provider == "anthropic":
        result = record.get("result") or {}
        message = result.get("message") or {}
        return str(message.get("model") or "").strip()
    raise BatchIntegrityError(f"unsupported compact API provider: {provider}")


def _load_responses(
    path: Path,
    *,
    provider: str,
    manifest_by_custom_id: Mapping[str, Mapping[str, str]],
    stage: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    records: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(
                    f"{stage} response JSONL line {line_no} is invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise BatchIntegrityError(
                    f"{stage} response JSONL line {line_no} is not an object"
                )
            custom_id = str(record.get("custom_id") or "")
            if not custom_id:
                raise BatchIntegrityError(
                    f"{stage} response JSONL line {line_no} has no custom_id"
                )
            if custom_id in records:
                raise BatchIntegrityError(
                    f"{stage} responses have duplicate custom_id: {custom_id}"
                )
            manifest_row = manifest_by_custom_id.get(custom_id)
            if manifest_row is None:
                raise BatchIntegrityError(
                    f"{stage} responses have an unknown custom_id: {custom_id}"
                )
            returned = _reported_model(record, provider)
            validate_response_model(
                returned or None,
                manifest_row,
                context=f"{stage} response line {line_no} {custom_id}",
            )
            if returned:
                identities.add(returned)
            records[custom_id] = record
    expected = set(manifest_by_custom_id)
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise BatchIntegrityError(
            f"{stage} responses do not match logical requests: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return records, identities


def _is_missing_parse_error(value: Any) -> bool:
    return str(value).strip() in {"", "nan", "None"}


def _gold_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, int):
        try:
            return int(float(str(observed))) == expected
        except (TypeError, ValueError):
            return False
    return str(observed) == str(expected)


def _load_predictions(
    path: Path,
    *,
    stage: str,
    prepared: PreparedBatch,
    request_manifest: RequestManifest,
    response_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    required = {
        "task",
        "model",
        "item_id",
        "parse_error",
        "response_model",
        "model_identity_sha256",
        "panel_id",
        "panel_sha256",
        "request_stage",
        "request_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BatchIntegrityError(
            f"{stage} predictions are missing columns: {missing}"
        )
    if len(frame) != len(prepared.requests):
        raise BatchIntegrityError(
            f"{stage} prediction count mismatch: {len(frame)} != {len(prepared.requests)}"
        )
    if frame.duplicated(["task", "item_id"]).any():
        raise BatchIntegrityError(
            f"{stage} predictions have duplicate task/item keys"
        )

    prediction_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw in frame.to_dict(orient="records"):
        row = {str(key): str(value) for key, value in raw.items()}
        prediction_by_key[(row["task"], row["item_id"])] = row

    output: list[dict[str, str]] = []
    for manifest_row in prepared.manifest_rows:
        custom_id = manifest_row["custom_id"]
        key = (manifest_row["task"], manifest_row["item_id"])
        row = prediction_by_key.get(key)
        if row is None:
            raise BatchIntegrityError(
                f"{stage} predictions are missing {key[0]} {key[1]}"
            )
        expected_values = {
            "model": prepared.requested_model,
            "model_identity_sha256": prepared.model_identity_sha256,
            "panel_id": prepared.panel_id,
            "panel_sha256": prepared.panel_sha256,
            "request_stage": stage,
            "request_index": manifest_row["request_index"],
        }
        for field, expected in expected_values.items():
            if row.get(field, "") != str(expected):
                raise BatchIntegrityError(
                    f"{stage} prediction {field} mismatch for {custom_id}"
                )
        if row.get("custom_id") not in {None, "", custom_id}:
            raise BatchIntegrityError(
                f"{stage} prediction custom_id mismatch for {custom_id}"
            )
        if row.get("request_sha256") not in {
            None,
            "",
            manifest_row["request_sha256"],
        }:
            raise BatchIntegrityError(
                f"{stage} prediction request hash mismatch for {custom_id}"
            )
        returned = _reported_model(response_records[custom_id], prepared.provider)
        if row["response_model"].strip() != returned:
            raise BatchIntegrityError(
                f"{stage} prediction returned-model provenance mismatch for {custom_id}"
            )
        if not returned and _is_missing_parse_error(row["parse_error"]):
            raise BatchIntegrityError(
                f"{stage} response has no returned model but is not retained as malformed: "
                f"{custom_id}"
            )

        _task, item = request_manifest.lookup[custom_id]
        for label, expected in item["gt"].items():
            column = f"gt_{label}"
            if column not in row or not _gold_equal(row[column], expected):
                raise BatchIntegrityError(
                    f"{stage} prediction gold-label mismatch for {custom_id}: {column}"
                )
            prediction_column = f"pred_{label}"
            if prediction_column not in row:
                raise BatchIntegrityError(
                    f"{stage} prediction is missing {prediction_column} for {custom_id}"
                )
        row["custom_id"] = custom_id
        row["request_sha256"] = manifest_row["request_sha256"]
        row["source_request_stage"] = stage
        row["request_stage"] = "all"
        output.append(row)
    return output


def finalize_compact_api_evidence(
    *,
    panel_manifest_path: str | Path,
    pilot_request_jsonl_path: str | Path,
    pilot_request_manifest_path: str | Path,
    pilot_responses_path: str | Path,
    pilot_predictions_path: str | Path,
    remainder_request_jsonl_path: str | Path,
    remainder_request_manifest_path: str | Path,
    remainder_responses_path: str | Path,
    remainder_predictions_path: str | Path,
    output_dir: str | Path,
    tasks_dir: str | Path = REPO / "tasks",
) -> dict[str, Any]:
    """Validate and atomically publish one compact API evidence bundle."""
    panel_manifest_path = Path(panel_manifest_path)
    tasks = load_task_definitions(tasks_dir=tasks_dir)
    selection = load_task_selection(tasks, panel_manifest_path)
    if (
        selection.panel_id != COMPACT_PANEL_ID
        or selection.expected_items != EXPECTED_ITEMS
        or selection.expected_tasks != 8
    ):
        raise BatchIntegrityError(
            "API finalization requires the frozen 8-task, 4,000-item compact manifest"
        )
    selected_tasks = list(selection.tasks)

    pilot_request_jsonl_path = Path(pilot_request_jsonl_path)
    pilot_request_manifest_path = Path(pilot_request_manifest_path)
    remainder_request_jsonl_path = Path(remainder_request_jsonl_path)
    remainder_request_manifest_path = Path(remainder_request_manifest_path)
    pilot_responses_path = Path(pilot_responses_path)
    remainder_responses_path = Path(remainder_responses_path)
    pilot_predictions_path = Path(pilot_predictions_path)
    remainder_predictions_path = Path(remainder_predictions_path)

    pilot = validate_prepared_batch(
        pilot_request_jsonl_path, pilot_request_manifest_path
    )
    remainder = validate_prepared_batch(
        remainder_request_jsonl_path, remainder_request_manifest_path
    )
    _assert_same_batch_identity(pilot, remainder)
    if pilot.panel_sha256 != selection.panel_sha256:
        raise BatchIntegrityError("staged request compact-manifest hash mismatch")
    if pilot.task_item_keys_sha256 != selection.task_item_keys_sha256:
        raise BatchIntegrityError("staged request task/item-key hash mismatch")

    pilot_manifest = load_request_manifest(
        pilot_request_manifest_path,
        selected_tasks,
        model_name=pilot.requested_model,
        provider=pilot.provider,
        selection=selection,
        require_enhanced=True,
        require_complete=True,
    )
    remainder_manifest = load_request_manifest(
        remainder_request_manifest_path,
        selected_tasks,
        model_name=remainder.requested_model,
        provider=remainder.provider,
        selection=selection,
        require_enhanced=True,
        require_complete=True,
    )
    manifest_rows = _manifest_rows_by_index(
        pilot,
        remainder,
        benchmark_commit=selection.benchmark_commit,
    )
    expected_response_model = _single_manifest_value(
        manifest_rows, "expected_response_model"
    )

    pilot_manifest_by_id = {
        row["custom_id"]: row for row in pilot.manifest_rows
    }
    remainder_manifest_by_id = {
        row["custom_id"]: row for row in remainder.manifest_rows
    }
    pilot_responses, pilot_identities = _load_responses(
        pilot_responses_path,
        provider=pilot.provider,
        manifest_by_custom_id=pilot_manifest_by_id,
        stage="pilot",
    )
    remainder_responses, remainder_identities = _load_responses(
        remainder_responses_path,
        provider=remainder.provider,
        manifest_by_custom_id=remainder_manifest_by_id,
        stage="remainder",
    )
    returned_identities = pilot_identities | remainder_identities
    if returned_identities != {expected_response_model}:
        raise BatchIntegrityError(
            "provider returned identity evidence mismatch: "
            f"{sorted(returned_identities)} != {[expected_response_model]}"
        )

    pilot_predictions = _load_predictions(
        pilot_predictions_path,
        stage="pilot",
        prepared=pilot,
        request_manifest=pilot_manifest,
        response_records=pilot_responses,
    )
    pilot_malformed, pilot_schema_invalid = prediction_malformed_masks(
        pd.DataFrame(pilot_predictions), selection.tasks
    )
    pilot_malformed_count = int(pilot_malformed.sum())
    pilot_schema_invalid_count = int(pilot_schema_invalid.sum())
    pilot_malformed_rate = pilot_malformed_count / PILOT_ITEMS
    pilot_reliability_pass = pilot_malformed_rate <= MALFORMED_LIMIT
    if not pilot_reliability_pass:
        raise BatchIntegrityError(
            "compact API pilot malformed rate exceeds 5%: "
            f"{pilot_malformed_count}/{PILOT_ITEMS} ({pilot_malformed_rate:.6f})"
        )
    remainder_predictions = _load_predictions(
        remainder_predictions_path,
        stage="remainder",
        prepared=remainder,
        request_manifest=remainder_manifest,
        response_records=remainder_responses,
    )
    predictions_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in [*pilot_predictions, *remainder_predictions]:
        key = (row["task"], row["item_id"])
        if key in predictions_by_key:
            raise BatchIntegrityError(
                f"combined predictions have duplicate task/item key: {key[0]} {key[1]}"
            )
        predictions_by_key[key] = row

    ordered_predictions: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        key = (manifest_row["task"], manifest_row["item_id"])
        row = predictions_by_key.get(key)
        if row is None:
            raise BatchIntegrityError(
                f"combined predictions are missing {key[0]} {key[1]}"
            )
        if row["custom_id"] != manifest_row["custom_id"]:
            raise BatchIntegrityError(
                f"combined prediction custom_id mismatch for {key[0]} {key[1]}"
            )
        ordered_predictions.append(row)
    if len(ordered_predictions) != EXPECTED_ITEMS:
        raise BatchIntegrityError(
            f"combined prediction count mismatch: {len(ordered_predictions)}"
        )

    request_serializations = {
        **_request_serializations_by_custom_id(pilot),
        **_request_serializations_by_custom_id(remainder),
    }
    if len(request_serializations) != EXPECTED_ITEMS:
        raise BatchIntegrityError("combined request JSONL has duplicate or missing requests")
    combined_request_jsonl = (
        "\n".join(
            request_serializations[row["custom_id"]] for row in manifest_rows
        )
        + "\n"
    ).encode("utf-8")

    all_responses = {**pilot_responses, **remainder_responses}
    if len(all_responses) != EXPECTED_ITEMS:
        raise BatchIntegrityError("combined responses have duplicate or missing custom_ids")
    combined_responses_jsonl = (
        "\n".join(
            json.dumps(all_responses[row["custom_id"]], ensure_ascii=False)
            for row in manifest_rows
        )
        + "\n"
    ).encode("utf-8")

    manifest_input_fields = list(pilot.manifest_rows[0].keys())
    if set(manifest_input_fields) != set(remainder.manifest_rows[0].keys()):
        raise BatchIntegrityError("pilot/remainder request manifest columns differ")
    manifest_fields = [*manifest_input_fields, "source_request_stage"]
    combined_manifest_csv = _csv_bytes(manifest_rows, manifest_fields)

    prediction_fields: list[str] = []
    for row in ordered_predictions:
        for field in row:
            if field not in prediction_fields:
                prediction_fields.append(field)
    validation_frame = pd.DataFrame(ordered_predictions)
    malformed, schema_invalid = prediction_malformed_masks(
        validation_frame, selection.tasks
    )
    schema_invalid_count = int(schema_invalid.sum())
    for position, row in enumerate(ordered_predictions):
        if bool(schema_invalid.iloc[position]) and _is_missing_parse_error(
            row["parse_error"]
        ):
            row["parse_error"] = "invalid_schema_output"
    combined_predictions_csv = _csv_bytes(ordered_predictions, prediction_fields)
    malformed_count = int(malformed.sum())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".compact-api-finalize-", dir=output_dir))
    try:
        staged_requests = staging_dir / "requests.jsonl"
        staged_manifest = staging_dir / "request_manifest.csv"
        staged_responses = staging_dir / "responses.jsonl"
        staged_predictions = staging_dir / "predictions.csv"
        staged_metadata = staging_dir / "metadata.json"
        _write_bytes(staged_requests, combined_request_jsonl)
        _write_bytes(staged_manifest, combined_manifest_csv)
        _write_bytes(staged_responses, combined_responses_jsonl)
        _write_bytes(staged_predictions, combined_predictions_csv)

        combined_prepared = validate_prepared_batch(staged_requests, staged_manifest)
        if (
            combined_prepared.request_stage != "all"
            or len(combined_prepared.requests) != EXPECTED_ITEMS
        ):
            raise BatchIntegrityError("combined request evidence failed final validation")
        load_request_manifest(
            staged_manifest,
            selected_tasks,
            model_name=pilot.requested_model,
            provider=pilot.provider,
            selection=selection,
            require_enhanced=True,
            require_complete=True,
        )
        final_predictions = pd.read_csv(
            staged_predictions, dtype=str, keep_default_na=False, low_memory=False
        )
        if len(final_predictions) != EXPECTED_ITEMS or final_predictions.duplicated(
            ["task", "item_id"]
        ).any():
            raise BatchIntegrityError("combined prediction artifact failed final validation")

        metadata = {
            "schema_version": 1,
            "evidence_type": "compact8_api_pilot_remainder_final",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "provider": pilot.provider,
            "benchmark_commit": selection.benchmark_commit,
            "panel_id": selection.panel_id,
            "panel_sha256": selection.panel_sha256,
            "task_item_keys_sha256": selection.task_item_keys_sha256,
            "requested_model": pilot.requested_model,
            "expected_response_model": expected_response_model,
            "returned_model_identity": expected_response_model,
            "provider_returned_model_identities": sorted(returned_identities),
            "model_identity_sha256": pilot.model_identity_sha256,
            "model_manifest_sha256": _single_manifest_value(
                manifest_rows, "model_manifest_sha256"
            ),
            "expected_tasks": selection.expected_tasks,
            "expected_items": selection.expected_items,
            "pilot_items": PILOT_ITEMS,
            "remainder_items": REMAINDER_ITEMS,
            "pilot_malformed_count": pilot_malformed_count,
            "pilot_schema_invalid_response_count": pilot_schema_invalid_count,
            "pilot_malformed_rate": pilot_malformed_rate,
            "pilot_malformed_limit": MALFORMED_LIMIT,
            "pilot_reliability_pass": pilot_reliability_pass,
            "logical_responses": EXPECTED_ITEMS,
            "rows_written": EXPECTED_ITEMS,
            "malformed_count": malformed_count,
            "schema_invalid_response_count": schema_invalid_count,
            "malformed_rate": malformed_count / EXPECTED_ITEMS,
            "malformed_limit": MALFORMED_LIMIT,
            "malformed_scoring_rule": (
                "parse_error_or_invalid_label_value_whole_response"
            ),
            "reliability_pass": malformed_count / EXPECTED_ITEMS <= MALFORMED_LIMIT,
            "requests_path": str((output_dir / "requests.jsonl").resolve()),
            "requests_sha256": file_sha256(staged_requests),
            "requests_manifest_path": str(
                (output_dir / "request_manifest.csv").resolve()
            ),
            "requests_manifest_sha256": file_sha256(staged_manifest),
            "responses_path": str((output_dir / "responses.jsonl").resolve()),
            "responses_sha256": file_sha256(staged_responses),
            "predictions_path": str((output_dir / "predictions.csv").resolve()),
            "predictions_sha256": file_sha256(staged_predictions),
            "source_stages": {
                "pilot": {
                    "request_count": PILOT_ITEMS,
                    "request_jsonl_path": str(pilot_request_jsonl_path.resolve()),
                    "request_jsonl_sha256": pilot.request_jsonl_sha256,
                    "request_manifest_path": str(
                        pilot_request_manifest_path.resolve()
                    ),
                    "request_manifest_sha256": pilot.request_manifest_sha256,
                    "responses_path": str(pilot_responses_path.resolve()),
                    "responses_sha256": file_sha256(pilot_responses_path),
                    "predictions_path": str(pilot_predictions_path.resolve()),
                    "predictions_sha256": file_sha256(pilot_predictions_path),
                },
                "remainder": {
                    "request_count": REMAINDER_ITEMS,
                    "request_jsonl_path": str(remainder_request_jsonl_path.resolve()),
                    "request_jsonl_sha256": remainder.request_jsonl_sha256,
                    "request_manifest_path": str(
                        remainder_request_manifest_path.resolve()
                    ),
                    "request_manifest_sha256": remainder.request_manifest_sha256,
                    "responses_path": str(remainder_responses_path.resolve()),
                    "responses_sha256": file_sha256(remainder_responses_path),
                    "predictions_path": str(remainder_predictions_path.resolve()),
                    "predictions_sha256": file_sha256(remainder_predictions_path),
                },
            },
        }
        _write_bytes(staged_metadata, _json_bytes(metadata))

        destinations = {
            "requests.jsonl": staged_requests,
            "request_manifest.csv": staged_manifest,
            "responses.jsonl": staged_responses,
            "predictions.csv": staged_predictions,
        }
        for name, source in destinations.items():
            _atomic_replace(source, output_dir / name)
        _atomic_replace(staged_metadata, output_dir / "metadata.json")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    result = dict(metadata)
    result["metadata_path"] = str((output_dir / "metadata.json").resolve())
    result["metadata_sha256"] = file_sha256(output_dir / "metadata.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument("--pilot-request-jsonl", required=True)
    parser.add_argument("--pilot-request-manifest", required=True)
    parser.add_argument("--pilot-responses", required=True)
    parser.add_argument("--pilot-predictions", required=True)
    parser.add_argument("--remainder-request-jsonl", required=True)
    parser.add_argument("--remainder-request-manifest", required=True)
    parser.add_argument("--remainder-responses", required=True)
    parser.add_argument("--remainder-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = finalize_compact_api_evidence(
        panel_manifest_path=args.panel_manifest,
        tasks_dir=args.tasks_dir,
        pilot_request_jsonl_path=args.pilot_request_jsonl,
        pilot_request_manifest_path=args.pilot_request_manifest,
        pilot_responses_path=args.pilot_responses,
        pilot_predictions_path=args.pilot_predictions,
        remainder_request_jsonl_path=args.remainder_request_jsonl,
        remainder_request_manifest_path=args.remainder_request_manifest,
        remainder_responses_path=args.remainder_responses,
        remainder_predictions_path=args.remainder_predictions,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
