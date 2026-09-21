#!/usr/bin/env python3
"""Shared integrity and provenance helpers for provider batch jobs.

This module is deliberately provider-client free.  Importing it and running any
of its validation helpers cannot contact a remote API.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


class BatchIntegrityError(ValueError):
    """Raised when prepared requests, manifests, or responses disagree."""


REQUEST_MANIFEST_FIELDS = [
    "custom_id",
    "task",
    "model",
    "item_id",
    "original_custom_id",
    "provider",
    "requested_model",
    "expected_response_model",
    "model_identity_sha256",
    "model_manifest_sha256",
    "panel_id",
    "panel_sha256",
    "panel_task_fingerprint",
    "benchmark_commit",
    "task_item_keys_sha256",
    "request_sha256",
]

# These columns are additive so archival request manifests remain readable.  A
# compact benchmark stage uses them to prove that the pilot and remainder are
# an exact, non-overlapping partition of the frozen manifest order.
STAGED_REQUEST_MANIFEST_FIELDS = [
    "request_stage",
    "request_index",
    "planned_request_count",
    "pilot_size",
]
VALID_REQUEST_STAGES = {"all", "pilot", "remainder"}
COMPACT_API_HARD_BUDGET_USD = Decimal("50")


@dataclass(frozen=True)
class BatchTaskSelection:
    tasks: tuple[dict[str, Any], ...]
    panel_id: str
    panel_sha256: str
    benchmark_commit: str
    task_item_keys_sha256: str
    expected_tasks: int
    expected_items: int
    task_fingerprints: Mapping[str, str]

    @property
    def is_frozen_panel(self) -> bool:
        return bool(self.panel_sha256)


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    requested_model: str
    expected_response_model: str
    model_identity_sha256: str
    model_manifest_sha256: str


@dataclass(frozen=True)
class RequestManifest:
    rows: tuple[dict[str, str], ...]
    row_by_custom_id: Mapping[str, dict[str, str]]
    lookup: Mapping[str, tuple[dict[str, Any], dict[str, Any]]]
    provider: str
    requested_model: str
    expected_response_model: str
    model_identity_sha256: str
    panel_id: str
    panel_sha256: str
    task_item_keys_sha256: str


@dataclass(frozen=True)
class PreparedBatch:
    requests: tuple[dict[str, Any], ...]
    request_serializations: tuple[str, ...]
    manifest_rows: tuple[dict[str, str], ...]
    provider: str
    requested_model: str
    model_identity_sha256: str
    panel_id: str
    panel_sha256: str
    task_item_keys_sha256: str
    request_jsonl_sha256: str
    request_manifest_sha256: str
    request_stage: str = ""
    planned_request_count: int = 0
    pilot_size: int = 0


@dataclass(frozen=True)
class RequestStagePlan:
    """An exact slice of the frozen request order."""

    name: str
    planned_request_count: int
    pilot_size: int
    expected_request_count: int
    first_request_index: int
    last_request_index: int

    def includes(self, request_index: int) -> bool:
        return self.first_request_index <= request_index <= self.last_request_index


def build_request_stage_plan(
    request_stage: str,
    planned_request_count: int,
    *,
    pilot_size: int = 16,
) -> RequestStagePlan:
    """Resolve ``all``, ``pilot``, or ``remainder`` without sampling."""
    stage = str(request_stage).strip().lower()
    if stage not in VALID_REQUEST_STAGES:
        raise BatchIntegrityError(
            f"request_stage must be one of {sorted(VALID_REQUEST_STAGES)}"
        )
    if planned_request_count < 1:
        raise BatchIntegrityError("planned_request_count must be positive")
    if pilot_size < 1 or pilot_size >= planned_request_count:
        raise BatchIntegrityError(
            "pilot_size must be positive and smaller than planned_request_count"
        )
    if stage == "pilot":
        first, last = 1, pilot_size
    elif stage == "remainder":
        first, last = pilot_size + 1, planned_request_count
    else:
        first, last = 1, planned_request_count
    return RequestStagePlan(
        name=stage,
        planned_request_count=planned_request_count,
        pilot_size=pilot_size,
        expected_request_count=last - first + 1,
        first_request_index=first,
        last_request_index=last,
    )


def staged_manifest_fields() -> list[str]:
    return [*REQUEST_MANIFEST_FIELDS, *STAGED_REQUEST_MANIFEST_FIELDS]


def add_request_stage_fields(
    row: Mapping[str, Any],
    *,
    stage: RequestStagePlan,
    request_index: int,
) -> dict[str, Any]:
    if not stage.includes(request_index):
        raise BatchIntegrityError(
            f"request index {request_index} is outside the {stage.name} stage"
        )
    return {
        **dict(row),
        "request_stage": stage.name,
        "request_index": request_index,
        "planned_request_count": stage.planned_request_count,
        "pilot_size": stage.pilot_size,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_sha256(request: Mapping[str, Any]) -> str:
    return canonical_sha256(request)


def _task_item_records(tasks: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        task_name = str(task["name"])
        for item in task["loader"]():
            key = (task_name, str(item["item_id"]))
            if key in seen:
                raise BatchIntegrityError(
                    f"duplicate task/item key in loaded benchmark: {key[0]} {key[1]}"
                )
            seen.add(key)
            records.append({"task": key[0], "item_id": key[1]})
    return records


def task_item_keys_sha256(tasks: Iterable[dict[str, Any]]) -> str:
    records = sorted(
        _task_item_records(tasks), key=lambda row: (row["task"], row["item_id"])
    )
    return canonical_sha256({"schema_version": 1, "task_item_keys": records})


def load_task_selection(
    tasks: Sequence[dict[str, Any]],
    panel_manifest_path: str | Path | None,
) -> BatchTaskSelection:
    """Resolve a frozen panel while retaining a deterministic exact-key hash."""
    if panel_manifest_path:
        # Kept local so legacy non-panel batch helpers do not require the panel
        # module at import time.
        from panel_manifest import assert_panel_checkout, load_panel_manifest

        panel = load_panel_manifest(
            panel_manifest_path,
            task_definitions=tasks,
        )
        assert_panel_checkout(panel, Path(__file__).resolve().parent.parent)
        selected = tuple(panel.tasks)
        observed_key_hash = task_item_keys_sha256(selected)
        return BatchTaskSelection(
            tasks=selected,
            panel_id=panel.panel_id,
            panel_sha256=panel.panel_sha256,
            benchmark_commit=panel.benchmark_commit,
            task_item_keys_sha256=observed_key_hash,
            expected_tasks=panel.expected_tasks,
            expected_items=panel.expected_items,
            task_fingerprints=panel.task_fingerprints,
        )

    selected = tuple(tasks)
    records = _task_item_records(selected)
    return BatchTaskSelection(
        tasks=selected,
        panel_id="",
        panel_sha256="",
        benchmark_commit="",
        task_item_keys_sha256=canonical_sha256(
            {
                "schema_version": 1,
                "task_item_keys": sorted(
                    records, key=lambda row: (row["task"], row["item_id"])
                ),
            }
        ),
        expected_tasks=len(selected),
        expected_items=len(records),
        task_fingerprints={},
    )


def _provider_for_model(model: Mapping[str, Any]) -> str:
    backend = str(model.get("backend", "")).strip()
    provider = str(model.get("provider") or backend).strip()
    if not provider:
        raise BatchIntegrityError("model has no provider/backend identity")
    return provider


def build_model_identity(
    model: Mapping[str, Any],
    model_manifest_path: str | Path | None = None,
) -> ModelIdentity:
    """Hash the semantic model manifest and inference configuration."""
    requested_model = str(model.get("name", "")).strip()
    if not requested_model:
        raise BatchIntegrityError("model name is required")

    raw: Mapping[str, Any]
    manifest_hash = ""
    if model_manifest_path is not None:
        path = Path(model_manifest_path)
        loaded = yaml.safe_load(path.read_text())
        if not isinstance(loaded, dict):
            raise BatchIntegrityError(f"{path} must contain a mapping")
        raw = loaded
        manifest_hash = file_sha256(path)
    else:
        raw = {
            key: value
            for key, value in model.items()
            if key
            not in {
                "api_key_file",
                "manifest_path",
                "ollama_url",
                "base_url",
            }
        }

    expected_response_model = str(raw.get("expected_response_model") or "").strip()
    payload = {
        "schema_version": 1,
        "provider": _provider_for_model(model),
        "requested_model": requested_model,
        "expected_response_model": expected_response_model,
        "model_manifest": raw,
    }
    return ModelIdentity(
        provider=_provider_for_model(model),
        requested_model=requested_model,
        expected_response_model=expected_response_model,
        model_identity_sha256=canonical_sha256(payload),
        model_manifest_sha256=manifest_hash,
    )


def request_manifest_row(
    *,
    custom_id: str,
    task_name: str,
    item_id: str,
    request: Mapping[str, Any],
    model_identity: ModelIdentity,
    selection: BatchTaskSelection,
    original_custom_id: str = "",
) -> dict[str, str]:
    return {
        "custom_id": custom_id,
        "task": task_name,
        "model": model_identity.requested_model,
        "item_id": str(item_id),
        "original_custom_id": original_custom_id,
        "provider": model_identity.provider,
        "requested_model": model_identity.requested_model,
        "expected_response_model": model_identity.expected_response_model,
        "model_identity_sha256": model_identity.model_identity_sha256,
        "model_manifest_sha256": model_identity.model_manifest_sha256,
        "panel_id": selection.panel_id,
        "panel_sha256": selection.panel_sha256,
        "panel_task_fingerprint": selection.task_fingerprints.get(task_name, ""),
        "benchmark_commit": selection.benchmark_commit,
        "task_item_keys_sha256": selection.task_item_keys_sha256,
        "request_sha256": request_sha256(request),
    }


def _single_value(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    required: bool,
) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    if required and (not values or "" in values):
        raise BatchIntegrityError(f"request manifest has missing {field}")
    nonempty = values - {""}
    if len(nonempty) > 1:
        raise BatchIntegrityError(
            f"request manifest has inconsistent {field}: {sorted(nonempty)}"
        )
    return next(iter(nonempty), "")


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], set[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise BatchIntegrityError(f"request manifest is empty: {path}")
    return rows, fields


def _parse_stage_columns(
    rows: Sequence[Mapping[str, str]],
    fields: set[str],
) -> tuple[str, int, int, set[int]]:
    """Validate additive stage columns and return their exact index set."""
    staged_fields = set(STAGED_REQUEST_MANIFEST_FIELDS)
    present = staged_fields & fields
    if not present:
        return "", 0, 0, set()
    if present != staged_fields:
        raise BatchIntegrityError(
            "request manifest has an incomplete request-stage provenance block"
        )
    stage = _single_value(rows, "request_stage", required=True)
    if stage not in VALID_REQUEST_STAGES:
        raise BatchIntegrityError(f"request manifest has invalid stage: {stage}")
    planned_raw = _single_value(rows, "planned_request_count", required=True)
    pilot_raw = _single_value(rows, "pilot_size", required=True)
    try:
        planned = int(planned_raw)
        pilot_size = int(pilot_raw)
        indices = {int(row["request_index"]) for row in rows}
    except (TypeError, ValueError, KeyError) as exc:
        raise BatchIntegrityError(
            "request manifest has invalid request-stage counts"
        ) from exc
    if len(indices) != len(rows):
        raise BatchIntegrityError("request manifest has duplicate request_index values")
    plan = build_request_stage_plan(stage, planned, pilot_size=pilot_size)
    expected_indices = set(
        range(plan.first_request_index, plan.last_request_index + 1)
    )
    if indices != expected_indices:
        raise BatchIntegrityError(
            f"request manifest is not the exact {stage} request-index partition"
        )
    return stage, planned, pilot_size, indices


def load_request_manifest(
    path: str | Path,
    tasks: Sequence[dict[str, Any]],
    *,
    model_name: str | None = None,
    provider: str | None = None,
    selection: BatchTaskSelection | None = None,
    require_enhanced: bool = False,
    require_complete: bool = False,
) -> RequestManifest:
    """Load the request manifest as the authoritative response key set."""
    rows, fields = _read_csv_rows(path)
    request_stage, planned_count, _pilot_size, request_indices = _parse_stage_columns(
        rows, fields
    )
    base_required = {"custom_id", "task", "model", "item_id"}
    missing_base = sorted(base_required - fields)
    if missing_base:
        raise BatchIntegrityError(
            f"request manifest missing columns: {missing_base}"
        )
    if require_enhanced:
        missing = sorted(set(REQUEST_MANIFEST_FIELDS) - fields)
        if missing:
            raise BatchIntegrityError(
                f"panel request manifest missing provenance columns: {missing}"
            )

    manifest_model = _single_value(rows, "requested_model", required=require_enhanced)
    if not manifest_model:
        manifest_model = _single_value(rows, "model", required=True)
    legacy_model = _single_value(rows, "model", required=True)
    if legacy_model != manifest_model:
        raise BatchIntegrityError(
            f"request manifest model mismatch: {legacy_model} != {manifest_model}"
        )
    if model_name is not None and manifest_model != model_name:
        raise BatchIntegrityError(
            f"request manifest model mismatch: {manifest_model} != {model_name}"
        )

    manifest_provider = _single_value(rows, "provider", required=require_enhanced)
    if provider is not None and manifest_provider and manifest_provider != provider:
        raise BatchIntegrityError(
            f"request manifest provider mismatch: {manifest_provider} != {provider}"
        )
    expected_response_model = _single_value(
        rows, "expected_response_model", required=False
    )
    model_identity = _single_value(
        rows, "model_identity_sha256", required=require_enhanced
    )
    panel_id = _single_value(rows, "panel_id", required=require_enhanced)
    panel_hash = _single_value(rows, "panel_sha256", required=require_enhanced)
    key_hash = _single_value(
        rows, "task_item_keys_sha256", required=require_enhanced
    )

    if selection is not None:
        if selection.is_frozen_panel:
            if panel_id != selection.panel_id:
                raise BatchIntegrityError(
                    f"request manifest panel id mismatch: {panel_id} != {selection.panel_id}"
                )
            if panel_hash != selection.panel_sha256:
                raise BatchIntegrityError(
                    "request manifest panel hash does not match the frozen panel"
                )
        if key_hash and key_hash != selection.task_item_keys_sha256:
            raise BatchIntegrityError(
                "request manifest task/item key hash does not match loaded tasks"
            )

    task_lookup: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    expected_keys: set[tuple[str, str]] = set()
    expected_key_by_index: dict[int, tuple[str, str]] = {}
    expected_index = 0
    for task in tasks:
        name = str(task["name"])
        per_task: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for item in task["loader"]():
            item_id = str(item["item_id"])
            if item_id in per_task:
                raise BatchIntegrityError(
                    f"loaded task has duplicate item_id: {name} {item_id}"
                )
            per_task[item_id] = (task, item)
            key = (name, item_id)
            expected_keys.add(key)
            expected_index += 1
            expected_key_by_index[expected_index] = key
        task_lookup[name] = per_task

    if request_stage:
        if planned_count != len(expected_keys):
            raise BatchIntegrityError(
                "request manifest planned_request_count does not match loaded tasks"
            )
        for row in rows:
            request_index = int(row["request_index"])
            observed_key = (row["task"], row["item_id"])
            if expected_key_by_index.get(request_index) != observed_key:
                raise BatchIntegrityError(
                    f"request manifest order/key mismatch at index {request_index}"
                )

    lookup: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    row_by_id: dict[str, dict[str, str]] = {}
    manifest_keys: set[tuple[str, str]] = set()
    for line_no, row in enumerate(rows, start=2):
        custom_id = row["custom_id"]
        key = (row["task"], row["item_id"])
        if not custom_id:
            raise BatchIntegrityError(
                f"request manifest line {line_no} has an empty custom_id"
            )
        if custom_id in row_by_id:
            raise BatchIntegrityError(
                f"request manifest has duplicate custom_id: {custom_id}"
            )
        if key in manifest_keys:
            raise BatchIntegrityError(
                f"request manifest has duplicate task/item key: {key[0]} {key[1]}"
            )
        try:
            task_item = task_lookup[key[0]][key[1]]
        except KeyError as exc:
            raise BatchIntegrityError(
                f"request manifest row does not match loaded tasks: "
                f"{custom_id} {key[0]} {key[1]}"
            ) from exc
        if require_enhanced and not row.get("request_sha256"):
            raise BatchIntegrityError(
                f"request manifest row has no request_sha256: {custom_id}"
            )
        if selection is not None and selection.is_frozen_panel:
            expected_fingerprint = selection.task_fingerprints[key[0]]
            if row.get("panel_task_fingerprint") != expected_fingerprint:
                raise BatchIntegrityError(
                    f"request manifest task fingerprint mismatch: {key[0]}"
                )
        row_by_id[custom_id] = row
        lookup[custom_id] = task_item
        manifest_keys.add(key)

    stage_is_exact_partition = bool(request_stage) and {
        expected_key_by_index[index] for index in request_indices
    } == manifest_keys
    if require_complete and manifest_keys != expected_keys and not stage_is_exact_partition:
        missing = sorted(expected_keys - manifest_keys)
        extra = sorted(manifest_keys - expected_keys)
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} (first {missing[0]})")
        if extra:
            detail.append(f"extra {len(extra)} (first {extra[0]})")
        raise BatchIntegrityError(
            "request manifest does not contain the exact loaded task/item set: "
            + "; ".join(detail)
        )

    return RequestManifest(
        rows=tuple(rows),
        row_by_custom_id=row_by_id,
        lookup=lookup,
        provider=manifest_provider or (provider or ""),
        requested_model=manifest_model,
        expected_response_model=expected_response_model,
        model_identity_sha256=model_identity,
        panel_id=panel_id,
        panel_sha256=panel_hash,
        task_item_keys_sha256=key_hash,
    )


_DATED_MODEL_RE = re.compile(r"(?:^|[-_])(?:20\d{2}-\d{2}-\d{2}|20\d{6})(?:$|[-_])")


def validate_response_model(
    returned_model: Any,
    manifest_row: Mapping[str, str],
    *,
    context: str,
) -> None:
    """Reject wrong provider model identities when the response reports one."""
    if returned_model is None or not str(returned_model).strip():
        return
    returned = str(returned_model).strip()
    requested = str(
        manifest_row.get("requested_model") or manifest_row.get("model") or ""
    ).strip()
    expected = str(manifest_row.get("expected_response_model") or "").strip()
    if expected:
        if returned != expected:
            raise BatchIntegrityError(
                f"{context}: response snapshot mismatch: {returned} != {expected}"
            )
        return
    if returned == requested:
        return
    # Mutable aliases commonly return a dated provider snapshot.  Accept only
    # that narrow alias expansion; an explicitly dated request remains exact.
    if not _DATED_MODEL_RE.search(requested) and returned.startswith(f"{requested}-"):
        return
    raise BatchIntegrityError(
        f"{context}: response model mismatch: {returned} != {requested}"
    )


def _request_model(request: Mapping[str, Any]) -> str:
    payload = request.get("body") if "body" in request else request.get("params")
    if not isinstance(payload, Mapping):
        raise BatchIntegrityError("request has neither a body nor params mapping")
    return str(payload.get("model") or "")


def validate_prepared_batch(
    request_jsonl_path: str | Path,
    request_manifest_path: str | Path,
) -> PreparedBatch:
    """Cross-check exact JSONL requests against their provenance manifest."""
    manifest_rows, fields = _read_csv_rows(request_manifest_path)
    request_stage, planned_count, pilot_size, _request_indices = _parse_stage_columns(
        manifest_rows, fields
    )
    missing = sorted(set(REQUEST_MANIFEST_FIELDS) - fields)
    if missing:
        raise BatchIntegrityError(
            f"request manifest missing provenance columns: {missing}"
        )
    row_by_id: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        custom_id = row["custom_id"]
        if custom_id in row_by_id:
            raise BatchIntegrityError(
                f"request manifest has duplicate custom_id: {custom_id}"
            )
        row_by_id[custom_id] = row

    requests: list[dict[str, Any]] = []
    request_serializations: list[str] = []
    seen: set[str] = set()
    with Path(request_jsonl_path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(
                    f"request JSONL line {line_no} is invalid JSON"
                ) from exc
            custom_id = str(request.get("custom_id") or "")
            if not custom_id:
                raise BatchIntegrityError(
                    f"request JSONL line {line_no} has no custom_id"
                )
            if custom_id in seen:
                raise BatchIntegrityError(
                    f"request JSONL has duplicate custom_id: {custom_id}"
                )
            if custom_id not in row_by_id:
                raise BatchIntegrityError(
                    f"request JSONL has extra custom_id: {custom_id}"
                )
            row = row_by_id[custom_id]
            observed_model = _request_model(request)
            expected_model = row.get("requested_model") or row.get("model")
            if observed_model != expected_model:
                raise BatchIntegrityError(
                    f"{custom_id}: request model mismatch: "
                    f"{observed_model} != {expected_model}"
                )
            observed_hash = request_sha256(request)
            if observed_hash != row.get("request_sha256"):
                raise BatchIntegrityError(
                    f"{custom_id}: request body hash does not match manifest"
                )
            seen.add(custom_id)
            requests.append(request)
            request_serializations.append(line.rstrip("\r\n"))

    missing_ids = set(row_by_id) - seen
    if missing_ids:
        first = sorted(missing_ids)[0]
        raise BatchIntegrityError(
            f"request JSONL is missing {len(missing_ids)} manifest custom_ids; first={first}"
        )

    provider = _single_value(manifest_rows, "provider", required=True)
    requested_model = _single_value(
        manifest_rows, "requested_model", required=True
    )
    model_identity = _single_value(
        manifest_rows, "model_identity_sha256", required=True
    )
    panel_id = _single_value(manifest_rows, "panel_id", required=False)
    panel_hash = _single_value(manifest_rows, "panel_sha256", required=False)
    key_hash = _single_value(
        manifest_rows, "task_item_keys_sha256", required=True
    )
    return PreparedBatch(
        requests=tuple(requests),
        request_serializations=tuple(request_serializations),
        manifest_rows=tuple(manifest_rows),
        provider=provider,
        requested_model=requested_model,
        model_identity_sha256=model_identity,
        panel_id=panel_id,
        panel_sha256=panel_hash,
        task_item_keys_sha256=key_hash,
        request_jsonl_sha256=file_sha256(request_jsonl_path),
        request_manifest_sha256=file_sha256(request_manifest_path),
        request_stage=request_stage,
        planned_request_count=planned_count,
        pilot_size=pilot_size,
    )


def write_metadata(path: str | Path, metadata: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n"
    )


def validate_cost_approval(
    preflight_report_path: str | Path,
    request_jsonl_path: str | Path,
    request_manifest_path: str | Path,
    approved_budget_usd: str | Decimal,
) -> dict[str, Any]:
    """Fail closed unless an exact token report matches and fits the acknowledged cap."""
    prepared = validate_prepared_batch(request_jsonl_path, request_manifest_path)
    report = json.loads(Path(preflight_report_path).read_text())
    token_estimator = report.get("token_estimator")
    valid_count_method = report.get("offline_preflight") is True or (
        prepared.provider == "anthropic"
        and isinstance(token_estimator, Mapping)
        and token_estimator.get("type") == "anthropic_token_count"
        and isinstance(report.get("token_count_provenance"), Mapping)
        and bool(report["token_count_provenance"].get("sha256"))
    )
    if report.get("schema_version") != 1 or not valid_count_method:
        raise BatchIntegrityError("invalid exact cost-preflight report")
    expected = {
        "request_jsonl_sha256": prepared.request_jsonl_sha256,
        "request_manifest_sha256": prepared.request_manifest_sha256,
        "provider": prepared.provider,
        "requested_model": prepared.requested_model,
        "model_identity_sha256": prepared.model_identity_sha256,
        "panel_sha256": prepared.panel_sha256,
        "task_item_keys_sha256": prepared.task_item_keys_sha256,
        "request_count": len(prepared.requests),
        "request_stage": prepared.request_stage,
        "planned_request_count": prepared.planned_request_count,
        "pilot_size": prepared.pilot_size,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise BatchIntegrityError(
                f"preflight report {field} does not match prepared batch"
            )
    try:
        approved = Decimal(str(approved_budget_usd))
        maximum = Decimal(str(report["maximum_cost_usd"]))
    except (InvalidOperation, KeyError) as exc:
        raise BatchIntegrityError("invalid approved or preflight dollar amount") from exc
    if approved < 0:
        raise BatchIntegrityError("approved budget must be non-negative")
    if maximum > approved:
        raise BatchIntegrityError(
            f"preflight maximum ${maximum} exceeds approved budget ${approved}"
        )
    approved_report = dict(report)
    approved_report["_validated_cost_approval"] = True
    approved_report["approved_budget_usd"] = str(approved)
    return approved_report


def validate_aggregate_approval(
    aggregate_report_path: str | Path,
    approval_record_path: str | Path,
    preflight_report_path: str | Path,
    per_model_approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a model preflight to the exact user-approved aggregate cost report."""
    aggregate_path = Path(aggregate_report_path)
    approval_path = Path(approval_record_path)
    preflight_path = Path(preflight_report_path)
    aggregate = json.loads(aggregate_path.read_text())
    approval = json.loads(approval_path.read_text())
    if (
        aggregate.get("schema_version") != 1
        or aggregate.get("approval_ready") is not True
        or aggregate.get("submission_enabled") is not False
    ):
        raise BatchIntegrityError("aggregate cost report is not approval-ready")
    hard_budget_raw = aggregate.get("hard_budget_usd")
    compact_approval = per_model_approval.get("panel_id") == "frontier_compact_8"
    if compact_approval and aggregate.get("report_type") != (
        "frontier_compact8_api_cost_approval"
    ):
        raise BatchIntegrityError(
            "compact8 submission requires the compact aggregate cost report"
        )
    if compact_approval and hard_budget_raw is None:
        raise BatchIntegrityError("compact8 aggregate cost report has no hard budget")
    if hard_budget_raw is not None:
        try:
            hard_budget = Decimal(str(hard_budget_raw))
            overall_maximum = Decimal(str(aggregate["overall_maximum_cost_usd"]))
        except (InvalidOperation, KeyError) as exc:
            raise BatchIntegrityError(
                "aggregate cost report has an invalid hard budget"
            ) from exc
        if hard_budget > COMPACT_API_HARD_BUDGET_USD:
            raise BatchIntegrityError(
                "compact API aggregate hard budget exceeds $50"
            )
        if overall_maximum > hard_budget:
            raise BatchIntegrityError(
                "aggregate maximum exceeds its hard budget"
            )
        if aggregate.get("budget_compliant") is not True:
            raise BatchIntegrityError("aggregate cost report is not budget compliant")
    if approval.get("schema_version") != 1 or approval.get("approval_status") != "approved":
        raise BatchIntegrityError("explicit aggregate approval record is missing or invalid")
    expected_approval = {
        "aggregate_report_sha256": file_sha256(aggregate_path),
        "overall_maximum_cost_usd": aggregate.get("overall_maximum_cost_usd"),
        "approved_model_ids": [row["model_id"] for row in aggregate.get("models", [])],
    }
    for field, value in expected_approval.items():
        if approval.get(field) != value:
            raise BatchIntegrityError(
                f"aggregate approval {field} does not match the cost report"
            )
    requested_model = str(per_model_approval.get("requested_model") or "")
    matching = [
        row
        for row in aggregate.get("models", [])
        if row.get("model_id") == requested_model
    ]
    if len(matching) != 1:
        raise BatchIntegrityError("requested model is not unique in aggregate approval")
    row = matching[0]
    staged_reports = row.get("preflight_reports")
    if isinstance(staged_reports, list):
        matching_reports = [
            report
            for report in staged_reports
            if isinstance(report, Mapping)
            and report.get("sha256") == file_sha256(preflight_path)
            and Path(str(report.get("path") or "")).resolve()
            == preflight_path.resolve()
            and str(report.get("maximum_cost_usd"))
            == str(per_model_approval.get("maximum_cost_usd"))
        ]
        preflight_matches = len(matching_reports) == 1
    else:
        preflight_matches = (
            row.get("preflight_report_sha256") == file_sha256(preflight_path)
            and Path(str(row.get("preflight_report_path") or "")).resolve()
            == preflight_path.resolve()
            and str(row.get("maximum_cost_usd"))
            == str(per_model_approval.get("maximum_cost_usd"))
        )
    if not preflight_matches:
        raise BatchIntegrityError(
            "model preflight does not match the approved aggregate report"
        )
    validated = dict(per_model_approval)
    validated["_validated_aggregate_approval"] = True
    validated["aggregate_report_sha256"] = file_sha256(aggregate_path)
    validated["approval_record_sha256"] = file_sha256(approval_path)
    return validated


def require_validated_cost_approval(approval: Mapping[str, Any] | None) -> None:
    if (
        not approval
        or approval.get("_validated_cost_approval") is not True
        or approval.get("_validated_aggregate_approval") is not True
    ):
        raise BatchIntegrityError(
            "provider submission requires a locally validated cost approval and "
            "matching aggregate approval"
        )
