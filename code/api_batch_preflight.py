#!/usr/bin/env python3
"""Offline token and maximum-cost preflight for prepared provider batches."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping

import yaml

from api_batch_common import (
    BatchIntegrityError,
    PreparedBatch,
    file_sha256,
    validate_prepared_batch,
    write_metadata,
)


_DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_MILLION = Decimal("1000000")


def _decimal(value: Any, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise BatchIntegrityError(f"pricing field {field} must be numeric") from exc
    if not result.is_finite() or result < minimum:
        raise BatchIntegrityError(
            f"pricing field {field} must be at least {minimum}"
        )
    return result


def load_pricing(
    path: str | Path,
    prepared: PreparedBatch,
) -> dict[str, Any]:
    pricing_path = Path(path)
    raw = yaml.safe_load(pricing_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise BatchIntegrityError("pricing manifest schema_version must be 1")
    if raw.get("currency") != "USD":
        raise BatchIntegrityError("pricing manifest currency must be USD")
    models = raw.get("models")
    if not isinstance(models, dict):
        raise BatchIntegrityError("pricing manifest models must be a mapping")
    entry = models.get(prepared.requested_model)
    if not isinstance(entry, dict):
        raise BatchIntegrityError(
            f"pricing manifest has no exact entry for {prepared.requested_model}"
        )
    if str(entry.get("provider") or "") != prepared.provider:
        raise BatchIntegrityError("pricing provider does not match prepared batch")
    pinned_identity = str(entry.get("model_identity_sha256") or "")
    if pinned_identity and pinned_identity != prepared.model_identity_sha256:
        raise BatchIntegrityError("pricing model identity does not match prepared batch")

    source_url = str(entry.get("source_url") or "")
    if not source_url.startswith("https://"):
        raise BatchIntegrityError("pricing source_url must be an https URL")
    effective_date = str(entry.get("effective_date") or "")
    if not _DATE_RE.fullmatch(effective_date):
        raise BatchIntegrityError("pricing effective_date must be YYYY-MM-DD")

    tokenizer = entry.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise BatchIntegrityError("pricing tokenizer must be a mapping")
    tokenizer_type = str(tokenizer.get("type") or "")
    if tokenizer_type not in {"tiktoken", "anthropic_token_count"}:
        raise BatchIntegrityError(
            "pricing tokenizer.type must be tiktoken or anthropic_token_count"
        )
    if prepared.provider == "openai" and tokenizer_type != "tiktoken":
        raise BatchIntegrityError("OpenAI preflight requires a pinned local tokenizer")
    if prepared.provider == "anthropic" and tokenizer_type != "anthropic_token_count":
        raise BatchIntegrityError("Anthropic preflight requires official token counts")
    if tokenizer_type == "tiktoken" and not str(tokenizer.get("encoding") or ""):
        raise BatchIntegrityError("pricing tokenizer.encoding is required")
    if tokenizer_type == "anthropic_token_count" and not str(
        tokenizer.get("endpoint") or ""
    ):
        raise BatchIntegrityError("pricing tokenizer.endpoint is required")

    return {
        "provider": prepared.provider,
        "requested_model": prepared.requested_model,
        "model_identity_sha256": pinned_identity,
        "batch_input_usd_per_million_tokens": _decimal(
            entry.get("batch_input_usd_per_million_tokens"),
            "batch_input_usd_per_million_tokens",
        ),
        "batch_output_usd_per_million_tokens": _decimal(
            entry.get("batch_output_usd_per_million_tokens"),
            "batch_output_usd_per_million_tokens",
        ),
        "input_token_safety_multiplier": _decimal(
            entry.get("input_token_safety_multiplier"),
            "input_token_safety_multiplier",
            minimum=Decimal("1"),
        ),
        "estimated_output_tokens_per_request": int(
            entry["estimated_output_tokens_per_request"]
        ),
        "tokenizer": dict(tokenizer),
        "source_url": source_url,
        "effective_date": effective_date,
        "pricing_manifest_path": str(pricing_path.resolve()),
        "pricing_manifest_sha256": file_sha256(pricing_path),
    }


def _load_encoding(name: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise BatchIntegrityError(
            "offline preflight requires the pinned tiktoken dependency"
        ) from exc
    try:
        return tiktoken.get_encoding(name)
    except ValueError as exc:
        raise BatchIntegrityError(f"unknown tiktoken encoding: {name}") from exc


def _load_anthropic_token_counts(
    path: str | Path,
    prepared: PreparedBatch,
) -> tuple[list[int], dict[str, Any]]:
    cache_path = Path(path)
    raw = json.loads(cache_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise BatchIntegrityError("Anthropic token-count cache schema_version must be 1")
    expected = {
        "provider": "anthropic",
        "requested_model": prepared.requested_model,
        "request_jsonl_sha256": prepared.request_jsonl_sha256,
        "request_manifest_sha256": prepared.request_manifest_sha256,
        "endpoint": "messages.count_tokens",
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise BatchIntegrityError(
                f"Anthropic token-count cache {field} does not match prepared batch"
            )
    records = raw.get("counts")
    if not isinstance(records, list):
        raise BatchIntegrityError("Anthropic token-count cache counts must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise BatchIntegrityError("Anthropic token-count record must be a mapping")
        custom_id = str(record.get("custom_id") or "")
        if not custom_id or custom_id in by_id:
            raise BatchIntegrityError("Anthropic token-count cache has duplicate custom_id")
        by_id[custom_id] = record
    values: list[int] = []
    for request, manifest_row in zip(
        prepared.requests, prepared.manifest_rows, strict=True
    ):
        custom_id = str(request.get("custom_id") or "")
        if custom_id != str(manifest_row["custom_id"]):
            raise BatchIntegrityError("prepared request/manifest order mismatch")
        record = by_id.get(custom_id)
        if record is None:
            raise BatchIntegrityError(
                f"Anthropic token-count cache is missing {custom_id}"
            )
        if str(record.get("request_sha256") or "") != str(
            manifest_row["request_sha256"]
        ):
            raise BatchIntegrityError(
                f"Anthropic token-count request hash mismatch for {custom_id}"
            )
        try:
            count = int(record["input_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BatchIntegrityError(
                f"Anthropic token-count value is invalid for {custom_id}"
            ) from exc
        if count < 0:
            raise BatchIntegrityError(
                f"Anthropic token-count value is negative for {custom_id}"
            )
        values.append(count)
    if set(by_id) != {str(request["custom_id"]) for request in prepared.requests}:
        raise BatchIntegrityError("Anthropic token-count cache has extra request IDs")
    provenance = {
        "path": str(cache_path.resolve()),
        "sha256": file_sha256(cache_path),
        "counted_at": raw.get("completed_at"),
        "sdk_version": raw.get("sdk_version"),
        "endpoint": raw.get("endpoint"),
    }
    return values, provenance


def _max_output_tokens(request: Mapping[str, Any]) -> int:
    payload = request.get("body") if "body" in request else request.get("params")
    if not isinstance(payload, Mapping):
        raise BatchIntegrityError("request has neither a body nor params mapping")
    value = payload.get("max_completion_tokens", payload.get("max_tokens"))
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise BatchIntegrityError("request has no valid output-token limit") from exc
    if tokens < 1:
        raise BatchIntegrityError("request output-token limit must be positive")
    return tokens


def _usd(value: Decimal) -> str:
    # Round ceilings upward so the report never understates its displayed cap.
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def build_preflight_report(
    request_jsonl_path: str | Path,
    request_manifest_path: str | Path,
    pricing_manifest_path: str | Path,
    token_count_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Price exact prepared requests from a local tokenizer or cached official counts."""
    prepared = validate_prepared_batch(request_jsonl_path, request_manifest_path)
    pricing = load_pricing(pricing_manifest_path, prepared)
    tokenizer = pricing["tokenizer"]
    token_count_provenance: dict[str, Any] | None = None
    if tokenizer["type"] == "tiktoken":
        encoding = _load_encoding(str(tokenizer["encoding"]))
        per_request_input = [
            len(encoding.encode(serialized_request))
            for serialized_request in prepared.request_serializations
        ]
        token_method = "exact_jsonl_request_line"
        token_note = (
            "Pinned local tokenizer over the exact serialized request; the input cap "
            "applies the declared safety multiplier."
        )
    else:
        if token_count_cache_path is None:
            raise BatchIntegrityError(
                "Anthropic preflight requires --token-count-cache from the official endpoint"
            )
        per_request_input, token_count_provenance = _load_anthropic_token_counts(
            token_count_cache_path,
            prepared,
        )
        token_method = "official_messages_count_tokens_per_exact_request"
        token_note = "Cached response from Anthropic's free token-counting endpoint."
    estimated_input_tokens = sum(per_request_input)
    max_input_tokens = math.ceil(
        Decimal(estimated_input_tokens)
        * pricing["input_token_safety_multiplier"]
    )
    per_request_output_caps = [_max_output_tokens(request) for request in prepared.requests]
    maximum_output_tokens = sum(per_request_output_caps)
    estimated_per_request = pricing["estimated_output_tokens_per_request"]
    if estimated_per_request < 0:
        raise BatchIntegrityError(
            "estimated_output_tokens_per_request must be non-negative"
        )
    estimated_output_tokens = sum(
        min(estimated_per_request, cap) for cap in per_request_output_caps
    )

    input_rate = pricing["batch_input_usd_per_million_tokens"]
    output_rate = pricing["batch_output_usd_per_million_tokens"]
    estimated_input_cost = Decimal(estimated_input_tokens) * input_rate / _MILLION
    maximum_input_cost = Decimal(max_input_tokens) * input_rate / _MILLION
    estimated_output_cost = Decimal(estimated_output_tokens) * output_rate / _MILLION
    maximum_output_cost = Decimal(maximum_output_tokens) * output_rate / _MILLION

    task_counts = Counter(row["task"] for row in prepared.manifest_rows)
    report = {
        "schema_version": 1,
        "offline_preflight": tokenizer["type"] == "tiktoken",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "request_jsonl_path": str(Path(request_jsonl_path).resolve()),
        "request_jsonl_sha256": prepared.request_jsonl_sha256,
        "request_manifest_path": str(Path(request_manifest_path).resolve()),
        "request_manifest_sha256": prepared.request_manifest_sha256,
        "pricing_manifest_path": pricing["pricing_manifest_path"],
        "pricing_manifest_sha256": pricing["pricing_manifest_sha256"],
        "pricing_source_url": pricing["source_url"],
        "pricing_effective_date": pricing["effective_date"],
        "provider": prepared.provider,
        "requested_model": prepared.requested_model,
        "model_identity_sha256": prepared.model_identity_sha256,
        "panel_id": prepared.panel_id,
        "panel_sha256": prepared.panel_sha256,
        "task_item_keys_sha256": prepared.task_item_keys_sha256,
        "request_count": len(prepared.requests),
        "request_stage": prepared.request_stage,
        "planned_request_count": prepared.planned_request_count,
        "pilot_size": prepared.pilot_size,
        "task_count": len(task_counts),
        "task_counts": dict(sorted(task_counts.items())),
        "token_estimator": {
            **pricing["tokenizer"],
            "method": token_method,
            "note": token_note,
        },
        "token_count_provenance": token_count_provenance,
        "estimated_input_tokens": estimated_input_tokens,
        "input_token_safety_multiplier": str(
            pricing["input_token_safety_multiplier"]
        ),
        "maximum_input_tokens": max_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "batch_input_usd_per_million_tokens": str(input_rate),
        "batch_output_usd_per_million_tokens": str(output_rate),
        "estimated_input_cost_usd": _usd(estimated_input_cost),
        "estimated_output_cost_usd": _usd(estimated_output_cost),
        "estimated_total_cost_usd": _usd(
            estimated_input_cost + estimated_output_cost
        ),
        "maximum_input_cost_usd": _usd(maximum_input_cost),
        "maximum_output_cost_usd": _usd(maximum_output_cost),
        "maximum_cost_usd": _usd(maximum_input_cost + maximum_output_cost),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate exact prepared batch request tokens and maximum cost offline. "
            "This command never creates a provider client."
        )
    )
    parser.add_argument("--request-jsonl", required=True)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--pricing-manifest", required=True)
    parser.add_argument(
        "--token-count-cache",
        help="Anthropic official token-count cache for the exact prepared requests.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_preflight_report(
        args.request_jsonl,
        args.request_manifest,
        args.pricing_manifest,
        token_count_cache_path=args.token_count_cache,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
