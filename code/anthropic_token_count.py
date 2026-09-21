#!/usr/bin/env python3
"""Cache free Anthropic input-token counts for exact prepared panel requests."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from api_batch_common import (
    BatchIntegrityError,
    canonical_sha256,
    validate_prepared_batch,
)


TOKEN_COUNT_INPUT_FIELDS = {
    "messages",
    "model",
    "output_config",
    "output_format",
    "system",
    "thinking",
    "tool_choice",
    "tools",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".anthropic_api_key"
    if path.is_file():
        key = path.read_text().strip()
    if not key:
        raise BatchIntegrityError(
            "Anthropic token counting requires ANTHROPIC_API_KEY or ~/.anthropic_api_key"
        )
    return key


def token_count_params(request_params: dict[str, Any]) -> dict[str, Any]:
    """Select every input-bearing field accepted by ``messages.count_tokens``."""
    result = {
        key: value
        for key, value in request_params.items()
        if key in TOKEN_COUNT_INPUT_FIELDS
    }
    if "model" not in result or "messages" not in result:
        raise BatchIntegrityError("Anthropic request lacks model or messages")
    unsupported = set(request_params) - TOKEN_COUNT_INPUT_FIELDS - {"max_tokens"}
    if unsupported:
        raise BatchIntegrityError(
            f"Anthropic count-token projection would omit unknown fields: {sorted(unsupported)}"
        )
    return result


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_existing(path: Path, expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    for field, value in expected.items():
        if raw.get(field) != value:
            raise BatchIntegrityError(
                f"existing Anthropic token-count cache {field} does not match"
            )
    records = raw.get("counts", [])
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        custom_id = str(record.get("custom_id") or "")
        if not custom_id or custom_id in by_id:
            raise BatchIntegrityError(
                "existing Anthropic token-count cache has duplicate custom_id"
            )
        by_id[custom_id] = dict(record)
    return by_id


async def count_requests(
    request_jsonl_path: str | Path,
    request_manifest_path: str | Path,
    output_path: str | Path,
    *,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Call only ``messages.count_tokens`` and persist resumable provenance."""
    if concurrency < 1 or concurrency > 32:
        raise BatchIntegrityError("concurrency must be between 1 and 32")
    prepared = validate_prepared_batch(request_jsonl_path, request_manifest_path)
    if prepared.provider != "anthropic":
        raise BatchIntegrityError("official Anthropic counting requires an Anthropic batch")
    output = Path(output_path)
    expected = {
        "schema_version": 1,
        "provider": "anthropic",
        "requested_model": prepared.requested_model,
        "model_identity_sha256": prepared.model_identity_sha256,
        "request_jsonl_sha256": prepared.request_jsonl_sha256,
        "request_manifest_sha256": prepared.request_manifest_sha256,
        "endpoint": "messages.count_tokens",
    }
    existing = _load_existing(output, expected)
    manifest_by_id = {
        str(row["custom_id"]): row for row in prepared.manifest_rows
    }
    for custom_id, record in existing.items():
        expected_hash = manifest_by_id.get(custom_id, {}).get("request_sha256")
        if expected_hash is None or record.get("request_sha256") != expected_hash:
            raise BatchIntegrityError(
                f"stale Anthropic token-count record for {custom_id}"
            )

    client = anthropic.AsyncAnthropic(
        api_key=load_api_key(),
        timeout=120.0,
        max_retries=5,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def count_one(request: dict[str, Any]) -> dict[str, Any]:
        custom_id = str(request["custom_id"])
        count_params = token_count_params(request["params"])
        async with semaphore:
            response = await client.messages.count_tokens(**count_params)
        payload = response.model_dump(mode="json")
        return {
            "custom_id": custom_id,
            "request_sha256": manifest_by_id[custom_id]["request_sha256"],
            "counted_payload_sha256": canonical_sha256(count_params),
            "input_tokens": int(response.input_tokens),
            "counted_at": now_utc(),
            "response_sha256": canonical_sha256(payload),
            "response": payload,
        }

    pending = [
        request
        for request in prepared.requests
        if str(request["custom_id"]) not in existing
    ]
    started_at = now_utc()
    try:
        for start in range(0, len(pending), concurrency * 4):
            chunk = pending[start : start + concurrency * 4]
            records = await asyncio.gather(*(count_one(request) for request in chunk))
            existing.update({record["custom_id"]: record for record in records})
            partial = {
                **expected,
                "sdk_version": importlib.metadata.version("anthropic"),
                "started_at": started_at,
                "completed_at": None,
                "complete": False,
                "request_count": len(prepared.requests),
                "counted_requests": len(existing),
                "counts": [existing[key] for key in sorted(existing)],
            }
            _atomic_write(output, partial)
    finally:
        await client.close()

    if len(existing) != len(prepared.requests):
        raise BatchIntegrityError("Anthropic token counting ended with incomplete coverage")
    result = {
        **expected,
        "sdk_version": importlib.metadata.version("anthropic"),
        "started_at": started_at,
        "completed_at": now_utc(),
        "complete": True,
        "request_count": len(prepared.requests),
        "counted_requests": len(existing),
        "counts": [existing[key] for key in sorted(existing)],
    }
    _atomic_write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-jsonl", required=True)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    result = asyncio.run(
        count_requests(
            args.request_jsonl,
            args.request_manifest,
            args.output,
            concurrency=args.concurrency,
        )
    )
    print(
        json.dumps(
            {
                "provider": result["provider"],
                "requested_model": result["requested_model"],
                "request_count": result["request_count"],
                "completed_at": result["completed_at"],
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
