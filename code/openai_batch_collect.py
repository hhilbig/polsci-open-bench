#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import benchmark
from api_batch_common import (
    BatchIntegrityError,
    RequestManifest,
    load_request_manifest,
    load_task_selection,
    validate_response_model,
)
from task_registry import load_task_definitions


REPO = Path(__file__).resolve().parent.parent


def _load_collection_manifest(
    tasks_dir: str | Path,
    model_name: str | None,
    manifest_path: str | Path | None,
    panel_manifest_path: str | Path | None,
) -> RequestManifest:
    tasks = load_task_definitions(tasks_dir=tasks_dir)
    selection = load_task_selection(tasks, panel_manifest_path)
    tasks = list(selection.tasks)
    if manifest_path is not None:
        return load_request_manifest(
            manifest_path,
            tasks,
            model_name=model_name,
            provider="openai",
            selection=selection,
            require_enhanced=selection.is_frozen_panel,
            require_complete=selection.is_frozen_panel,
        )
    if selection.is_frozen_panel:
        raise BatchIntegrityError(
            "--panel-manifest requires the authoritative --request-manifest"
        )
    if not model_name:
        raise BatchIntegrityError("model_name is required without a request manifest")

    rows = []
    lookup = {}
    row_by_id = {}
    for task in tasks:
        for item in task["loader"]():
            custom_id = f"{task['name']}|{model_name}|{item['item_id']}"
            row = {
                "custom_id": custom_id,
                "task": task["name"],
                "model": model_name,
                "item_id": str(item["item_id"]),
                "requested_model": model_name,
            }
            rows.append(row)
            lookup[custom_id] = (task, item)
            row_by_id[custom_id] = row
    return RequestManifest(
        rows=tuple(rows),
        row_by_custom_id=row_by_id,
        lookup=lookup,
        provider="openai",
        requested_model=model_name,
        expected_response_model="",
        model_identity_sha256="",
        panel_id="",
        panel_sha256="",
        task_item_keys_sha256=selection.task_item_keys_sha256,
    )


def build_item_lookup(
    tasks_dir: str | Path,
    model_name: str,
    manifest_path: str | Path | None = None,
    panel_manifest_path: str | Path | None = None,
) -> dict[str, tuple[dict, dict]]:
    return dict(
        _load_collection_manifest(
            tasks_dir,
            model_name,
            manifest_path,
            panel_manifest_path,
        ).lookup
    )


def response_content(result: dict) -> tuple[str, int | None, str | None]:
    error = result.get("error")
    if error:
        return "", None, f"api_error: {error}"
    response = result.get("response") or {}
    status_code = response.get("status_code")
    if status_code != 200:
        return "", None, f"api_error_status_{status_code}: {response.get('body')}"
    body = response.get("body") or {}
    choices = body.get("choices") or []
    content = ""
    if choices:
        content = ((choices[0].get("message") or {}).get("content") or "")
    usage = body.get("usage") or {}
    eval_count = usage.get("completion_tokens")
    return content, eval_count, None


def reported_model(result: dict) -> str | None:
    response = result.get("response") or {}
    body = response.get("body") or {}
    return body.get("model")


def collect(
    batch_jsonl: Path,
    tasks_dir: str | Path,
    model_name: str | None = None,
    manifest_path: str | Path | None = None,
    panel_manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    request_manifest = _load_collection_manifest(
        tasks_dir,
        model_name,
        manifest_path,
        panel_manifest_path,
    )
    lookup = request_manifest.lookup
    resolved_model = request_manifest.requested_model
    rows = []
    seen = set()
    with batch_jsonl.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            if custom_id not in lookup:
                raise KeyError(f"Unknown custom_id on line {line_no}: {custom_id}")
            if custom_id in seen:
                raise BatchIntegrityError(
                    f"Duplicate custom_id on line {line_no}: {custom_id}"
                )
            task, item = lookup[custom_id]
            manifest_row = request_manifest.row_by_custom_id[custom_id]
            response_model = reported_model(result)
            validate_response_model(
                response_model,
                manifest_row,
                context=f"line {line_no} {custom_id}",
            )
            content, eval_count, api_err = response_content(result)
            if api_err:
                preds = {}
                parse_err = api_err
            else:
                preds, parse_err = benchmark.parse_content(content, task)
            row = {
                "task": task["name"],
                "model": resolved_model,
                "item_id": item["item_id"],
                "latency_s": None,
                "eval_count": eval_count,
                "parse_error": parse_err,
                "raw_content_preview": content[:200],
                "response_model": response_model,
                "model_identity_sha256": request_manifest.model_identity_sha256,
                "panel_id": request_manifest.panel_id,
                "panel_sha256": request_manifest.panel_sha256,
                "request_stage": manifest_row.get("request_stage", ""),
                "request_index": manifest_row.get("request_index", ""),
            }
            if api_err:
                if task.get("label_key"):
                    row[f"pred_{task['label_key']}"] = None
                else:
                    for label in task["labels"]:
                        row[f"pred_{label}"] = None
            else:
                for key, value in preds.items():
                    row[f"pred_{key}"] = value
            for key, value in item["gt"].items():
                row[f"gt_{key}"] = value
            rows.append(row)
            seen.add(custom_id)
    missing = set(lookup) - seen
    if missing:
        raise ValueError(f"Batch output missing {len(missing)} custom_ids; first missing: {sorted(missing)[0]}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert OpenAI Batch API output JSONL into benchmark prediction CSV.")
    parser.add_argument("--batch-jsonl", required=True)
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument(
        "--panel-manifest",
        help="Frozen panel YAML; requires the exact authoritative request manifest.",
    )
    parser.add_argument(
        "--request-manifest",
        "--manifest",
        dest="request_manifest",
        help="Prepared request manifest CSV; authoritative for expected responses.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Expected model (otherwise read from --request-manifest).",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_name = args.model
    if model_name is None and args.request_manifest is None:
        model_name = "gpt-5.5"
    df = collect(
        Path(args.batch_jsonl),
        args.tasks_dir,
        model_name,
        args.request_manifest,
        args.panel_manifest,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    parse_errors = df["parse_error"].notna().sum()
    print(f"wrote {out} ({len(df)} rows, {parse_errors} parse/api errors)")
    print(df.groupby("task").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
