#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import benchmark
from api_batch_common import (
    REQUEST_MANIFEST_FIELDS,
    BatchTaskSelection,
    ModelIdentity,
    RequestStagePlan,
    add_request_stage_fields,
    build_request_stage_plan,
    build_model_identity,
    file_sha256,
    load_task_selection,
    request_manifest_row,
    require_validated_cost_approval,
    staged_manifest_fields,
    validate_aggregate_approval,
    validate_cost_approval,
    validate_prepared_batch,
    write_metadata,
)
from api_pilot_qualification import validate_pilot_qualification
from model_registry import load_model_definition
from task_registry import load_task_definitions


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUTDIR = REPO / "output" / "openai_batch"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_chat_body(model: dict, task: dict, system_prompt: str, user_content: str) -> dict:
    schema_mode = benchmark.response_format_type(model)
    if schema_mode == "json_object":
        system_prompt = benchmark.augment_system_prompt_for_json_object(system_prompt, task)

    body = {
        "model": model["name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if schema_mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": task["json_schema"],
            },
        }
    else:
        body["response_format"] = {"type": "json_object"}

    if model["name"].startswith("gpt-5"):
        body["max_completion_tokens"] = int(model.get("max_output_tokens") or 2000)
    else:
        body["max_tokens"] = int(model.get("max_output_tokens") or 1024)
        body["temperature"] = 0.1

    if model.get("reasoning_effort"):
        body["reasoning_effort"] = model["reasoning_effort"]

    extra_body = benchmark.extra_body_for_openai_model(model)
    if extra_body:
        body.update(extra_body)

    return body


def iter_requests(
    tasks: list[dict],
    model: dict,
    request_stage: RequestStagePlan | None = None,
):
    request_index = 0
    for task in tasks:
        system_prompt = Path(task["prompt_path"]).read_text()
        for item in task["loader"]():
            request_index += 1
            if request_stage is not None and not request_stage.includes(request_index):
                continue
            custom_id = f"{task['name']}|{model['name']}|{item['item_id']}"
            body = build_chat_body(model, task, system_prompt, item["user_content"])
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            yield request_index, task, item, request


def write_batch_files(
    tasks: list[dict],
    model: dict,
    outdir: Path,
    prefix: str,
    *,
    model_identity: ModelIdentity | None = None,
    selection: BatchTaskSelection | None = None,
    request_stage: str | None = None,
    pilot_size: int = 16,
) -> tuple[Path, Path, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{prefix}.jsonl"
    manifest_path = outdir / f"{prefix}_manifest.csv"
    metadata_path = outdir / f"{prefix}_metadata.json"
    identity = model_identity or build_model_identity(model)
    selected = selection or load_task_selection(tasks, None)
    stage_plan = (
        build_request_stage_plan(
            request_stage,
            selected.expected_items,
            pilot_size=pilot_size,
        )
        if request_stage is not None
        else None
    )

    counts: dict[str, int] = {}
    n_requests = 0
    with jsonl_path.open("w") as jf, manifest_path.open("w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=(
                staged_manifest_fields()
                if stage_plan is not None
                else REQUEST_MANIFEST_FIELDS
            ),
        )
        writer.writeheader()
        for request_index, task, item, request in iter_requests(
            tasks, model, stage_plan
        ):
            jf.write(json.dumps(request, ensure_ascii=False) + "\n")
            manifest_row = request_manifest_row(
                    custom_id=request["custom_id"],
                    task_name=task["name"],
                    item_id=item["item_id"],
                    request=request,
                    model_identity=identity,
                    selection=selected,
                    original_custom_id=request["custom_id"],
                )
            if stage_plan is not None:
                manifest_row = add_request_stage_fields(
                    manifest_row,
                    stage=stage_plan,
                    request_index=request_index,
                )
            writer.writerow(manifest_row)
            counts[task["name"]] = counts.get(task["name"], 0) + 1
            n_requests += 1

    expected_requests = (
        stage_plan.expected_request_count
        if stage_plan is not None
        else selected.expected_items
    )
    if selected.is_frozen_panel and n_requests != expected_requests:
        raise ValueError(
            f"prepared {n_requests} requests, expected {expected_requests}"
        )
    meta = {
        "schema_version": 1,
        "provider": identity.provider,
        "model": identity.requested_model,
        "requested_model": identity.requested_model,
        "expected_response_model": identity.expected_response_model,
        "model_identity_sha256": identity.model_identity_sha256,
        "model_manifest_sha256": identity.model_manifest_sha256,
        "panel_id": selected.panel_id,
        "panel_sha256": selected.panel_sha256,
        "benchmark_commit": selected.benchmark_commit,
        "task_item_keys_sha256": selected.task_item_keys_sha256,
        "tasks": len(counts),
        "requests": n_requests,
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": file_sha256(jsonl_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "metadata_path": str(metadata_path),
        "task_counts": counts,
        "request_stage": stage_plan.name if stage_plan else "",
        "planned_request_count": (
            stage_plan.planned_request_count if stage_plan else n_requests
        ),
        "pilot_size": stage_plan.pilot_size if stage_plan else None,
        "expected_stage_requests": expected_requests,
    }
    write_metadata(metadata_path, meta)
    return jsonl_path, manifest_path, meta


def submit_batch(
    jsonl_path: Path,
    metadata: dict[str, str],
    *,
    cost_approval: dict | None = None,
) -> dict:
    require_validated_cost_approval(cost_approval)
    client = OpenAI()
    uploaded = client.files.create(file=jsonl_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )
    return {
        "file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "created_at": batch.created_at,
    }


def retrieve_batch(batch_id: str) -> dict:
    batch = OpenAI().batches.retrieve(batch_id)
    return batch.model_dump(mode="json")


def download_file(file_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = OpenAI().files.content(file_id)
    content.write_to_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and submit OpenAI Batch API jobs for benchmark tasks.")
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument(
        "--panel-manifest",
        help="Frozen panel YAML; selects and validates its exact task/item set.",
    )
    parser.add_argument("--model-manifest", default=str(REPO / "models" / "gpt_5_5.yaml"))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--request-stage",
        choices=["all", "pilot", "remainder"],
        default=None,
        help="Prepare the exact full order, first 16-request pilot, or remaining requests.",
    )
    parser.add_argument("--pilot-size", type=int, default=16)
    parser.add_argument("--submit", action="store_true", help="Upload JSONL and create the OpenAI batch.")
    parser.add_argument("--status", help="Retrieve and print status for an existing OpenAI batch id.")
    parser.add_argument("--download-file-id", help="Download an OpenAI file id to --download-output.")
    parser.add_argument("--download-output", help="Path for --download-file-id output.")
    parser.add_argument(
        "--preflight-report",
        help="Offline preflight JSON matching the prepared requests; required with --submit.",
    )
    parser.add_argument(
        "--budget-usd",
        default=None,
        help="Explicitly acknowledged maximum dollar cost; required with --submit.",
    )
    parser.add_argument(
        "--aggregate-report",
        help="Exact aggregate cost report reviewed by the user; required with --submit.",
    )
    parser.add_argument(
        "--approval-record",
        help="Explicit approval record bound to --aggregate-report; required with --submit.",
    )
    parser.add_argument(
        "--pilot-qualification",
        help="Passed, hash-pinned pilot qualification; required to submit compact8 remainder.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.download_file_id:
        if not args.download_output:
            raise ValueError("--download-output is required with --download-file-id")
        download_file(args.download_file_id, Path(args.download_output))
        print(f"Wrote {args.download_output}")
        return 0

    if args.status:
        outdir.mkdir(parents=True, exist_ok=True)
        status = retrieve_batch(args.status)
        status_path = outdir / f"{args.status}_status.json"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(json.dumps(status, indent=2, sort_keys=True))
        print(f"Wrote {status_path}")
        return 0

    tasks = load_task_definitions(tasks_dir=args.tasks_dir)
    selection = load_task_selection(tasks, args.panel_manifest)
    tasks = list(selection.tasks)
    model_manifest_path = Path(args.model_manifest)
    model = load_model_definition(model_manifest_path)
    if model["backend"] != "openai" or model.get("provider") == "deepseek":
        raise ValueError("This helper is only for OpenAI-hosted OpenAI-compatible chat-completion models.")
    identity = build_model_identity(model, model_manifest_path)

    prefix = args.prefix or f"{model['name'].replace(':', '_')}_tasks_batch_{now_stamp()}"
    jsonl_path, manifest_path, meta = write_batch_files(
        tasks,
        model,
        outdir,
        prefix,
        model_identity=identity,
        selection=selection,
        request_stage=args.request_stage,
        pilot_size=args.pilot_size,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))

    if args.submit:
        if selection.panel_id == "frontier_compact_8" and args.request_stage not in {
            "pilot",
            "remainder",
        }:
            raise ValueError(
                "compact8 submission must use --request-stage pilot or remainder"
            )
        if (
            not args.preflight_report
            or args.budget_usd is None
            or not args.aggregate_report
            or not args.approval_record
        ):
            raise ValueError(
                "--submit requires --preflight-report, --budget-usd, "
                "--aggregate-report, and --approval-record; "
                "prepare the batch, run api_batch_preflight.py, review its cap, "
                "then explicitly acknowledge that cap"
            )
        preflight = validate_cost_approval(
            args.preflight_report,
            jsonl_path,
            manifest_path,
            args.budget_usd,
        )
        preflight = validate_aggregate_approval(
            args.aggregate_report,
            args.approval_record,
            args.preflight_report,
            preflight,
        )
        pilot_qualification = None
        if selection.panel_id == "frontier_compact_8" and args.request_stage == "remainder":
            if not args.pilot_qualification:
                raise ValueError(
                    "compact8 remainder submission requires --pilot-qualification"
                )
            pilot_qualification = validate_pilot_qualification(
                args.pilot_qualification,
                validate_prepared_batch(jsonl_path, manifest_path),
            )
        meta["preflight_report_path"] = str(Path(args.preflight_report).resolve())
        meta["preflight_report_sha256"] = file_sha256(args.preflight_report)
        meta["preflight_maximum_cost_usd"] = preflight["maximum_cost_usd"]
        meta["approved_budget_usd"] = str(args.budget_usd)
        if pilot_qualification is not None:
            meta["pilot_qualification_path"] = str(
                Path(args.pilot_qualification).resolve()
            )
            meta["pilot_qualification_sha256"] = pilot_qualification[
                "qualification_report_sha256"
            ]
        write_metadata(meta["metadata_path"], meta)
        batch_meta = {
            "project": "polsci-open-bench",
            "scope": selection.panel_id or Path(args.tasks_dir).name,
            "model": identity.requested_model,
            "requests": str(meta["requests"]),
            "model_identity": identity.model_identity_sha256,
            "task_item_keys": selection.task_item_keys_sha256,
            "request_manifest": meta["manifest_sha256"],
            "approved_budget_usd": str(args.budget_usd),
        }
        if selection.panel_sha256:
            batch_meta["panel_sha256"] = selection.panel_sha256
        submit_meta = submit_batch(
            jsonl_path,
            batch_meta,
            cost_approval=preflight,
        )
        result_path = outdir / f"{prefix}_batch.json"
        result = {**meta, **submit_meta}
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
