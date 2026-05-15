#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import benchmark
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
        body["max_completion_tokens"] = 2000
    else:
        body["max_tokens"] = 1024
        body["temperature"] = 0.1

    if model.get("reasoning_effort"):
        body["reasoning_effort"] = model["reasoning_effort"]

    extra_body = benchmark.extra_body_for_openai_model(model)
    if extra_body:
        body.update(extra_body)

    return body


def iter_requests(tasks: list[dict], model: dict):
    for task in tasks:
        system_prompt = Path(task["prompt_path"]).read_text()
        for item in task["loader"]():
            custom_id = f"{task['name']}|{model['name']}|{item['item_id']}"
            body = build_chat_body(model, task, system_prompt, item["user_content"])
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            yield task, item, request


def write_batch_files(tasks: list[dict], model: dict, outdir: Path, prefix: str) -> tuple[Path, Path, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{prefix}.jsonl"
    manifest_path = outdir / f"{prefix}_manifest.csv"

    counts: dict[str, int] = {}
    n_requests = 0
    with jsonl_path.open("w") as jf, manifest_path.open("w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["custom_id", "task", "model", "item_id"],
        )
        writer.writeheader()
        for task, item, request in iter_requests(tasks, model):
            jf.write(json.dumps(request, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "custom_id": request["custom_id"],
                    "task": task["name"],
                    "model": model["name"],
                    "item_id": item["item_id"],
                }
            )
            counts[task["name"]] = counts.get(task["name"], 0) + 1
            n_requests += 1

    meta = {
        "model": model["name"],
        "tasks": len(counts),
        "requests": n_requests,
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "task_counts": counts,
    }
    return jsonl_path, manifest_path, meta


def submit_batch(jsonl_path: Path, metadata: dict[str, str]) -> dict:
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
    parser.add_argument("--model-manifest", default=str(REPO / "models" / "gpt_5_5.yaml"))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--submit", action="store_true", help="Upload JSONL and create the OpenAI batch.")
    parser.add_argument("--status", help="Retrieve and print status for an existing OpenAI batch id.")
    parser.add_argument("--download-file-id", help="Download an OpenAI file id to --download-output.")
    parser.add_argument("--download-output", help="Path for --download-file-id output.")
    parser.add_argument("--budget-usd", default=None, help="Budget/cost cap recorded in batch metadata.")
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
    model = load_model_definition(Path(args.model_manifest))
    if model["backend"] != "openai" or model.get("provider") == "deepseek":
        raise ValueError("This helper is only for OpenAI-hosted OpenAI-compatible chat-completion models.")

    prefix = args.prefix or f"{model['name'].replace(':', '_')}_tasks_batch_{now_stamp()}"
    jsonl_path, manifest_path, meta = write_batch_files(tasks, model, outdir, prefix)
    print(json.dumps(meta, indent=2, sort_keys=True))

    if args.submit:
        batch_meta = {
            "project": "polsci-open-bench",
            "scope": Path(args.tasks_dir).name,
            "model": model["name"],
            "requests": str(meta["requests"]),
        }
        if args.budget_usd:
            batch_meta["budget_usd"] = str(args.budget_usd)
        submit_meta = submit_batch(jsonl_path, batch_meta)
        result_path = outdir / f"{prefix}_batch.json"
        result = {**meta, **submit_meta}
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
