#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

from benchmark import _load_api_key
from model_registry import load_model_definition
from task_registry import load_task_definitions


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUTDIR = REPO / "output" / "anthropic_batch"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_client(model: dict) -> anthropic.Anthropic:
    key = _load_api_key(
        model,
        default_env="ANTHROPIC_API_KEY",
        default_file=Path.home() / ".anthropic_api_key",
    )
    if not key:
        raise RuntimeError("Missing Anthropic API key in ANTHROPIC_API_KEY or ~/.anthropic_api_key")
    return anthropic.Anthropic(api_key=key, timeout=300.0, max_retries=2)


def anthropic_safe_key(label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", label)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "label"
    if safe[0].isdigit():
        safe = f"label_{safe}"
    return safe[:64]


def anthropic_key_map(task: dict) -> dict[str, str]:
    if task["label_kind"] != "multi_binary":
        return {}
    out: dict[str, str] = {}
    used: set[str] = set()
    for label in task["labels"]:
        base = anthropic_safe_key(label)
        safe = base
        i = 2
        while safe in used:
            suffix = f"_{i}"
            safe = f"{base[:64 - len(suffix)]}{suffix}"
            i += 1
        used.add(safe)
        out[label] = safe
    return out


def anthropic_input_schema(task: dict) -> dict:
    key_map = anthropic_key_map(task)
    if not key_map:
        return task["json_schema"]

    schema = json.loads(json.dumps(task["json_schema"]))
    schema["properties"] = {
        key_map[label]: schema["properties"][label] for label in task["labels"]
    }
    schema["required"] = [key_map[label] for label in task["labels"]]
    return schema


def anthropic_system_prompt(task: dict, system_prompt: str) -> str:
    lines = [system_prompt.rstrip()]

    if task["label_kind"] == "categorical":
        labels = ", ".join(f"`{label}`" for label in task["labels"])
        lines.extend(
            [
                "",
                f"For the Anthropic tool call, `{task['label_key']}` must be exactly one of: {labels}.",
                "If your preferred wording differs, choose the closest allowed value. Do not invent or paraphrase labels.",
            ]
        )

    key_map = anthropic_key_map(task)
    if key_map and any(label != safe for label, safe in key_map.items()):
        lines.extend(
            [
                "",
                "For the Anthropic tool call, use these schema field names:",
            ]
        )
        for label in task["labels"]:
            safe = key_map[label]
            if safe != label:
                lines.append(f"- `{safe}` means `{label}`.")
        lines.append("Return values using the schema field names above.")
    return "\n".join(lines)


def build_message_params(model: dict, task: dict, system_prompt: str, user_content: str) -> dict:
    return {
        "model": model["name"],
        "max_tokens": 2000,
        "system": anthropic_system_prompt(task, system_prompt),
        "messages": [{"role": "user", "content": user_content}],
        "tools": [
            {
                "name": "classify",
                "description": "Return the classification for the input.",
                "input_schema": anthropic_input_schema(task),
            }
        ],
        "tool_choice": {"type": "tool", "name": "classify"},
    }


def iter_requests(tasks: list[dict], model: dict, retry_items: set[tuple[str, str]] | None = None):
    request_idx = 0
    for task in tasks:
        system_prompt = Path(task["prompt_path"]).read_text()
        for item in task["loader"]():
            if retry_items is not None and (task["name"], str(item["item_id"])) not in retry_items:
                continue
            request_idx += 1
            custom_id = f"req_{request_idx:06d}"
            yield task, item, {
                "custom_id": custom_id,
                "params": build_message_params(model, task, system_prompt, item["user_content"]),
            }


def load_retry_items(path: str | Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    if "parse_error" not in df.columns:
        raise ValueError(f"{path} does not contain a parse_error column")
    failed = df[df["parse_error"].notna()]
    return {(row.task, str(row.item_id)) for row in failed.itertuples()}


def write_batch_files(
    tasks: list[dict],
    model: dict,
    outdir: Path,
    prefix: str,
    retry_items: set[tuple[str, str]] | None = None,
) -> tuple[Path, Path, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{prefix}.jsonl"
    manifest_path = outdir / f"{prefix}_manifest.csv"

    counts: dict[str, int] = {}
    n_requests = 0
    with jsonl_path.open("w") as jf, manifest_path.open("w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["custom_id", "task", "model", "item_id", "original_custom_id"],
        )
        writer.writeheader()
        for task, item, request in iter_requests(tasks, model, retry_items=retry_items):
            jf.write(json.dumps(request, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "custom_id": request["custom_id"],
                    "task": task["name"],
                    "model": model["name"],
                    "item_id": item["item_id"],
                    "original_custom_id": f"{task['name']}|{model['name']}|{item['item_id']}",
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


def _load_requests(jsonl_path: Path) -> list[dict]:
    requests = []
    with jsonl_path.open() as handle:
        for line in handle:
            if line.strip():
                requests.append(json.loads(line))
    return requests


def submit_batch(client: anthropic.Anthropic, jsonl_path: Path) -> dict:
    batch = client.messages.batches.create(requests=_load_requests(jsonl_path))
    return batch.model_dump(mode="json")


def retrieve_batch(client: anthropic.Anthropic, batch_id: str) -> dict:
    return client.messages.batches.retrieve(batch_id).model_dump(mode="json")


def download_results(client: anthropic.Anthropic, batch_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in client.messages.batches.results(batch_id):
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, submit, poll, and download Anthropic Message Batch jobs.")
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument("--model-manifest", default=str(REPO / "models" / "claude_sonnet_4_6.yaml"))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--submit", action="store_true", help="Create the Anthropic message batch.")
    parser.add_argument("--status", help="Retrieve and print status for an existing Anthropic message batch id.")
    parser.add_argument("--download-batch-id", help="Download result JSONL for an ended Anthropic message batch id.")
    parser.add_argument("--download-output", help="Path for --download-batch-id output.")
    parser.add_argument("--budget-usd", default=None, help="Budget/cost cap recorded in local metadata.")
    parser.add_argument(
        "--retry-errors-from",
        help="Prediction CSV; include only rows whose parse_error is non-missing.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    model = load_model_definition(Path(args.model_manifest))
    if model["backend"] != "anthropic":
        raise ValueError("This helper is only for Anthropic model manifests.")
    client = make_client(model)

    if args.download_batch_id:
        if not args.download_output:
            raise ValueError("--download-output is required with --download-batch-id")
        download_results(client, args.download_batch_id, Path(args.download_output))
        print(f"Wrote {args.download_output}")
        return 0

    if args.status:
        outdir.mkdir(parents=True, exist_ok=True)
        status = retrieve_batch(client, args.status)
        status_path = outdir / f"{args.status}_status.json"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(json.dumps(status, indent=2, sort_keys=True))
        print(f"Wrote {status_path}")
        return 0

    tasks = load_task_definitions(tasks_dir=args.tasks_dir)
    prefix = args.prefix or f"{model['name'].replace(':', '_')}_tasks_batch_{now_stamp()}"
    retry_items = load_retry_items(args.retry_errors_from)
    jsonl_path, manifest_path, meta = write_batch_files(
        tasks,
        model,
        outdir,
        prefix,
        retry_items=retry_items,
    )
    if retry_items is not None:
        meta["retry_source"] = args.retry_errors_from
        meta["retry_items"] = len(retry_items)
    if args.budget_usd:
        meta["budget_usd"] = str(args.budget_usd)
    print(json.dumps(meta, indent=2, sort_keys=True))

    if args.submit:
        submit_meta = submit_batch(client, jsonl_path)
        result_path = outdir / f"{prefix}_batch.json"
        result = {**meta, **submit_meta}
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
