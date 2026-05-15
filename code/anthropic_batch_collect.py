#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

import benchmark
from task_registry import load_task_definitions


REPO = Path(__file__).resolve().parent.parent


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


def restore_anthropic_keys(tool_input: object, task: dict) -> object:
    key_map = anthropic_key_map(task)
    if not key_map or not isinstance(tool_input, dict):
        return tool_input
    restored = {}
    for label in task["labels"]:
        safe = key_map[label]
        if label in tool_input:
            restored[label] = tool_input[label]
        else:
            restored[label] = tool_input.get(safe, 0)
    return restored


def build_item_lookup(
    tasks_dir: str | Path,
    model_name: str,
    manifest_path: str | Path | None = None,
) -> dict[str, tuple[dict, dict]]:
    tasks = load_task_definitions(tasks_dir=tasks_dir)
    if manifest_path is None:
        lookup = {}
        for task in tasks:
            for item in task["loader"]():
                custom_id = f"{task['name']}|{model_name}|{item['item_id']}"
                lookup[custom_id] = (task, item)
        return lookup

    task_lookup = {
        task["name"]: {str(item["item_id"]): (task, item) for item in task["loader"]()}
        for task in tasks
    }
    lookup = {}
    with Path(manifest_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model_name:
                raise ValueError(f"Manifest model mismatch: {row['model']} != {model_name}")
            try:
                lookup[row["custom_id"]] = task_lookup[row["task"]][row["item_id"]]
            except KeyError as exc:
                raise KeyError(
                    f"Manifest row does not match loaded tasks: {row['custom_id']} "
                    f"{row['task']} {row['item_id']}"
                ) from exc
    return lookup


def response_content(result_line: dict, task: dict) -> tuple[str, int | None, str | None]:
    result = result_line.get("result") or {}
    result_type = result.get("type")
    if result_type != "succeeded":
        return "", None, f"api_{result_type}: {result}"

    message = result.get("message") or {}
    content_blocks = message.get("content") or []
    tool_input = None
    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") == "classify":
            tool_input = restore_anthropic_keys(block.get("input"), task)
            break
    content = json.dumps(tool_input, ensure_ascii=False) if tool_input is not None else ""
    usage = message.get("usage") or {}
    eval_count = usage.get("output_tokens")
    if tool_input is None:
        return content, eval_count, f"missing_tool_use: {content_blocks}"
    return content, eval_count, None


def collect(
    batch_jsonl: Path,
    tasks_dir: str | Path,
    model_name: str,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    lookup = build_item_lookup(tasks_dir, model_name, manifest_path)
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
            task, item = lookup[custom_id]
            content, eval_count, api_err = response_content(result, task)
            if api_err:
                preds = {}
                parse_err = api_err
            else:
                preds, parse_err = benchmark.parse_content(content, task)
            row = {
                "task": task["name"],
                "model": model_name,
                "item_id": item["item_id"],
                "latency_s": None,
                "eval_count": eval_count,
                "parse_error": parse_err,
                "raw_content_preview": content[:200],
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
    parser = argparse.ArgumentParser(description="Convert Anthropic Message Batch output JSONL into benchmark prediction CSV.")
    parser.add_argument("--batch-jsonl", required=True)
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--manifest", help="Batch manifest CSV mapping Anthropic custom_ids to task items.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = collect(Path(args.batch_jsonl), args.tasks_dir, args.model, args.manifest)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    parse_errors = df["parse_error"].notna().sum()
    print(f"wrote {out} ({len(df)} rows, {parse_errors} parse/api errors)")
    print(df.groupby("task").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
