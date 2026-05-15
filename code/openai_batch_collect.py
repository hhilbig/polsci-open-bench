#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import benchmark
from task_registry import load_task_definitions


REPO = Path(__file__).resolve().parent.parent


def build_item_lookup(tasks_dir: str | Path, model_name: str) -> dict[str, tuple[dict, dict]]:
    lookup = {}
    for task in load_task_definitions(tasks_dir=tasks_dir):
        for item in task["loader"]():
            custom_id = f"{task['name']}|{model_name}|{item['item_id']}"
            lookup[custom_id] = (task, item)
    return lookup


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


def collect(batch_jsonl: Path, tasks_dir: str | Path, model_name: str) -> pd.DataFrame:
    lookup = build_item_lookup(tasks_dir, model_name)
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
            content, eval_count, api_err = response_content(result)
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
    parser = argparse.ArgumentParser(description="Convert OpenAI Batch API output JSONL into benchmark prediction CSV.")
    parser.add_argument("--batch-jsonl", required=True)
    parser.add_argument("--tasks-dir", default=str(REPO / "tasks"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = collect(Path(args.batch_jsonl), args.tasks_dir, args.model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    parse_errors = df["parse_error"].notna().sum()
    print(f"wrote {out} ({len(df)} rows, {parse_errors} parse/api errors)")
    print(df.groupby("task").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
