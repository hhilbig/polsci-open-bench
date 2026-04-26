#!/usr/bin/env python3
"""
Batched classification benchmark (Pipal-style). Runs the same tasks and
models as benchmark.py but at multiple batch sizes, writing to a
separate predictions CSV so the main benchmark outputs stay canonical.

Grid (default):
  tasks      = 6 active + state_adaptation (7 total; reinstated here)
  models     = benchmark.py MODELS (4 Ollama + gpt-5.4-nano + gpt-5.4)
  batch_sizes = {1, 10, 20}

Per-batch semantics:
  - b=1 is the classic single-item path (identical to benchmark.py's
    classify_* functions; included so this run is self-contained).
  - b>1 builds one API call with all N items numbered and asks the
    model to return a JSON array of N objects in order.
  - OpenAI strict schema wraps the per-item schema as
    {"results": [<object>, ...]} with minItems = maxItems = b.
  - Per-item latency is reported as batch_wall_clock / b so the paper's
    speedup math works directly.
  - Per-cell checkpoint: after each (task, model, b) completes, the
    full accumulated CSV is rewritten, so a mid-run failure loses at
    most one cell.

Usage:
  python3 code/batch_benchmark.py
  python3 code/batch_benchmark.py --only-task gilardi_relevance --batch-sizes 10,20
  python3 code/batch_benchmark.py --only-task state_adaptation --only-model gemma4:31b-it-q4_K_M --N 5 --batch-sizes 10
  python3 code/batch_benchmark.py --resume    # skip cells already in output CSV
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark import (  # noqa: E402
    TASKS as ACTIVE_TASKS,
    _STATE_ADAPTATION_TASK_DEF,
    MODELS,
    PROMPTS, OUT,
    OLLAMA_URL,
    parse_content as parse_single_content,
    warmup_ollama,
)


# State_adaptation is reinstated for the batching study. Sits at the front so
# --only-task state_adaptation works via the standard lookup.
TASKS = [_STATE_ADAPTATION_TASK_DEF] + ACTIVE_TASKS

BATCH_SIZES_DEFAULT = [1, 10, 20]

# (model, task, batch_size) combos to skip. gemma × state_adaptation at b=1
# is known unusable (~150 s/item, ~10 h for the cell); the rescue story only
# needs b=10 and b=20 for that cell.
BATCH_SKIP_COMBOS = {
    ("gemma4:31b-it-q4_K_M", "state_adaptation", 1),
}


# --------- Prompt construction ---------

def build_user_content(task_def, batch):
    """For b=1, pass through the single item's user_content unchanged (identical
    to the main benchmark's payload). For b>1, build a numbered-list prompt with
    an explicit JSON-array output instruction."""
    if len(batch) == 1:
        return batch[0]["user_content"]

    kind = task_def["label_kind"]
    labels = task_def.get("labels", [])
    if kind == "binary":
        key = task_def["label_key"]
        fmt = f'{{"{key}": 0 or 1}}'
    elif kind == "categorical":
        key = task_def["label_key"]
        fmt = f'{{"{key}": "<one of: ' + ", ".join(labels) + '>"}}'
    elif kind == "multi_binary":
        fields = ", ".join([f'"{l}": 0_or_1' for l in labels])
        fmt = f"{{{fields}}}"
    else:
        fmt = "{}"

    header = (
        f"You will receive {len(batch)} items below. Apply the classification "
        f"rules from the instructions above to EACH item. Return ONLY a JSON "
        f"array of {len(batch)} objects in the same order as the input items. "
        f"Each object must have EXACTLY the format: {fmt}. Do not add any "
        f"extra fields (no 'confidence', no 'reasoning', no comments). Do not "
        f"include any prose before or after the JSON array."
    )
    blocks = [header, ""]
    for i, it in enumerate(batch):
        blocks.append(f"Item {i + 1}:\n{it['user_content']}")
        blocks.append("")
    return "\n".join(blocks).strip()


# --------- Parsing ---------

def _strip_fences(content: str) -> str:
    c = content.strip()
    if "```json" in c:
        c = c.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in c:
        c = c.split("```", 1)[1].split("```", 1)[0].strip()
    return c


def _empty_pred(task_def):
    kind = task_def["label_kind"]
    if kind == "multi_binary":
        return {k: None for k in task_def["labels"]}
    return {task_def["label_key"]: None}


def _pred_from_obj(obj, task_def):
    """Extract pred_dict from a single JSON object. Returns (pred_dict, err_or_None)."""
    kind = task_def["label_kind"]
    if kind == "multi_binary":
        return {k: int(bool(obj.get(k, 0))) for k in task_def["labels"]}, None
    if kind == "binary":
        key = task_def["label_key"]
        return {key: int(bool(obj.get(key, 0)))}, None
    if kind == "categorical":
        key = task_def["label_key"]
        v = obj.get(key, "")
        if v in task_def["labels"]:
            return {key: v}, None
        return {key: None}, f"invalid_label: {v!r}"
    return _empty_pred(task_def), "unknown_label_kind"


def parse_response(content: str, task_def, expected_n: int):
    """Return list of (pred_dict, err) tuples for the expected_n items."""
    if expected_n == 1:
        pred, err = parse_single_content(content, task_def)
        return [(pred, err)]

    try:
        arr, _ = json.JSONDecoder().raw_decode(_strip_fences(content))
    except Exception:
        return [(_empty_pred(task_def),
                 f"batch_parse_fail: {content[:80]!r}")] * expected_n

    # OpenAI-wrapped array: {"results": [...]}
    if isinstance(arr, dict) and "results" in arr and isinstance(arr["results"], list):
        arr = arr["results"]

    if not isinstance(arr, list):
        return [(_empty_pred(task_def), "not_array")] * expected_n

    results = []
    for i in range(expected_n):
        if i < len(arr) and isinstance(arr[i], dict):
            pred, err = _pred_from_obj(arr[i], task_def)
            results.append((pred, err))
        else:
            results.append((_empty_pred(task_def), "missing_in_batch"))
    return results


# --------- Inference ---------

def classify_ollama_batched(model, system_prompt, batch, task_def, think=False):
    user = build_user_content(task_def, batch)
    # num_predict must cover N items worth of output JSON plus any extra fields
    # the model decides to emit (some models add "confidence" / "reasoning").
    # multi_binary needs ~200 tok/item; categorical/binary ~40 tok/item. Scale
    # generously so truncation never causes parse failures.
    if len(batch) == 1:
        num_predict = 256 if task_def["label_kind"] == "multi_binary" else 128
    else:
        per_item = 220 if task_def["label_kind"] == "multi_binary" else 80
        num_predict = 256 + per_item * len(batch)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=900) as c:
        r = c.post(f"{OLLAMA_URL}/api/chat", json=payload)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    content = d.get("message", {}).get("content", "").strip()
    return content, latency, d.get("eval_count")


def _build_batched_schema(task_def, batch_size):
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": task_def["json_schema"],
                "minItems": batch_size,
                "maxItems": batch_size,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def classify_openai_batched(client, model, system_prompt, batch, task_def):
    user = build_user_content(task_def, batch)
    if len(batch) == 1:
        schema = task_def["json_schema"]
        schema_name = "classification"
    else:
        schema = _build_batched_schema(task_def, len(batch))
        schema_name = "batched_classification"
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 256 + 80 * len(batch)
    else:
        kwargs["max_tokens"] = 128 + 64 * len(batch)
        kwargs["temperature"] = 0.1
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    eval_count = resp.usage.completion_tokens if resp.usage else None
    return content, latency, eval_count


# --------- Orchestration ---------

def run_cell(task, model_def, oai, batch_size, items):
    """Run one (task, model, batch_size) cell. Returns list of per-item row dicts."""
    system_prompt = (PROMPTS / task["prompt_file"]).read_text()
    rows = []
    n_batches = (len(items) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(items), batch_size)):
        batch = items[i:i + batch_size]
        try:
            if model_def["backend"] == "ollama":
                content, latency, eval_count = classify_ollama_batched(
                    model_def["name"], system_prompt, batch, task, think=model_def["think"]
                )
            else:
                content, latency, eval_count = classify_openai_batched(
                    oai, model_def["name"], system_prompt, batch, task
                )
            parsed = parse_response(content, task, len(batch))
        except Exception as e:
            parsed = [(_empty_pred(task), f"api_error: {e}")] * len(batch)
            content, latency, eval_count = "", None, None

        per_item_latency = (latency / len(batch)) if latency is not None else None
        for j, (pred, err) in enumerate(parsed):
            row = {
                "task": task["name"],
                "model": model_def["name"],
                "batch_size": batch_size,
                "item_id": batch[j]["item_id"],
                "latency_s": per_item_latency,
                "batch_latency_s": latency,
                "eval_count": eval_count,
                "parse_error": err,
                "raw_content_preview": content[:200],
            }
            for k, v in pred.items():
                row[f"pred_{k}"] = v
            for k, v in batch[j]["gt"].items():
                row[f"gt_{k}"] = v
            rows.append(row)

        if latency is not None:
            print(f"    batch {bi + 1}/{n_batches}: {latency:.1f}s "
                  f"(per-item {per_item_latency:.2f}s)", flush=True)
        else:
            print(f"    batch {bi + 1}/{n_batches}: FAILED", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only-task", help="Restrict to one task by name")
    ap.add_argument("--only-model", help="Restrict to one model by name")
    ap.add_argument("--batch-sizes", default=",".join(str(b) for b in BATCH_SIZES_DEFAULT),
                    help="Comma-separated list of batch sizes")
    ap.add_argument("--N", type=int, default=None,
                    help="Cap N items per task (use for smoke tests; items remain deterministic)")
    ap.add_argument("--output", default=str(OUT / "predictions_batched.csv"),
                    help="Output CSV path (default: output/predictions_batched.csv)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already present in the output CSV")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    oai = OpenAI()

    tasks = TASKS
    if args.only_task:
        tasks = [t for t in TASKS if t["name"] == args.only_task]
        if not tasks:
            print(f"Unknown task: {args.only_task}"); return

    models = MODELS
    if args.only_model:
        models = [m for m in MODELS if m["name"] == args.only_model]
        if not models:
            print(f"Unknown model: {args.only_model}"); return

    # Resume support
    existing = set()
    out_path = Path(args.output)
    all_rows = []
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        for _, r in prev[["task", "model", "batch_size"]].drop_duplicates().iterrows():
            existing.add((str(r["task"]), str(r["model"]), int(r["batch_size"])))
        all_rows = prev.to_dict("records")
        print(f"[resume] {len(existing)} cells already present; skipping those.", flush=True)

    print(f"Batch benchmark starting")
    print(f"  tasks      : {[t['name'] for t in tasks]}")
    print(f"  models     : {[m['name'] for m in models]}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  output     : {out_path}")
    if args.N is not None:
        print(f"  N cap      : {args.N}")
    print()

    for task in tasks:
        items = task["loader"]()
        if args.N is not None and len(items) > args.N:
            items = items[:args.N]

        print(f"\n########## TASK: {task['name']} ({len(items)} items) ##########", flush=True)

        for m in models:
            if m["backend"] == "ollama":
                print(f"\n  Warming up {m['name']}...", flush=True)
                warmup_ollama(m["name"])

            for b in batch_sizes:
                cell_key = (task["name"], m["name"], b)
                if (m["name"], task["name"], b) in BATCH_SKIP_COMBOS:
                    print(f"\n  --- SKIPPING {m['name']} × {task['name']} × b={b} "
                          f"(BATCH_SKIP_COMBOS) ---", flush=True)
                    continue
                if args.resume and cell_key in existing:
                    print(f"  [RESUME] {task['name']} × {m['name']} × b={b} already done",
                          flush=True)
                    continue

                print(f"\n  --- {m['name']} × {task['name']} × b={b} ---", flush=True)
                t0 = time.perf_counter()
                rows = run_cell(task, m, oai, b, items)
                cell_time = time.perf_counter() - t0
                n_parse_err = sum(1 for r in rows if r.get("parse_error"))
                print(f"    CELL DONE: {len(rows)} rows in {cell_time:.1f}s, "
                      f"{n_parse_err} parse errors", flush=True)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(out_path, index=False)

    print(f"\n=== ALL DONE ===", flush=True)
    print(f"  predictions: {out_path}", flush=True)


if __name__ == "__main__":
    main()
