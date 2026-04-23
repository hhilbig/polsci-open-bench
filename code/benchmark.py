#!/usr/bin/env python3
"""
polsci-open-bench: classification benchmark for local LLMs vs OpenAI API.

Currently covers 7 tasks × 7 models × N=50 items.

Tasks (see TASKS list below for canonical names):
  - state_adaptation        - 8-label bill classification (private data)
  - gilardi_relevance       - binary content-moderation relevance
  - gilardi_stance          - 3-class content-moderation stance
  - ballard_incivility      - binary congressional tweet incivility
  - ornstein_scotus_sentiment - 3-class SCOTUS tweet sentiment
  - halterman_ccc_protest   - 8-class protest event type
  - chae_semeval_stance     - 3-class SemEval-2016 political stance

Models (see MODELS list; supports any Ollama-hosted model + OpenAI API):
  Ollama (via localhost:11434):
    gemma4:31b-it-q4_K_M
    qwen3:14b-q4_K_M
    qwen3:30b-a3b-q4_K_M
    mistral-small:24b-instruct-2501-q4_K_M
  OpenAI:
    gpt-5.4-nano
    gpt-5.4-mini
    gpt-5.4

Basic usage:
  # Run full benchmark (all tasks × all models)
  python code/benchmark.py

  # Selective rerun of one (task, model) cell, merge into existing predictions
  python code/benchmark.py \\
    --only-model qwen3:30b-a3b-q4_K_M \\
    --only-task halterman_ccc_protest \\
    --merge-into output/predictions.csv

  # Run only one task across all models
  python code/benchmark.py --only-task gilardi_stance

Parse strategy:
  Primary — JSON via raw_decode (trailing content ignored).
  Fallback — case-insensitive bare-label match against the task's enum.
  Both API and local models are supported. API models use structured outputs
  (JSON schema enforced server-side).

Data files are expected at:
  data/{task_name}.csv                 (public tasks)
  data/private/bills_for_matthew.xlsx  (state_adaptation - not redistributed)

Prompts are expected at:
  prompts/{task_name}.txt              (or with explicit label-task suffix)

Set OPENAI_API_KEY in env for OpenAI calls. For local models, make sure Ollama
is running at http://localhost:11434 (or override with OLLAMA_URL env var).
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import httpx
import openpyxl
import pandas as pd
from openai import OpenAI


# --------- Config ---------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data"
PROMPTS = REPO / "prompts"
OUT = REPO / "output"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
N_SAMPLE = 50
SEED = 20260422

MODELS = [
    {"name": "gemma4:31b-it-q4_K_M",                     "backend": "ollama", "think": False},
    {"name": "qwen3:14b-q4_K_M",                         "backend": "ollama", "think": False},
    {"name": "qwen3:30b-a3b-q4_K_M",                     "backend": "ollama", "think": False},
    {"name": "mistral-small:24b-instruct-2501-q4_K_M",   "backend": "ollama", "think": False},
    {"name": "gpt-5.4-nano",                             "backend": "openai", "think": False},
    {"name": "gpt-5.4-mini",                             "backend": "openai", "think": False},
    {"name": "gpt-5.4",                                  "backend": "openai", "think": False},
]


# --------- Task loaders ---------

STATE_ADAPTATION_DIMS = [
    "adaptation", "mitigation", "relief", "irrelevant",
    "regulatory", "fiscal", "symbolic", "reversal",
]


def load_state_adaptation():
    path = DATA / "private" / "bills_for_matthew.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"state_adaptation task requires {path}\n"
            "This data is private (single RA's coding) and not redistributed. "
            "Contact the repo owner or supply your own labelled set."
        )
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Bills"]
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=rows[0])
    label_cols = [f"manual_{d}" for d in STATE_ADAPTATION_DIMS]
    df = df[df[label_cols].notna().all(axis=1)].reset_index(drop=True)
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(df)), N_SAMPLE))
    sub = df.iloc[idxs].reset_index(drop=True)
    items = []
    for _, r in sub.iterrows():
        items.append({
            "item_id": r["identifier"],
            "user_content": f"Title: {r['title']}\nText: {r['abstract'] or ''}",
            "gt": {d: int(r[f"manual_{d}"]) for d in STATE_ADAPTATION_DIMS},
        })
    return items


def _sample_csv(path, text_col, id_col, gt_builder):
    df = pd.read_csv(path)
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(df)), N_SAMPLE))
    sub = df.iloc[idxs].reset_index(drop=True)
    items = []
    for i, r in sub.iterrows():
        items.append({
            "item_id": str(r[id_col]) if id_col in r else f"{path.stem}_{i:03}",
            "user_content": text_col(r),
            "gt": gt_builder(r),
        })
    return items


def load_gilardi_relevance():
    return _sample_csv(DATA / "gilardi_relevance.csv",
                       text_col=lambda r: f"Tweet: {r['text']}",
                       id_col="status_id",
                       gt_builder=lambda r: {"relevant": int(r["gt_relevant"])})


def load_ballard_incivility():
    return _sample_csv(DATA / "ballard_incivility.csv",
                       text_col=lambda r: f"Tweet: {r['text']}",
                       id_col="status_id",
                       gt_builder=lambda r: {"uncivil": int(r["gt_uncivil"])})


def load_gilardi_stance():
    return _sample_csv(DATA / "gilardi_stance.csv",
                       text_col=lambda r: f"Tweet: {r['text']}",
                       id_col="status_id",
                       gt_builder=lambda r: {"stance": str(r["gt_stance"])})


def load_ornstein_scotus():
    return _sample_csv(DATA / "ornstein_scotus.csv",
                       text_col=lambda r: f"Case: {r['case']}\nTweet: {r['text']}",
                       id_col="tweet_id",
                       gt_builder=lambda r: {"sentiment": r["gt_sentiment"]})


def load_chae_semeval():
    df = pd.read_csv(DATA / "semeval_stance.csv")
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(df)), N_SAMPLE))
    sub = df.iloc[idxs].reset_index(drop=True)
    items = []
    for i, r in sub.iterrows():
        items.append({
            "item_id": f"semeval_{i:03}",
            "user_content": f"Target: {r['target_short']}\nTweet: {r['text']}",
            "gt": {"stance": r["gt_stance"]},
        })
    return items


def load_halterman_ccc():
    df = pd.read_csv(DATA / "halterman_ccc.csv")
    # Exactly 50 rows — use them all (deterministic, no sampling)
    items = []
    for i, r in df.iterrows():
        items.append({
            "item_id": f"ccc_{i:03}",
            "user_content": f"News story:\n{r['text']}",
            "gt": {"protest_type": r["gt_protest_type"]},
        })
    return items


# --------- Task definitions ---------

TASKS = [
    {
        "name": "state_adaptation",
        "loader": load_state_adaptation,
        "prompt_file": "state_adaptation.txt",
        "label_kind": "multi_binary",
        "labels": STATE_ADAPTATION_DIMS,
        "json_schema": {
            "type": "object",
            "properties": {d: {"type": "integer", "enum": [0, 1]} for d in STATE_ADAPTATION_DIMS},
            "required": STATE_ADAPTATION_DIMS,
            "additionalProperties": False,
        },
    },
    {
        "name": "gilardi_relevance",
        "loader": load_gilardi_relevance,
        "prompt_file": "gilardi_relevance.txt",
        "label_kind": "binary",
        "labels": ["relevant"],
        "label_key": "relevant",
        "json_schema": {
            "type": "object",
            "properties": {"relevant": {"type": "integer", "enum": [0, 1]}},
            "required": ["relevant"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ballard_incivility",
        "loader": load_ballard_incivility,
        "prompt_file": "ballard_incivility.txt",
        "label_kind": "binary",
        "labels": ["uncivil"],
        "label_key": "uncivil",
        "json_schema": {
            "type": "object",
            "properties": {"uncivil": {"type": "integer", "enum": [0, 1]}},
            "required": ["uncivil"],
            "additionalProperties": False,
        },
    },
    {
        "name": "gilardi_stance",
        "loader": load_gilardi_stance,
        "prompt_file": "gilardi_stance.txt",
        "label_kind": "categorical",
        "labels": ["pro", "neutral", "contra"],
        "label_key": "stance",
        "json_schema": {
            "type": "object",
            "properties": {"stance": {"type": "string", "enum": ["pro", "neutral", "contra"]}},
            "required": ["stance"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ornstein_scotus_sentiment",
        "loader": load_ornstein_scotus,
        "prompt_file": "ornstein_scotus_sentiment.txt",
        "label_kind": "categorical",
        "labels": ["Positive", "Negative", "Neutral"],
        "label_key": "sentiment",
        "json_schema": {
            "type": "object",
            "properties": {"sentiment": {"type": "string", "enum": ["Positive", "Negative", "Neutral"]}},
            "required": ["sentiment"],
            "additionalProperties": False,
        },
    },
    {
        "name": "halterman_ccc_protest",
        "loader": load_halterman_ccc,
        "prompt_file": "halterman_ccc_protest.txt",
        "label_kind": "categorical",
        "labels": ["PROTEST", "RALLY", "DEMONSTRATION", "MARCH",
                   "CARAVAN", "BICYCLE_RIDE", "DIRECT_ACTION", "COUNTER_PROTEST"],
        "label_key": "protest_type",
        "json_schema": {
            "type": "object",
            "properties": {"protest_type": {
                "type": "string",
                "enum": ["PROTEST", "RALLY", "DEMONSTRATION", "MARCH",
                         "CARAVAN", "BICYCLE_RIDE", "DIRECT_ACTION", "COUNTER_PROTEST"]}},
            "required": ["protest_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "chae_semeval_stance",
        "loader": load_chae_semeval,
        "prompt_file": "chae_semeval_stance.txt",
        "label_kind": "categorical",
        "labels": ["FAVOR", "AGAINST", "NONE"],
        "label_key": "stance",
        "json_schema": {
            "type": "object",
            "properties": {"stance": {"type": "string", "enum": ["FAVOR", "AGAINST", "NONE"]}},
            "required": ["stance"],
            "additionalProperties": False,
        },
    },
]


# --------- Parsing (JSON + bare-label fallback) ---------

def extract_json(content: str) -> str:
    c = content.strip()
    if "```json" in c:
        c = c.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in c:
        c = c.split("```", 1)[1].split("```", 1)[0].strip()
    return c


def parse_content(content: str, task: dict):
    """Parse model output. JSON first, case-insensitive bare-label match as fallback."""
    kind = task["label_kind"]
    # JSON attempt
    try:
        obj, _ = json.JSONDecoder().raw_decode(extract_json(content))
        if kind == "multi_binary":
            return {k: int(bool(obj.get(k, 0))) for k in task["labels"]}, None
        elif kind == "binary":
            k = task["label_key"]
            return {k: int(bool(obj.get(k, 0)))}, None
        else:
            k = task["label_key"]
            v = obj.get(k, "")
            if v in task["labels"]:
                return {k: v}, None
            # fall through
    except Exception:
        pass

    # Bare-label fallback (categorical only; multi-binary has no single-label equivalent)
    if kind == "categorical":
        k = task["label_key"]
        upper = content.strip().upper()
        for lbl in task["labels"]:
            if upper == lbl.upper() or upper.strip(' "\'.,!') == lbl.upper():
                return {k: lbl}, None
        hits = [lbl for lbl in task["labels"] if lbl.upper() in upper]
        if len(hits) == 1:
            return {k: hits[0]}, None
        return {k: None}, f"parse_fail: {content[:80]!r}"

    if kind == "multi_binary":
        return {k: None for k in task["labels"]}, f"parse_fail: {content[:80]!r}"

    k = task["label_key"]
    return {k: None}, f"parse_fail: {content[:80]!r}"


# --------- Inference ---------

def classify_ollama(model, system_prompt, user_content, think=False):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=600) as c:
        r = c.post(f"{OLLAMA_URL}/api/chat", json=payload)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    return {
        "content": d.get("message", {}).get("content", "").strip(),
        "latency_s": latency,
        "eval_count": d.get("eval_count"),
    }


def classify_openai(client, model, system_prompt, user_content, json_schema):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "classification", "strict": True, "schema": json_schema},
        },
    }
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["max_tokens"] = 1024
        kwargs["temperature"] = 0.1
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0
    return {
        "content": resp.choices[0].message.content or "",
        "latency_s": latency,
        "eval_count": resp.usage.completion_tokens if resp.usage else None,
    }


def warmup_ollama(model):
    try:
        with httpx.Client(timeout=600) as c:
            c.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "think": False,
                "options": {"num_predict": 1},
            })
    except Exception as e:
        print(f"  warmup failed: {e}", flush=True)


# --------- Orchestration ---------

def run_task(task, models, oai, checkpoint_path):
    system_prompt = (PROMPTS / task["prompt_file"]).read_text()
    items = task["loader"]()
    print(f"\n########## TASK: {task['name']} ({len(items)} items) ##########", flush=True)
    rows = []
    for m in models:
        if m["backend"] == "ollama":
            print(f"\n--- Warming up {m['name']} ---", flush=True)
            warmup_ollama(m["name"])
        print(f"\n--- {m['name']} on {task['name']} ---", flush=True)
        for i, it in enumerate(items):
            try:
                if m["backend"] == "ollama":
                    r = classify_ollama(m["name"], system_prompt, it["user_content"], think=m["think"])
                else:
                    r = classify_openai(oai, m["name"], system_prompt, it["user_content"], task["json_schema"])
                preds, parse_err = parse_content(r["content"], task)
                row = {
                    "task": task["name"], "model": m["name"],
                    "item_id": it["item_id"],
                    "latency_s": r["latency_s"], "eval_count": r.get("eval_count"),
                    "parse_error": parse_err,
                    "raw_content_preview": r["content"][:200],
                }
                for k, v in preds.items(): row[f"pred_{k}"] = v
                for k, v in it["gt"].items(): row[f"gt_{k}"] = v
                rows.append(row)
                print(f"  [{i+1:2}/{len(items)}] {str(it['item_id'])[:14]:<14} "
                      f"{r['latency_s']:5.1f}s tok={r.get('eval_count')}"
                      + (" [parse_err]" if parse_err else ""), flush=True)
            except Exception as e:
                print(f"  [{i+1:2}/{len(items)}] {str(it['item_id'])[:14]:<14} ERROR: {e}", flush=True)
                row = {
                    "task": task["name"], "model": m["name"], "item_id": it["item_id"],
                    "latency_s": None, "eval_count": None,
                    "parse_error": f"api_error: {e}", "raw_content_preview": "",
                }
                key = task.get("label_key")
                if key:
                    row[f"pred_{key}"] = None
                else:
                    for k in task["labels"]:
                        row[f"pred_{k}"] = None
                for k, v in it["gt"].items():
                    row[f"gt_{k}"] = v
                rows.append(row)

            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return rows


def merge_into(existing_csv: Path, new_rows: list,
               key_cols=("task", "model", "item_id")):
    """Replace matching (task, model, item_id) rows in existing_csv with new_rows."""
    new_df = pd.DataFrame(new_rows)
    if not existing_csv.exists():
        new_df.to_csv(existing_csv, index=False)
        print(f"[merge] wrote {len(new_df)} rows to new {existing_csv}", flush=True)
        return
    old = pd.read_csv(existing_csv)
    new_keys = set(zip(*[new_df[c] for c in key_cols]))
    mask = list(zip(*[old[c] for c in key_cols]))
    keep = [k not in new_keys for k in mask]
    filtered = old[keep]
    for c in filtered.columns:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[filtered.columns.tolist()]
    merged = pd.concat([filtered, new_df], ignore_index=True)
    merged.to_csv(existing_csv, index=False)
    print(f"[merge] {existing_csv}: replaced {len(old)-len(filtered)} rows with "
          f"{len(new_df)}; total now {len(merged)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only-model", help="Restrict to one model by name")
    ap.add_argument("--only-task",  help="Restrict to one task by name")
    ap.add_argument("--output",     default=str(OUT / "predictions.csv"),
                    help="Where to write this run's predictions (default output/predictions.csv).")
    ap.add_argument("--merge-into", dest="merge_into", default=None,
                    help="After running, merge this run's rows into an existing CSV "
                         "(replacing matching task/model/item_id rows).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    oai = OpenAI()

    tasks = TASKS
    if args.only_task:
        tasks = [t for t in TASKS if t["name"] == args.only_task]
        if not tasks:
            print(f"Unknown task: {args.only_task}"); return
        print(f"[selective] Running ONLY task: {args.only_task}", flush=True)

    models = MODELS
    if args.only_model:
        models = [m for m in MODELS if m["name"] == args.only_model]
        if not models:
            print(f"Unknown model: {args.only_model}"); return
        print(f"[selective] Running ONLY model: {args.only_model}", flush=True)

    all_rows = []
    for task in tasks:
        all_rows.extend(run_task(task, models, oai, args.output))
        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    if args.merge_into:
        merge_into(Path(args.merge_into), all_rows)

    print("\n=== ALL DONE ===", flush=True)
    print(f"  predictions: {args.output}", flush=True)
    if args.merge_into:
        print(f"  merged into: {args.merge_into}", flush=True)


if __name__ == "__main__":
    main()
