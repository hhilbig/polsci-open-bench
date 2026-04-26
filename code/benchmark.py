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

# v2 sampling: keep v1's 250 + add 250 disjoint new items for total N=500.
# v1 used N_SAMPLE=250 with SEED. v2 uses (v1 sample) ∪ (new sample with SEED+1).
N_V1 = 250
N_V2_NEW = 250
N_SAMPLE = N_V1 + N_V2_NEW   # = 500
SEED = 20260422
SEED_V2 = SEED + 1            # secondary seed for v2-new draw

MODELS = [
    {"name": "gemma4:31b-it-q4_K_M",                     "backend": "ollama", "think": False},
    {"name": "qwen3:14b-q4_K_M",                         "backend": "ollama", "think": False},
    {"name": "qwen3:30b-a3b-q4_K_M",                     "backend": "ollama", "think": False},
    {"name": "mistral-small:24b-instruct-2501-q4_K_M",   "backend": "ollama", "think": False},
    {"name": "gpt-5.5",                                  "backend": "openai", "think": False, "reasoning_effort": "medium"},
    {"name": "gpt-5.4-nano",                             "backend": "openai", "think": False, "reasoning_effort": "medium"},
]

# (model_name, task_name) pairs to skip. Populated empirically — some Ollama
# models are unusably slow on specific tasks due to long system prompts that
# Ollama re-prefills on every call. Dropped cells are reported as absent (not
# as parse errors) so the report can flag them explicitly.
#
# Note: state_adaptation is currently disabled at the TASKS-list level, so the
# previously-needed ("gemma4:31b-it-q4_K_M", "state_adaptation") entry is gone
# from this set. Reinstate it if state_adaptation is re-enabled.
SKIP_COMBOS = set()


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
    v1_idxs, v2_idxs = _v1_v2_indices(len(df))
    items = []
    for idx in v1_idxs + v2_idxs:
        r = df.iloc[idx]
        items.append({
            "item_id": r["identifier"],
            "user_content": f"Title: {r['title']}\nText: {r['abstract'] or ''}",
            "gt": {d: int(r[f"manual_{d}"]) for d in STATE_ADAPTATION_DIMS},
        })
    return items


def _v1_v2_indices(n_total: int):
    """Return (v1_idxs, v2_new_idxs) — both sorted, disjoint subsets of range(n_total).

    v1 uses SEED + sample size N_V1; v2-new uses SEED_V2 sampling N_V2_NEW from the
    remainder. If n_total < N_V1+N_V2_NEW, v2_new is shrunk to fit.
    """
    rng_v1 = random.Random(SEED)
    n_v1 = min(N_V1, n_total)
    v1_idxs = sorted(rng_v1.sample(range(n_total), n_v1))
    remaining = sorted(set(range(n_total)) - set(v1_idxs))
    n_v2 = min(N_V2_NEW, len(remaining))
    rng_v2 = random.Random(SEED_V2)
    v2_idxs = sorted(rng_v2.sample(remaining, n_v2)) if n_v2 > 0 else []
    return v1_idxs, v2_idxs


def _sample_csv(path, text_col, id_col, gt_builder):
    df = pd.read_csv(path)
    v1_idxs, v2_idxs = _v1_v2_indices(len(df))
    items = []
    # v1 items first — preserves existing item_ids (positional fallback uses i in 0..N_V1-1)
    for i, idx in enumerate(v1_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": str(r[id_col]) if id_col in r else f"{path.stem}_{i:03}",
            "user_content": text_col(r),
            "gt": gt_builder(r),
        })
    # v2-new items — positional fallback continues from N_V1 (250..)
    for j, idx in enumerate(v2_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": str(r[id_col]) if id_col in r else f"{path.stem}_{N_V1 + j:03}",
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
    v1_idxs, v2_idxs = _v1_v2_indices(len(df))
    items = []
    for i, idx in enumerate(v1_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": f"semeval_{i:03}",
            "user_content": f"Target: {r['target_short']}\nTweet: {r['text']}",
            "gt": {"stance": r["gt_stance"]},
        })
    for j, idx in enumerate(v2_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": f"semeval_{N_V1 + j:03}",
            "user_content": f"Target: {r['target_short']}\nTweet: {r['text']}",
            "gt": {"stance": r["gt_stance"]},
        })
    return items


def load_halterman_ccc():
    # Upgraded 2026-04-24 to Halterman & Keith (2025) Dataverse ccc_test.tab
    # (1,010 rows, 4-class). Previous version was N=50 with 8-class schema;
    # archived as data/halterman_ccc.csv (legacy), still on disk.
    df = pd.read_csv(DATA / "halterman_ccc_hk2025.csv")
    v1_idxs, v2_idxs = _v1_v2_indices(len(df))
    items = []
    for i, idx in enumerate(v1_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": f"ccc_{i:03}",
            "user_content": f"News story:\n{r['text']}",
            "gt": {"protest_type": r["gt_protest_type"]},
        })
    for j, idx in enumerate(v2_idxs):
        r = df.iloc[idx]
        items.append({
            "item_id": f"ccc_{N_V1 + j:03}",
            "user_content": f"News story:\n{r['text']}",
            "gt": {"protest_type": r["gt_protest_type"]},
        })
    return items


def load_halterman_keith_bfrs():
    return _sample_csv(DATA / "halterman_keith_bfrs.csv",
                       text_col=lambda r: f"News story from Pakistan:\n{r['text']}",
                       id_col="__index__",  # no id col; fallback to row index
                       gt_builder=lambda r: {"event_type": r["gt_event_type"]})


def load_halterman_keith_cmp():
    return _sample_csv(DATA / "halterman_keith_cmp.csv",
                       text_col=lambda r: f"Manifesto quasi-sentence:\n{r['text']}",
                       id_col="__index__",
                       gt_builder=lambda r: {"policy_domain": r["gt_policy_domain"]})


BES_MII_LABELS = [
    "Referendum unspecified", "Coronavirus", "COVID-economy", "BLM and responses",
    "health", "education", "election outcome", "pol-neg", "partisan-neg",
    "societal divides", "morals", "nat ident, goals-loss", "racism/discrimination",
    "welfare", "terrorism", "immigration", "asylum", "crime", "europe",
    "constitutional", "international trade", "devolution", "scot-ind", "constitution",
    "foreign affairs", "war", "defence", "foreign emergency", "domestic emergency",
    "economy-general", "economy-personal", "unemployment", "taxation",
    "debt/deficit", "inflation", "living costs", "poverty", "austerity",
    "inequality", "housing", "social care", "pensions/ageing",
    "transport/infrastructure", "environment", "pol value-auth", "pol values-liberal",
    "pol values-right", "pol values-left", "other", "uncoded",
]


def load_mellon_bes_mii():
    return _sample_csv(DATA / "mellon_bes_mii_2024.csv",
                       text_col=lambda r: f"Open-ended response: {r['text']}",
                       id_col="item_id",
                       gt_builder=lambda r: {"issue": r["gt_issue"]})


def load_wesleyan_creative_ads():
    # Cap user content to keep Ollama prefill tractable. 90%+ of ads fit in 4000 chars.
    MAX_CHARS = 4000
    def text_col(r):
        full = r["full_text"]
        if len(full) > MAX_CHARS:
            full = full[:MAX_CHARS] + f"... [truncated at {MAX_CHARS} chars]"
        return (
            f"Ad sponsor (page name): {r['page_name']}\n"
            f"Candidate being evaluated: {r['candidate']}\n"
            f"Ad content:\n{full}"
        )
    return _sample_csv(DATA / "wesleyan_creative_ads_2022.csv",
                       text_col=text_col,
                       id_col="ad_id",
                       gt_builder=lambda r: {"tone": r["gt_tone"]})


# --------- Task definitions ---------

# state_adaptation is temporarily disabled for the N=250 run. Its 24 KB
# multi-binary prompt made some Ollama cells unusable (~150 s/item on gemma4)
# and substantially lengthened the full grid even on faster models. Task
# config is preserved below, commented out, so it can be reinstated for a
# dedicated single-task run later. Labels and loader still defined above.
_STATE_ADAPTATION_TASK_DEF = {
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
}

TASKS = [
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
        "labels": ["PROTEST", "RALLY", "DEMONSTRATION", "MARCH"],
        "label_key": "protest_type",
        "json_schema": {
            "type": "object",
            "properties": {"protest_type": {
                "type": "string",
                "enum": ["PROTEST", "RALLY", "DEMONSTRATION", "MARCH"]}},
            "required": ["protest_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "halterman_keith_bfrs",
        "loader": load_halterman_keith_bfrs,
        "prompt_file": "halterman_keith_bfrs.txt",
        "label_kind": "categorical",
        "labels": ["ASSASSINATION", "DRONE_ASSASSINATION", "ATTACK_ON_STATE",
                   "CONVENTIONAL_ATTACK_ON_GOV_FORCES", "GUERILLA_ATTACK_ON_GOV_FORCES",
                   "GOV_ATTACK_ON_NONSTATE_COMBATANTS", "GOV_ATTACK_ON_CIVILIANS",
                   "RIOT", "TERRORISM", "THREAT_OF_VIOLENCE",
                   "VIOLENT_POLITICAL_DEMONSTRATION", "OTHER"],
        "label_key": "event_type",
        "json_schema": {
            "type": "object",
            "properties": {"event_type": {
                "type": "string",
                "enum": ["ASSASSINATION", "DRONE_ASSASSINATION", "ATTACK_ON_STATE",
                         "CONVENTIONAL_ATTACK_ON_GOV_FORCES", "GUERILLA_ATTACK_ON_GOV_FORCES",
                         "GOV_ATTACK_ON_NONSTATE_COMBATANTS", "GOV_ATTACK_ON_CIVILIANS",
                         "RIOT", "TERRORISM", "THREAT_OF_VIOLENCE",
                         "VIOLENT_POLITICAL_DEMONSTRATION", "OTHER"]}},
            "required": ["event_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "halterman_keith_cmp",
        "loader": load_halterman_keith_cmp,
        "prompt_file": "halterman_keith_cmp.txt",
        "label_kind": "categorical",
        "labels": ["External Relations", "Freedom and Democracy", "Political System",
                   "Economy", "Welfare and Quality of Life", "Fabric of Society",
                   "Social Groups"],
        "label_key": "policy_domain",
        "json_schema": {
            "type": "object",
            "properties": {"policy_domain": {
                "type": "string",
                "enum": ["External Relations", "Freedom and Democracy", "Political System",
                         "Economy", "Welfare and Quality of Life", "Fabric of Society",
                         "Social Groups"]}},
            "required": ["policy_domain"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mellon_bes_mii_2024",
        "loader": load_mellon_bes_mii,
        "prompt_file": "mellon_bes_mii_2024.txt",
        "label_kind": "categorical",
        "labels": BES_MII_LABELS,
        "label_key": "issue",
        "json_schema": {
            "type": "object",
            "properties": {"issue": {"type": "string", "enum": BES_MII_LABELS}},
            "required": ["issue"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wesleyan_creative_ads_2022",
        "loader": load_wesleyan_creative_ads,
        "prompt_file": "wesleyan_creative_ads_2022.txt",
        "label_kind": "categorical",
        "labels": ["promote", "attack", "unclear"],
        "label_key": "tone",
        "json_schema": {
            "type": "object",
            "properties": {"tone": {"type": "string", "enum": ["promote", "attack", "unclear"]}},
            "required": ["tone"],
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


def classify_openai(client, model, system_prompt, user_content, json_schema,
                    reasoning_effort=None):
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
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
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

def run_task(task, models, oai, checkpoint_path, only_new_items=False):
    system_prompt = (PROMPTS / task["prompt_file"]).read_text()
    items = task["loader"]()
    if only_new_items:
        # Items are returned as (v1, then v2-new). Skip the v1 items.
        skipped = items[:N_V1]
        items = items[N_V1:]
        print(f"\n[only-new-items] skipping first {len(skipped)} v1 items; running {len(items)} new items.", flush=True)
    print(f"\n########## TASK: {task['name']} ({len(items)} items) ##########", flush=True)
    rows = []
    for m in models:
        if (m["name"], task["name"]) in SKIP_COMBOS:
            print(f"\n--- SKIPPING {m['name']} on {task['name']} (SKIP_COMBOS) ---", flush=True)
            continue
        if m["backend"] == "ollama":
            print(f"\n--- Warming up {m['name']} ---", flush=True)
            warmup_ollama(m["name"])
        print(f"\n--- {m['name']} on {task['name']} ---", flush=True)
        for i, it in enumerate(items):
            try:
                if m["backend"] == "ollama":
                    r = classify_ollama(m["name"], system_prompt, it["user_content"], think=m["think"])
                else:
                    r = classify_openai(oai, m["name"], system_prompt, it["user_content"], task["json_schema"],
                                        reasoning_effort=m.get("reasoning_effort"))
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
    """Replace matching (task, model, item_id) rows in existing_csv with new_rows.

    Takes the UNION of columns from old and new. New columns introduced by
    new_rows (e.g., gt_event_type for a freshly-added task) are preserved.
    Old columns absent from new_rows stay populated for non-overwritten rows.
    """
    new_df = pd.DataFrame(new_rows)
    if not existing_csv.exists():
        new_df.to_csv(existing_csv, index=False)
        print(f"[merge] wrote {len(new_df)} rows to new {existing_csv}", flush=True)
        return
    old = pd.read_csv(existing_csv)
    # Force string dtype on key cols on both sides to avoid int64-vs-str mismatch
    # that silently duplicates rows on key comparison.
    for c in key_cols:
        old[c] = old[c].astype(str)
        new_df[c] = new_df[c].astype(str)
    new_keys = set(zip(*[new_df[c] for c in key_cols]))
    mask = list(zip(*[old[c] for c in key_cols]))
    keep = [k not in new_keys for k in mask]
    filtered = old[keep]
    # Take the union of columns. Preserve old order, append new-only columns at the end.
    union_cols = list(filtered.columns) + [c for c in new_df.columns if c not in filtered.columns]
    for c in union_cols:
        if c not in filtered.columns:
            filtered = filtered.assign(**{c: None})
        if c not in new_df.columns:
            new_df[c] = None
    filtered = filtered[union_cols]
    new_df = new_df[union_cols]
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
    ap.add_argument("--only-new-items", action="store_true",
                    help="Only run the v2-new 250 items (skip the v1 250). Used for "
                         "Ollama carry-over: existing v1 predictions are kept; only the "
                         "new items 250..499 get model calls.")
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
        all_rows.extend(run_task(task, models, oai, args.output, only_new_items=args.only_new_items))
        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    if args.merge_into:
        merge_into(Path(args.merge_into), all_rows)

    print("\n=== ALL DONE ===", flush=True)
    print(f"  predictions: {args.output}", flush=True)
    if args.merge_into:
        print(f"  merged into: {args.merge_into}", flush=True)


if __name__ == "__main__":
    main()
