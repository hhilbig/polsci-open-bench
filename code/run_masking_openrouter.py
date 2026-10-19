#!/usr/bin/env python3
"""
Party-masking experiment for commercial models through OpenRouter.

Each sampled item is sent twice in immediate succession, once as written
(tasks_ext/) and once with party and politician names masked
(tasks_ext_masked/), so both versions of an item hit the same provider state.
The system prompts, JSON schemas and parsing are the benchmark's own
(task_registry, benchmark.parse_content); outputs use the column layout of the
Hive runs so code/analyze_masking_experiment.py can read them.

Sample: every item whose passage contained a name, plus a seeded random 25% of
the items without names (the placebo) in each task.

Spending: OpenRouter reports the charged cost of every call; the run stops
before the next call once the total across models reaches --cost-cap-usd.

With --ignore-names, each sampled item is sent once more as written, with
IGNORE_NOTE appended to the system prompt (condition "ignore"), into
IGNORE_RUN_DIR. Same items, so the results compare item by item with the
as-written and masked conditions of the main run.

Usage:
  python3 code/run_masking_openrouter.py --pilot 5      # 5 items per task per model
  python3 code/run_masking_openrouter.py                # full sample, resumes
  python3 code/run_masking_openrouter.py --ignore-names # instruction-only variant
"""
import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import task_registry  # noqa: E402
from benchmark import augment_system_prompt_for_json_object, parse_content  # noqa: E402

RUN_DIR = REPO / "output" / "sidecar" / "task_ext_20261011_masked_api"
IGNORE_RUN_DIR = REPO / "output" / "sidecar" / "task_ext_20261018_ignore_api"
IGNORE_NOTE = ("The text may name political parties or politicians. Judge only the position "
               "the text itself expresses. Do not let the names, or anything you know about "
               "those parties or people, affect your answer.")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TASKS = ["benoit_manifesto_economic_position", "benoit_manifesto_social_position",
         "lemens_gallego_tweet_position"]
PLACEBO_SHARE = 0.25
SEED = 20261010
# Same versions as the September panel; reasoning off as in September. DeepSeek
# models are pinned to one host (DeepInfra, fp8): OpenRouter otherwise spreads
# calls over ~30 hosts with fp4/fp8 quantizations, and DeepSeek's own endpoint
# is excluded by the account's no-training-on-paid-data setting.
DEEPINFRA = {"order": ["DeepInfra"], "allow_fallbacks": False}
MODELS = {
    "deepseek_v4_flash": {"id": "deepseek/deepseek-v4.1-flash", "temperature": 0,
                          "reasoning": {"enabled": False}, "provider": DEEPINFRA},
    "deepseek_v4_pro": {"id": "deepseek/deepseek-v4-pro-0813", "temperature": 0,
                        "reasoning": {"enabled": False}, "provider": DEEPINFRA},
    "gpt_5_6_terra": {"id": "openai/gpt-5.6-terra", "reasoning": {"effort": "none"}},
    # Gemini reasoning cannot be disabled on OpenRouter; "minimal" still uses
    # ~100 output tokens, so it gets a larger output limit.
    "gemini_3_8_flash": {"id": "google/gemini-3.8-flash", "reasoning": {"effort": "minimal"},
                         "max_tokens": 1024},
    "claude_sonnet_5": {"id": "anthropic/claude-sonnet-5", "reasoning": {"enabled": False}},
}

lock = threading.Lock()
spent = {"usd": 0.0}


def load_task(task, masked):
    path = (REPO / "tasks_ext_masked" / f"{task}_masked.yaml") if masked else (REPO / "tasks_ext" / f"{task}.yaml")
    return task_registry.load_task_definitions(task_manifest=path)[0]


def sample_items(task, pilot):
    orig, mask = load_task(task, False), load_task(task, True)
    a, b = orig["loader"](), mask["loader"]()
    assert [x["item_id"] for x in a] == [x["item_id"] for x in b]
    changed = pd.read_csv(REPO / "data" / f"{task}_masked.csv").set_index("source_id").masked_changed
    rng = random.Random(f"{SEED}-{task}")
    keep = []
    for x, y in zip(a, b):
        named = bool(changed[x["item_id"]])
        if named or rng.random() < PLACEBO_SHARE:
            keep.append((x, y, named))
    if pilot:
        keep = [k for k in keep if k[2]][:pilot - pilot // 2] + [k for k in keep if not k[2]][:pilot // 2]
    return orig, mask, keep


def call(key, model, system_prompt, user_content, task_def):
    spec = MODELS[model]
    if spec.get("format") == "json_object":
        system_prompt = augment_system_prompt_for_json_object(system_prompt, task_def)
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "json_schema", "json_schema": {
            "name": "classification", "strict": True, "schema": task_def["json_schema"]}}
    body = {
        "model": spec["id"],
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}],
        "response_format": response_format,
        "max_tokens": spec.get("max_tokens", 256),
        "reasoning": spec["reasoning"],
        "usage": {"include": True},
        "provider": {"require_parameters": True, **spec.get("provider", {})},
    }
    if "temperature" in spec:
        body["temperature"] = spec["temperature"]
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "Political Science LLM Benchmark"})
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
    if "error" in d:
        raise RuntimeError(f"OpenRouter error: {str(d['error'])[:300]}")
    usage = d.get("usage") or {}
    choice = d["choices"][0]
    return {
        "content": (choice["message"].get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": time.perf_counter() - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "provider": d.get("provider"),
        "returned_model": d.get("model"),
    }


def run_one(key, model, task_def, system_prompt, item, condition, cap):
    with lock:
        if spent["usd"] >= cap:
            raise SystemExit(f"cost cap ${cap} reached")
    r = call(key, model, system_prompt, item["user_content"], task_def)
    with lock:
        spent["usd"] += r["cost_usd"] or 0.0
    pred, err = parse_content(r["content"], task_def)
    row = {"task": task_def["name"], "model": model, "model_id": MODELS[model]["id"],
           "item_id": item["item_id"], "condition": condition, "parse_error": err,
           "raw_content_preview": r["content"][:200], **{k: v for k, v in r.items() if k != "content"}}
    row.update({f"pred_{k}": v for k, v in pred.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="items per task per model")
    # DeepSeek V4 Flash was run by mistake for part of the sample; Gemini 3.8
    # Flash replaces it. Its rows stay in the ledger but are not in the default set.
    ap.add_argument("--models", default="deepseek_v4_pro,gpt_5_6_terra,gemini_3_8_flash,claude_sonnet_5")
    ap.add_argument("--cost-cap-usd", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--job-id", default=None, help="jobstatus id for menu-bar progress")
    ap.add_argument("--ignore-names", action="store_true",
                    help="as-written text with IGNORE_NOTE added to the prompt")
    args = ap.parse_args()
    key = (REPO / ".openrouter_api_key").read_text().strip()
    out_root = (IGNORE_RUN_DIR if args.ignore_names else RUN_DIR) / (f"pilot{args.pilot}" if args.pilot else "full")
    ledger = out_root / "calls.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["task"], r["item_id"], r["condition"]))
            spent["usd"] += r.get("cost_usd") or 0.0
    print(f"resuming with {len(done)} calls done, ${spent['usd']:.4f} spent", flush=True)

    jobs = []
    for task in TASKS:
        orig, mask, keep = sample_items(task, args.pilot)
        print(f"{task}: {len(keep)} items ({sum(k[2] for k in keep)} named)", flush=True)
        prompts = {"orig": orig["prompt_path"].read_text(), "masked": mask["prompt_path"].read_text()}
        prompts["ignore"] = prompts["orig"].rstrip() + "\n\n" + IGNORE_NOTE
        for model in args.models.split(","):
            for x, y, _ in keep:
                conds = (("ignore", orig, x),) if args.ignore_names else (("orig", orig, x), ("masked", mask, y))
                for cond, td, it in conds:
                    if (model, td["name"], it["item_id"], cond) not in done:
                        jobs.append((model, td, prompts[cond], it, cond))
    print(f"{len(jobs)} calls to make", flush=True)

    def status(done_n, **kw):
        if args.job_id:
            extra = [f"--{k}={v}" for k, v in kw.items()]
            subprocess.run(["jobstatus", "set", args.job_id, "--done", str(done_n),
                            "--note", f"${spent['usd']:.2f} spent of ${args.cost_cap_usd:.0f} cap", *extra],
                           check=False)

    def work(job):
        try:
            return run_one(key, *job, args.cost_cap_usd)
        except SystemExit:
            raise
        except Exception as e:  # recorded, not retried automatically
            return {"task": job[1]["name"], "model": job[0], "item_id": job[3]["item_id"],
                    "condition": job[4], "parse_error": f"call_failed: {str(e)[:200]}", "cost_usd": None}

    n = 0
    with ThreadPoolExecutor(args.workers) as pool, ledger.open("a") as fh:
        for row in pool.map(work, jobs):
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} calls, ${spent['usd']:.3f} spent", flush=True)
                status(n)
    print(f"finished: {n} calls, ${spent['usd']:.4f} spent in total", flush=True)
    status(n, state="finished")

    # Write per-model task files in the Hive layout for the analysis script.
    calls = pd.read_json(ledger, lines=True)
    calls = calls[calls.model.isin(args.models.split(","))]
    for (model, task, cond), g in calls.groupby(["model", "task", "condition"]):
        d = out_root / cond / model / "tasks"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns="condition").to_csv(d / f"{task}.csv", index=False)


if __name__ == "__main__":
    main()
