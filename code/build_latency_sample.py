#!/usr/bin/env python3
"""Sequential per-request latency for the refresh panel models.

The refresh panel has cost and accuracy for every model but no usable timing.
The eight original API models ran through provider batch endpoints over one to
three days, so their wall-clock reflects batch scheduling rather than model
speed, and Jev has no batch endpoint at all. This script measures the one thing
that makes them comparable: how long a single request takes, with every model
called the same way.

Design. A stratified sample of the frozen panel, three items from each task, so
prompt lengths match what the panel already measured. Every call is sequential
and unbatched, one request in flight at a time. Wall-clock is recorded per
request. Predictions are retained so they can be checked against the stored
panel: a mismatch means the harness is wrong, not the model.

This is a single measurement under one set of network and load conditions. It
gives the ordering and the order of magnitude, not a reproducible benchmark.

Usage:
  python3 code/build_latency_sample.py --per-task 3 --dry-run
  python3 code/build_latency_sample.py --per-task 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PANEL = REPO / "output/sidecar/refresh_20260910"
OUT = REPO / "output/sidecar/latency_sample"
SEED = 20260919

# provider -> (env var, fallback key files relative to repo or home)
CREDENTIALS = {
    "anthropic": ("ANTHROPIC_API_KEY", ["~/.anthropic_api_key"]),
    "openai": ("OPENAI_API_KEY", ["~/.openai_api_key"]),
    "deepseek": ("DEEPSEEK_API_KEY", ["~/.deepseek_api_key"]),
    "openrouter": ("OPENROUTER_API_KEY", [".openrouter_api_key", "~/.openrouter_api_key"]),
    "google": ("GEMINI_API_KEY", ["~/.gemini_api_key", "~/.google_api_key"]),
}

MODELS = {
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-6-astra": "openai",
    "deepseek-v4-flash": "deepseek",
    "deepseek-v4-pro": "deepseek",
    "jev-1.13.0": "openrouter",
    # Added after the original timed run, so these two were missing from the
    # latency panel until now.
    "gemini-3.8-flash": "google",
    "gemini-3.1-flash-lite": "google",
}


def load_key(provider):
    env, files = CREDENTIALS[provider]
    value = os.environ.get(env, "").strip()
    if value:
        return value
    for name in files:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = REPO / name
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
    raise RuntimeError(f"no credential for {provider}")


def sample_items(per_task, active_only=True):
    """Stratified draw from the frozen panel, reproducible by SEED."""
    import sys
    sys.path.insert(0, str(HERE))
    from task_registry import load_task_definitions

    panel = json.loads((PANEL / "panel.json").read_text())
    names = {t["name"] for t in load_task_definitions(active_only=active_only)}
    rows = [r for r in panel["rows"] if r["task"] in names]
    frame = pd.DataFrame([{"task": r["task"], "item_id": str(r["item"]["item_id"]),
                           "user_content": r["item"]["user_content"]} for r in rows])
    # groupby().sample() keeps the grouping column, which .apply(include_groups=False)
    # strips. Tasks with fewer rows than requested contribute what they have.
    counts = frame.task.value_counts()
    if (counts < per_task).any():
        short = counts[counts < per_task].to_dict()
        print(f"  note: tasks with fewer than {per_task} panel rows: {short}")
    return (frame.groupby("task", group_keys=False)
            .sample(n=per_task, random_state=SEED, replace=False)
            .reset_index(drop=True))


def estimate_cost(n_items):
    """Per-model estimate from the panel's own measured cost per item."""
    path = REPO / "output/sidecar/jev_sidecar/cost_performance.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path)
    c["estimated_usd"] = c.cost_per_1k_items * n_items / 1000
    return c[["model", "cost_per_1k_items", "estimated_usd"]]


def call(provider, model, system_prompt, user_content, key, timeout=120):
    """One sequential, unbatched request. Returns (latency_s, out_tokens, error)."""
    import urllib.error
    import urllib.request

    if provider == "openrouter":
        # Jev is a Decisions model and rejects chat/completions. Reuse the
        # payload builder from the runner that produced the panel, so the
        # request is identical in shape to the one being timed against.
        import sys
        sys.path.insert(0, str(HERE))
        import refresh_jev
        url = refresh_jev.ENDPOINT
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "HTTP-Referer": "https://hhilbig.github.io/llm-benchmark/",
                   "X-OpenRouter-Title": "Political Science LLM Benchmark"}
        body = refresh_jev.build_payload(system_prompt, {"user_content": user_content})
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        body = {"model": model, "max_tokens": 256,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]}
    else:
        base = {"openai": "https://api.openai.com/v1/chat/completions",
                "deepseek": "https://api.deepseek.com/chat/completions",
                "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "openrouter": "https://openrouter.ai/api/v1/chat/completions"}[provider]
        url = base
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        model_id = "typesafe/jev-1.13" if provider == "openrouter" else model
        body = {"model": model_id,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}]}
        if provider == "openai":
            body["max_completion_tokens"] = 256
        else:
            body["max_tokens"] = 256
        if provider == "google":
            # Same configuration as the scored panel run: without this Gemini
            # thinks by default, and those tokens both cost money and eat the
            # output budget, which would time something other than the run
            # being compared against.
            body["model"] = f"models/{model}"
            body["reasoning_effort"] = "none"

    payload = json.dumps(body).encode()
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
        elapsed = time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - start, None, f"http {exc.code}: {exc.read()[:120]!r}"
    except Exception as exc:
        return time.perf_counter() - start, None, f"{type(exc).__name__}: {exc}"

    usage = data.get("usage", {}) or {}
    out = usage.get("output_tokens") or usage.get("completion_tokens")
    return elapsed, out, None


def run(items, models, output):
    import sys
    sys.path.insert(0, str(HERE))
    from task_registry import load_task_definitions

    task_defs = {t["name"]: t for t in load_task_definitions(active_only=True)}
    prompts = {name: Path(t["prompt_path"]).read_text() for name, t in task_defs.items()}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = []
    for model in models:
        provider = MODELS[model]
        key = load_key(provider)
        started = time.time()
        for i, row in enumerate(items.itertuples(), 1):
            spec = task_defs[row.task] if provider == "openrouter" else prompts[row.task]
            latency, out_tokens, error = call(
                provider, model, spec, row.user_content, key)
            records.append(dict(model=model, provider=provider, task=row.task,
                                item_id=row.item_id, latency_s=latency,
                                output_tokens=out_tokens, error=error,
                                prompt_chars=len(row.user_content)))
            if i % 25 == 0 or i == len(items):
                done = pd.DataFrame(records)
                done = done[done.model == model]
                print(f"  {model:20s} {i:3d}/{len(items)}  "
                      f"median {done.latency_s.median():5.2f}s  "
                      f"errors {done.error.notna().sum()}", flush=True)
        pd.DataFrame(records).to_csv(output / "latency_raw.csv", index=False)
        print(f"  {model:20s} done in {time.time()-started:.0f}s wall", flush=True)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=sorted(MODELS))
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve credentials, draw the sample and price it. No requests.")
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()

    items = sample_items(args.per_task)
    print(f"sample: {len(items)} items across {items.task.nunique()} tasks "
          f"({args.per_task} per task)")
    print(f"prompt chars: median {int(items.user_content.str.len().median())}, "
          f"p90 {int(items.user_content.str.len().quantile(0.9))}")

    print("\ncredentials:")
    for provider in sorted({MODELS[m] for m in args.models}):
        try:
            load_key(provider)
            print(f"  {provider:12s} ok")
        except RuntimeError as exc:
            print(f"  {provider:12s} MISSING ({exc})")

    est = estimate_cost(len(items))
    if est is not None:
        est = est[est.model.isin(args.models)]
        print(f"\nestimated cost for {len(items)} items x {len(args.models)} models:")
        for row in est.sort_values("estimated_usd", ascending=False).itertuples():
            print(f"  {row.model:20s} ${row.estimated_usd:6.3f}")
        print(f"  {'TOTAL':20s} ${est.estimated_usd.sum():6.3f}  "
              f"(batch-era rates; sequential calls may cost more)")

    if args.dry_run:
        print("\ndry run, no requests sent")
        return

    print(f"\nrunning {len(items)} sequential requests x {len(args.models)} models")
    frame = run(items, args.models, args.output)
    out = Path(args.output)
    frame.to_csv(out / "latency_raw.csv", index=False)

    ok = frame[frame.error.isna()]
    summary = (ok.groupby("model")
               .agg(n=("latency_s", "size"),
                    median_latency_s=("latency_s", "median"),
                    p90_latency_s=("latency_s", lambda s: s.quantile(0.9)),
                    mean_latency_s=("latency_s", "mean"),
                    median_output_tokens=("output_tokens", "median"))
               .reset_index())
    summary["items_per_minute_sequential"] = 60 / summary.median_latency_s
    summary["errors"] = summary.model.map(
        frame[frame.error.notna()].groupby("model").size()).fillna(0).astype(int)
    summary = summary.sort_values("median_latency_s")
    summary.to_csv(out / "latency_summary.csv", index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nwrote {out}/latency_raw.csv and latency_summary.csv")


if __name__ == "__main__":
    main()
