#!/usr/bin/env python3
"""Claude Sonnet 5 on the 2026-11-14 task pool through Anthropic's Message Batches API.

OpenRouter serves Sonnet 5's batch endpoint only with the user's own Anthropic
key, so this script sends the same requests to Anthropic directly with
~/.anthropic_api_key (the route of Sonnet 5's November run). Requests are the
API draw items from code/ext_api_batch.load_requests (same items, prompts and
moderation exclusions as the OpenRouter models). Settings follow November:
no extended thinking, 256 output tokens, the answer returned as a forced tool
call whose input schema is the task schema (code/anthropic_batch_submit.py).
The task instructions are the system message with a one-hour cache marker.

Results are rewritten into the OpenRouter result layout, so
refresh_openrouter_batch.collect writes predictions.csv as for the other models.

Usage (from the repository root):
    python3 experiments/task_ext_20261114/anthropic_direct.py submit --stage pilot --cost-cap-usd 6
    python3 experiments/task_ext_20261114/anthropic_direct.py watch  --stage pilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
import anthropic  # noqa: E402

import ext_api_batch as e  # noqa: E402
from anthropic_batch_submit import anthropic_input_schema, anthropic_key_map, anthropic_system_prompt  # noqa: E402

MODEL = "claude-sonnet-5"
# Anthropic batch prices per million tokens (half the list price of $2/$10).
# One-hour cache writes cost 2x and cache reads 0.1x the input price.
USD_IN, USD_OUT = 1.0, 5.0
MAX_TOKENS = 256


def now():
    return datetime.now(timezone.utc).isoformat()


def client():
    key = (Path.home() / ".anthropic_api_key").read_text().strip()
    return anthropic.Anthropic(api_key=key, timeout=300.0, max_retries=3)


def paths(stage):
    d = e.ROOT / MODEL / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / "state.json", d / "results.json"


def params(task, item):
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": anthropic_system_prompt(task, Path(task["prompt_path"]).read_text()),
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": item["user_content"]}],
        "tools": [{"name": "classify", "description": "Return the classification for the input.",
                   "input_schema": anthropic_input_schema(task)}],
        "tool_choice": {"type": "tool", "name": "classify"},
    }


def worst_case_usd(requests, tasks):
    """Every input token at the cache-write price and every answer at the limit."""
    chars = sum(len(Path(tasks[r["task"]]["prompt_path"]).read_text()) + len(r["item"]["user_content"])
                for r in requests)
    return (chars / 3.0 * USD_IN * 2 + len(requests) * MAX_TOKENS * USD_OUT) / 1e6


def submit(stage, cap):
    state_path, _ = paths(stage)
    if state_path.exists():
        raise SystemExit(f"{MODEL} {stage}: already {json.loads(state_path.read_text())['status']}")
    requests, tasks = e.load_requests(MODEL, stage)
    worst = worst_case_usd(requests, tasks)
    pilot = e.ROOT / MODEL / "pilot" / "state.json"
    if stage == "remainder" and pilot.exists():
        # As in refresh_openrouter_batch: reserve 1.5 times the pilot's cost per
        # request (pilot items are the longest, so this overstates the remainder).
        p = json.loads(pilot.read_text())
        worst = min(worst, 1.5 * p["cost_usd"] / p["n_requests"] * len(requests))
    spent = e.rob.committed_usd()
    if spent + worst > cap:
        raise SystemExit(f"cap: committed ${spent:.2f} + worst case ${worst:.2f} > ${cap:.2f}")
    batch = client().messages.batches.create(requests=[
        {"custom_id": r["custom_id"], "params": params(tasks[r["task"]], r["item"])} for r in requests])
    state = {"model": MODEL, "stage": stage, "mode": "anthropic_direct", "status": "submitted",
             "n_requests": len(requests), "reserved_usd": round(worst, 4), "created": now(),
             "batch_ids": [batch.id], "refused": []}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    e.register(f"Submitted {MODEL} {stage} to Anthropic directly: {len(requests)} requests, batch {batch.id}, "
               f"worst case ${worst:.2f}", cap=cap)
    print(f"{MODEL} {stage}: batch {batch.id}, {len(requests)} requests, worst case ${worst:.2f}", flush=True)


def to_openrouter_entry(result, tasks_by_id):
    """One Anthropic batch result in the layout refresh_openrouter_batch.decode reads."""
    cid = result.custom_id
    if result.result.type != "succeeded":
        return {"custom_id": cid, "error": f"{result.result.type}: {getattr(result.result, 'error', '')}"[:300]}
    msg = result.result.message
    task = tasks_by_id[cid]
    back = {safe: label for label, safe in anthropic_key_map(task).items()}
    calls = [b for b in msg.content if b.type == "tool_use"]
    content = json.dumps({back.get(k, k): v for k, v in calls[0].input.items()}) if calls else ""
    u = msg.usage
    read = u.cache_read_input_tokens or 0
    written = u.cache_creation_input_tokens or 0
    prompt = u.input_tokens + read + written
    cost = (u.input_tokens * USD_IN + read * USD_IN * 0.1 + written * USD_IN * 2
            + u.output_tokens * USD_OUT) / 1e6
    return {"custom_id": cid, "response": {"body": {
        "model": msg.model, "provider": "Anthropic (direct)",
        "choices": [{"message": {"content": content},
                     "finish_reason": "length" if msg.stop_reason == "max_tokens" else "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": u.output_tokens, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": written}}}}}


def watch(stage, poll):
    state_path, results_path = paths(stage)
    state = json.loads(state_path.read_text())
    c = client()
    requests, tasks = e.load_requests(MODEL, stage)
    tasks_by_id = {r["custom_id"]: tasks[r["task"]] for r in requests}
    while True:
        try:
            b = c.messages.batches.retrieve(state["batch_ids"][0])
        except Exception as exc:  # network drops: retry
            print(f"{MODEL} {stage}: poll failed, will retry: {exc}", flush=True)
            time.sleep(poll)
            continue
        counts = b.request_counts
        state["request_counts"] = {"processing": counts.processing, "succeeded": counts.succeeded,
                                   "errored": counts.errored, "expired": counts.expired}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if b.processing_status == "ended":
            break
        time.sleep(poll)
    entries = [to_openrouter_entry(r, tasks_by_id) for r in c.messages.batches.results(b.id)]
    results_path.write_text(json.dumps({"results": entries}) + "\n")
    state.update(status="completed", finished=now(),
                 cost_usd=sum(((x.get("response") or {}).get("body") or {}).get("usage", {}).get("cost", 0)
                              for x in entries))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    summary = e.rob.collect(MODEL, stage)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    e.install()
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["submit", "watch"])
    ap.add_argument("--stage", choices=["pilot", "remainder"], required=True)
    ap.add_argument("--cost-cap-usd", type=float, default=45.0)
    ap.add_argument("--poll-seconds", type=int, default=180)
    a = ap.parse_args()
    submit(a.stage, a.cost_cap_usd) if a.command == "submit" else watch(a.stage, a.poll_seconds)
