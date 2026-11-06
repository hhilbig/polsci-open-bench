#!/usr/bin/env python3
"""November 2026 API additions through OpenRouter, on the frozen September panel.

Models with a batch version go through the OpenRouter Batch API (one batch per
model and stage). The others are sent one request at a time to one pinned host.
Prompts, schemas and items come from the September panel; outputs stay in the
sidecar until an explicit release step uses them.

Spending:
- A batch cannot be cancelled, so the cap is applied before submission: spent +
  reserved + this reservation must stay at or under --cost-cap-usd.
- The pilot reserves the worst case: every request reaches max_tokens.
- The remainder reserves 1.5 times the observed pilot cost per request, and needs
  --remainder-approved. This is an estimate, not a hard limit: a batch can cost
  more than its reservation, and the overrun shows up only when it is collected.
- No paid retries happen automatically.

A state file is written before each paid POST. If the process dies in between,
the next submit refuses to send that batch again until someone checks the account.

Usage:
  python3 code/refresh_openrouter_batch.py budget --stage pilot
  python3 code/refresh_openrouter_batch.py submit --stage pilot
  python3 code/refresh_openrouter_batch.py watch --stage pilot
  python3 code/refresh_openrouter_batch.py report --stage pilot
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import benchmark  # noqa: E402
from run_registry import DEFAULT_STATUS_MD, append_event, render_markdown  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

ROOT = REPO / "output" / "sidecar" / "refresh_openrouter_20261104"
PANEL = REPO / "output" / "sidecar" / "refresh_20260910" / "panel.json"
KEY_PATH = REPO / ".openrouter_api_key"
API = "https://openrouter.ai/api/v1"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
RUN_ID = "refresh-openrouter-20261104"
# Worst-case input tokens per byte of request body. Prompts tokenize at more than
# 2 bytes per token in practice, so this overstates input cost on purpose.
TOKENS_PER_BYTE = 0.5
REMAINDER_MARGIN = 1.5

# Output limit: 256 as in September when reasoning can be off. Models that require
# reasoning get 2048, because reasoning tokens count against the limit.
OFF = {"effort": "none"}
LOW = {"effort": "low"}
MODELS = {
    # Batch API. These models have one batch provider each.
    "claude-opus-5.5": dict(id="anthropic/claude-opus-5.5", mode="batch", reasoning=LOW,
                            max_tokens=2048, usd_in=2.0, usd_out=10.0),
    "claude-sonnet-5.5": dict(id="anthropic/claude-sonnet-5.5", mode="batch", reasoning=LOW,
                              max_tokens=2048, usd_in=1.0, usd_out=5.0),
    "claude-haiku-5.5": dict(id="anthropic/claude-haiku-5.5", mode="batch", reasoning=OFF,
                             max_tokens=256, usd_in=0.05, usd_out=0.25),
    "gpt-6.1-sol": dict(id="openai/gpt-6.1-sol", mode="batch", reasoning=LOW,
                        max_tokens=2048, usd_in=1.0, usd_out=5.0),
    "gpt-6-luna": dict(id="openai/gpt-6-luna", mode="batch", reasoning=OFF,
                       max_tokens=256, usd_in=0.05, usd_out=0.25),
    "gemini-4-argon": dict(id="google/gemini-4-argon", mode="batch", reasoning=LOW,
                           max_tokens=2048, temperature=0, usd_in=1.0, usd_out=5.0, split_by_task=True),
    "grok-4.7": dict(id="x-ai/grok-4.7", mode="batch", reasoning=LOW,
                     max_tokens=2048, temperature=0, usd_in=1.6, usd_out=4.8),
    "glm-5.3": dict(id="z-ai/glm-5.3", mode="batch", reasoning=LOW,
                    max_tokens=2048, temperature=0, usd_in=0.45, usd_out=2.0),
    # No batch version: one request at a time, pinned to one host and quantization.
    "kimi-k3.1-pro": dict(id="moonshotai/kimi-k3.1-pro", mode="sync", reasoning=OFF,
                          max_tokens=256, temperature=0, host="moonshotai/mxfp4",
                          usd_in=2.0, usd_out=10.0),
    "deepseek-v4.1-flash-1017": dict(id="deepseek/deepseek-v4.1-flash-1017", mode="sync",
                                     reasoning=OFF, max_tokens=256, temperature=0,
                                     host="deepinfra/fp8", usd_in=0.098, usd_out=0.294),
    # OpenRouter lists effort "none" here, but Mistral rejects it with HTTP 400.
    "mistral-large-4": dict(id="mistralai/mistral-large-4-0", mode="sync", reasoning=LOW,
                            max_tokens=2048, temperature=0, host="mistral/zdr",
                            usd_in=0.7, usd_out=2.1),
    "beam-1.0": dict(id="reflection/beam-1.0-501b-a23b", mode="sync", reasoning=LOW,
                     max_tokens=4096, temperature=0, host="coreweave/nvfp4",
                     usd_in=0.3, usd_out=1.0),
}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def build_body(spec, task, item):
    """One chat-completions body; the task's own prompt and schema, as in September."""
    body = {
        "messages": [{"role": "system", "content": Path(task["prompt_path"]).read_text()},
                     {"role": "user", "content": item["user_content"]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "classification", "strict": True, "schema": task["json_schema"]}},
        "max_tokens": spec["max_tokens"],
        "reasoning": spec["reasoning"],
    }
    if "temperature" in spec:
        body["temperature"] = spec["temperature"]
    return body


def load_requests(model, stage):
    """Frozen panel rows for this stage, after checking the task sources still match."""
    panel = read(PANEL)
    tasks = {t["name"]: t for t in load_task_definitions()}
    assert set(panel["sources"]) == set(tasks), "Task set differs from the frozen panel"
    for name, task in tasks.items():
        current = digest([task["loader"](), Path(task["prompt_path"]).read_text(), task["json_schema"]])
        assert current == panel["sources"][name], f"Source changed: {name}"
    out = []
    for i, row in enumerate(panel["rows"]):
        if row["pilot"] == (stage == "pilot"):
            out.append({"custom_id": f"r{i:04d}", "task": row["task"], "item": row["item"],
                        "body": build_body(MODELS[model], tasks[row["task"]], row["item"])})
    return out, tasks


def worst_case_usd(spec, requests):
    total = 0.0
    for r in requests:
        input_tokens = len(json.dumps(r["body"], ensure_ascii=False).encode()) * TOKENS_PER_BYTE
        total += (input_tokens * spec["usd_in"] + spec["max_tokens"] * spec["usd_out"]) / 1e6
    return total


def reservation(model, stage, requests, root=None):
    if stage == "pilot":
        return worst_case_usd(MODELS[model], requests)
    state = read_state(model, "pilot", root)
    if not state or state.get("status") != "collected" or not state.get("n_requests"):
        raise SystemExit(f"{model}: pilot not collected, cannot size the remainder")
    return state["cost_usd"] / state["n_requests"] * len(requests) * REMAINDER_MARGIN


SYNC_RETRIES = 6
NO_BALANCE_EXIT = 75


class NoBalance(SystemExit):
    def __init__(self, message):
        print(message, flush=True)
        super().__init__(NO_BALANCE_EXIT)


def state_path(model, stage, root=None):
    return (root or ROOT) / model / stage / "state.json"


def read_state(model, stage, root=None):
    path = state_path(model, stage, root)
    return read(path) if path.exists() else None


def committed_usd(root=None):
    """Actual cost of everything collected, plus reservations for the rest."""
    total = 0.0
    for path in sorted((root or ROOT).glob("*/*/state.json")):
        s = read(path)
        total += s["cost_usd"] if s.get("status") == "collected" else s.get("reserved_usd", 0.0)
    return total


def check_cap(new_usd, cap, root=None):
    so_far = committed_usd(root)
    if so_far + new_usd > cap:
        raise SystemExit(f"Refusing: ${so_far:.2f} committed + ${new_usd:.2f} new exceeds the ${cap:.2f} cap")
    return so_far


def http(key, method, url, payload=None, timeout=300):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "X-Title": "Political Science LLM Benchmark"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check_prices(models):
    """Stop if a listed price moved, because every reservation depends on it."""
    req = urllib.request.Request(f"{API}/models", headers={"User-Agent": "polsci-open-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        listed = {m["id"]: m for m in json.load(r)["data"]}
    for model in models:
        spec = MODELS[model]
        if spec["mode"] == "batch":
            m = listed[f"{spec['id']}:batch"]["pricing"]
            usd_in, usd_out = float(m["prompt"]) * 1e6, float(m["completion"]) * 1e6
            assert math.isclose(usd_in, spec["usd_in"], rel_tol=.01) and math.isclose(
                usd_out, spec["usd_out"], rel_tol=.01), f"{model}: price is now {usd_in}/{usd_out}"


def batch_payload(spec, requests):
    # The docs require `requests` to come last, or the submit returns 400.
    return {"endpoint": "/v1/chat/completions", "model": spec["id"], "completion_window": "24h",
            "requests": [{"custom_id": r["custom_id"], "body": r["body"]} for r in requests]}


def batch_groups(spec, requests):
    """Requests per batch. Google needs one batch per JSON schema, which here means per task."""
    if not spec.get("split_by_task"):
        return [requests]
    groups = {}
    for r in requests:
        groups.setdefault(r["task"], []).append(r)
    return list(groups.values())


def post_batch(key, payload, attempts=6):
    """A 429 means the batch was refused, so waiting and sending it again cannot charge twice."""
    for attempt in range(attempts):
        try:
            return http(key, "POST", f"{API}/batches", payload)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(120)


def submit_batch(key, model, stage, requests, reserved):
    spec, path = MODELS[model], state_path(model, stage)
    payloads = [batch_payload(spec, group) for group in batch_groups(spec, requests)]
    state = {"model": model, "stage": stage, "mode": "batch", "status": "submitting",
             "n_requests": len(requests), "reserved_usd": reserved, "created": now(),
             "request_sha256": digest(payloads), "batch_ids": [], "refused": []}
    save(path, state)
    for payload in payloads:
        try:
            response = post_batch(key, payload)
        except urllib.error.HTTPError as e:
            if not state["batch_ids"]:
                raise  # Nothing went through: main() clears the state file.
            state["refused"].append(f"HTTP {e.code}: {e.read()[:300]!r}")
            save(path, state)
            continue
        state["batch_ids"].append(response["id"])
        save(path, state)
    state.update(status="submitted", submitted=now())
    save(path, state)
    print(f"{model}: {len(state['batch_ids'])} batch(es), {len(requests)} requests, "
          f"{len(state['refused'])} refused, reserved ${reserved:.3f}", flush=True)


def retry_missing(key, model, stage, cap):
    """Resubmit requests with no successful result (refused batches, provider errors)."""
    spec, path = MODELS[model], state_path(model, stage)
    state = read(path)
    if state.get("mode") != "batch" or state["status"] not in ("collected", "completed", "ended"):
        raise SystemExit(f"{model}: retry needs a finished batch run, status is {state['status']}")
    raw = read(path.parent / "results.json")
    ok = {e["custom_id"] for e in raw.get("results") or []
          if ((e.get("response") or {}).get("body") or {}).get("choices")}
    requests, _ = load_requests(model, stage)
    todo = [r for r in requests if r["custom_id"] not in ok]
    if not todo:
        print(f"{model}: nothing to retry", flush=True)
        return
    new = reservation(model, stage, todo)
    check_cap(new + (state.get("cost_usd") or 0.0) - committed_of(state), cap)
    payloads = [batch_payload(spec, group) for group in batch_groups(spec, todo)]
    state.setdefault("refused_earlier", []).extend(state.get("refused", []))
    state.update(status="submitting", refused=[], reserved_usd=(state.get("cost_usd") or 0.0) + new,
                 retry_requests=len(todo), retried=now())
    save(path, state)
    posted = 0
    for payload in payloads:
        try:
            response = post_batch(key, payload)
        except urllib.error.HTTPError as e:
            state["refused"].append(f"HTTP {e.code}: {e.read()[:300]!r}")
            save(path, state)
            continue
        state["batch_ids"].append(response["id"])
        posted += 1
        save(path, state)
    # Later batches come last, so their results replace earlier failures for the same custom_id.
    state["status"] = "submitted" if posted else "collected"
    save(path, state)
    print(f"{model}: retried {len(todo)} requests in {posted}/{len(payloads)} batch(es)", flush=True)
    if len(state["refused"]):
        raise NoBalance(f"{model}: {len(state['refused'])} retry batch(es) refused")


def committed_of(state):
    return state["cost_usd"] if state.get("status") == "collected" else state.get("reserved_usd", 0.0)


def run_sync(key, model, stage, requests, reserved, cap):
    spec, path = MODELS[model], state_path(model, stage)
    calls = path.parent / "calls.jsonl"
    done, spent = {}, 0.0
    if calls.exists():
        for line in calls.read_text().splitlines():
            row = json.loads(line)
            done[row["custom_id"]] = row
            spent += ((row.get("response") or {}).get("usage") or {}).get("cost") or 0.0
    save(path, {"model": model, "stage": stage, "mode": "sync", "status": "running",
                "n_requests": len(requests), "reserved_usd": reserved, "created": now()})
    todo = [r for r in requests if r["custom_id"] not in done]
    base = committed_usd() - reserved

    def one(r):
        body = {**r["body"], "model": spec["id"], "usage": {"include": True},
                "provider": {"order": [spec["host"]], "allow_fallbacks": False, "require_parameters": True}}
        # Retry transient failures so a network drop or rate limit doesn't become a failed row.
        for attempt in range(SYNC_RETRIES + 1):
            try:
                return {"custom_id": r["custom_id"],
                        "response": http(key, "POST", f"{API}/chat/completions", body, timeout=180)}
            except urllib.error.HTTPError as e:
                error = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code == 402:
                    return {"custom_id": r["custom_id"], "no_balance": error}
                if e.code != 429 and e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                error = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < SYNC_RETRIES:
                time.sleep(min(300, 15 * 2 ** attempt))
        return {"custom_id": r["custom_id"], "error": error}

    with ThreadPoolExecutor(4) as pool, calls.open("a") as fh:
        for start in range(0, len(todo), 4):
            if base + spent > cap:
                raise SystemExit(f"{model}: stopped at the ${cap:.2f} cap")
            rows = list(pool.map(one, todo[start:start + 4]))
            for row in rows:
                if "no_balance" in row:
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                spent += ((row.get("response") or {}).get("usage") or {}).get("cost") or 0.0
            if any("no_balance" in row for row in rows):
                raise NoBalance(f"{model}: OpenRouter has no balance left for this key; resume later")
    results = [json.loads(x) for x in calls.read_text().splitlines()]
    state = read(path)
    state.update(status="completed", cost_usd=spent, submitted=state["created"])
    save(path, state)
    save(path.parent / "results.json", {"results": results})
    print(f"{model}: {len(results)} calls, ${spent:.4f}", flush=True)


def poll_batch(key, model, stage):
    """Save results once every batch is finished. Returns True when nothing is left to wait for."""
    path = state_path(model, stage)
    state = read(path)
    if state["status"] in ("completed", "collected", "ended"):
        return True
    if state["status"] != "submitted":
        return False
    ids = state.get("batch_ids") or [state["batch_id"]]
    batches = [http(key, "GET", f"{API}/batches/{i}") for i in ids]
    state["remote_status"] = sorted({b.get("status") for b in batches})
    state["request_counts"] = {k: sum((b.get("request_counts") or {}).get(k, 0) for b in batches)
                               for k in ("total", "completed", "failed")}
    # The docs say results can arrive shortly after a batch reports completed.
    if any(b.get("status") not in TERMINAL or (b["status"] == "completed" and not b.get("results"))
           for b in batches):
        save(path, state)
        return False
    errors = [b["error"] for b in batches if b.get("error")] + state.get("refused", [])
    save(path.parent / "results.json", {"results": [e for b in batches for e in b.get("results") or []],
                                        "batches": [{k: v for k, v in b.items() if k != "results"}
                                                    for b in batches]})
    state.update(status="completed" if all(b["status"] == "completed" for b in batches) else "ended",
                 cost_usd=sum((b.get("usage") or {}).get("cost") or 0.0 for b in batches),
                 provider=sorted({str(b.get("provider")) for b in batches}), error=errors, finished=now())
    save(path, state)
    print(f"{model}: {state['status']}, ${state['cost_usd']:.4f}", flush=True)
    return True


def decode(model, request, entry, task):
    """One prediction row. A failed call stays in the output as a counted failure."""
    row = {"task": task["name"], "model": model, "model_id": MODELS[model]["id"],
           "item_id": request["item"]["item_id"], "custom_id": request["custom_id"]}
    row.update({f"gt_{k}": v for k, v in request["item"]["gt"].items()})
    response = (entry or {}).get("response")
    body = response.get("body", response) if isinstance(response, dict) else None
    if not body or not body.get("choices"):
        problem = (entry or {}).get("error") or (body or {}).get("error") or "no result returned"
        row.update(parse_error=f"call_failed: {str(problem)[:300]}", truncated=False, raw_content="")
        return row
    choice = body["choices"][0]
    text = (choice.get("message") or {}).get("content") or ""
    stop = choice.get("finish_reason")
    usage = body.get("usage") or {}
    predictions, error = benchmark.parse_content(text, task)
    try:
        obj = json.loads(text)
        expected = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
        if not isinstance(obj, dict) or set(obj) != set(expected):
            raise ValueError("Missing or unexpected output fields")
        if task["label_kind"] == "categorical":
            if obj[task["label_key"]] not in task["labels"]:
                raise ValueError("Unknown categorical label")
        elif any(type(v) is not int or v not in (0, 1) for v in obj.values()):
            raise ValueError("Binary fields must be integers 0 or 1")
    except ValueError as exc:
        error = "schema_invalid: " + str(exc)[:160]
    details = usage.get("completion_tokens_details") or {}
    row.update(parse_error=error, truncated=stop in ("length", "max_tokens"), stop_reason=stop,
               returned_model=body.get("model"), provider=body.get("provider"),
               input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
               reasoning_tokens=details.get("reasoning_tokens"), raw_content=text)
    row.update({f"pred_{k}": v for k, v in predictions.items()})
    return row


def collect(model, stage):
    """Write predictions.csv and return a summary for this model and stage."""
    import pandas as pd

    path = state_path(model, stage)
    state = read(path)
    requests, tasks = load_requests(model, stage)
    raw = read(path.parent / "results.json") if (path.parent / "results.json").exists() else {}
    entries = {e["custom_id"]: e for e in raw.get("results") or []}
    rows = [decode(model, r, entries.get(r["custom_id"]), tasks[r["task"]]) for r in requests]
    df = pd.DataFrame(rows)
    df.to_csv(path.parent / "predictions.csv", index=False)
    if state["status"] == "completed":
        state["status"] = "collected"
        save(path, state)
    bad = df.parse_error.notna() & (df.parse_error.astype(str) != "")
    summary = {"model": model, "status": state["status"], "requests": len(df), "failed": int(bad.sum()),
               "truncated": int(df.truncated.fillna(False).astype(bool).sum()),
               "cost_usd": round(state.get("cost_usd") or 0.0, 4),
               "mean_input": round(float(pd.to_numeric(df.get("input_tokens"), errors="coerce").mean()), 1)
               if "input_tokens" in df else None,
               "mean_output": round(float(pd.to_numeric(df.get("output_tokens"), errors="coerce").mean()), 1)
               if "output_tokens" in df else None,
               "mean_reasoning": round(float(pd.to_numeric(df.get("reasoning_tokens"), errors="coerce").mean()), 1)
               if "reasoning_tokens" in df else None,
               "providers": sorted(str(x) for x in df.get("provider", pd.Series(dtype=str)).dropna().unique()),
               "returned_models": sorted(str(x) for x in df.get("returned_model", pd.Series(dtype=str)).dropna().unique())}
    return summary


def register(note, status="running", cap=None):
    append_event(run_id=RUN_ID, event="update", status=status, host="local-mac",
                 runner="code/refresh_openrouter_batch.py",
                 model_scope=",".join(MODELS), task_scope="34_tasks:frozen_panel:68_pilot_then_3332_remainder",
                 batch_sizes="one_batch_per_model_and_stage", output=str(ROOT.relative_to(REPO)),
                 log=str((ROOT / "watch.log").relative_to(REPO)), cost_cap_usd=cap, note=note)
    render_markdown(status_path=DEFAULT_STATUS_MD)


def jobstatus(**kw):
    args = [f"--{k}={v}" for k, v in kw.items()]
    subprocess.run(["jobstatus", "set", "polsci-openrouter-additions", "--project", "polsci-open-bench",
                    "--task", "Run the new API models on the benchmark through OpenRouter",
                    "--unit", "models", "--host", "this-mac", *args], check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["budget", "submit", "retry", "watch", "report"])
    ap.add_argument("--interim", action="store_true",
                    help="watch: leave the menu-bar status running; a later watch reports the end")
    ap.add_argument("--stage", choices=["pilot", "remainder"], default="pilot")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--cost-cap-usd", type=float, default=50.0)
    ap.add_argument("--remainder-approved", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=120)
    args = ap.parse_args()
    models = args.models.split(",")
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    if args.stage == "remainder" and args.command == "submit" and not args.remainder_approved:
        raise SystemExit("The remainder needs Hanno's approval after the pilot: pass --remainder-approved")

    if args.command == "budget":
        total = 0.0
        for model in models:
            requests, _ = load_requests(model, args.stage)
            usd = reservation(model, args.stage, requests)
            total += usd
            print(f"{model:28s} {MODELS[model]['mode']:6s} {len(requests):5d} requests  reserve ${usd:7.3f}")
        print(f"{'total':28s} {'':6s} {'':5s}           reserve ${total:7.3f}  (committed ${committed_usd():.2f})")
        return

    if args.command == "submit":
        check_prices(models)
        plan = []
        for model in models:
            existing = read_state(model, args.stage)
            if existing and existing["status"] == "submitting":
                raise SystemExit(f"{model}: an earlier submit may have reached OpenRouter. "
                                 "Check GET /batches before doing anything else.")
            if existing and existing["status"] != "running":
                print(f"{model}: already {existing['status']}, skipping", flush=True)
                continue
            requests, _ = load_requests(model, args.stage)
            plan.append((model, requests, reservation(model, args.stage, requests)))
        check_cap(sum(p[2] for p in plan), args.cost_cap_usd)
        key = KEY_PATH.read_text().strip()
        register(f"Submitting {args.stage}: {[p[0] for p in plan]}, ${sum(p[2] for p in plan):.2f} reserved",
                 cap=args.cost_cap_usd)
        no_balance = False
        for model, requests, reserved in plan:
            try:
                if MODELS[model]["mode"] == "batch":
                    submit_batch(key, model, args.stage, requests, reserved)
                else:
                    run_sync(key, model, args.stage, requests, reserved, args.cost_cap_usd)
            except urllib.error.HTTPError as e:
                # The state file stays at "submitting" only if the POST outcome is unknown.
                # An HTTP error means the batch was refused, so it is safe to clear.
                state_path(model, args.stage).unlink(missing_ok=True)
                print(f"{model}: refused, HTTP {e.code}: {e.read()[:400]!r}", flush=True)
                no_balance |= e.code == 402
        if no_balance:
            raise NoBalance("Some batches were refused for lack of balance; resubmit later")
        return

    if args.command == "retry":
        if args.stage == "remainder" and not args.remainder_approved:
            raise SystemExit("The remainder needs Hanno's approval: pass --remainder-approved")
        key = KEY_PATH.read_text().strip()
        for model in models:
            retry_missing(key, model, args.stage, args.cost_cap_usd)
        return

    if args.command == "watch":
        key = KEY_PATH.read_text().strip()
        states = [m for m in models if read_state(m, args.stage)]
        jobstatus(state="running", total=len(states), done=0, log=ROOT / "watch.log",
                  note=f"{args.stage}: waiting for batches")
        try:
            while True:
                waiting = []
                for model in states:
                    try:
                        if not poll_batch(key, model, args.stage):
                            waiting.append(model)
                    except (urllib.error.URLError, TimeoutError) as e:
                        print(f"{model}: poll failed, will retry: {e}", flush=True)
                        waiting.append(model)
                jobstatus(state="running", total=len(states), done=len(states) - len(waiting),
                          note=f"{args.stage}: still waiting for {', '.join(waiting) or 'nothing'}")
                if not waiting:
                    break
                time.sleep(args.poll_seconds)
        except BaseException as e:
            jobstatus(state="failed", note=f"{args.stage}: watcher stopped: {str(e)[:120]}")
            raise
        jobstatus(state="running" if args.interim else "finished", total=len(states), done=len(states),
                  note=f"{args.stage}: all results returned; ${committed_usd():.2f} committed")
        args.command = "report"

    if args.command == "report":
        summaries = [collect(m, args.stage) for m in models if read_state(m, args.stage)
                     and read_state(m, args.stage)["status"] in ("completed", "collected", "ended")]
        save(ROOT / f"{args.stage}_report.json", {"created": now(), "committed_usd": committed_usd(),
                                                  "models": summaries})
        for s in summaries:
            print(json.dumps(s, ensure_ascii=False))
        print(f"committed ${committed_usd():.4f}")
        register(f"{args.stage}: {len(summaries)} models collected; ${committed_usd():.2f} committed")


if __name__ == "__main__":
    main()
