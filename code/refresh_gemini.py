#!/usr/bin/env python3
"""Run the Gemini Flash models on the frozen September 2026 benchmark panel.

Added because the panel had no Google model in it, which is the first thing a
reader notices about a comparison of commercial classification APIs. Two models
were named in the request: gemini-3.8-flash and gemini-3.1-flash-lite.

Requests go through Google's OpenAI-compatible surface, so the body comes from
the same `build_chat_body` the OpenAI and DeepSeek panel runs used. Two fields
are stripped because Gemini rejects them: `thinking`, which is DeepSeek's, and
`reasoning_effort`, which is OpenAI's. Everything else, including the prompt,
the JSON response format and the 256-token output cap, is identical.

Cost is metered against a hard cap before every request, responses are
checkpointed individually so a restart resumes rather than repeats, and only
429 and 5xx are retried. Any other failure stops the run rather than risk
repeating a billed request.

Usage:
  python3 code/refresh_gemini.py plan
  python3 code/refresh_gemini.py run --model gemini-3.8-flash
  python3 code/refresh_gemini.py collect
  python3 code/refresh_gemini.py manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark                                     # noqa: E402
from openai_batch_submit import build_chat_body      # noqa: E402
from task_registry import load_task_definitions      # noqa: E402

SOURCE = Path("output/sidecar/refresh_20260910")
ROOT = Path("output/sidecar/refresh_gemini_20260920")
PANEL = SOURCE / "panel.json"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# name -> (request id, USD per 1M input, USD per 1M output)
# Standard paid-tier prices read from ai.google.dev/gemini-api/docs/pricing on
# 2026-09-20. Both Flash tiers are promotional through 2026-12-31; 3.8-flash
# doubles to 1.50/7.50 on 2027-01-01. The Batch API is a flat 50% discount, so
# the batch figure is these halved, not a separately measured rate.
MODELS = {
    "gemini-3.8-flash": ("models/gemini-3.8-flash", 0.75, 3.75),
    "gemini-3.1-flash-lite": ("models/gemini-3.1-flash-lite", 0.25, 1.50),
}
BATCH_DISCOUNT = 0.5
# Budget for the live artifacts only. A first gemini-3.8-flash run was billed
# $3.0866 and discarded (see _discarded/README.md); that money is spent and is
# deliberately not counted here, because this cap exists to bound what the
# runner may still spend. User approved ~$1.70 more on 2026-09-20 to redo that
# model with thinking disabled; flash-lite's valid run cost $0.4742.
APPROVED_TOTAL_USD = 2.30
WORKERS = 8
MAX_ATTEMPTS = 4


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_key() -> str:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    for name in ("~/.gemini_api_key", "~/.google_api_key"):
        path = Path(name).expanduser()
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
    raise RuntimeError("no Google credential found")


def panel_rows() -> list[dict]:
    panel = json.loads(PANEL.read_text())
    rows = panel["rows"]
    if not rows:
        raise RuntimeError("frozen panel is empty")
    return rows


def task_map() -> dict[str, dict]:
    return {t["name"]: t for t in load_task_definitions()}


def request_body(model: str, task: dict, user_content: str) -> dict:
    request_id, _, _ = MODELS[model]
    config = {"name": request_id, "max_output_tokens": 256, "provider": "google",
              "reasoning_effort": None, "thinking_mode": "disabled",
              "response_format_type": "json_object"}
    prompt = Path(task["prompt_path"]).read_text()
    body = build_chat_body(config, task, prompt, user_content)
    # `thinking` is DeepSeek's field and Gemini rejects it with HTTP 400.
    body.pop("thinking", None)
    # Gemini 3.x Flash thinks by default, and those tokens are billed and count
    # against max_tokens while being invisible in completion_tokens. Left on,
    # they ate the 256-token budget and truncated 10.4% of responses mid-JSON.
    # "none" is also the configuration the rest of the panel ran under, so this
    # makes Gemini comparable rather than favoured.
    body["reasoning_effort"] = "none"
    body["model"] = request_id
    return body


def response_path(model: str, task: str, item_id: str) -> Path:
    safe = str(item_id).replace("/", "_")
    return ROOT / model / "responses" / task / f"{safe}.json"


def spent(model: str | None = None) -> float:
    total = 0.0
    models = [model] if model else list(MODELS)
    for name in models:
        for path in (ROOT / name / "responses").glob("*/*.json"):
            total += json.loads(path.read_text()).get("cost_usd", 0.0)
    return total


def billable_output_tokens(usage: dict) -> int:
    """Output tokens actually charged, including hidden thinking tokens.

    Gemini reports thinking tokens in neither `completion_tokens` nor
    `completion_tokens_details.reasoning_tokens`, but they do appear in
    `total_tokens` and they are billed. Pricing off `completion_tokens` alone
    understated one run by a factor of two.
    """
    completion = usage.get("completion_tokens", 0)
    implied = usage.get("total_tokens", 0) - usage.get("prompt_tokens", 0)
    return max(completion, implied)


def price(model: str, usage: dict) -> float:
    _, inp, out = MODELS[model]
    return (usage.get("prompt_tokens", 0) * inp
            + billable_output_tokens(usage) * out) / 1e6


def post(body: dict, key: str) -> dict:
    """One request. Retries only 429 and 5xx; anything else raises."""
    payload = json.dumps(body).encode()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            ENDPOINT, data=payload, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200]
            if exc.code == 429 or exc.code >= 500:
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"http {exc.code} after {attempt} attempts: {detail!r}")
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"http {exc.code}: {detail!r}")
    raise RuntimeError("unreachable")


def plan() -> None:
    rows = panel_rows()
    tasks = task_map()
    lengths = [len(r["item"]["user_content"]) for r in rows]
    approx_in = sum(l / 4 + 400 for l in lengths) / len(lengths)
    print(f"panel: {len(rows)} rows, {len({r['task'] for r in rows})} tasks")
    print(f"missing task definitions: "
          f"{sorted({r['task'] for r in rows} - set(tasks)) or 'none'}")
    print(f"\nestimate at ~{approx_in:.0f} input and 28 output tokens per item:")
    total = 0.0
    for model, (_, inp, out) in MODELS.items():
        est = (approx_in * inp + 28 * out) / 1e6 * len(rows)
        total += est
        done = spent(model)
        print(f"  {model:24s} ${est:5.2f} sequential   ${est * BATCH_DISCOUNT:5.2f} at batch rate"
              f"   already spent ${done:.4f}")
    print(f"  {'TOTAL':24s} ${total:5.2f}")
    print(f"\ncap ${APPROVED_TOTAL_USD:.2f}; spent so far ${spent():.4f}")


def run(model: str) -> None:
    key = load_key()
    rows = panel_rows()
    tasks = task_map()
    already = spent()
    print(f"{now()}  {model}: {len(rows)} rows, ${already:.4f} spent, "
          f"cap ${APPROVED_TOTAL_USD:.2f}")

    pending = [r for r in rows
               if not response_path(model, r["task"], r["item"]["item_id"]).exists()]
    print(f"  {len(rows) - len(pending)} already checkpointed, {len(pending)} to send")
    if not pending:
        return

    lock = threading.Lock()
    state = {"spent": already, "done": 0, "stop": None}

    def one(row: dict) -> None:
        if state["stop"]:
            return
        with lock:
            if state["spent"] >= APPROVED_TOTAL_USD:
                state["stop"] = "cost cap reached"
                return
        task = tasks[row["task"]]
        try:
            result = post(request_body(model, task, row["item"]["user_content"]), key)
        except Exception as exc:                      # noqa: BLE001 - stop the run
            with lock:
                state["stop"] = f"{row['task']}/{row['item']['item_id']}: {exc}"
            return
        usage = result.get("usage", {}) or {}
        cost = price(model, usage)
        path = response_path(model, row["task"], row["item"]["item_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "task": row["task"],
                                    "item_id": row["item"]["item_id"],
                                    "cost_usd": cost, "captured": now(),
                                    "response": result}))
        with lock:
            state["spent"] += cost
            state["done"] += 1
            if state["done"] % 250 == 0:
                print(f"  {state['done']:5d}/{len(pending)}  ${state['spent']:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, pending))

    print(f"{now()}  {model}: {state['done']} sent, ${state['spent']:.4f} total")
    if state["stop"]:
        raise SystemExit(f"stopped: {state['stop']}")


def decode(model: str, task: dict, saved: dict) -> dict:
    """Mirror of refresh_pilot.decode for the OpenAI-compatible branch."""
    result = saved["response"]
    choice = (result.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = result.get("usage", {}) or {}
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
    row = {"task": task["name"], "model": model, "item_id": saved["item_id"],
           "parse_error": error,
           "truncated": choice.get("finish_reason") in ("length", "max_tokens"),
           "stop_reason": choice.get("finish_reason"),
           "returned_model": result.get("model"),
           "input_tokens": usage.get("prompt_tokens", 0),
           "output_tokens": billable_output_tokens(usage),
           "visible_output_tokens": usage.get("completion_tokens", 0),
           "cost_usd_upper": saved["cost_usd"],
           "raw_content": text, "usage_present": bool(usage)}
    return row


def collect() -> None:
    rows = {(r["task"], str(r["item"]["item_id"])): r for r in panel_rows()}
    tasks = task_map()
    out = []
    for model in MODELS:
        for path in sorted((ROOT / model / "responses").glob("*/*.json")):
            saved = json.loads(path.read_text())
            key = (saved["task"], str(saved["item_id"]))
            record = decode(model, tasks[saved["task"]], saved)
            record.update({"gt_" + k: v for k, v in rows[key]["item"]["gt"].items()})
            predictions, _ = benchmark.parse_content(record["raw_content"],
                                                     tasks[saved["task"]])
            record.update({"pred_" + k: v for k, v in predictions.items()})
            out.append(record)
    frame = pd.DataFrame(out)
    ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "predictions.csv", index=False)
    summary = frame.groupby("model").agg(
        n=("item_id", "size"),
        tasks=("task", "nunique"),
        malformed=("parse_error", lambda s: s.notna().sum()),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd_upper", "sum"))
    summary["cost_usd_batch_rate"] = summary.cost_usd * BATCH_DISCOUNT
    summary["malformed_rate"] = summary.malformed / summary.n
    summary.to_csv(ROOT / "run_summary.csv")
    print(summary.to_string())
    print(f"\nwrote {ROOT}/predictions.csv and run_summary.csv")


def manifest() -> None:
    """Write the audited provenance manifest build_refresh_release.py reads.

    The release builder takes one prediction file per API addition, so each
    model's rows are split out of the combined predictions.csv. Coverage is
    checked against the frozen panel before anything is written.
    """
    panel = json.loads(PANEL.read_text())
    panel_sha = hashlib.sha256(json.dumps(panel, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = json.loads((SOURCE / "comparison_report.json").read_text())
    if report["panel_sha256"] != panel_sha:
        raise ValueError("Frozen panel hash does not match the comparison report")
    expected = {(r["task"], str(r["item"]["item_id"])) for r in panel["rows"]}
    frame = pd.read_csv(ROOT / "predictions.csv", dtype={"item_id": str}, low_memory=False)
    entries = []
    for model, (request_id, _, _) in MODELS.items():
        rows = frame[frame.model == model].sort_values(["task", "item_id"])
        if (len(rows) != len(expected) or rows.duplicated(["task", "item_id"]).any()
                or set(zip(rows.task, rows.item_id)) != expected):
            raise ValueError(f"{model} predictions do not exactly cover the frozen panel")
        if not rows.usage_present.all():
            raise ValueError(f"{model} has responses without usage")
        out = ROOT / model / "predictions.csv"
        rows.to_csv(out, index=False)
        saved = sorted((ROOT / model / "responses").glob("*/*.json"))
        dates = sorted({json.loads(p.read_text()).get("captured", "")[:10] for p in saved} - {""})
        entries.append({
            "model": model,
            "model_id": request_id,
            "provider": "google",
            "predictions": str(out),
            "panel_sha256": panel_sha,
            "prediction_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "revision": model,
            "hardware_tier": "api",
            "settings": {"model": request_id, "max_tokens": 256, "reasoning_effort": "none",
                         "structured_output": "json_object",
                         "native_interface": "Gemini OpenAI-compatible chat completions"},
            "run_dates": dates,
            "input_tokens": int(rows.input_tokens.sum()),
            "output_tokens": int(rows.output_tokens.sum()),
            "cost_usd_upper": float(rows.cost_usd_upper.sum()),
            "malformed": int(rows.parse_error.notna().sum()),
            "truncated": int(rows.truncated.sum()),
            "returned_models": sorted(set(rows.returned_model.dropna())),
            "documented_versions": [model],
            "cost_note": "Token counts times the published standard rate, not a provider invoice.",
            "provenance_validated": True,
        })
    (ROOT / "api_manifest.json").write_text(json.dumps({"models": entries}, indent=2) + "\n")
    print(json.dumps(entries, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "run", "collect", "manifest"])
    ap.add_argument("--model", choices=list(MODELS))
    args = ap.parse_args()
    if args.command == "plan":
        plan()
    elif args.command == "run":
        if not args.model:
            raise SystemExit("--model is required for run")
        run(args.model)
    elif args.command == "manifest":
        manifest()
    else:
        collect()


if __name__ == "__main__":
    main()
