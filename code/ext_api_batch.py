#!/usr/bin/env python3
"""API models on the 2026-11-14 task pool through the OpenRouter Batch API.

This reuses code/refresh_openrouter_batch.py unchanged (its cap check, state
files, worst-case pilot reservation, batch submit, polling and decoding) and
swaps in three things: the output root, the request list, and the request body.

- Requests come from output/sidecar/task_ext_20261114_api/draw.json (built by
  experiments/task_ext_20261114/build_api_draw.py): 100 items per task, 50 for
  the six longest tasks, a stratified subset of the items the open models ran.
- Prompts and schemas come from tasks_ext_v2_20261114.
- Prompt caching: the fixed task instructions are the system message and come
  before the item text, so providers with automatic prefix caching can reuse
  them. Anthropic caches only marked content, so for Anthropic models the
  system message carries a one-hour cache marker.

The pilot (40 requests: 20 items from each of two tasks with long instructions)
exists to measure whether batch requests are billed at the cached-input price.
`report` prints cached input tokens and billed cost per model.

Usage (from the repository root):
  python3 code/ext_api_batch.py budget --stage pilot
  python3 code/ext_api_batch.py submit --stage pilot --cost-cap-usd 4
  python3 code/ext_api_batch.py watch  --stage pilot
  python3 code/ext_api_batch.py cache-report --stage pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import refresh_openrouter_batch as rob  # noqa: E402
from run_registry import DEFAULT_STATUS_MD, append_event, render_markdown  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

ROOT = REPO / "output" / "sidecar" / "task_ext_20261114_api"
DRAW = ROOT / "draw.json"
TASK_DIR = REPO / "tasks_ext_v2_20261114"
RUN_ID = "task-ext-20261114-api"

# Same models and settings as the November OpenRouter batch run.
BATCH_MODELS = ["claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5", "gpt-6.1-sol",
                "gpt-6-luna", "gemini-4-argon", "glm-5.3"]
MODELS = {m: dict(rob.MODELS[m]) for m in BATCH_MODELS}
# Added 2026-11-17 (Hanno approved about $17): models on the November page that
# had not run on the new task pool. Settings follow their November runs (no
# reasoning, 256 output tokens); those ran through the Anthropic and Google APIs
# directly, these go through OpenRouter like the other API models here.
# Prices are OpenRouter's batch listings on 2026-11-17.
OFF = rob.OFF
MODELS.update({
    "claude-opus-5": dict(id="anthropic/claude-opus-5", mode="batch", reasoning=OFF,
                          max_tokens=256, usd_in=2.5, usd_out=12.5),
    "claude-sonnet-5": dict(id="anthropic/claude-sonnet-5", mode="batch", reasoning=OFF,
                            max_tokens=256, usd_in=1.0, usd_out=5.0),
    # OpenRouter refuses reasoning off for Gemini 3.8 Flash in batch and one at a
    # time ("Reasoning is mandatory"; refused pilots in gemini-3.8-flash/pilot_*_refused,
    # no charge). It runs at the lowest effort, "minimal". November's run, through
    # Google with reasoning "none", averaged 42 output tokens against 10 for
    # Flash-Lite, so Google probably applied minimal reasoning then too. The output
    # limit is 1024 because reasoning tokens count against it.
    "gemini-3.8-flash": dict(id="google/gemini-3.8-flash", mode="batch", reasoning={"effort": "minimal"},
                             max_tokens=1024, temperature=0, usd_in=0.375, usd_out=1.875,
                             split_by_task=True),
    "gemini-3.1-flash-lite": dict(id="google/gemini-3.1-flash-lite", mode="batch", reasoning=OFF,
                                  max_tokens=256, temperature=0, usd_in=0.125, usd_out=0.75,
                                  split_by_task=True),
    "deepseek-v4.1-flash-1017": dict(rob.MODELS["deepseek-v4.1-flash-1017"]),
    "kimi-k3.1-pro": dict(rob.MODELS["kimi-k3.1-pro"]),
})
for m in MODELS:
    if MODELS[m]["id"].startswith("anthropic/"):
        MODELS[m]["cache_marker"] = True
# OpenRouter lowered GLM-5.3's batch price before its remainder batch was sent
# (listed 2026-11-16: $0.281 in, $1.25 out per million tokens; the pilot ran at
# $0.45/$2.00). The price check refuses a submit at a stale price.
MODELS["glm-5.3"].update(usd_in=0.281, usd_out=1.25)
# OpenRouter holds a worst-case estimate (max_tokens on every request) when it
# accepts a batch, so 2048 needed ~$239 of balance for the refused remainder
# batches. In November no answer from these models exceeded 522 tokens (Opus; the
# others at most 160), so the remainder uses 1024 for the four models whose
# batches were refused on 2026-11-15. Their pilots ran at 2048. The other models
# keep their November limits in both stages (Haiku and Luna 256, Gemini 4 Argon
# 2048, whose first 30 remainder batches were already accepted at 2048). Answers
# that hit the limit are flagged.
REMAINDER_MAX_TOKENS = {"claude-opus-5.5": 1024, "claude-sonnet-5.5": 1024,
                        "gpt-6.1-sol": 1024, "glm-5.3": 1024}


def spec_for(model, stage):
    spec = dict(MODELS[model])
    if stage == "remainder" and model in REMAINDER_MAX_TOKENS:
        spec["max_tokens"] = REMAINDER_MAX_TOKENS[model]
    return spec


# OpenRouter sends a batch to the cheapest batch provider unless told otherwise.
# GLM-5.3's pilot ran on DeepInfra (fp4, low reasoning); by 2026-11-16 Morph was
# cheaper, and Morph returned no reasoning tokens. Its remainder therefore runs
# pinned to DeepInfra (the Morph run is kept in glm-5.3/remainder_morph).
PROVIDER_ONLY = {"glm-5.3": ["deepinfra/fp4"]}


def batch_payload(spec, requests):
    payload = _rob_batch_payload(spec, requests)
    only = next((v for m, v in PROVIDER_ONLY.items() if MODELS[m]["id"] == spec["id"]), None)
    if only:
        requests_ = payload.pop("requests")  # `requests` must stay last
        payload["provider"] = {"only": only}
        payload["requests"] = requests_
    return payload


_rob_batch_payload = rob.batch_payload


def build_body(spec, task, item):
    body = rob.build_body(spec, task, item)
    only = next((v for m, v in PROVIDER_ONLY.items() if MODELS[m]["id"] == spec["id"]), None)
    if only:
        body["provider"] = {"only": only, "allow_fallbacks": False}
    if spec.get("cache_marker"):
        text = body["messages"][0]["content"]
        body["messages"][0]["content"] = [
            {"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    return body


MODERATION_EXCLUDED = ROOT / "moderation_excluded.json"


def moderation_excluded(model):
    """custom_ids OpenRouter's preflight moderation rejects for this provider.

    One flagged request makes OpenRouter fail the whole batch (no charge), so
    flagged items are left out of that provider's batches and listed here."""
    if not MODERATION_EXCLUDED.exists():
        return set()
    provider = MODELS[model]["id"].split("/")[0]
    return {cid for cid, row in json.loads(MODERATION_EXCLUDED.read_text()).items()
            if row["provider"] == provider}


def load_requests(model, stage):
    skip = moderation_excluded(model)
    draw = json.loads(DRAW.read_text())
    tasks = {t["name"]: t for t in load_task_definitions(tasks_dir=TASK_DIR)}
    out = []
    for i, row in enumerate(draw["rows"]):
        if row["pilot"] == (stage == "pilot") and f"x{i:05d}" not in skip:
            item = row["item"]
            if item["gt"] is None:
                raise ValueError(f"{row['task']}: missing gold")
            out.append({"custom_id": f"x{i:05d}", "task": row["task"], "item": item,
                        "body": build_body(spec_for(model, stage), tasks[row["task"]], item)})
    return out, tasks


def register(note, status="running", cap=None):
    append_event(run_id=RUN_ID, event="update", status=status, host="local-mac",
                 runner="code/ext_api_batch.py", model_scope=",".join(MODELS),
                 task_scope="47_tasks:task_ext_20261114_api_draw:40_pilot_then_4360_remainder",
                 batch_sizes="one_batch_per_model_and_stage (Google: per task)",
                 output=str(ROOT.relative_to(REPO)), log=str((ROOT / "watch.log").relative_to(REPO)),
                 cost_cap_usd=cap, note=note)
    render_markdown(status_path=DEFAULT_STATUS_MD)


def jobstatus(**kw):
    import subprocess
    args = [f"--{k}={v}" for k, v in kw.items()]
    subprocess.run(["jobstatus", "set", "polsci-ext-api", "--project", "polsci-open-bench",
                    "--task", "Run API models on the new benchmark tasks through OpenRouter",
                    "--unit", "models", "--host", "this-mac", *args], check=False)


def cache_report(stage):
    """Cached input tokens and billed cost against the uncached batch price."""
    rows = []
    for model, spec in MODELS.items():
        path = rob.state_path(model, stage).parent / "results.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text()).get("results") or []
        n = prompt = cached = written = out = 0
        cost = 0.0
        for e in results:
            body = ((e.get("response") or {}).get("body")) or {}
            usage = body.get("usage") or {}
            if not usage:
                continue
            n += 1
            prompt += usage.get("prompt_tokens") or 0
            out += usage.get("completion_tokens") or 0
            details = usage.get("prompt_tokens_details") or {}
            cached += details.get("cached_tokens") or 0
            written += details.get("cache_write_tokens") or 0
            cost += usage.get("cost") or 0.0
        state = rob.read_state(model, stage) or {}
        uncached = (prompt * spec["usd_in"] + out * spec["usd_out"]) / 1e6
        rows.append(dict(model=model, requests=n, prompt_tokens=prompt, cached_tokens=cached,
                         cache_written_tokens=written, cached_share=round(cached / prompt, 3) if prompt else None,
                         billed_usd_per_request_sum=round(cost, 4),
                         billed_usd_batch_total=round(state.get("cost_usd") or 0.0, 4),
                         uncached_price_usd=round(uncached, 4),
                         billed_over_uncached=round((state.get("cost_usd") or cost) / uncached, 3) if uncached else None))
    print(json.dumps(rows, indent=1))
    (ROOT / f"{stage}_cache_report.json").write_text(json.dumps(rows, indent=1) + "\n")


_rob_decode = rob.decode


def decode(model, request, entry, task):
    """rob.decode's extra schema check knows only binary, categorical and
    multi-binary tasks, so it marks every answer to the score task
    (lemens_gallego_tweet_position) invalid. For that task keep the error from
    benchmark.parse_content, as the open-model runner does."""
    row = _rob_decode(model, request, entry, task)
    if task["label_kind"] == "score" and str(row.get("parse_error") or "").startswith("schema_invalid: Binary"):
        _, error = rob.benchmark.parse_content(row.get("raw_content") or "", task)
        row["parse_error"] = error
    return row


def install():
    rob.decode = decode
    rob.batch_payload = batch_payload
    rob.ROOT = ROOT
    rob.MODELS = MODELS
    rob.RUN_ID = RUN_ID
    rob.load_requests = load_requests
    rob.register = register
    rob.jobstatus = jobstatus


if __name__ == "__main__":
    install()
    if len(sys.argv) > 1 and sys.argv[1] == "cache-report":
        stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "pilot"
        cache_report(stage)
    else:
        rob.main()
