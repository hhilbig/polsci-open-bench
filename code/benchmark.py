#!/usr/bin/env python3
"""
polsci-open-bench: classification benchmark for local and commercial LLMs.

Covers the built-in political-science classification task manifests across
the built-in model manifests, with N=500 items per task.

Default tasks are loaded from YAML manifests in `tasks/`. You can also supply a
single custom task manifest or task directory at the CLI.

Default models are loaded from YAML manifests in `models/`. You can also supply
your own model manifest or model-manifest directory at the CLI.

Basic usage:
  # Run full benchmark (all tasks x all models)
  python3 code/benchmark.py

  # Selective rerun of one (task, model) cell, merge into existing predictions
  python3 code/benchmark.py \\
    --only-model qwen3:30b-a3b-q4_K_M \\
    --only-task halterman_ccc_protest \\
    --merge-into output/predictions.csv

  # Run only one task across all models
  python3 code/benchmark.py --only-task gilardi_stance

  # Run a self-contained custom task directory with an explicit local model
  python3 code/benchmark.py \\
    --task-dir examples/minimal_custom_task \\
    --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \\
    --output output/custom_predictions.csv

Parse strategy:
  Primary  - JSON via raw_decode (trailing content ignored).
  Fallback - bare 0/1 for binary tasks, or case-insensitive bare-label match
             against categorical task enums.
  Both API and local models are supported. API models use structured outputs
  (JSON schema enforced server-side).

Set OPENAI_API_KEY in the environment for OpenAI calls. For Anthropic, either
set ANTHROPIC_API_KEY or write the key to ~/.anthropic_api_key. For other
OpenAI-compatible API providers such as DeepSeek, set the provider-specific key
named in the model manifest. For local models, ensure Ollama is running at
http://localhost:11434 (or override OLLAMA_URL).
"""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd
from openai import OpenAI
from model_registry import (add_model_loading_args, load_model_definitions,
                            load_model_definitions_from_args)
from run_registry import DEFAULT_LEDGER, DEFAULT_STATUS_MD, append_event, render_markdown
from task_registry import (DEFAULT_SAMPLE_N_V1 as N_V1, add_task_loading_args,
                           load_task_definitions,
                           load_task_definitions_from_args)

try:
    import anthropic
except ImportError:
    anthropic = None


# --------- Config ---------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# (model_name, task_name) pairs to skip. Reserved for empirically-discovered
# unusable cells (long system prompts that an Ollama backend re-prefills on
# every call). Dropped cells are reported as absent (not as parse errors) so
# the report can flag them explicitly.
SKIP_COMBOS = set()
TASKS = load_task_definitions()
MODELS = load_model_definitions()


# --------- Parsing (JSON + bare-label fallback) ---------

def extract_json(content: str) -> str:
    c = content.strip()
    if "```json" in c:
        c = c.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in c:
        c = c.split("```", 1)[1].split("```", 1)[0].strip()
    return c


def _coerce_binary_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().strip(" \"'.,!").lower()
        if normalized in {"0", "false", "no"}:
            return 0
        if normalized in {"1", "true", "yes"}:
            return 1
    raise ValueError(f"not a binary label: {value!r}")


def _normalize_json_prediction(obj):
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return obj


def parse_content(content: str, task: dict):
    """Parse model output. JSON first, then compact task-specific fallbacks."""
    kind = task["label_kind"]
    # JSON attempt
    try:
        obj, _ = json.JSONDecoder().raw_decode(extract_json(content))
        obj = _normalize_json_prediction(obj)
        if kind == "multi_binary":
            return {k: _coerce_binary_label(obj.get(k, 0)) for k in task["labels"]}, None
        elif kind == "binary":
            k = task["label_key"]
            if isinstance(obj, dict):
                return {k: _coerce_binary_label(obj.get(k, 0))}, None
            return {k: _coerce_binary_label(obj)}, None
        else:
            k = task["label_key"]
            v = obj.get(k, "")
            if v in task["labels"]:
                return {k: v}, None
            # fall through
    except Exception:
        pass

    if kind == "binary":
        k = task["label_key"]
        try:
            return {k: _coerce_binary_label(extract_json(content))}, None
        except ValueError:
            return {k: None}, f"parse_fail: {content[:80]!r}"

    # Bare-label fallback (categorical only; multi-binary has no single-label equivalent)
    if kind == "categorical":
        k = task["label_key"]
        upper = extract_json(content).strip().upper()
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

def classify_ollama(model_def, system_prompt, user_content):
    ollama_url = model_def.get("ollama_url") or OLLAMA_URL
    payload = {
        "model": model_def["name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": model_def.get("think", False),
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=600) as c:
        r = c.post(f"{ollama_url}/api/chat", json=payload)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    return {
        "content": d.get("message", {}).get("content", "").strip(),
        "latency_s": latency,
        "eval_count": d.get("eval_count"),
    }


def classify_openai(client, model_def, task, system_prompt, user_content):
    model = model_def["name"]
    schema_mode = response_format_type(model_def)
    if schema_mode == "json_object":
        system_prompt = augment_system_prompt_for_json_object(system_prompt, task)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if schema_mode == "json_schema":
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": task["json_schema"],
            },
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["max_tokens"] = 1024
        kwargs["temperature"] = 0.1
    if model_def.get("reasoning_effort"):
        kwargs["reasoning_effort"] = model_def["reasoning_effort"]
    extra_body = extra_body_for_openai_model(model_def)
    if extra_body:
        kwargs["extra_body"] = extra_body
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0
    return {
        "content": resp.choices[0].message.content or "",
        "latency_s": latency,
        "eval_count": resp.usage.completion_tokens if resp.usage else None,
    }


def _load_api_key(model_def, default_env=None, default_file=None):
    env_name = model_def.get("api_key_env") or default_env
    if env_name:
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
    key_file = model_def.get("api_key_file") or default_file
    if key_file:
        key_path = Path(key_file).expanduser()
        if key_path.exists():
            return key_path.read_text().strip()
    return ""


def _is_local_base_url(raw_url):
    if not raw_url:
        return False
    host = urlparse(raw_url).hostname
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def is_deepseek_model(model_def):
    if model_def.get("provider") == "deepseek":
        return True
    raw_url = model_def.get("base_url")
    if not raw_url:
        return False
    return urlparse(raw_url).hostname == "api.deepseek.com"


def response_format_type(model_def):
    if model_def.get("response_format_type"):
        return model_def["response_format_type"]
    if is_deepseek_model(model_def):
        return "json_object"
    return "json_schema"


def extra_body_for_openai_model(model_def):
    thinking_mode = model_def.get("thinking_mode")
    if thinking_mode:
        return {"thinking": {"type": thinking_mode}}
    if is_deepseek_model(model_def):
        return {"thinking": {"type": "disabled"}}
    return None


def example_payload_for_task(task):
    kind = task["label_kind"]
    if kind == "binary":
        return {task["label_key"]: 0}
    if kind == "categorical":
        return {task["label_key"]: task["labels"][0]}
    return {label: 0 for label in task["labels"]}


def augment_system_prompt_for_json_object(system_prompt, task):
    example = json.dumps(example_payload_for_task(task), ensure_ascii=False)
    reminder = (
        "\n\nOutput format reminder: return JSON only. "
        f"Use exactly this JSON object shape:\n{example}\n"
        "Do not include prose, markdown fences, or extra fields."
    )
    if reminder.strip() in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + reminder


def make_openai_client(model_def):
    kwargs = {}
    api_key = _load_api_key(model_def, default_env="OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    elif _is_local_base_url(model_def.get("base_url")):
        # OpenAI-compatible local servers often ignore the key but the SDK expects one.
        kwargs["api_key"] = "DUMMY"
    else:
        return None
    if model_def.get("base_url"):
        kwargs["base_url"] = model_def["base_url"]
    return OpenAI(**kwargs)


def make_anthropic_client(model_def):
    """Return an Anthropic client, or None if SDK or key unavailable.

    300s timeout per request: long enough for any single classification call,
    short enough that a hung connection bubbles up as an exception (caught by
    run_task's try/except) instead of stalling the whole grid.
    """
    if anthropic is None:
        return None
    key = _load_api_key(
        model_def,
        default_env="ANTHROPIC_API_KEY",
        default_file=Path.home() / ".anthropic_api_key",
    )
    if not key:
        return None
    return anthropic.Anthropic(api_key=key, timeout=300.0, max_retries=2)


def classify_anthropic(client, model, system_prompt, user_content, json_schema):
    """One-shot classification via Anthropic Messages API with tool-use forced
    to return JSON matching the task schema. Mirrors classify_openai's interface."""
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        tools=[{
            "name": "classify",
            "description": "Return the classification for the input.",
            "input_schema": json_schema,
        }],
        tool_choice={"type": "tool", "name": "classify"},
    )
    latency = time.perf_counter() - t0
    tool_use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
    content = json.dumps(tool_use.input) if tool_use is not None else ""
    eval_count = resp.usage.output_tokens if resp.usage else None
    return {"content": content, "latency_s": latency, "eval_count": eval_count}


def warmup_ollama(model_def):
    ollama_url = model_def.get("ollama_url") or OLLAMA_URL
    try:
        with httpx.Client(timeout=600) as c:
            c.post(f"{ollama_url}/api/chat", json={
                "model": model_def["name"],
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "think": False,
                "options": {"num_predict": 1},
            })
    except Exception as e:
        print(f"  warmup failed: {e}", flush=True)


# --------- Orchestration ---------

def run_task(task, models, model_clients, checkpoint_path, only_new_items=False):
    system_prompt = Path(task["prompt_path"]).read_text()
    items = task["loader"]()
    if only_new_items:
        # Items are returned as baseline items followed by added items.
        skipped = items[:N_V1]
        items = items[N_V1:]
        print(f"\n[only-new-items] skipping first {len(skipped)} baseline items; running {len(items)} added items.", flush=True)
    print(f"\n########## TASK: {task['name']} ({len(items)} items) ##########", flush=True)
    rows = []
    for m in models:
        if (m["name"], task["name"]) in SKIP_COMBOS:
            print(f"\n--- SKIPPING {m['name']} on {task['name']} (SKIP_COMBOS) ---", flush=True)
            continue
        if m["backend"] == "ollama":
            print(f"\n--- Warming up {m['name']} ---", flush=True)
            warmup_ollama(m)
        print(f"\n--- {m['name']} on {task['name']} ---", flush=True)
        for i, it in enumerate(items):
            try:
                if m["backend"] == "ollama":
                    r = classify_ollama(m, system_prompt, it["user_content"])
                elif m["backend"] == "anthropic":
                    client = model_clients.get(m["name"])
                    if client is None:
                        raise RuntimeError("Anthropic backend selected but no client (SDK or key missing).")
                    r = classify_anthropic(client, m["name"], system_prompt,
                                           it["user_content"], task["json_schema"])
                else:
                    client = model_clients[m["name"]]
                    r = classify_openai(client, m, task, system_prompt, it["user_content"])
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


def has_usable_rows(rows):
    return any(not row.get("parse_error") for row in rows)


def require_usable_rows(rows, output_path, context="run"):
    if not rows:
        raise RuntimeError(f"No prediction rows were written for {context}.")
    if not has_usable_rows(rows):
        raise RuntimeError(
            f"No usable predictions were written for {context}: all {len(rows)} rows "
            f"in {output_path} have parse_error set. Check model availability, API keys, "
            "or the model manifest before treating this run as successful."
        )


def _is_public_predictions_path(path):
    if not path:
        return False
    try:
        return Path(path).expanduser().resolve() == (OUT / "predictions.csv").resolve()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_loading_args(ap)
    add_model_loading_args(ap)
    ap.add_argument("--only-model", help="Restrict to one model by name")
    ap.add_argument("--only-task",  help="Restrict to one task by name")
    ap.add_argument("--output",     default=str(OUT / "predictions.csv"),
                    help="Where to write this run's predictions (default output/predictions.csv).")
    ap.add_argument("--merge-into", dest="merge_into", default=None,
                    help="After running, merge this run's rows into an existing CSV "
                         "(replacing matching task/model/item_id rows).")
    ap.add_argument("--only-new-items", action="store_true",
                    help="Only run items after the first N_V1 baseline items. Used for "
                         "incremental extension runs where existing baseline predictions "
                         "are kept and only added items get model calls.")
    ap.add_argument("--run-id", default=None,
                    help="Optional run id for output/run_registry.jsonl bookkeeping.")
    ap.add_argument("--run-ledger", default=str(DEFAULT_LEDGER),
                    help="Run ledger path used when --run-id is supplied.")
    ap.add_argument("--run-note", default=None,
                    help="Human note stored in the run ledger.")
    ap.add_argument("--run-log", default=None,
                    help="Log path stored in the run ledger, useful for tmux/remote runs.")
    ap.add_argument("--run-tmux-session", default=None,
                    help="tmux session name stored in the run ledger.")
    ap.add_argument("--run-cost-cap-usd", default=None,
                    help="Cost ceiling stored in the run ledger for paid API runs; metadata only, not enforced.")
    ap.add_argument("--render-run-status", action="store_true",
                    help=f"Render {DEFAULT_STATUS_MD.relative_to(REPO)} after ledger updates.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    tasks = load_task_definitions_from_args(args)
    if args.only_task:
        tasks = [t for t in tasks if t["name"] == args.only_task]
        if not tasks:
            print(f"Unknown task: {args.only_task}"); return
        print(f"[selective] Running ONLY task: {args.only_task}", flush=True)

    models = load_model_definitions_from_args(args)
    if args.only_model:
        models = [m for m in models if m["name"] == args.only_model]
        if not models:
            print(f"Unknown model: {args.only_model}"); return
        print(f"[selective] Running ONLY model: {args.only_model}", flush=True)

    model_clients = {}
    unavailable = {}
    for model in models:
        if model["backend"] == "openai":
            client = make_openai_client(model)
            if client is None:
                unavailable[model["name"]] = (
                    f"missing API key for backend=openai "
                    f"({model.get('api_key_env') or 'OPENAI_API_KEY'})"
                )
            else:
                model_clients[model["name"]] = client
        elif model["backend"] == "anthropic":
            client = make_anthropic_client(model)
            if client is None:
                unavailable[model["name"]] = (
                    f"missing API key or SDK for backend=anthropic "
                    f"({model.get('api_key_env') or 'ANTHROPIC_API_KEY'})"
                )
            else:
                model_clients[model["name"]] = client

    if unavailable:
        for model_name, reason in unavailable.items():
            print(f"[skip unavailable model] {model_name}: {reason}", flush=True)
        models = [m for m in models if m["name"] not in unavailable]
        if not models:
            if args.run_id:
                append_event(
                    args.run_ledger,
                    event="fail",
                    run_id=args.run_id,
                    status="failed",
                    runner="benchmark.py",
                    output=args.output,
                    note="No runnable models after filtering unavailable API models.",
                )
                if args.render_run_status:
                    render_markdown(args.run_ledger, DEFAULT_STATUS_MD)
            print("No runnable models after filtering unavailable API models.", flush=True)
            return

    if args.run_id:
        append_event(
            args.run_ledger,
            event="start",
            run_id=args.run_id,
            status="running",
            runner="benchmark.py",
            task_scope=[t["name"] for t in tasks],
            model_scope=[m["name"] for m in models],
            output=args.output,
            merge_into=args.merge_into,
            log=args.run_log,
            tmux_session=args.run_tmux_session,
            cost_cap_usd=args.run_cost_cap_usd,
            note=args.run_note,
        )
        if args.render_run_status:
            render_markdown(args.run_ledger, DEFAULT_STATUS_MD)

    all_rows = []
    try:
        for task_idx, task in enumerate(tasks, start=1):
            task_rows = run_task(task, models, model_clients, args.output,
                                 only_new_items=args.only_new_items)
            all_rows.extend(task_rows)
            pd.DataFrame(all_rows).to_csv(args.output, index=False)
            # Per-task merge so a mid-run kill preserves completed tasks.
            if args.merge_into:
                require_usable_rows(task_rows, args.output, context=task["name"])
                merge_into(Path(args.merge_into), task_rows)
            if args.run_id:
                append_event(
                    args.run_ledger,
                    event="update",
                    run_id=args.run_id,
                    status="running",
                    runner="benchmark.py",
                    current_task=task["name"],
                    completed_tasks=task_idx,
                    total_tasks=len(tasks),
                    rows_written=len(all_rows),
                    output=args.output,
                    merge_into=args.merge_into,
                )
                if args.render_run_status:
                    render_markdown(args.run_ledger, DEFAULT_STATUS_MD)
    except Exception as exc:
        if args.run_id:
            append_event(
                args.run_ledger,
                event="fail",
                run_id=args.run_id,
                status="failed",
                runner="benchmark.py",
                output=args.output,
                merge_into=args.merge_into,
                error=repr(exc),
            )
            if args.render_run_status:
                render_markdown(args.run_ledger, DEFAULT_STATUS_MD)
        raise

    try:
        require_usable_rows(all_rows, args.output)
    except Exception as exc:
        if args.run_id:
            append_event(
                args.run_ledger,
                event="fail",
                run_id=args.run_id,
                status="failed",
                runner="benchmark.py",
                output=args.output,
                merge_into=args.merge_into,
                error=repr(exc),
                rows_written=len(all_rows),
            )
            if args.render_run_status:
                render_markdown(args.run_ledger, DEFAULT_STATUS_MD)
        raise

    print("\n=== ALL DONE ===", flush=True)
    print(f"  predictions: {args.output}", flush=True)
    if args.merge_into:
        print(f"  merged into: {args.merge_into}", flush=True)
    if args.run_id:
        append_event(
            args.run_ledger,
            event="finish",
            run_id=args.run_id,
            status="completed",
            runner="benchmark.py",
            output=args.output,
            merge_into=args.merge_into,
            rows_written=len(all_rows),
            completed_tasks=len(tasks),
            total_tasks=len(tasks),
        )
        if args.render_run_status:
            render_markdown(args.run_ledger, DEFAULT_STATUS_MD)

    if _is_public_predictions_path(args.output) or _is_public_predictions_path(args.merge_into):
        try:
            from build_coverage_matrix import refresh as refresh_coverage
            refresh_coverage(verbose=False)
            print("  coverage matrix: docs/coverage_matrix.md", flush=True)
        except Exception as exc:
            print(f"  coverage matrix refresh skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
