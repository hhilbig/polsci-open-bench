#!/usr/bin/env python3
"""Run pinned Jev 1.13 on the frozen September 2026 benchmark panel.

The provider has no batch endpoint. Requests are checkpointed individually.
Only explicit 429/529 responses are retried; ambiguous transport failures stop
the run so a possibly billed request is not silently repeated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from api_prediction_validation import prediction_malformed_masks
from run_registry import append_event, render_markdown, DEFAULT_STATUS_MD
from task_registry import load_task_definitions


SOURCE = Path("output/sidecar/refresh_20260910")
ROOT = Path("output/sidecar/refresh_jev_20260918")
PANEL = SOURCE / "panel.json"
MODEL = "jev-1.13.0"
REQUEST_MODEL = "typesafe/jev-1.13"
PROVIDER = "openrouter_typesafe"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
INPUT_USD_PER_MILLION = 0.042
APPROVED_TOTAL_USD = 0.25
PILOT_RESERVATION_USD = 0.02
EXPECTED_ITEMS = 3400
EXPECTED_TASKS = 34
EXPECTED_PILOT = 68
ADAPTER_VERSION = "jev_choice_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def task_map() -> dict[str, dict[str, Any]]:
    tasks = load_task_definitions()
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    return {task["name"]: task for task in tasks}


def validate_frozen_panel(panel: dict[str, Any]) -> None:
    if panel.get("seed") != 20260910 or len(panel.get("rows", [])) != EXPECTED_ITEMS:
        raise ValueError("Unexpected frozen panel identity or size")
    counts: dict[str, int] = {}
    keys: set[tuple[str, str]] = set()
    pilots = 0
    for row in panel["rows"]:
        task = str(row["task"])
        item_id = str(row["item"]["item_id"])
        key = (task, item_id)
        if key in keys:
            raise ValueError(f"Duplicate frozen key: {key}")
        keys.add(key)
        counts[task] = counts.get(task, 0) + 1
        pilots += bool(row["pilot"])
    if len(counts) != EXPECTED_TASKS or set(counts.values()) != {100}:
        raise ValueError("Frozen panel must contain 100 items in each of 34 tasks")
    if pilots != EXPECTED_PILOT:
        raise ValueError(f"Expected {EXPECTED_PILOT} pilot items, found {pilots}")


def label_values(schema: dict[str, Any]) -> list[Any]:
    values = schema.get("enum")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("Jev adapter requires a finite enum with at least two values")
    rendered = [str(value) for value in values]
    if len(set(rendered)) != len(rendered):
        raise ValueError("Jev string labels are not unique")
    if len(values) > 255:
        raise ValueError("Jev Choice supports at most 255 options")
    return values


def build_payload(task: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    prompt = Path(task["prompt_path"]).read_text()
    properties = task["json_schema"].get("properties") or {}
    expected = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
    if list(properties) != list(expected):
        raise ValueError(f"Schema field order differs from scorer for {task['name']}")
    questions: dict[str, Any] = {}
    for field in expected:
        values = label_values(properties[field])
        questions[field] = {
            "type": "choice",
            "instructions": (
                f"Using benchmark_instructions, assign the value of output field "
                f"{field!r} to text_to_code. Follow every label definition and "
                "boundary rule in benchmark_instructions exactly."
            ),
            "criteria": {str(value): None for value in values},
        }
    return {
        "model": REQUEST_MODEL,
        "state": {
            "benchmark_instructions": prompt,
            "text_to_code": item["user_content"],
        },
        "questions": questions,
    }


def prepare() -> dict[str, Any]:
    panel = read(PANEL)
    validate_frozen_panel(panel)
    tasks = task_map()
    requests = []
    total_bytes = pilot_bytes = 0
    for index, row in enumerate(panel["rows"]):
        task = tasks[row["task"]]
        payload = build_payload(task, row["item"])
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        payload_bytes = len(serialized.encode())
        request = {
            "custom_id": f"jev-{index:04d}",
            "request_index": index,
            "task": row["task"],
            "item_id": str(row["item"]["item_id"]),
            "pilot": bool(row["pilot"]),
            "gold": row["item"]["gt"],
            "payload": payload,
            "payload_sha256": canonical_sha(payload),
            "payload_bytes": payload_bytes,
        }
        requests.append(request)
        total_bytes += payload_bytes
        pilot_bytes += payload_bytes if row["pilot"] else 0
    output = {
        "created_at": now(),
        "model": MODEL,
        "provider": PROVIDER,
        "endpoint": ENDPOINT,
        "adapter_version": ADAPTER_VERSION,
        "panel_sha256": canonical_sha(panel),
        "requests": requests,
    }
    request_path = ROOT / "requests.json"
    if request_path.exists():
        old = read(request_path)
        for value in (old, output):
            value.pop("created_at", None)
        if old != output:
            raise ValueError("Prepared Jev requests changed")
    else:
        save(request_path, output)
    audit = {
        "model": MODEL,
        "requests": len(requests),
        "pilot_requests": sum(row["pilot"] for row in requests),
        "payload_bytes": total_bytes,
        "pilot_payload_bytes": pilot_bytes,
        "one_byte_per_token_cost_upper": total_bytes * INPUT_USD_PER_MILLION / 1e6,
        "pilot_one_byte_per_token_cost_upper": pilot_bytes * INPUT_USD_PER_MILLION / 1e6,
        "approved_total_usd": APPROVED_TOTAL_USD,
        "pilot_reservation_usd": PILOT_RESERVATION_USD,
        "note": (
            "Byte-based values are deliberately conservative preflight bounds, not token "
            "counts. The continuation gate uses billed pilot input tokens."
        ),
    }
    save(ROOT / "preflight.json", audit)
    print(json.dumps(audit, indent=2))
    return audit


def load_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    paths = [Path(".openrouter_api_key"), Path.home() / ".openrouter_api_key"]
    for path in paths:
        if not value and path.exists():
            value = path.read_text().strip()
    if not value:
        raise RuntimeError(
            "Missing OpenRouter credential. Set OPENROUTER_API_KEY or create "
            ".openrouter_api_key with mode 600."
        )
    return value


def stage_requests(stage: str) -> list[dict[str, Any]]:
    prepared = read(ROOT / "requests.json")
    if prepared["panel_sha256"] != canonical_sha(read(PANEL)):
        raise ValueError("Prepared requests do not match the frozen panel")
    expected = stage == "pilot"
    return [row for row in prepared["requests"] if row["pilot"] is expected]


def completed_cost() -> float:
    total = 0.0
    paths = list(ROOT.glob("*/responses/*.json")) + list(
        ROOT.glob("*/failed_attempts/*.json")
    )
    for path in paths:
        record = read(path)
        if record.get("status") in {"completed", "completed_response_lost"}:
            usage = (record.get("response") or {}).get("usage") or {}
            total += int(
                usage.get("input_tokens") or usage.get("inputTokens") or 0
            ) * INPUT_USD_PER_MILLION / 1e6
    return total


def failed_attempt_records(stage: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((ROOT / stage / "failed_attempts").glob("*.json")):
        record = read(path)
        usage = (record.get("response") or {}).get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
        records.append(
            {
                "file": str(path),
                "status": record.get("status"),
                "generation_id": record.get("generation_id"),
                "provider_status": record.get("provider_status"),
                "input_tokens": input_tokens,
                "output_tokens": int(
                    usage.get("output_tokens") or usage.get("outputTokens") or 0
                ),
                "cost_usd": input_tokens * INPUT_USD_PER_MILLION / 1e6,
                "failure": record.get("failure"),
            }
        )
    return records


def pilot_projection() -> dict[str, float]:
    path = ROOT / "pilot" / "report.json"
    if not path.exists():
        raise ValueError("Pilot report is required before the remainder")
    report = read(path)
    if not report.get("passed"):
        raise ValueError("Pilot did not pass structural validation")
    by_task = report.get("input_tokens_by_task") or {}
    if set(by_task) != set(task_map()):
        raise ValueError("Pilot token report does not cover every task")
    # Longest and lower-median input per task makes the mean conservative for
    # the fixed 100-item sample unless provider tokenization behaves anomalously.
    projected_tokens = sum(sum(values) / len(values) * 100 for values in by_task.values())
    projected_cost = projected_tokens * INPUT_USD_PER_MILLION / 1e6
    return {"projected_input_tokens": projected_tokens, "projected_cost_usd": projected_cost}


def post(payload: dict[str, Any], key: str) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hhilbig.github.io/llm-benchmark/",
            "X-OpenRouter-Title": "Political Science LLM Benchmark",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(errors="replace")
            if exc.code not in {429, 529} or attempt == 2:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body[:400]}") from exc
            retry_after = exc.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(min(max(delay, 0.1), 15))
        except (TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "Ambiguous OpenRouter transport failure; request was not retried automatically"
            ) from exc
    raise AssertionError("unreachable")


def register(stage: str, status: str, note: str) -> None:
    append_event(
        run_id=f"refresh-jev-1-13-{stage}-20260918",
        event="update",
        status=status,
        host="local",
        runner="refresh_jev.py",
        model_scope=MODEL,
        task_scope=f"34_tasks:{EXPECTED_PILOT if stage == 'pilot' else EXPECTED_ITEMS-EXPECTED_PILOT}_items",
        batch_sizes="individual_checkpointed_requests",
        output=str(ROOT / stage),
        log=str(ROOT / stage / "submission.json"),
        cost_cap_usd=PILOT_RESERVATION_USD if stage == "pilot" else APPROVED_TOTAL_USD,
        note=note,
    )
    render_markdown(status_path=DEFAULT_STATUS_MD)


def run(stage: str) -> None:
    if not (ROOT / "requests.json").exists():
        prepare()
    rows = stage_requests(stage)
    if stage == "pilot":
        if read(ROOT / "preflight.json")["pilot_one_byte_per_token_cost_upper"] > PILOT_RESERVATION_USD:
            raise ValueError("Pilot byte-based reservation exceeds its approved cap")
    else:
        projection = pilot_projection()
        already = completed_cost()
        if projection["projected_cost_usd"] > APPROVED_TOTAL_USD or already >= APPROVED_TOTAL_USD:
            raise ValueError(
                f"Projected full cost ${projection['projected_cost_usd']:.6f} exceeds approved cap"
            )
    key = load_key()
    destination = ROOT / stage
    responses = destination / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    state_path = destination / "submission.json"
    state = {
        "stage": stage,
        "model": MODEL,
        "status": "running",
        "created_at": now(),
        "request_count": len(rows),
        "requests_sha256": canonical_sha(rows),
        "approved_total_usd": APPROVED_TOTAL_USD,
    }
    if state_path.exists():
        old = read(state_path)
        if old["requests_sha256"] != state["requests_sha256"]:
            raise ValueError("Saved stage request fingerprint changed")
        state["created_at"] = old["created_at"]
    save(state_path, state)
    register(stage, "running", f"Pinned {MODEL}; approved total cap ${APPROVED_TOTAL_USD:.2f}")
    try:
        for position, row in enumerate(rows, start=1):
            target = responses / f"{row['custom_id']}.json"
            if target.exists():
                saved = read(target)
                if saved.get("status") == "completed" and saved.get("request_sha256") == canonical_sha(row):
                    continue
                raise ValueError(f"Incomplete or changed checkpoint: {target}")
            if completed_cost() >= APPROVED_TOTAL_USD:
                raise ValueError("Actual billed-token estimate reached the approved cap")
            save(target, {"status": "submitting", "request_sha256": canonical_sha(row)})
            result = post(row["payload"], key)
            save(
                target,
                {
                    "status": "completed",
                    "request_sha256": canonical_sha(row),
                    "completed_at": now(),
                    "response": result,
                },
            )
            if position % 50 == 0 or position == len(rows):
                print(
                    f"{stage}: {position}/{len(rows)} checked; cost ${completed_cost():.6f}",
                    flush=True,
                )
        state.update(status="responses_complete", completed_at=now())
        save(state_path, state)
        register(stage, "running", "Responses complete; structural collection pending")
    except Exception as exc:
        state.update(status="needs_attention", error_type=type(exc).__name__, error=str(exc), stopped_at=now())
        save(state_path, state)
        register(stage, "needs_attention", f"Stopped: {type(exc).__name__}: {str(exc)[:240]}")
        raise


def decode(row: dict[str, Any], task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    answers = response.get("answers")
    usage = response.get("usage")
    returned_model = str(response.get("model") or "")
    allowed_returned_models = {
        MODEL,
        REQUEST_MODEL,
        "typesafe/jev-1.13-20260917",
    }
    if returned_model not in allowed_returned_models:
        errors.append(f"returned_model={returned_model!r}")
    if not isinstance(answers, dict):
        errors.append("missing answers")
        answers = {}
    if not isinstance(usage, dict) or int(
        usage.get("input_tokens") or usage.get("inputTokens") or 0
    ) <= 0:
        errors.append("missing input token usage")
        usage = usage if isinstance(usage, dict) else {}
    expected = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
    if set(answers) != set(expected):
        errors.append("missing or unexpected answer fields")
    predictions: dict[str, Any] = {}
    for field in expected:
        schema = task["json_schema"]["properties"][field]
        mapping = {str(value): value for value in label_values(schema)}
        answer = answers.get(field)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            errors.append(f"{field}: invalid answer type")
            continue
        choice = answer.get("choice")
        if choice not in mapping:
            errors.append(f"{field}: unknown choice")
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(mapping):
            errors.append(f"{field}: invalid probabilities")
            continue
        predictions[field] = mapping[choice]
    input_tokens = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)
    parse_error = "; ".join(errors)
    output = {
        "task": task["name"],
        "model": MODEL,
        "item_id": row["item_id"],
        "parse_error": parse_error,
        "truncated": False,
        "stop_reason": "typed_complete",
        "returned_model": returned_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd_upper": input_tokens * INPUT_USD_PER_MILLION / 1e6,
        "raw_content": json.dumps(predictions, sort_keys=True, ensure_ascii=False),
        "usage_present": not any(error == "missing input token usage" for error in errors),
        "documented_version": MODEL,
        "version_documentation_checked": "2026-09-18",
        "provider_created": None,
    }
    output.update({f"gt_{key}": value for key, value in row["gold"].items()})
    output.update({f"pred_{key}": value for key, value in predictions.items()})
    return output


def collect(stage: str) -> dict[str, Any]:
    rows = stage_requests(stage)
    tasks = task_map()
    state_path = ROOT / stage / "submission.json"
    if not state_path.exists() or read(state_path).get("requests_sha256") != canonical_sha(rows):
        raise ValueError("Missing or changed stage submission record")
    decoded = []
    token_by_task: dict[str, list[int]] = {}
    for row in rows:
        path = ROOT / stage / "responses" / f"{row['custom_id']}.json"
        if not path.exists():
            continue
        saved = read(path)
        if saved.get("status") != "completed" or saved.get("request_sha256") != canonical_sha(row):
            raise ValueError(f"Invalid response checkpoint: {path}")
        record = decode(row, tasks[row["task"]], saved["response"])
        decoded.append(record)
        token_by_task.setdefault(row["task"], []).append(record["input_tokens"])
    frame = pd.DataFrame(decoded)
    predictions_path = ROOT / stage / "predictions.csv"
    if len(frame):
        malformed, _ = prediction_malformed_masks(frame, list(tasks.values()))
        frame["malformed"] = malformed
        frame["stage"] = stage
        frame["observed_completed_at"] = [
            read(ROOT / stage / "responses" / f"{row['custom_id']}.json")["completed_at"]
            for row in rows
            if (ROOT / stage / "responses" / f"{row['custom_id']}.json").exists()
        ]
        frame.to_csv(predictions_path, index=False)
        bad = int(malformed.sum())
        returned = sorted(frame.returned_model.dropna().unique().tolist())
        successful_cost = float(frame.cost_usd_upper.sum())
    else:
        bad, returned, successful_cost = 0, [], 0.0
    failed_attempts = failed_attempt_records(stage)
    failed_cost = sum(record["cost_usd"] for record in failed_attempts)
    cost = successful_cost + failed_cost
    report = {
        "stage": stage,
        "model": MODEL,
        "expected": len(rows),
        "observed": len(frame),
        "malformed": bad,
        "truncated": 0,
        "input_tokens": int(frame.input_tokens.sum()) if len(frame) else 0,
        "output_tokens": int(frame.output_tokens.sum()) if len(frame) else 0,
        "cost_usd": cost,
        "successful_response_cost_usd": successful_cost,
        "failed_attempt_cost_usd": failed_cost,
        "infrastructure_failures": failed_attempts,
        "returned_models": returned,
        "input_tokens_by_task": token_by_task,
        "passed": (
            len(frame) == len(rows)
            and bad == 0
            and bool(returned)
            and set(returned) <= {MODEL, REQUEST_MODEL, "typesafe/jev-1.13-20260917"}
        ),
        "collected_at": now(),
    }
    save(ROOT / stage / "report.json", report)
    state = read(state_path)
    state.update(status="completed" if report["passed"] else "needs_attention", collected_at=now())
    save(state_path, state)
    register(
        stage,
        "completed" if report["passed"] else "needs_attention",
        f"{len(frame)}/{len(rows)} rows; {bad} malformed; cost ${cost:.6f}",
    )
    print(json.dumps(report, indent=2))
    return report


def finalize() -> dict[str, Any]:
    panel = read(PANEL)
    validate_frozen_panel(panel)
    reports = [read(ROOT / stage / "report.json") for stage in ("pilot", "remainder")]
    if not all(report.get("passed") for report in reports):
        raise ValueError("Both Jev stages must pass before finalization")
    frames = [
        pd.read_csv(ROOT / stage / "predictions.csv", dtype={"item_id": str}, low_memory=False)
        for stage in ("pilot", "remainder")
    ]
    frame = pd.concat(frames, ignore_index=True)
    expected = {(row["task"], str(row["item"]["item_id"])) for row in panel["rows"]}
    actual = set(zip(frame.task, frame.item_id))
    if len(frame) != EXPECTED_ITEMS or frame.duplicated(["task", "item_id"]).any() or actual != expected:
        raise ValueError("Jev predictions do not exactly cover the frozen panel")
    if (
        frame.malformed.any()
        or frame.truncated.any()
        or not set(frame.returned_model)
        or not set(frame.returned_model)
        <= {MODEL, REQUEST_MODEL, "typesafe/jev-1.13-20260917"}
    ):
        raise ValueError("Jev final panel has invalid model output")
    output = ROOT / "predictions.csv"
    frame.sort_values(["task", "item_id"]).to_csv(output, index=False)
    run_dates = sorted({str(value)[:10] for value in frame.observed_completed_at if str(value)})
    manifest = {
        "models": [
            {
                "model": MODEL,
                "model_id": REQUEST_MODEL,
                "provider": PROVIDER,
                "predictions": str(output),
                "panel_sha256": canonical_sha(panel),
                "prediction_sha256": file_sha(output),
                "revision": MODEL,
                "hardware_tier": "api",
                "settings": {
                    "model": REQUEST_MODEL,
                    "structured_output": "native_choice",
                    "native_interface": "OpenRouter Decisions API backed by TypeSafe",
                    "prompt_adapter_sha256": canonical_sha(
                        {"adapter_version": ADAPTER_VERSION, "requests": read(ROOT / "requests.json")["requests"]}
                    ),
                },
                "run_dates": run_dates,
                "input_tokens": int(frame.input_tokens.sum()),
                "output_tokens": int(frame.output_tokens.sum()),
                "cost_usd_upper": completed_cost(),
                "successful_response_cost_usd": float(frame.cost_usd_upper.sum()),
                "infrastructure_failures": sum(
                    (failed_attempt_records(stage) for stage in ("pilot", "remainder")),
                    [],
                ),
                "returned_models": sorted(set(frame.returned_model)),
                "documented_versions": [MODEL, "typesafe/jev-1.13-20260917"],
                "provenance_validated": True,
            }
        ]
    }
    save(ROOT / "api_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("stage", choices=["pilot", "remainder"])
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("stage", choices=["pilot", "remainder"])
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args.stage)
    elif args.command == "collect":
        collect(args.stage)
    else:
        finalize()


if __name__ == "__main__":
    main()
