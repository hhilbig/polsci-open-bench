#!/usr/bin/env python3
"""Append-only run ledger and Markdown status renderer.

The benchmark often runs in sidecars, on remote hosts, or inside tmux sessions.
This module keeps a small machine-readable ledger so a later session can answer
"what is running, what finished, and what still needs attention" without
reconstructing it from shell history.
"""
import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = REPO / "output" / "run_registry.jsonl"
DEFAULT_STATUS_MD = REPO / "docs" / "run_status.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _repo_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO / path


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if v is not None and v != ""}


def append_event(ledger_path: Path | str | None = None, **record: Any) -> dict[str, Any]:
    """Append one JSONL event and return the normalized record."""
    ledger = _repo_path(ledger_path) or DEFAULT_LEDGER
    event = _clean_record(
        {
            "timestamp": now_utc(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **record,
        }
    )
    if "run_id" not in event:
        raise ValueError("run_id is required")
    if "event" not in event:
        raise ValueError("event is required")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def read_events(ledger_path: Path | str | None = None) -> list[dict[str, Any]]:
    ledger = _repo_path(ledger_path) or DEFAULT_LEDGER
    if not ledger.exists():
        return []
    events: list[dict[str, Any]] = []
    with ledger.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {ledger} line {line_no}: {exc}") from exc
    return events


def latest_runs(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold append-only events into latest run state by run_id."""
    runs: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue
        current = runs.setdefault(str(run_id), {})
        current.update(event)
        current.setdefault("started_at", event.get("timestamp"))
        if event.get("event") == "start":
            current["started_at"] = event.get("timestamp", current.get("started_at"))
        if event.get("event") in {"finish", "fail"}:
            current["finished_at"] = event.get("timestamp")
    return runs


def _fmt_value(value: Any) -> str:
    if isinstance(value, list):
        if len(value) > 8:
            shown = ", ".join(str(v) for v in value[:8])
            return f"{shown}, ... ({len(value)} total)"
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _run_line(run: dict[str, Any]) -> str:
    fields = []
    status = run.get("status")
    base_fields = [
        ("status", "status"),
        ("runner", "runner"),
        ("host", "host"),
        ("task_scope", "task scope"),
        ("model_scope", "model scope"),
        ("batch_sizes", "batch sizes"),
        ("tmux_session", "tmux"),
        ("output", "output"),
        ("log", "log"),
        ("batch_id", "batch id"),
        ("input_file_id", "input file id"),
        ("output_file_id", "output file id"),
        ("error_file_id", "error file id"),
        ("rows_written", "rows"),
        ("cost_cap_usd", "cost cap"),
    ]
    active_fields = [
        ("current_task", "task"),
        ("current_model", "model"),
        ("current_batch_size", "b"),
        ("completed_cells", "cells"),
    ]
    for key, label in base_fields:
        if key in run:
            fields.append(f"{label}: `{_fmt_value(run[key])}`")
    if status in {"queued", "running"}:
        for key, label in active_fields:
            if key in run:
                fields.append(f"{label}: `{_fmt_value(run[key])}`")
    note = run.get("note") or run.get("notes")
    if note:
        fields.append(f"note: {_fmt_value(note)}")
    return "; ".join(fields) if fields else "(no details)"


def render_markdown(
    ledger_path: Path | str | None = None,
    status_path: Path | str | None = None,
    max_recent: int = 12,
) -> str:
    events = read_events(ledger_path)
    runs = latest_runs(events)
    ordered = sorted(runs.values(), key=lambda r: r.get("timestamp", ""), reverse=True)
    active = [r for r in ordered if r.get("status") in {"queued", "running"}]
    failed = [r for r in ordered if r.get("status") in {"failed", "error", "needs_attention", "blocked"}]
    tabled = [r for r in ordered if r.get("status") == "tabled"]
    completed = [r for r in ordered if r.get("status") in {"completed", "cancelled"}]
    other = [r for r in ordered if r not in active and r not in failed and r not in tabled and r not in completed]

    lines: list[str] = []
    lines.append("# Run status")
    lines.append("")
    lines.append(f"Last rendered: {now_utc()}")
    lines.append("")
    lines.append("Open this file at the start of a session before reconstructing runs from logs.")
    lines.append("")
    lines.append("## Active Runs")
    lines.append("")
    if active:
        for run in active:
            lines.append(f"- **{run['run_id']}**: {_run_line(run)}")
    else:
        lines.append("- None recorded.")
    lines.append("")
    lines.append("## Failed Or Needs Attention")
    lines.append("")
    if failed:
        for run in failed[:max_recent]:
            lines.append(f"- **{run['run_id']}**: {_run_line(run)}")
    else:
        lines.append("- None recorded.")
    lines.append("")
    lines.append("## Tabled Runs")
    lines.append("")
    if tabled:
        for run in tabled[:max_recent]:
            lines.append(f"- **{run['run_id']}**: {_run_line(run)}")
    else:
        lines.append("- None recorded.")
    lines.append("")
    lines.append("## Recent Completed Runs")
    lines.append("")
    if completed:
        for run in completed[:max_recent]:
            lines.append(f"- **{run['run_id']}**: {_run_line(run)}")
    else:
        lines.append("- None recorded.")
    if other:
        lines.append("")
        lines.append("## Other Runs")
        lines.append("")
        for run in other[:max_recent]:
            lines.append(f"- **{run['run_id']}**: {_run_line(run)}")
    lines.append("")
    lines.append("## Ledger")
    lines.append("")
    ledger = _repo_path(ledger_path) or DEFAULT_LEDGER
    lines.append(f"- Source: `{ledger.relative_to(REPO) if ledger.is_relative_to(REPO) else ledger}`")
    lines.append(f"- Events: {len(events)}")
    lines.append(f"- Runs: {len(runs)}")
    lines.append("")
    lines.append("## Operating Rule")
    lines.append("")
    lines.append("- Register every sidecar, API, local, droplet, or mac2 run before launch.")
    lines.append("- Record the host, tmux session, log path, output path, model scope, task scope, and cost cap when applicable.")
    lines.append("- Render this file before ending a session.")
    lines.append("")

    text = "\n".join(lines)
    if status_path is not None:
        out = _repo_path(status_path) or DEFAULT_STATUS_MD
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return text


def parse_meta(items: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE metadata, got: {item}")
        key, value = item.split("=", 1)
        meta[key.replace("-", "_")] = value
    return meta


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--runner")
    parser.add_argument("--host")
    parser.add_argument("--task-scope")
    parser.add_argument("--model-scope")
    parser.add_argument("--batch-sizes")
    parser.add_argument("--output")
    parser.add_argument("--log")
    parser.add_argument("--tmux-session")
    parser.add_argument("--cost-cap-usd")
    parser.add_argument("--note")
    parser.add_argument("--meta", action="append", default=[], help="Additional KEY=VALUE field. Can be repeated.")


def command_record(args: argparse.Namespace, event: str, status: str | None = None) -> dict[str, Any]:
    record = {
        "event": event,
        "run_id": args.run_id,
        "status": status,
        "runner": args.runner,
        "host": args.host,
        "task_scope": args.task_scope,
        "model_scope": args.model_scope,
        "batch_sizes": args.batch_sizes,
        "output": args.output,
        "log": args.log,
        "tmux_session": args.tmux_session,
        "cost_cap_usd": args.cost_cap_usd,
        "note": args.note,
    }
    record.update(parse_meta(args.meta))
    return append_event(args.ledger, **record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Register a queued/running run.")
    add_common_run_args(p_start)
    p_start.add_argument("--initial-status", default="running", choices=["queued", "running"])

    p_update = sub.add_parser("update", help="Append progress for a run.")
    add_common_run_args(p_update)
    p_update.add_argument("--run-status", default="running")

    p_finish = sub.add_parser("finish", help="Mark a run completed, failed, or cancelled.")
    add_common_run_args(p_finish)
    p_finish.add_argument("--final-status", default="completed", choices=["completed", "failed", "cancelled"])

    p_render = sub.add_parser("render-status", help="Render docs/run_status.md from the ledger.")
    p_render.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p_render.add_argument("--output", default=str(DEFAULT_STATUS_MD))
    p_render.add_argument("--max-recent", type=int, default=12)
    p_render.add_argument("--print", action="store_true", dest="print_output")

    args = parser.parse_args()
    if args.command == "start":
        command_record(args, event="start", status=args.initial_status)
    elif args.command == "update":
        command_record(args, event="update", status=args.run_status)
    elif args.command == "finish":
        event = "fail" if args.final_status == "failed" else "finish"
        command_record(args, event=event, status=args.final_status)
    elif args.command == "render-status":
        text = render_markdown(args.ledger, args.output, max_recent=args.max_recent)
        if args.print_output:
            print(text)


if __name__ == "__main__":
    main()
