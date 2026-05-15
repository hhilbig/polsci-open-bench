from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOC_PATH = REPO / "docs" / "task_inventory.md"


def _rel(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _manifest_rows(repo: Path, directory: str) -> list[dict]:
    root = repo / directory
    rows = []
    for path in sorted(root.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text()) or {}
        labels = spec.get("labels") or []
        label_kind = spec.get("label_kind", "unknown")
        if label_kind == "binary":
            label_summary = "binary"
        elif label_kind == "categorical":
            label_summary = f"{len(labels)} labels"
        elif label_kind == "multi_binary":
            label_summary = f"{len(labels)} binary labels"
        else:
            label_summary = label_kind
        rows.append(
            {
                "name": spec.get("name", path.stem),
                "directory": directory,
                "path": _rel(path, repo),
                "order": int(spec.get("order", 999)),
                "family": spec.get("family", "Custom"),
                "source": spec.get("source", "Unspecified"),
                "label_kind": label_kind,
                "label_summary": label_summary,
            }
        )
    return sorted(rows, key=lambda row: (row["order"], row["name"]))


def _count_csv(path: Path, batch_size: int | None = None) -> dict | None:
    if not path.exists():
        return None
    tasks = set()
    models = set()
    batches = set()
    cells = set()
    row_count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "task" not in reader.fieldnames:
            return None
        for row in reader:
            if batch_size is not None and row.get("batch_size") != str(batch_size):
                continue
            row_count += 1
            task = row.get("task") or ""
            model = row.get("model") or ""
            batch = row.get("batch_size") or ""
            if task:
                tasks.add(task)
            if model:
                models.add(model)
            if batch:
                batches.add(batch)
            cells.add((task, model, batch))
    return {
        "rows": row_count,
        "tasks": len(tasks),
        "models": len(models),
        "batches": sorted(batches, key=lambda value: int(value) if value.isdigit() else value),
        "cells": len(cells),
    }


def _coverage_rows(repo: Path) -> list[dict]:
    rows = []
    specs = [
        ("Canonical serial predictions", repo / "output" / "predictions.csv", None),
        ("Canonical serial summary", repo / "output" / "summary.csv", None),
        ("Canonical batched predictions", repo / "output" / "predictions_batched.csv", None),
        ("Canonical batched summary", repo / "output" / "summary_batched.csv", None),
        ("Local b=10 summary", repo / "output" / "summary_batched_local_b10.csv", 10),
    ]
    for label, path, batch_size in specs:
        counts = _count_csv(path, batch_size=batch_size)
        if counts:
            rows.append({"label": label, "path": _rel(path, repo), **counts})
    return rows


def _table(headers: list[str], rows: Iterable[Iterable[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _family_table(rows: list[dict]) -> list[str]:
    counts = Counter(row["family"] for row in rows)
    return _table(
        ["Family", "Tasks"],
        ((family, str(count)) for family, count in sorted(counts.items())),
    )


def _manifest_table(rows: list[dict]) -> list[str]:
    return _table(
        ["Task", "Family", "Labels", "Manifest"],
        (
            [
                f"`{row['name']}`",
                row["family"],
                row["label_summary"],
                f"`{row['path']}`",
            ]
            for row in rows
        ),
    )


def render_markdown(repo: Path = REPO) -> str:
    tasks = _manifest_rows(repo, "tasks")

    lines: list[str] = [
        "# Task Inventory",
        "",
        "This file is generated from task manifests. Regenerate it after adding, removing, or changing a task:",
        "",
        "```bash",
        "python3 code/task_inventory.py --write",
        "```",
        "",
        "## Counts",
        "",
    ]
    lines.extend(
        _table(
            ["Category", "Directory", "Count", "Meaning"],
            [
                [
                    "Canonical benchmark tasks",
                    "`tasks/`",
                    str(len(tasks)),
                    "Public 34-task benchmark manifests.",
                ],
            ],
        )
    )
    lines.extend(["", "## Task Families", ""])
    lines.extend(_family_table(tasks))

    coverage = _coverage_rows(repo)
    if coverage:
        lines.extend(["", "## Output Coverage", ""])
        lines.extend(
            _table(
                ["Artifact", "Path", "Tasks", "Models", "Cells", "Rows", "Batch sizes"],
                (
                    [
                        row["label"],
                        f"`{row['path']}`",
                        str(row["tasks"]),
                        str(row["models"]),
                        str(row["cells"]),
                        f"{row['rows']:,}",
                        ", ".join(row["batches"]) if row["batches"] else "",
                    ]
                    for row in coverage
                ),
            )
        )

    lines.extend(["", "## Tasks", ""])
    lines.extend(_manifest_table(tasks))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render or check the generated task inventory doc.")
    parser.add_argument("--write", action="store_true", help="Write docs/task_inventory.md.")
    parser.add_argument("--check", action="store_true", help="Fail if docs/task_inventory.md is stale.")
    args = parser.parse_args()

    rendered = render_markdown(REPO)
    if args.write:
        DOC_PATH.write_text(rendered)
        print(f"Wrote {_rel(DOC_PATH, REPO)}")
    if args.check:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != rendered:
            print(f"{_rel(DOC_PATH, REPO)} is stale; run `python3 code/task_inventory.py --write`.")
            return 1
        print(f"{_rel(DOC_PATH, REPO)} is current.")
    if not args.write and not args.check:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
