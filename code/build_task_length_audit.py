#!/usr/bin/env python3
"""
Build lightweight task-length and relative-performance audit artifacts.

Outputs:
  - output/task_length_audit.csv
  - output/task_length_analysis.md
"""
import argparse
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from model_registry import add_model_loading_args, load_model_definitions_from_args
from task_registry import (
    add_task_loading_args,
    load_task_definitions,
    load_task_definitions_from_args,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"


def _word_count(text):
    return len(str(text).split())


def _series_summary(values):
    s = pd.Series(values, dtype="float64")
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p90": float(s.quantile(0.9)),
        "max": float(s.max()),
    }


def _effective_label_count(task, items):
    counts = Counter()
    if task["label_kind"] == "multi_binary":
        labels = task["labels"]
        for item in items:
            counts[tuple(int(item["gt"][label]) for label in labels)] += 1
    else:
        label_key = task["label_key"]
        for item in items:
            counts[item["gt"][label_key]] += 1

    total = sum(counts.values())
    if total == 0:
        return np.nan
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(math.exp(entropy))


def _coding_complexity(task, effective_label_count, prompt_words):
    if task["label_kind"] == "multi_binary":
        return "High"
    if effective_label_count >= 8:
        return "High"
    if effective_label_count >= 3 or prompt_words >= 300:
        return "Medium"
    return "Low"


def _task_length_row(task):
    items = task["loader"]()
    prompt_text = Path(task["prompt_path"]).read_text()
    prompt_words = _word_count(prompt_text)
    item_chars = [len(item["user_content"]) for item in items]
    item_words = [_word_count(item["user_content"]) for item in items]
    item_lines = [item["user_content"].count("\n") + 1 for item in items]

    char_stats = _series_summary(item_chars)
    word_stats = _series_summary(item_words)
    line_stats = _series_summary(item_lines)
    effective_label_count = _effective_label_count(task, items)

    return {
        "task": task["name"],
        "family": task.get("family"),
        "source": task.get("source"),
        "label_kind": task["label_kind"],
        "label_count": len(task["labels"]),
        "effective_label_count": effective_label_count,
        "coding_complexity": _coding_complexity(task, effective_label_count, prompt_words),
        "sample_n": len(items),
        "prompt_chars": len(prompt_text),
        "prompt_words": prompt_words,
        "item_chars_mean": char_stats["mean"],
        "item_chars_median": char_stats["median"],
        "item_chars_p90": char_stats["p90"],
        "item_chars_max": char_stats["max"],
        "item_words_mean": word_stats["mean"],
        "item_words_median": word_stats["median"],
        "item_words_p90": word_stats["p90"],
        "item_lines_mean": line_stats["mean"],
        "item_lines_median": line_stats["median"],
    }


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _with_complexity_proxy(df):
    out = df.copy()
    parts = []
    for col in ["prompt_chars", "item_chars_median", "label_count"]:
        s = np.log1p(out[col].astype(float))
        std = s.std(ddof=0)
        if std == 0:
            parts.append(pd.Series(np.zeros(len(out)), index=out.index))
        else:
            parts.append((s - s.mean()) / std)
    out["complexity_proxy_z"] = sum(parts) / len(parts)
    return out


def _merge_relative_performance(task_df, summary_path, local_models, api_models):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return task_df

    summary = pd.read_csv(summary_path, low_memory=False)
    rows = []
    for task_name, g in summary.groupby("task"):
        local = g[g["model"].isin(local_models)]
        api = g[g["model"].isin(api_models)]
        row = {"task": task_name}
        if len(local):
            local_best = local.sort_values("headline_f1", ascending=False).iloc[0]
            row["local_best_model"] = local_best["model"]
            row["local_best_f1"] = float(local_best["headline_f1"])
            row["local_mean_f1"] = float(local["headline_f1"].mean())
        if len(api):
            api_best = api.sort_values("headline_f1", ascending=False).iloc[0]
            row["api_best_model"] = api_best["model"]
            row["api_best_f1"] = float(api_best["headline_f1"])
            row["api_mean_f1"] = float(api["headline_f1"].mean())
        if "local_best_f1" in row and "api_best_f1" in row:
            row["local_minus_api_best_f1"] = row["local_best_f1"] - row["api_best_f1"]
        if "local_mean_f1" in row and "api_mean_f1" in row:
            row["local_minus_api_mean_f1"] = row["local_mean_f1"] - row["api_mean_f1"]
        rows.append(row)

    perf = pd.DataFrame(rows)
    return task_df.merge(perf, on="task", how="left")


def _write_markdown(df, path):
    corr_cols = [
        ("item_chars_median", "Median sampled item chars"),
        ("prompt_chars", "Prompt chars"),
        ("label_count", "Label count"),
        ("complexity_proxy_z", "Complexity proxy"),
    ]
    lines = [
        "# Task length and complexity audit",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "This is exploratory. The current benchmark still has a small number of tasks, so the length/performance relationships should be read as descriptive rather than definitive.",
        "",
        "## Cross-task correlations",
        "",
    ]
    target = "local_minus_api_best_f1"
    if target in df.columns:
        for col, label in corr_cols:
            corr = _safe_corr(df[col], df[target])
            lines.append(f"- `{label}` vs `local_minus_api_best_f1`: `{corr:.3f}`" if not np.isnan(corr) else f"- `{label}` vs `local_minus_api_best_f1`: `NA`")
    else:
        lines.append("- Relative performance columns unavailable because no summary file was provided.")

    table_cols = [
        c
        for c in [
            "task",
            "family",
            "coding_complexity",
            "effective_label_count",
            "label_count",
            "prompt_chars",
            "item_chars_median",
            "item_chars_p90",
            "local_best_model",
            "local_best_f1",
            "api_best_model",
            "api_best_f1",
            "local_minus_api_best_f1",
        ]
        if c in df.columns
    ]
    lines.extend(["", "## Task table", ""])
    if table_cols:
        lines.append("| " + " | ".join(table_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(table_cols)) + "|")
        for _, row in df[table_cols].round(3).iterrows():
            vals = [str(row[col]) if not pd.isna(row[col]) else "" for col in table_cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_task_loading_args(ap)
    add_model_loading_args(ap)
    ap.add_argument("--summary", default=str(OUT / "summary.csv"))
    ap.add_argument("--output-csv", default=str(OUT / "task_length_audit.csv"))
    ap.add_argument("--output-md", default=str(OUT / "task_length_analysis.md"))
    ap.add_argument(
        "--extra-tasks-dir",
        action="append",
        default=[],
        help="Additional directory of task manifests to append to the primary task set.",
    )
    args = ap.parse_args()

    tasks = load_task_definitions_from_args(args)
    seen = {task["name"] for task in tasks}
    for extra_dir in args.extra_tasks_dir:
        for task in load_task_definitions(tasks_dir=extra_dir):
            if task["name"] in seen:
                continue
            tasks.append(task)
            seen.add(task["name"])
    tasks = sorted(tasks, key=lambda task: (task["order"], task["name"]))
    models = load_model_definitions_from_args(args)
    local_models = {
        model["name"] for model in models if model.get("compute_class") == "local"
    }
    api_models = {
        model["name"] for model in models if model.get("compute_class") == "api"
    }
    rows = [_task_length_row(task) for task in tasks]
    df = pd.DataFrame(rows).sort_values(["family", "task"]).reset_index(drop=True)
    df = _with_complexity_proxy(df)
    df = _merge_relative_performance(df, args.summary, local_models, api_models)

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    df.to_csv(out_csv, index=False)
    _write_markdown(df, out_md)
    print(f"wrote {out_csv} ({len(df)} rows)")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
