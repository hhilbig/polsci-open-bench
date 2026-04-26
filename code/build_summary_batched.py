#!/usr/bin/env python3
"""
Build output/summary_batched.csv from output/predictions_batched.csv.

Same metric layout as build_summary.py, but rows are grouped by
(task, model, batch_size). Adds a `median_batch_latency_s` column showing
the actual per-call wall-clock (not divided by batch size) — useful for
tracing the "what does one API call cost" axis.

Also emits a small "agreement" metric per cell: fraction of items whose
predictions at b>1 match the corresponding b=1 predictions for the same
(task, model, item_id). For cells with batch_size==1, agreement is 1.0
by definition. This captures how much batching perturbs outputs.

Usage:
  python3 code/build_summary_batched.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from benchmark import TASKS as ACTIVE_TASKS, _STATE_ADAPTATION_TASK_DEF

# Include state_adaptation (it was reinstated for the batching study).
TASKS = [_STATE_ADAPTATION_TASK_DEF] + ACTIVE_TASKS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"


def _metrics_for_group(task_def, g):
    kind = task_def["label_kind"]
    clean = g[g.parse_error.isna()]
    row = {
        "n": len(g),
        "parse_ok": len(clean),
        "parse_err_rate": 1 - len(clean) / len(g) if len(g) else np.nan,
        "mean_latency_s": g.latency_s.mean(),
        "median_latency_s": g.latency_s.median(),
        "median_batch_latency_s": g.batch_latency_s.median() if "batch_latency_s" in g else np.nan,
    }
    labels = task_def["labels"]

    if kind == "multi_binary":
        per_label_f1 = {}
        for lbl in labels:
            gt_col, pred_col = f"gt_{lbl}", f"pred_{lbl}"
            sub = clean[[gt_col, pred_col]].dropna()
            if len(sub) == 0:
                per_label_f1[f"f1_{lbl}"] = np.nan
                continue
            per_label_f1[f"f1_{lbl}"] = f1_score(
                sub[gt_col].astype(int), sub[pred_col].astype(int),
                pos_label=1, zero_division=0,
            )
        row.update(per_label_f1)
        vals = [v for v in per_label_f1.values() if not np.isnan(v)]
        row["avg_f1"] = float(np.mean(vals)) if vals else np.nan
        row["headline_f1"] = row["avg_f1"]

    elif kind == "binary":
        key = task_def["label_key"]
        gt_col, pred_col = f"gt_{key}", f"pred_{key}"
        sub = clean[[gt_col, pred_col]].dropna()
        f1 = f1_score(sub[gt_col].astype(int), sub[pred_col].astype(int),
                      pos_label=1, zero_division=0) if len(sub) else np.nan
        row[f"f1_{key}"] = f1
        row["headline_f1"] = f1

    elif kind == "categorical":
        key = task_def["label_key"]
        gt_col, pred_col = f"gt_{key}", f"pred_{key}"
        sub = clean[[gt_col, pred_col]].dropna()
        per_class = {}
        for lbl in labels:
            per_class[f"f1_{lbl}"] = (
                f1_score(sub[gt_col], sub[pred_col], labels=[lbl], average="macro", zero_division=0)
                if len(sub) else np.nan
            )
        row.update(per_class)
        vals = [v for v in per_class.values() if not np.isnan(v)]
        row["avg_f1"] = float(np.mean(vals)) if vals else np.nan
        row["accuracy"] = (sub[pred_col] == sub[gt_col]).mean() if len(sub) else np.nan
        row["headline_f1"] = row["avg_f1"]

    else:
        raise ValueError(f"Unknown label_kind: {kind}")
    return row


def _compute_agreement(preds, task_def):
    """For each (task, model, b>1, item_id), check whether the pred matches
    the corresponding (task, model, b=1, item_id) prediction. Returns a
    mapping (task, model, b) -> mean agreement (float or nan when either side
    missing/unparseable)."""
    pred_cols_for_task = []
    kind = task_def["label_kind"]
    if kind == "multi_binary":
        pred_cols_for_task = [f"pred_{l}" for l in task_def["labels"]]
    else:
        pred_cols_for_task = [f"pred_{task_def['label_key']}"]

    # Build (task, model, item_id) -> b=1 pred-tuple dict
    baseline = preds[(preds.task == task_def["name"]) & (preds.batch_size == 1)]
    b1_lookup = {}
    for _, r in baseline.iterrows():
        key = (r.model, r.item_id)
        b1_lookup[key] = tuple(r[c] for c in pred_cols_for_task)

    results = {}
    for (model, b), g in preds[preds.task == task_def["name"]].groupby(["model", "batch_size"]):
        if b == 1:
            results[(model, b)] = 1.0
            continue
        matches = 0
        comparable = 0
        for _, r in g.iterrows():
            b1 = b1_lookup.get((model, r.item_id))
            if b1 is None:
                continue
            this = tuple(r[c] for c in pred_cols_for_task)
            if any(pd.isna(v) for v in b1) or any(pd.isna(v) for v in this):
                continue
            comparable += 1
            if b1 == this:
                matches += 1
        results[(model, b)] = matches / comparable if comparable else np.nan
    return results


def main():
    preds = pd.read_csv(OUT / "predictions_batched.csv", low_memory=False)
    task_defs = {t["name"]: t for t in TASKS}

    # Agreement computed per task (needs access to the full task's prediction grid).
    per_task_agreement = {}
    for task_name, task_def in task_defs.items():
        if task_name not in preds.task.unique():
            continue
        per_task_agreement[task_name] = _compute_agreement(preds, task_def)

    rows = []
    for (task, model, b), g in preds.groupby(["task", "model", "batch_size"]):
        if task not in task_defs:
            print(f"[skip] unknown task in predictions: {task}")
            continue
        r = {"task": task, "model": model, "batch_size": int(b),
             **_metrics_for_group(task_defs[task], g)}
        r["agreement_vs_b1"] = per_task_agreement.get(task, {}).get((model, int(b)), np.nan)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["task", "model", "batch_size"]).reset_index(drop=True)

    leading = ["task", "model", "batch_size", "n", "parse_ok", "parse_err_rate",
               "mean_latency_s", "median_latency_s", "median_batch_latency_s"]
    trailing = ["avg_f1", "accuracy", "headline_f1", "agreement_vs_b1"]
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    ordered = leading + f1_cols + [c for c in trailing if c in df.columns]
    df = df[[c for c in ordered if c in df.columns]]

    out_path = OUT / "summary_batched.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    print()
    print(df[["task", "model", "batch_size", "parse_err_rate", "headline_f1",
              "agreement_vs_b1", "median_latency_s"]]
          .round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
