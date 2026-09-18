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
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scoring import headline_f1, per_class_f1, scored_labels
from task_registry import add_task_loading_args, load_task_definitions_from_args

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"


def _metrics_for_group(task_def, g, support_only=True, label_subset=None):
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
    f1_by_label = per_class_f1(
        task_def, clean, support_only=support_only, label_subset=label_subset
    )
    row.update({f"f1_{lbl}": val for lbl, val in f1_by_label.items()})
    row["headline_f1"] = headline_f1(
        task_def, clean, support_only=support_only, label_subset=label_subset
    )

    if kind == "multi_binary":
        row["avg_f1"] = row["headline_f1"]
        return row

    if kind == "binary":
        return row

    if kind == "categorical":
        key = task_def["label_key"]
        gt_col, pred_col = f"gt_{key}", f"pred_{key}"
        sub = clean[[gt_col, pred_col]].dropna()
        row["avg_f1"] = row["headline_f1"]
        row["accuracy"] = (sub[pred_col] == sub[gt_col]).mean() if len(sub) else np.nan
        return row

    raise ValueError(f"Unknown label_kind: {kind}")


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


def _load_b1_from_serial(serial_path):
    """Read v2 serial predictions and tag them as batch_size=1 for join with the
    batched grid. Adds the missing batch_latency_s column (= latency_s for serial)."""
    serial_path = Path(serial_path)
    if not serial_path.exists():
        return pd.DataFrame()
    s = pd.read_csv(serial_path, low_memory=False)
    s["batch_size"] = 1
    s["batch_latency_s"] = s["latency_s"]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_task_loading_args(ap)
    ap.add_argument("--predictions", default=str(OUT / "predictions_batched.csv"))
    ap.add_argument("--serial-predictions", default=str(OUT / "predictions.csv"))
    ap.add_argument("--output", default=str(OUT / "summary_batched.csv"))
    ap.add_argument(
        "--legacy-all-labels",
        action="store_true",
        help=(
            "Average macro F1 over every manifest label, including labels with no "
            "gold support in the sample. Reproduces the published v1 summaries."
        ),
    )
    args = ap.parse_args()
    support_only = not args.legacy_all_labels

    preds = pd.read_csv(args.predictions, low_memory=False)
    serial_b1 = _load_b1_from_serial(args.serial_predictions)
    if len(serial_b1):
        preds = pd.concat([preds, serial_b1], ignore_index=True, sort=False)
        print(f"[serial b=1] merged {len(serial_b1)} rows from predictions.csv into baseline")
    task_defs = {t["name"]: t for t in load_task_definitions_from_args(args)}

    # Agreement computed per task (needs access to the full task's prediction grid).
    per_task_agreement = {}
    for task_name, task_def in task_defs.items():
        if task_name not in preds.task.unique():
            continue
        per_task_agreement[task_name] = _compute_agreement(preds, task_def)

    # Pin the scored label set per task from the full task frame, so every
    # (model, batch_size) cell is averaged over the same classes.
    label_subsets = {
        task: scored_labels(task_defs[task], gtask, support_only=support_only)
        for task, gtask in preds.groupby("task")
        if task in task_defs
    }

    rows = []
    for (task, model, b), g in preds.groupby(["task", "model", "batch_size"]):
        if task not in task_defs:
            print(f"[skip] unknown task in predictions: {task}")
            continue
        r = {"task": task, "model": model, "batch_size": int(b),
             **_metrics_for_group(task_defs[task], g,
                                  support_only=support_only,
                                  label_subset=label_subsets.get(task))}
        r["agreement_vs_b1"] = per_task_agreement.get(task, {}).get((model, int(b)), np.nan)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["task", "model", "batch_size"]).reset_index(drop=True)

    leading = ["task", "model", "batch_size", "n", "parse_ok", "parse_err_rate",
               "mean_latency_s", "median_latency_s", "median_batch_latency_s"]
    trailing = ["avg_f1", "accuracy", "headline_f1", "agreement_vs_b1"]
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    ordered = leading + f1_cols + [c for c in trailing if c in df.columns]
    df = df[[c for c in ordered if c in df.columns]]

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    print()
    print(df[["task", "model", "batch_size", "parse_err_rate", "headline_f1",
              "agreement_vs_b1", "median_latency_s"]]
          .round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
