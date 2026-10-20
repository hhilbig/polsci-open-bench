#!/usr/bin/env python3
"""
Chance baselines for the main-release tasks.

A headline F1 means little without the score a model would get by ignoring the
text. Two baselines are computed on the exact items each task was scored on,
with `scoring.headline_f1` and the same support-only label set as
`output/summary.csv`:

  majority    -- every item gets the most frequent gold value (for multi-label
                 tasks, the most frequent value of each label). On binary tasks
                 whose positive class is the minority this scores 0.
  stratified  -- labels drawn at random in proportion to the gold labels. Its
                 expected per-class F1 equals that class's share of the items,
                 so the headline is the mean share over the scored classes
                 (positive-class share for binary tasks).

Usage:
  python3 code/build_chance_baselines.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import headline_f1, scored_labels
from task_registry import load_task_definitions

ROOT = Path(__file__).resolve().parents[1]


def gold_frame(predictions: pd.DataFrame, task_def: dict) -> pd.DataFrame:
    """One row per item: the gold columns of the first model scored on the task."""
    rows = predictions[predictions["task"] == task_def["name"]]
    first = rows[rows["model"] == rows["model"].iloc[0]]
    gt_cols = [c for c in first.columns if c.startswith("gt_") and first[c].notna().any()]
    return first[["item_id"] + gt_cols].reset_index(drop=True)


def baselines(task_def: dict, gold: pd.DataFrame) -> dict:
    kind = task_def["label_kind"]
    keys = task_def["labels"] if kind == "multi_binary" else [task_def["label_key"]]
    frame = gold.copy()
    for key in keys:
        frame[f"pred_{key}"] = frame[f"gt_{key}"].mode().iloc[0]
    labels = scored_labels(task_def, frame)
    if kind == "categorical":
        share = frame[f"gt_{task_def['label_key']}"].astype(str).value_counts(normalize=True)
        stratified = float(np.mean([share.get(str(label), 0.0) for label in labels]))
        accuracy = float(share.max())
        majority = str(share.idxmax())
    else:
        stratified = float(np.mean([frame[f"gt_{key}"].astype(float).mean() for key in
                                    (labels if kind == "multi_binary" else keys)]))
        accuracy = float(np.mean([(frame[f"gt_{k}"] == frame[f"pred_{k}"]).mean() for k in keys]))
        majority = ";".join(str(int(float(frame[f"pred_{k}"].iloc[0]))) for k in keys)
    return {"task": task_def["name"], "label_kind": kind, "n": len(frame),
            "majority_value": majority,
            "majority_f1": headline_f1(task_def, frame, label_subset=labels),
            "majority_accuracy": accuracy, "stratified_f1": stratified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", default=str(ROOT / "output/predictions.csv"))
    parser.add_argument("--summary", default=str(ROOT / "output/summary.csv"))
    parser.add_argument("--output", default=str(ROOT / "output/chance_baselines.csv"))
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    tasks = {t["name"]: t for t in load_task_definitions() if t["name"] in set(summary["task"])}
    predictions = pd.read_csv(args.predictions, low_memory=False)
    out = pd.DataFrame([baselines(t, gold_frame(predictions, t)) for t in tasks.values()])
    best = summary.groupby("task")["headline_f1"].max().rename("best_model_f1")
    out = out.merge(best, left_on="task", right_index=True)
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"wrote {args.output} ({len(out)} tasks)")
    print(out[["majority_f1", "majority_accuracy", "stratified_f1", "best_model_f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
