#!/usr/bin/env python3
"""
Build output/summary.csv from output/predictions.csv.

Per-(task, model) row includes headline_f1, accuracy, parse_err_rate,
latency percentiles, GPU-hours-per-1000-items (manifest `compute_class:
local`), and 95%
paired-bootstrap CIs on the headline metric (1000 iterations, paired
by item across models so the CIs support model-vs-model comparison).

Usage:
  python3 code/build_summary.py           # reads output/predictions.csv, writes output/summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

from model_registry import add_model_loading_args, load_model_definitions_from_args
from scoring import headline_f1, macro_f1_arrays, per_class_f1, scored_labels
from task_registry import add_task_loading_args, load_task_definitions_from_args

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20260424


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "output"


def _metrics_for_group(task_def, g, support_only=True, label_subset=None):
    """Per-(task, model) metrics.

    `support_only` restricts the macro average to labels with gold support; see
    `code/scoring.py`. `label_subset` pins that set explicitly, so every model on
    a task is scored over the same classes even if a model's own rows happen to
    drop an item.
    """
    kind = task_def["label_kind"]
    clean = g[g.parse_error.isna()]
    row = {
        "n": len(g),
        "parse_ok": len(clean),
        "parse_err_rate": 1 - len(clean) / len(g) if len(g) else np.nan,
        "mean_latency_s": g.latency_s.mean(),
        "median_latency_s": g.latency_s.median(),
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

    key = task_def["label_key"]
    gt_col, pred_col = f"gt_{key}", f"pred_{key}"
    sub = clean[[gt_col, pred_col]].dropna()

    if kind == "binary":
        # accuracy + MCC for binary tasks
        if len(sub):
            row["accuracy"] = (sub[pred_col].astype(int) == sub[gt_col].astype(int)).mean()
            try:
                row["mcc"] = matthews_corrcoef(sub[gt_col].astype(int), sub[pred_col].astype(int))
            except ValueError:
                row["mcc"] = np.nan
        else:
            row["accuracy"] = np.nan
            row["mcc"] = np.nan
        return row

    if kind == "categorical":
        row["avg_f1"] = row["headline_f1"]
        row["accuracy"] = (sub[pred_col] == sub[gt_col]).mean() if len(sub) else np.nan
        # MCC handles class imbalance better than macro F1; same data + mask as accuracy.
        try:
            row["mcc"] = (
                matthews_corrcoef(sub[gt_col], sub[pred_col]) if len(sub) else np.nan
            )
        except ValueError:
            row["mcc"] = np.nan
        return row

    raise ValueError(f"Unknown label_kind: {kind}")


def _bootstrap_cis(preds, task_defs, support_only=True, label_subsets=None):
    """Paired-by-item bootstrap on headline F1 per (task, model). Returns dict
    keyed by (task, model) -> (low, high). Optimized: pre-extract numpy arrays
    once per cell, then index into them inside the bootstrap loop.

    The scored label set is fixed once from the full sample and reused in every
    replicate, so the CI is computed over the same classes as the point estimate
    rather than a set that shifts with each resample.

    Note: resampling is by item, and several tasks contain the same text many
    times over (see docs/task_source_fidelity_audit.md), so repeated texts are
    treated as independent draws and these intervals are narrower than the
    effective sample supports."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for task, gtask in preds.groupby("task"):
        if task not in task_defs:
            continue
        td = task_defs[task]
        kind = td["label_kind"]
        if label_subsets is not None and task in label_subsets:
            labels = list(label_subsets[task])
        else:
            labels = scored_labels(td, gtask, support_only=support_only)
        item_ids = sorted(gtask["item_id"].astype(str).unique())
        n = len(item_ids)
        if n == 0:
            continue
        # Pre-build numpy arrays per (task, model)
        # For categorical: gt_arr (string), pred_arr (string)
        # For binary: gt_arr (0/1), pred_arr (0/1)
        # For multi_binary: gt_arr (n × k), pred_arr (n × k)
        cells = {}
        for model, gm in gtask.groupby("model"):
            gm = gm.copy()
            gm["item_id"] = gm["item_id"].astype(str)
            gm = gm.drop_duplicates("item_id").set_index("item_id").reindex(item_ids)
            if kind == "multi_binary":
                gt_arr = np.stack([gm[f"gt_{l}"].astype("float").values for l in labels], axis=1)
                pred_arr = np.stack([gm[f"pred_{l}"].astype("float").values for l in labels], axis=1)
                # row-wise mask: include rows where all labels are non-null on both sides
                mask = ~(np.isnan(gt_arr).any(axis=1) | np.isnan(pred_arr).any(axis=1))
                cells[model] = ("multi_binary", gt_arr.astype(np.int8), pred_arr.astype(np.int8), mask)
            elif kind == "binary":
                key = td["label_key"]
                gt_arr = gm[f"gt_{key}"].astype("float").values
                pred_arr = gm[f"pred_{key}"].astype("float").values
                mask = ~(np.isnan(gt_arr) | np.isnan(pred_arr))
                cells[model] = ("binary", gt_arr.astype(np.int8), pred_arr.astype(np.int8), mask)
            else:  # categorical
                key = td["label_key"]
                gt_arr = gm[f"gt_{key}"].astype("string").values
                pred_arr = gm[f"pred_{key}"].astype("string").values
                mask = (gt_arr != pd.NA) & (pred_arr != pd.NA)
                # pd.NA → object; convert to plain strings, mark None as a sentinel
                gt_arr = np.array([str(x) if not (x is pd.NA) else "__NA__" for x in gt_arr], dtype=object)
                pred_arr = np.array([str(x) if not (x is pd.NA) else "__NA__" for x in pred_arr], dtype=object)
                mask = (gt_arr != "__NA__") & (pred_arr != "__NA__")
                cells[model] = ("categorical", gt_arr, pred_arr, mask)

        # Run bootstrap iterations
        boot_f1 = {m: np.empty(BOOTSTRAP_ITERS) for m in cells}
        for b in range(BOOTSTRAP_ITERS):
            idx = rng.integers(0, n, n)
            for m, (k, gt_arr, pred_arr, mask) in cells.items():
                if k == "multi_binary":
                    sub_gt = gt_arr[idx]
                    sub_pred = pred_arr[idx]
                    sub_mask = mask[idx]
                    if sub_mask.sum() == 0:
                        boot_f1[m][b] = np.nan
                        continue
                    sub_gt = sub_gt[sub_mask]
                    sub_pred = sub_pred[sub_mask]
                    per = []
                    for j in range(sub_gt.shape[1]):
                        per.append(f1_score(sub_gt[:, j], sub_pred[:, j], pos_label=1, zero_division=0))
                    boot_f1[m][b] = float(np.mean(per))
                elif k == "binary":
                    sub_gt = gt_arr[idx]
                    sub_pred = pred_arr[idx]
                    sub_mask = mask[idx]
                    if sub_mask.sum() == 0:
                        boot_f1[m][b] = np.nan
                        continue
                    boot_f1[m][b] = f1_score(sub_gt[sub_mask], sub_pred[sub_mask],
                                             pos_label=1, zero_division=0)
                else:  # categorical
                    sub_gt = gt_arr[idx]
                    sub_pred = pred_arr[idx]
                    sub_mask = mask[idx]
                    if sub_mask.sum() == 0:
                        boot_f1[m][b] = np.nan
                        continue
                    sg = sub_gt[sub_mask]
                    sp = sub_pred[sub_mask]
                    boot_f1[m][b] = macro_f1_arrays(sg, sp, labels)
        for m, arr in boot_f1.items():
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                out[(task, m)] = (np.nan, np.nan)
            else:
                out[(task, m)] = (float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5)))
    return out


def _model_lookup(models):
    return {model["name"]: model for model in models}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_task_loading_args(ap)
    add_model_loading_args(ap)
    ap.add_argument("--predictions", default=str(OUT / "predictions.csv"))
    ap.add_argument("--output", default=str(OUT / "summary.csv"))
    ap.add_argument(
        "--legacy-all-labels",
        action="store_true",
        help=(
            "Average macro F1 over every manifest label, including labels with no "
            "gold support in the sample. This is how the published v1 summaries "
            "were computed; kept only to reproduce them."
        ),
    )
    args = ap.parse_args()
    support_only = not args.legacy_all_labels

    preds = pd.read_csv(args.predictions, low_memory=False)
    task_defs = {t["name"]: t for t in load_task_definitions_from_args(args)}
    model_lookup = _model_lookup(load_model_definitions_from_args(args))

    # Fix the scored label set per task from the full sample, so every model on a
    # task is averaged over the same classes and the bootstrap matches the point
    # estimate.
    label_subsets = {
        task: scored_labels(task_defs[task], gtask, support_only=support_only)
        for task, gtask in preds.groupby("task")
        if task in task_defs
    }

    rows = []
    for (task, model), g in preds.groupby(["task", "model"]):
        if task not in task_defs:
            print(f"[skip] unknown task in predictions: {task}")
            continue
        r = {
            "task": task,
            "model": model,
            **_metrics_for_group(
                task_defs[task], g,
                support_only=support_only,
                label_subset=label_subsets.get(task),
            ),
        }
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)

    # Bootstrap CIs (paired-by-item, 1000 iters, 95%)
    print(f"[bootstrap] computing {BOOTSTRAP_ITERS}-iter paired CIs ...")
    cis = _bootstrap_cis(
        preds, task_defs, support_only=support_only, label_subsets=label_subsets
    )
    df["headline_f1_lo"] = df.apply(lambda r: cis.get((r["task"], r["model"]), (np.nan, np.nan))[0], axis=1)
    df["headline_f1_hi"] = df.apply(lambda r: cis.get((r["task"], r["model"]), (np.nan, np.nan))[1], axis=1)

    # GPU-hours per 1000 for self-hosted / local models only.
    df["gpu_hours_per_1000"] = df.apply(
        lambda r: (
            r["median_latency_s"] * 1000 / 3600
            if model_lookup.get(r["model"], {}).get("compute_class") == "local"
            else np.nan
        ),
        axis=1,
    )
    # USD per 1000 calls for any model manifest that provides a per-call cost.
    df["usd_per_1000"] = df.apply(
        lambda r: (
            model_lookup[r["model"]]["cost_per_call_usd"] * 1000
            if model_lookup.get(r["model"], {}).get("cost_per_call_usd") is not None
            else np.nan
        ),
        axis=1,
    )
    # Cost per 1000 *correct* predictions = cost_per_1000 / accuracy.
    # Two parallel columns since units differ across backends.
    df["gpu_hours_per_1000_correct"] = df["gpu_hours_per_1000"] / df["accuracy"]
    df["usd_per_1000_correct"] = df["usd_per_1000"] / df["accuracy"]

    # Move identifier + summary cols to the front; f1_* sparse columns after.
    leading = ["task", "model", "n", "parse_ok", "parse_err_rate",
               "mean_latency_s", "median_latency_s",
               "gpu_hours_per_1000", "gpu_hours_per_1000_correct",
               "usd_per_1000", "usd_per_1000_correct"]
    trailing = ["avg_f1", "accuracy", "mcc",
                "headline_f1", "headline_f1_lo", "headline_f1_hi"]
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    ordered = leading + f1_cols + [c for c in trailing if c in df.columns]
    df = df[[c for c in ordered if c in df.columns]]

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    print(df[["task", "model", "headline_f1", "headline_f1_lo", "headline_f1_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
