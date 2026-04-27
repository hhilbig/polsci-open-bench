#!/usr/bin/env python3
"""
Build output/summary.csv from output/predictions.csv.

Per-(task, model) row includes headline_f1, accuracy, parse_err_rate,
latency percentiles, GPU-hours-per-1000-items (Ollama only), and 95%
paired-bootstrap CIs on the headline metric (1000 iterations, paired
by item across models so the CIs support model-vs-model comparison).

Usage:
  python3 code/build_summary.py           # reads output/predictions.csv, writes output/summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

from benchmark import TASKS  # reuse the canonical task definitions

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20260424
OLLAMA_MODELS = {
    "gemma4:31b-it-q4_K_M",
    "qwen3:14b-q4_K_M",
    "qwen3:30b-a3b-q4_K_M",
    "mistral-small:24b-instruct-2501-q4_K_M",
}

# Observed average $/call for OpenAI models in the v2 (2026-04-25) run.
# Ground truth: /v1/organization/costs ledger entries for the v2 grid only,
# divided by call count (5,000 per model = 10 tasks × 500 items).
# Use as observed averages for cost_per_1000_correct estimates.
USD_PER_CALL = {
    # v2 medium reasoning, ground truth from cost ledger:
    "gpt-5.5":      0.00433,    # $21.66 / 5,000 calls
    "gpt-5.4-nano": 0.00024,    # $1.20 / 5,000 calls
    # not in v2 grid (historical / placeholder):
    "gpt-5.4":      0.00300,
    "gpt-5.4-mini": 0.00100,
}


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
        # MCC handles class imbalance better than macro F1; same data + mask as accuracy.
        try:
            row["mcc"] = (
                matthews_corrcoef(sub[gt_col], sub[pred_col]) if len(sub) else np.nan
            )
        except ValueError:
            row["mcc"] = np.nan
        row["headline_f1"] = row["avg_f1"]

    else:
        raise ValueError(f"Unknown label_kind: {kind}")
    return row


def _bootstrap_cis(preds, task_defs):
    """Paired-by-item bootstrap on headline F1 per (task, model). Returns dict
    keyed by (task, model) -> (low, high). Optimized: pre-extract numpy arrays
    once per cell, then index into them inside the bootstrap loop."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for task, gtask in preds.groupby("task"):
        if task not in task_defs:
            continue
        td = task_defs[task]
        kind = td["label_kind"]
        labels = td["labels"]
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
                    boot_f1[m][b] = f1_score(sg, sp, labels=labels,
                                             average="macro", zero_division=0)
        for m, arr in boot_f1.items():
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                out[(task, m)] = (np.nan, np.nan)
            else:
                out[(task, m)] = (float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5)))
    return out


def main():
    preds = pd.read_csv(OUT / "predictions.csv")
    task_defs = {t["name"]: t for t in TASKS}
    rows = []
    for (task, model), g in preds.groupby(["task", "model"]):
        if task not in task_defs:
            print(f"[skip] unknown task in predictions: {task}")
            continue
        r = {"task": task, "model": model, **_metrics_for_group(task_defs[task], g)}
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)

    # Bootstrap CIs (paired-by-item, 1000 iters, 95%)
    print(f"[bootstrap] computing {BOOTSTRAP_ITERS}-iter paired CIs ...")
    cis = _bootstrap_cis(preds, task_defs)
    df["headline_f1_lo"] = df.apply(lambda r: cis.get((r["task"], r["model"]), (np.nan, np.nan))[0], axis=1)
    df["headline_f1_hi"] = df.apply(lambda r: cis.get((r["task"], r["model"]), (np.nan, np.nan))[1], axis=1)

    # GPU-hours per 1000 (Ollama only; OpenAI gets NaN)
    df["gpu_hours_per_1000"] = df.apply(
        lambda r: (r["median_latency_s"] * 1000 / 3600) if r["model"] in OLLAMA_MODELS else np.nan,
        axis=1,
    )
    # USD per 1000 calls (OpenAI only; Ollama gets NaN). Uses observed averages
    # from /v1/organization/costs in the v2 run (see USD_PER_CALL dict at the top).
    df["usd_per_1000"] = df.apply(
        lambda r: (USD_PER_CALL[r["model"]] * 1000) if r["model"] in USD_PER_CALL else np.nan,
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

    out_path = OUT / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    print(df[["task", "model", "headline_f1", "headline_f1_lo", "headline_f1_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
