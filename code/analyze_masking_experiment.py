#!/usr/bin/env python3
"""
Party-masking experiment: does naming a party or politician in the passage
shift the position a model assigns?

Compares each item's prediction in the masked run (task_ext_20261010_masked,
names replaced by [PARTY] / [POLITICIAN]) with its prediction for the identical
item in the unmasked run (task_ext_20261002_position). Items whose passage
contained no name are word-for-word identical in both runs and measure
run-to-run noise (the placebo).

Benoit manifesto tasks (codes -2..2, positive = right / conservative):
- change = masked code minus unmasked code, per item.
- Main estimate: mean change for Conservative sentences minus mean change for
  Labour sentences, among passages that named someone. If names pushed
  Conservative sentences right and Labour sentences left, masking moves them
  back and this is negative. The same contrast on unchanged passages is the
  placebo. 95% intervals from a bootstrap over manifestos.
- Also the Conservative-minus-Labour gap in error (model minus expert mean,
  with expert modal-code fixed effects) before and after masking.

Le Mens-Gallego tweets (0-100, NA scored as 50):
- Mean change for Republican minus Democratic tweets, changed vs unchanged
  tweets, item bootstrap; NA rates by party before and after masking.

Writes CSVs to output/sidecar/task_ext_20261010_masked/analysis/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(20261010)
REPO = Path(__file__).resolve().parents[1]
ORIG = REPO / "output" / "sidecar" / "task_ext_20261002"
MASK = REPO / "output" / "sidecar" / "task_ext_20261010_masked"
OUT = MASK / "analysis"
MODELS = {
    "gemma4_31b_it_qat_w4a16": "Gemma 4 31B",
    "qwen3_6_27b_fp8": "Qwen3.6 27B",
    "qwen3_6_35b_a3b_fp8": "Qwen3.6 35B-A3B",
    "qwen3_30b_a3b_instruct_2507_fp8": "Qwen3 30B",
}
CODES = {
    "economic": {"very left": -2, "somewhat left": -1, "neither left nor right": 0,
                 "somewhat right": 1, "very right": 2},
    "social": {"very liberal": -2, "somewhat liberal": -1,
               "neither liberal nor conservative": 0,
               "somewhat conservative": 1, "very conservative": 2},
}
N_BOOT = 2000


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def benoit_items(domain, model_key):
    task = f"benoit_manifesto_{domain}_position"
    data = pd.read_csv(REPO / "data" / f"{task}_masked.csv")
    col = f"pred_{domain}_position"
    orig = pd.read_csv(ORIG / model_key / "tasks" / f"{task}.csv")[["item_id", col]]
    mask = pd.read_csv(MASK / model_key / "tasks" / f"{task}_masked.csv")[["item_id", col]]
    f = (orig.merge(mask, on="item_id", suffixes=("_orig", "_mask"), validate="one_to_one")
         .merge(data, left_on="item_id", right_on="source_id", validate="one_to_one"))
    assert len(f) == 500
    f["code_orig"] = f[f"{col}_orig"].map(CODES[domain])
    f["code_mask"] = f[f"{col}_mask"].map(CODES[domain])
    assert f[["code_orig", "code_mask"]].notna().all().all()
    f["change"] = f.code_mask - f.code_orig
    return f


def con_lab_change(fr):
    con = fr.loc[fr.source_party == "Conservatives", "change"].mean()
    lab = fr.loc[fr.source_party == "Labour", "change"].mean()
    return con - lab


def con_lab_error_gap(fr, code_col):
    X = pd.get_dummies(fr.gt_code.astype(int), dtype=float)
    X["mean"] = fr.mean_code_in_domain
    X["con"] = (fr.source_party == "Conservatives").astype(float)
    X["ld"] = (fr.source_party == "Liberal Democrats").astype(float)
    y = (fr[code_col] - fr.mean_code_in_domain).to_numpy(float)
    return ols(y, X.to_numpy(float))[-2]


def cluster_boot(fr, stat):
    groups = {k: g for k, g in fr.groupby("source_manifesto")}
    keys = list(groups)
    draws = []
    for _ in range(N_BOOT):
        b = pd.concat([groups[k] for k in np.random.choice(keys, len(keys))])
        if {"Conservatives", "Labour"} <= set(b.source_party):
            draws.append(stat(b))
    return np.nanpercentile(draws, [2.5, 97.5])


def benoit_results():
    rows = []
    for domain in CODES:
        for model_key, model in MODELS.items():
            f = benoit_items(domain, model_key)
            for named, g in f.groupby("masked_changed"):
                lo, hi = cluster_boot(g, con_lab_change)
                rows.append({
                    "domain": domain, "model": model,
                    "passage": "named" if named else "unchanged (placebo)",
                    "n": len(g),
                    "share_code_changed": (g.change != 0).mean(),
                    "con_minus_lab_change": con_lab_change(g), "lo": lo, "hi": hi,
                    "mean_change_con": g.loc[g.source_party == "Conservatives", "change"].mean(),
                    "mean_change_lab": g.loc[g.source_party == "Labour", "change"].mean(),
                    "mean_change_ld": g.loc[g.source_party == "Liberal Democrats", "change"].mean(),
                    "con_lab_error_gap_orig": con_lab_error_gap(g, "code_orig"),
                    "con_lab_error_gap_mask": con_lab_error_gap(g, "code_mask"),
                })
    return pd.DataFrame(rows)


def tweet_results():
    task = "lemens_gallego_tweet_position"
    data = pd.read_csv(REPO / "data" / f"{task}_masked.csv")
    rows = []
    for model_key, model in MODELS.items():
        cols = ["item_id", "pred_score", "parse_error"]
        orig = pd.read_csv(ORIG / model_key / "tasks" / f"{task}.csv")[cols]
        mask = pd.read_csv(MASK / model_key / "tasks" / f"{task}_masked.csv")[cols]
        f = (orig.merge(mask, on="item_id", suffixes=("_orig", "_mask"), validate="one_to_one")
             .merge(data, left_on="item_id", right_on="source_id", validate="one_to_one"))
        for v in ("orig", "mask"):
            f[f"na_{v}"] = f[f"parse_error_{v}"].astype(str).eq("not_applicable")
            f[f"score_{v}"] = f[f"pred_score_{v}"].where(~f[f"na_{v}"], 50.0)
        f["change"] = f.score_mask - f.score_orig
        f["rep"] = f.source_party == "Republican Party"
        stat = lambda g: g.loc[g.rep, "change"].mean() - g.loc[~g.rep, "change"].mean()
        for named, g in f.groupby("masked_changed"):
            boot = [stat(g.sample(len(g), replace=True)) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "model": model, "passage": "named" if named else "unchanged (placebo)",
                "n": len(g), "n_rep": int(g.rep.sum()),
                "rep_minus_dem_change": stat(g), "lo": lo, "hi": hi,
                "mean_change_rep": g.loc[g.rep, "change"].mean(),
                "mean_change_dem": g.loc[~g.rep, "change"].mean(),
                "na_rate_orig_rep": g.loc[g.rep, "na_orig"].mean(),
                "na_rate_mask_rep": g.loc[g.rep, "na_mask"].mean(),
                "na_rate_orig_dem": g.loc[~g.rep, "na_orig"].mean(),
                "na_rate_mask_dem": g.loc[~g.rep, "na_mask"].mean(),
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, frame in {"benoit_masking": benoit_results(), "tweet_masking": tweet_results()}.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
