#!/usr/bin/env python3
"""
Aggregate validity and directional bias of the four open-weight models on the
ideological-position extension tasks (run task_ext_20261002).

1. Manifesto-level validity (Benoit et al. 2016 sentences). Each model's codes
   are averaged within each of the 18 manifestos and correlated with (a) the
   expert average over the same sampled sentences and (b) the expert average
   over every coded sentence of that manifesto. The expert subsample's own
   correlation with (b) is the ceiling set by sampling only ~28 sentences.
2. Directional bias.
   - Benoit: signed error (model code minus expert modal code) split into a
     uniform shift (positive = rightward/conservative) and a pull toward the
     centre, plus the party gap in signed error at the same expert code
     (Conservative minus Labour), with a bootstrap over manifestos. A second
     version also controls for the expert mean code, since within one modal
     code Conservative sentences may sit further right in the experts' own
     averages.
   - Manifesto-level correlation within each party across the six elections,
     because the pooled correlation is mostly driven by separating parties.
   - Le Mens-Gallego tweets: signed error (model score minus mean human score)
     regressed on the human score and author party, plus NA rates by party and
     the Republican-Democrat gap in average placement.

Writes CSVs to output/sidecar/task_ext_20261002/analysis/ and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(20261010)
REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "output" / "sidecar" / "task_ext_20261002"
OUT = RUN / "analysis"
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


def benoit_frame(domain, model_key):
    data = pd.read_csv(REPO / "data" / f"benoit_manifesto_{domain}_position.csv")
    preds = pd.read_csv(RUN / model_key / "tasks" / f"benoit_manifesto_{domain}_position.csv")
    frame = preds[["item_id", f"pred_{domain}_position"]].merge(
        data, left_on="item_id", right_on="source_id", validate="one_to_one")
    frame["pred_code"] = frame[f"pred_{domain}_position"].map(CODES[domain])
    assert frame.pred_code.notna().all(), "unmapped prediction label"
    frame["error"] = frame.pred_code - frame.gt_code
    return data, frame


def manifesto_validity():
    rows = []
    for domain in CODES:
        for model_key, model in MODELS.items():
            data, frame = benoit_frame(domain, model_key)
            full = data.groupby("source_manifesto").mean_code_in_domain.mean()
            agg = frame.groupby("source_manifesto").agg(
                model=("pred_code", "mean"), expert_sample=("mean_code_in_domain", "mean"),
                n=("pred_code", "size"))
            agg["expert_full"] = full
            rows.append({
                "domain": domain, "model": model, "n_manifestos": len(agg),
                "sentences_per_manifesto": agg.n.median(),
                "r_sentence_level": np.corrcoef(frame.pred_code, frame.mean_code_in_domain)[0, 1],
                "r_model_vs_expert_same_sentences": agg[["model", "expert_sample"]].corr().iloc[0, 1],
                "r_model_vs_expert_all_sentences": agg[["model", "expert_full"]].corr().iloc[0, 1],
                "r_ceiling_expert_sample_vs_all": agg[["expert_sample", "expert_full"]].corr().iloc[0, 1],
                "r_party_means_model_vs_expert": frame.groupby("source_party")[["pred_code", "mean_code_in_domain"]]
                    .mean().corr().iloc[0, 1],
            })
    return pd.DataFrame(rows)


def within_party_validity():
    rows = []
    for domain in CODES:
        for model_key, model in MODELS.items():
            data, frame = benoit_frame(domain, model_key)
            full = data.groupby("source_manifesto").mean_code_in_domain.mean()
            agg = frame.groupby(["source_party", "source_manifesto"]).agg(
                model=("pred_code", "mean"),
                expert_sample=("mean_code_in_domain", "mean")).reset_index()
            agg["expert_full"] = agg.source_manifesto.map(full)
            for party, g in agg.groupby("source_party"):
                rows.append({
                    "domain": domain, "model": model, "party": party, "n_manifestos": len(g),
                    "r_model_vs_expert_all_sentences": g[["model", "expert_full"]].corr().iloc[0, 1],
                    "r_ceiling_expert_sample_vs_all": g[["expert_sample", "expert_full"]].corr().iloc[0, 1],
                })
    return pd.DataFrame(rows)


def benoit_bias():
    rows = []
    for domain in CODES:
        for model_key, model in MODELS.items():
            _, f = benoit_frame(domain, model_key)
            left, right, mid = f[f.gt_code < 0], f[f.gt_code > 0], f[f.gt_code == 0]

            def split(fr):
                el = fr.loc[fr.gt_code < 0, "error"].mean()
                er = fr.loc[fr.gt_code > 0, "error"].mean()
                return (el + er) / 2, (el - er) / 2

            def party_gap(fr):
                # Signed error on party dummies with expert-code fixed effects.
                X = pd.get_dummies(fr.gt_code.astype(int), prefix="g", dtype=float)
                X["con"] = (fr.source_party == "Conservatives").astype(float)
                X["ld"] = (fr.source_party == "Liberal Democrats").astype(float)
                return ols(fr.error.to_numpy(float), X.to_numpy())[-2]

            def party_gap_mean_control(fr):
                X = pd.get_dummies(fr.gt_code.astype(int), prefix="g", dtype=float)
                X["mean"] = fr.mean_code_in_domain
                X["con"] = (fr.source_party == "Conservatives").astype(float)
                X["ld"] = (fr.source_party == "Liberal Democrats").astype(float)
                y = (fr.pred_code - fr.mean_code_in_domain).to_numpy(float)
                return ols(y, X.to_numpy(float))[-2]

            shift, pull = split(f)
            gap = party_gap(f)
            gap_mean = party_gap_mean_control(f)
            manifestos = f.source_manifesto.unique()
            groups = {m: g for m, g in f.groupby("source_manifesto")}
            boot = []
            for _ in range(N_BOOT):
                b = pd.concat([groups[m] for m in np.random.choice(manifestos, len(manifestos))])
                if b.source_party.nunique() < 3:
                    continue
                boot.append((*split(b), party_gap(b), party_gap_mean_control(b)))
            boot = np.array(boot)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            rows.append({
                "domain": domain, "model": model,
                "shift": shift, "shift_lo": lo[0], "shift_hi": hi[0],
                "pull_to_centre": pull, "pull_lo": lo[1], "pull_hi": hi[1],
                "neutral_items_mean_error": mid.error.mean(),
                "con_minus_lab_error": gap, "gap_lo": lo[2], "gap_hi": hi[2],
                "con_minus_lab_error_mean_control": gap_mean,
                "gap_mc_lo": lo[3], "gap_mc_hi": hi[3],
                "n_left": len(left), "n_right": len(right), "n_neutral": len(mid),
            })
    return pd.DataFrame(rows)


def tweet_bias():
    data = pd.read_csv(REPO / "data" / "lemens_gallego_tweet_position.csv")
    rows = []
    for model_key, model in MODELS.items():
        preds = pd.read_csv(RUN / model_key / "tasks" / "lemens_gallego_tweet_position.csv")
        f = preds[["item_id", "pred_score", "parse_error"]].merge(
            data, left_on="item_id", right_on="source_id", validate="one_to_one")
        f["rep"] = (f.source_party == "Republican Party").astype(float)
        f["na"] = f.parse_error.astype(str).eq("not_applicable")
        f["pred_mid"] = f.pred_score.where(~f.na, 50.0)
        ans = f[~f.na].copy()
        ans["error"] = ans.pred_score - ans.human_position_mean

        def fit(fr):
            X = np.column_stack([np.ones(len(fr)), fr.human_position_mean - 50, fr.rep])
            return ols(fr.error.to_numpy(float), X)

        beta = fit(ans)
        boot = np.array([fit(ans.sample(len(ans), replace=True)) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        gap = lambda col, fr: fr.loc[fr.rep == 1, col].mean() - fr.loc[fr.rep == 0, col].mean()
        rows.append({
            "model": model,
            "human_na_share_dem": (f.n_not_applicable / f.n_raters)[f.rep == 0].mean(),
            "human_na_share_rep": (f.n_not_applicable / f.n_raters)[f.rep == 1].mean(),
            "na_rate_dem": f.loc[f.rep == 0, "na"].mean(),
            "na_rate_rep": f.loc[f.rep == 1, "na"].mean(),
            "slope_error_on_human": beta[1], "slope_lo": lo[1], "slope_hi": hi[1],
            "rep_minus_dem_error": beta[2], "gap_lo": lo[2], "gap_hi": hi[2],
            "rep_dem_gap_model_na50": gap("pred_mid", f),
            "rep_dem_gap_human": gap("human_position_mean", f),
            "n_answered": len(ans),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    results = {
        "manifesto_validity": manifesto_validity(),
        "within_party_validity": within_party_validity(),
        "benoit_bias": benoit_bias(),
        "tweet_bias": tweet_bias(),
    }
    for name, frame in results.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
