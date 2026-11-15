#!/usr/bin/env python3
"""
Pilot: is model error in existing predictions different across the groups a
researcher would compare?

No model is run. Uses predictions already on disk for the 500-item samples.

1. Benoit et al. (2016) UK manifesto sentences, economic and social position
   (codes -2..2, positive = right / conservative), six open-weight models,
   sentences as written.
   - Group contrast: mean code of Conservative sentences minus mean code of
     Labour sentences, from expert mean codes and from model codes. The
     difference between the two is the bias in the party contrast.
   - Conditional gap: the Conservative coefficient in a regression of
     (model code - expert mean) on party, the expert mean and dummies for the
     rounded expert code. This asks whether Conservative sentences get a more
     right-wing code than Labour sentences that experts placed at the same
     position; a model that only stretches the scale has a contrast bias but
     no conditional gap.
   - Both are reported for all sentences, sentences that name a party or
     politician, and sentences that do not.
   - 95% intervals: bootstrap over manifestos.

2. AgoraSpeech (Sermpezis et al. 2026) paragraphs from 2023 Greek election
   speeches, criticism vs political agenda, ten models from the serial panel.
   The prompt shows the speaker's name.
   - Group contrast: share of paragraphs coded criticism for the five
     opposition leaders minus the share for Mitsotakis (prime minister at the
     time), from gold labels and from model labels.
   - Error rates by group: share of gold-agenda paragraphs coded criticism and
     share of gold-criticism paragraphs coded agenda.
   - 95% intervals: bootstrap over speeches.

3. Manifesto contrast against title-hidden expert codes (needs --codings).
   The expert means in (1) come mostly from codings made with the manifesto
   title shown. Benoit et al. recorded `title_shown` for every coding. For
   sample sentences that experts coded in the domain both with and without the
   title, this recomputes the Conservative-minus-Labour contrast bias against
   the mean of title-hidden codes and against the mean of title-shown codes.
   Input: coding_all_long_2014-03-14.csv from "Data - Created/coding_all_long.zip"
   in github.com/kbenoit/CSTA-APSR.

Writes CSVs to output/sidecar/pilot_differential_error_20261114/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(20261114)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "sidecar" / "pilot_differential_error_20261114"
N_BOOT = 2000

BENOIT_RUNS = {
    "Gemma 4 31B": "task_ext_20261002/gemma4_31b_it_qat_w4a16",
    "Qwen3.6 27B": "task_ext_20261002/qwen3_6_27b_fp8",
    "Qwen3.6 35B-A3B": "task_ext_20261002/qwen3_6_35b_a3b_fp8",
    "Qwen3 30B": "task_ext_20261002/qwen3_30b_a3b_instruct_2507_fp8",
    "Qwen3.8 27B": "task_ext_20261018_newopen/qwen3_8_27b_fp8",
    "Nemotron 3.5 Lightning": "task_ext_20261018_newopen/nemotron3_5_lightning_fp8",
}
CODES = {
    "economic": {"very left": -2, "somewhat left": -1, "neither left nor right": 0,
                 "somewhat right": 1, "very right": 2},
    "social": {"very liberal": -2, "somewhat liberal": -1,
               "neither liberal nor conservative": 0,
               "somewhat conservative": 1, "very conservative": 2},
}


def boot(fr, cluster, stat):
    groups = [g for _, g in fr.groupby(cluster)]
    draws = []
    for _ in range(N_BOOT):
        idx = np.random.randint(0, len(groups), len(groups))
        b = pd.concat([groups[i] for i in idx])
        try:
            draws.append(stat(b))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return np.nanpercentile(np.array(draws, dtype=float), [2.5, 97.5], axis=0)


# ---------------------------------------------------------------- Benoit

def benoit_stats(fr):
    con = fr.source_party.eq("Conservatives")
    lab = fr.source_party.eq("Labour")
    if con.sum() < 5 or lab.sum() < 5:
        raise ValueError("too few sentences")
    gold = fr.loc[con, "mean_code_in_domain"].mean() - fr.loc[lab, "mean_code_in_domain"].mean()
    model = fr.loc[con, "code"].mean() - fr.loc[lab, "code"].mean()
    X = pd.get_dummies(fr.gt_code.astype(int), dtype=float)
    X["mean"] = fr.mean_code_in_domain
    X["con"] = con.astype(float)
    X["ld"] = fr.source_party.eq("Liberal Democrats").astype(float)
    y = (fr.code - fr.mean_code_in_domain).to_numpy(float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(float), y, rcond=None)
    cond_gap = beta[list(X.columns).index("con")]
    slope = np.polyfit(fr.mean_code_in_domain, fr.code, 1)[0]
    return np.array([gold, model, model - gold, cond_gap, slope])


def benoit():
    rows = []
    for domain in CODES:
        task = f"benoit_manifesto_{domain}_position"
        data = pd.read_csv(REPO / "data" / f"{task}_masked.csv")
        col = f"pred_{domain}_position"
        for model, run in BENOIT_RUNS.items():
            pred = pd.read_csv(REPO / "output" / "sidecar" / run / "tasks" / f"{task}.csv")
            f = pred[["item_id", col]].merge(data, left_on="item_id", right_on="source_id",
                                             validate="one_to_one")
            assert len(f) == 500
            f["code"] = f[col].map(CODES[domain])
            assert f.code.notna().all()
            subsets = {"all": f, "names a party or politician": f[f.masked_changed],
                       "names no one": f[~f.masked_changed]}
            for label, g in subsets.items():
                est = benoit_stats(g)
                lo, hi = boot(g, "source_manifesto", benoit_stats)
                rows.append({
                    "domain": domain, "model": model, "sentences": label, "n": len(g),
                    "n_con": int(g.source_party.eq("Conservatives").sum()),
                    "n_lab": int(g.source_party.eq("Labour").sum()),
                    "gold_con_minus_lab": est[0], "model_con_minus_lab": est[1],
                    "contrast_bias": est[2], "bias_lo": lo[2], "bias_hi": hi[2],
                    "conditional_gap": est[3], "cond_lo": lo[3], "cond_hi": hi[3],
                    "slope_on_expert_mean": est[4],
                })
    return pd.DataFrame(rows)


def title_stats(fr):
    con = fr.source_party.eq("Conservatives")
    lab = fr.source_party.eq("Labour")
    if con.sum() < 5 or lab.sum() < 5:
        raise ValueError("too few sentences")
    diff = lambda c: fr.loc[con, c].mean() - fr.loc[lab, c].mean()
    hidden, shown, model = diff("ref_hidden"), diff("ref_shown"), diff("code")
    return np.array([hidden, shown, model, model - hidden, model - shown, shown - hidden])


def benoit_title_reference(codings_path):
    codes = pd.read_csv(codings_path, low_memory=False,
                        usecols=["coderid", "sentenceid", "code", "title_shown", "scale", "source"])
    codes = codes[codes.source.eq("Experts") & codes.code.isin([-2, -1, 0, 1, 2])]
    rows = []
    for domain in CODES:
        task = f"benoit_manifesto_{domain}_position"
        data = pd.read_csv(REPO / "data" / f"{task}_masked.csv")
        d = codes[codes.scale.eq(domain.capitalize())]
        ref = d.groupby(["sentenceid", "title_shown"])["code"].mean().unstack()
        ref = ref.rename(columns={"No": "ref_hidden", "Yes": "ref_shown"}).dropna()
        col = f"pred_{domain}_position"
        for model, run in BENOIT_RUNS.items():
            pred = pd.read_csv(REPO / "output" / "sidecar" / run / "tasks" / f"{task}.csv")
            f = pred[["item_id", col]].merge(data, left_on="item_id", right_on="source_id",
                                             validate="one_to_one")
            f["code"] = f[col].map(CODES[domain])
            f = f.merge(ref, left_on="sentenceid", right_index=True, how="inner")
            subsets = {"all": f, "names a party or politician": f[f.masked_changed],
                       "names no one": f[~f.masked_changed]}
            for label, g in subsets.items():
                est = title_stats(g)
                lo, hi = boot(g, "source_manifesto", title_stats)
                rows.append({
                    "domain": domain, "model": model, "sentences": label, "n": len(g),
                    "n_con": int(g.source_party.eq("Conservatives").sum()),
                    "n_lab": int(g.source_party.eq("Labour").sum()),
                    "experts_hidden_con_minus_lab": est[0],
                    "experts_shown_con_minus_lab": est[1],
                    "model_con_minus_lab": est[2],
                    "bias_vs_hidden": est[3], "hidden_lo": lo[3], "hidden_hi": hi[3],
                    "bias_vs_shown": est[4], "shown_lo": lo[4], "shown_hi": hi[4],
                    "experts_shown_minus_hidden": est[5], "sh_lo": lo[5], "sh_hi": hi[5],
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- AgoraSpeech

def agora_stats(fr):
    opp = ~fr.source_politician.eq("Mitsotakis")
    gov = ~opp
    gold = fr.loc[opp, "gold"].mean() - fr.loc[gov, "gold"].mean()
    model = fr.loc[opp, "pred"].mean() - fr.loc[gov, "pred"].mean()
    out = [gold, model, model - gold]
    for grp in (opp, gov):
        agenda = grp & fr.gold.eq(0)
        crit = grp & fr.gold.eq(1)
        out.append(fr.loc[agenda, "pred"].mean())        # agenda coded criticism
        out.append(1 - fr.loc[crit, "pred"].mean())      # criticism coded agenda
    out.append(out[3] - out[5])                          # gap, agenda coded criticism
    out.append(out[4] - out[6])                          # gap, criticism coded agenda
    return np.array(out, dtype=float)


def agora():
    task = "agoraspeech_criticism_agenda"
    data = pd.read_csv(REPO / "data" / f"{task}.csv",
                       usecols=["source_id", "source_politician", "source_speech_id"])
    cols = ["task", "model", "item_id", "parse_error",
            "pred_criticism_or_agenda", "gt_criticism_or_agenda"]
    pred = pd.read_csv(REPO / "output" / "predictions.csv", usecols=cols, low_memory=False)
    pred = pred[pred.task.eq(task)]
    rows = []
    for model, p in pred.groupby("model"):
        f = p.merge(data, left_on="item_id", right_on="source_id", validate="one_to_one")
        assert len(f) == 500
        n_unparsed = int(f.pred_criticism_or_agenda.isna().sum())
        f = f[f.pred_criticism_or_agenda.notna()].copy()
        f["pred"] = f.pred_criticism_or_agenda.eq("criticism").astype(float)
        f["gold"] = f.gt_criticism_or_agenda.eq("criticism").astype(float)
        est = agora_stats(f)
        lo, hi = boot(f, "source_speech_id", agora_stats)
        rows.append({
            "model": model, "n": len(f), "n_unparsed": n_unparsed,
            "n_mitsotakis": int(f.source_politician.eq("Mitsotakis").sum()),
            "accuracy": float((f.pred == f.gold).mean()),
            "gold_opp_minus_gov": est[0], "model_opp_minus_gov": est[1],
            "contrast_bias": est[2], "bias_lo": lo[2], "bias_hi": hi[2],
            "opp_agenda_coded_criticism": est[3], "opp_criticism_coded_agenda": est[4],
            "gov_agenda_coded_criticism": est[5], "gov_criticism_coded_agenda": est[6],
            "gap_agenda_coded_criticism": est[7], "gap_ac_lo": lo[7], "gap_ac_hi": hi[7],
            "gap_criticism_coded_agenda": est[8], "gap_ca_lo": lo[8], "gap_ca_hi": hi[8],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codings", help="coding_all_long_2014-03-14.csv from kbenoit/CSTA-APSR")
    ap.add_argument("--only-title", action="store_true", help="run only part 3")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    if args.codings:
        t = benoit_title_reference(args.codings)
        t.to_csv(OUT / "benoit_party_contrast_title_reference.csv", index=False)
        print(t.round(3).to_string(index=False))
        print()
    if args.only_title:
        return
    b = benoit()
    b.to_csv(OUT / "benoit_party_contrast.csv", index=False)
    a = agora()
    a.to_csv(OUT / "agoraspeech_government_contrast.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(b.round(3).to_string(index=False))
    print()
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
