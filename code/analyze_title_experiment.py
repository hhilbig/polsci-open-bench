#!/usr/bin/env python3
"""
Manifesto title shown vs hidden: how much does knowing which party wrote a
sentence move its left-right code, for models and for people?

Measure: for each sentence, code with the title shown minus code with it hidden
(steps on the 5-point scale, -2..2). The party effect is the mean shift for
Conservative sentences minus the mean shift for Labour sentences. Positive means
that knowing the party pushes the two parties' sentences apart.

People (Benoit et al. 2016, coding_all_long_2014-03-14.csv):
- experts: same coder and sentence, sequential stage (title shown) vs random
  stage (title hidden), averaged per sentence;
- semi-experts: randomized title, per sentence mean(shown) - mean(hidden).
Models: tasks_party_title runs (one score per sentence and condition, temperature 0).

With --precision, also reports how wide the 95% interval of each model's effect
would be on random subsets of the economic expert-pair sentences, to size a
paid run on the commercial models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED = 20261019
CODE = {
    "economic": {"very left": -2, "somewhat left": -1, "neither left nor right": 0,
                 "somewhat right": 1, "very right": 2},
    "social": {"very liberal": -2, "somewhat liberal": -1, "neither liberal nor conservative": 0,
               "somewhat conservative": 1, "very conservative": 2},
}
SCALE = {"economic": "Economic", "social": "Social"}


def party_effect(shift: pd.DataFrame) -> float:
    m = shift.groupby("party")["shift"].mean()
    return m["Conservatives"] - m["Labour"]


def boot_ci(shift: pd.DataFrame, rng, reps=2000):
    con = shift.loc[shift.party.eq("Conservatives"), "shift"].to_numpy()
    lab = shift.loc[shift.party.eq("Labour"), "shift"].to_numpy()
    draws = [rng.choice(con, len(con)).mean() - rng.choice(lab, len(lab)).mean() for _ in range(reps)]
    return np.percentile(draws, [2.5, 97.5])


def human_shifts(codings: Path, domain: str, data: pd.DataFrame) -> dict:
    c = pd.read_csv(codings, low_memory=False,
                    usecols=["coderid", "sentenceid", "code", "title_shown", "scale", "party", "source"])
    c = c[c.scale.eq(SCALE[domain]) & c.code.isin([-2, -1, 0, 1, 2]) & c.sentenceid.isin(data.sentenceid)]
    out = {}
    e = c[c.source.eq("Experts")].groupby(["coderid", "sentenceid", "title_shown"], as_index=False).first()
    w = e.pivot_table(index=["coderid", "sentenceid"], columns="title_shown", values="code").dropna()
    w = (w["Yes"] - w["No"]).rename("shift").reset_index().groupby("sentenceid", as_index=False)["shift"].mean()
    out["Experts (same coder, a year apart)"] = w
    s = c[c.source.eq("SemiExperts")].groupby(["sentenceid", "title_shown"])["code"].mean().unstack().dropna()
    out["Semi-experts (randomized)"] = (s["Yes"] - s["No"]).rename("shift").reset_index()
    return out


def model_shifts(run_dir: Path, domain: str) -> dict:
    out = {}
    for model_dir in sorted(p for p in run_dir.iterdir() if (p / "tasks").is_dir()):
        frames = {}
        for cond in ["hidden", "shown"]:
            f = model_dir / "tasks" / f"benoit_title_{domain}_position_{cond}.csv"
            if not f.exists():
                break
            d = pd.read_csv(f)
            frames[cond] = d.set_index("item_id")[f"pred_{domain}_position"].map(CODE[domain])
        if len(frames) < 2:
            continue
        both = pd.concat(frames, axis=1).dropna()
        out[model_dir.name] = pd.DataFrame({
            "sentenceid": both.index.str.replace("benoit_", "").astype(int),
            "shift": (both["shown"] - both["hidden"]).to_numpy()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codings", required=True, help="coding_all_long_2014-03-14.csv")
    ap.add_argument("--run-dir", default=str(REPO / "output/sidecar/task_ext_20261019_title"))
    ap.add_argument("--precision", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for domain in ["economic", "social"]:
        data = pd.read_csv(REPO / "data" / f"benoit_title_{domain}_position.csv")
        party = data.set_index("sentenceid")["source_party"]
        sets = {"expert pairs": set(data.sentenceid[data.in_expert_pairs]),
                "semi-expert experiment": set(data.sentenceid[data.in_semi_expert_experiment])}
        coders = {**human_shifts(Path(args.codings), domain, data), **model_shifts(Path(args.run_dir), domain)}
        for who, shift in coders.items():
            shift = shift.assign(party=shift.sentenceid.map(party))
            for set_name, ids in sets.items():
                x = shift[shift.sentenceid.isin(ids)]
                if x.party.nunique() < 2:
                    continue
                lo, hi = boot_ci(x, rng)
                rows.append({"domain": domain, "sentences": set_name, "coder": who, "n": len(x),
                             "share_moved": round((x["shift"] != 0).mean(), 3),
                             "con_minus_lab_shift": round(party_effect(x), 3),
                             "lo": round(lo, 3), "hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    res.to_csv(Path(args.run_dir) / "title_effects.csv", index=False)

    if args.precision:
        data = pd.read_csv(REPO / "data" / "benoit_title_economic_position.csv")
        party = data.set_index("sentenceid")["source_party"]
        ids = data.sentenceid[data.in_expert_pairs].to_numpy()
        prec = []
        for model, shift in model_shifts(Path(args.run_dir), "economic").items():
            shift = shift.assign(party=shift.sentenceid.map(party))
            shift = shift[shift.sentenceid.isin(ids)]
            for k in [200, 300, 400, 600, 1000]:
                widths, ests = [], []
                for _ in range(20):
                    x = shift.sample(k, random_state=int(rng.integers(1e9)))
                    lo, hi = boot_ci(x, rng, reps=500)
                    widths.append((hi - lo) / 2)
                    ests.append(party_effect(x))
                prec.append({"model": model, "k": k, "full_effect": round(party_effect(shift), 3),
                             "median_half_width": round(float(np.median(widths)), 3),
                             "est_range": f"{min(ests):.2f} to {max(ests):.2f}"})
        print(pd.DataFrame(prec).to_string(index=False))


if __name__ == "__main__":
    main()
