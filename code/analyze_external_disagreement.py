#!/usr/bin/env python3
"""Estimate whether newer-model gains rise with within-dataset human agreement."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20260822
BOOTSTRAPS = 10_000
EXTERNAL_TASKS = {
    "hatexplain_votes": "HateXplain",
    "measuring_hate_speech_votes": "Measuring Hate Speech",
    "mfrc_votes": "Moral Foundations Reddit",
}


def external_items(data_dir: Path, llama_path: Path, qwen_path: Path) -> pd.DataFrame:
    llama = pd.read_csv(llama_path); qwen = pd.read_csv(qwen_path)
    for model, frame in [("Llama 3.1", llama), ("Qwen3.6", qwen)]:
        if frame.duplicated(["task", "item_id"]).any():
            raise ValueError(f"duplicate task/item keys in {model} predictions")
        if frame["parse_error"].notna().any():
            raise ValueError(f"malformed response in {model} predictions")
    rows = []
    for task, display in EXTERNAL_TASKS.items():
        source_name = {"hatexplain_votes": "hatexplain_votes.csv", "measuring_hate_speech_votes": "measuring_hate_speech_votes.csv", "mfrc_votes": "mfrc_votes.csv"}[task]
        source = pd.read_csv(data_dir / source_name).set_index("source_id")
        left = llama.loc[llama.task.eq(task)].set_index("item_id")
        right = qwen.loc[qwen.task.eq(task)].set_index("item_id")
        if set(source.index) != set(left.index) or set(source.index) != set(right.index):
            raise ValueError(f"prediction/source coverage mismatch for {task}")
        if task == "mfrc_votes":
            labels = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity", "Thin Morality", "Non-Moral"]
            old = np.mean([left[f"pred_{x}"].astype(str).str.lower().eq(left[f"gt_{x}"].astype(str).str.lower()).to_numpy() for x in labels], axis=0)
            new = np.mean([right[f"pred_{x}"].astype(str).str.lower().eq(right[f"gt_{x}"].astype(str).str.lower()).to_numpy() for x in labels], axis=0)
        else:
            old = left["pred_label"].eq(left["gt_label"]).astype(float).to_numpy()
            new = right["pred_label"].eq(right["gt_label"]).astype(float).to_numpy()
        frame = source.loc[left.index, ["vote_share_majority"]].reset_index().rename(columns={"source_id": "item_id", "vote_share_majority": "agreement"})
        frame["dataset"] = display; frame["older_score"] = old; frame["newer_score"] = new; frame["delta"] = new - old
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def legacy_items(root: Path) -> pd.DataFrame:
    rows = []
    for filename, display in [("semeval_item_human_agreement.csv", "SemEval stance"), ("alia_item_human_agreement.csv", "ALIA civic stance")]:
        frame = pd.read_csv(root / filename)
        rows.append(pd.DataFrame({"item_id": frame.get("item_id", frame.get("source_id")), "dataset": display,
                                  "agreement": frame.vote_share_majority, "older_score": frame.llama_correct.astype(float),
                                  "newer_score": frame.qwen_correct.astype(float),
                                  "delta": frame.qwen_correct.astype(float) - frame.llama_correct.astype(float)}))
    return pd.concat(rows, ignore_index=True)


def slope(frame: pd.DataFrame) -> float:
    z = (frame.agreement - frame.agreement.mean()) / frame.agreement.std(ddof=0)
    return float(np.dot(z, frame.delta - frame.delta.mean()) / np.dot(z, z))


def estimate(items: pd.DataFrame, draws: int = BOOTSTRAPS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed); datasets = sorted(items.dataset.unique()); boot = {d: [] for d in datasets}
    rows = []
    for dataset in datasets:
        group = items.loc[items.dataset.eq(dataset)].reset_index(drop=True)
        point = slope(group)
        for _ in range(draws):
            sample = group.iloc[rng.integers(0, len(group), len(group))]
            boot[dataset].append(slope(sample))
        values = np.asarray(boot[dataset])
        q25, q75 = group.agreement.quantile([.25, .75])
        rows.append({"dataset": dataset, "n_items": len(group), "slope_per_agreement_sd": point,
                     "ci_low": np.percentile(values, 2.5), "ci_high": np.percentile(values, 97.5),
                     "older_score_mean": group["older_score"].mean() if "older_score" in group else np.nan,
                     "newer_score_mean": group["newer_score"].mean() if "newer_score" in group else np.nan,
                     "mean_newer_minus_older": group.delta.mean(),
                     "low_agreement_delta": group.loc[group.agreement <= q25, "delta"].mean(),
                     "high_agreement_delta": group.loc[group.agreement >= q75, "delta"].mean()})
    points = np.array([row["slope_per_agreement_sd"] for row in rows])
    pooled_draws = np.mean(np.column_stack([boot[d] for d in datasets]), axis=1)
    rows.append({"dataset": "Equal-dataset pooled", "n_items": len(items), "slope_per_agreement_sd": points.mean(),
                 "ci_low": np.percentile(pooled_draws, 2.5), "ci_high": np.percentile(pooled_draws, 97.5),
                 "older_score_mean": items.groupby("dataset").older_score.mean().mean() if "older_score" in items else np.nan,
                 "newer_score_mean": items.groupby("dataset").newer_score.mean().mean() if "newer_score" in items else np.nan,
                 "mean_newer_minus_older": items.groupby("dataset").delta.mean().mean(),
                 "low_agreement_delta": np.nan, "high_agreement_delta": np.nan})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path) -> None:
    individual = summary.loc[summary.dataset.ne("Equal-dataset pooled")].copy()
    order = individual.sort_values("slope_per_agreement_sd").dataset.tolist() + ["Equal-dataset pooled"]
    shown = summary.set_index("dataset").loc[order].reset_index(); y = np.arange(len(shown))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; ax.axvline(0, color="0.75", lw=.8)
    for i, row in shown.iterrows():
        dark = row.dataset == "Equal-dataset pooled"; color = "#146c94" if dark else "0.25"
        ax.plot([100*row.ci_low, 100*row.ci_high], [i, i], color=color, lw=2 if dark else 1)
        ax.scatter(100*row.slope_per_agreement_sd, i, color=color, s=38 if dark else 22, zorder=3)
    ax.set_yticks(y, shown.dataset); ax.set_xlabel("Newer-model advantage per 1 SD higher human agreement\n(percentage points)")
    ax.set_title("Within-dataset relationship", loc="left", fontweight="bold")
    ax = axes[1]; ax.axvline(0, color="0.75", lw=.8)
    for i, row in individual.set_index("dataset").loc[order[:-1]].reset_index().iterrows():
        lo, hi = 100*row.low_agreement_delta, 100*row.high_agreement_delta
        ax.plot([lo, hi], [i, i], color="0.7", lw=1); ax.scatter(lo, i, facecolor="white", edgecolor="0.25", s=28); ax.scatter(hi, i, color="#146c94", s=28)
    ax.set_yticks(range(len(order)-1), order[:-1]); ax.set_xlabel("Qwen3.6 minus Llama 3.1 score (points)")
    ax.set_title("Agreement quartiles", loc="left", fontweight="bold")
    ax.legend(handles=[plt.Line2D([],[],marker='o',mfc='white',mec='0.25',ls='',label='Lowest'), plt.Line2D([],[],marker='o',color='#146c94',ls='',label='Highest')], frameon=False, ncol=2, loc="lower right")
    for ax in axes:
        ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y', length=0); ax.grid(False)
    fig.suptitle("Does human agreement predict newer-model improvement?", x=.07, ha="left", fontsize=14, fontweight="bold")
    fig.text(.07, .01, "Paired item comparisons; 10,000 within-dataset bootstrap draws. Positive values favor Qwen3.6 more as agreement rises.", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, .04, 1, .94]); fig.savefig(output.with_suffix('.png'), dpi=220); fig.savefig(output.with_suffix('.pdf'))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--external-root',type=Path,required=True); p.add_argument('--human-votes-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    items=pd.concat([external_items(Path('data/human_votes/external'), a.external_root/'llama3_1_70b_external_votes'/'predictions.csv', a.external_root/'qwen3_6_27b_external_votes'/'predictions.csv'), legacy_items(a.human_votes_root)],ignore_index=True)
    a.output_dir.mkdir(parents=True,exist_ok=True); summary=estimate(items); items.to_csv(a.output_dir/'item_disagreement_effects.csv',index=False); summary.to_csv(a.output_dir/'disagreement_effect_summary.csv',index=False); plot(summary,a.output_dir/'human_agreement_improvement')

if __name__=='__main__': main()
