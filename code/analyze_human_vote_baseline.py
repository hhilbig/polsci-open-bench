#!/usr/bin/env python3
"""Compare Llama and Qwen with released human annotation distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from task_registry import _v1_v2_indices  # noqa: E402

ROOT = REPO / "output" / "sidecar" / "frontier_2026" / "human_votes"
LLAMA_FULL = REPO / "output" / "sidecar" / "frontier_2026" / "broad18" / "holdout16" / "llama3_1_70b_instruct_fp8_dynamic_full34" / "predictions.csv"
QWEN_FULL = REPO / "output" / "sidecar" / "hive_model_bakeoff_20260804" / "qwen3_6_27b_fp8" / "predictions.csv"
ALIA_DATA = REPO / "data" / "human_votes" / "alia_stance_votes.csv"
SEMEVAL_DATA = REPO / "data" / "human_votes" / "semeval_stance_votes.csv"
ALIA_RUN = ROOT / "alia"
LABELS = ["FAVOR", "AGAINST", "NONE"]
SEED = 20260822


def macro_f1(gold: pd.Series, pred: pd.Series) -> float:
    scores = []
    for label in LABELS:
        g, p = gold == label, pred == label
        tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def sampled_votes(data: pd.DataFrame) -> pd.DataFrame:
    first, second = _v1_v2_indices(len(data), n_v1=250, n_v2_new=250, seed=20260422)
    selected = data.iloc[first + second].copy().reset_index(drop=True)
    return selected


def vote_baseline(frame: pd.DataFrame) -> dict[str, float]:
    human_gold, human_vote = [], []
    for row in frame.itertuples(index=False):
        counts = {label: int(getattr(row, f"votes_{label}")) for label in LABELS}
        majority = max(counts, key=counts.get)
        for label, count in counts.items():
            human_gold.extend([majority] * count)
            human_vote.extend([label] * count)
    return {
        "human_vote_accuracy_vs_item_majority": float(np.mean(np.array(human_gold) == np.array(human_vote))),
        "human_vote_macro_f1_vs_item_majority": macro_f1(pd.Series(human_gold), pd.Series(human_vote)),
    }


def semeval_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    votes = sampled_votes(pd.read_csv(SEMEVAL_DATA))
    votes["item_id"] = [f"semeval_{i:03}" for i in range(len(votes))]
    for label in LABELS:
        votes[f"votes_{label}"] = votes[f"votes_{label}"].astype(int)
    majority = votes[[f"votes_{x}" for x in LABELS]].idxmax(axis=1).str.removeprefix("votes_")
    votes["human_majority"] = majority
    predictions = {}
    for model, path in [("llama", LLAMA_FULL), ("qwen", QWEN_FULL)]:
        data = pd.read_csv(path, dtype={"item_id": str}, low_memory=False)
        data = data.loc[data["task"] == "chae_semeval_stance", ["item_id", "pred_stance"]]
        predictions[model] = data
        votes = votes.merge(data.rename(columns={"pred_stance": f"{model}_prediction"}), on="item_id", validate="one_to_one")
        votes[f"{model}_correct"] = votes[f"{model}_prediction"] == votes["human_majority"]
    votes["agreement_bin"] = pd.cut(votes["vote_share_majority"], [0, .75, .999999, 1.0], labels=["low (≤.75)", "high (.75–<1)", "unanimous"], include_lowest=True)
    rows = []
    human = vote_baseline(votes)
    rows.append({"group": "all", "model": "individual human vote", "n_items": len(votes), **human})
    for group_name, group in [("all", votes), *[(str(name), g) for name, g in votes.groupby("agreement_bin", observed=True)]]:
        for model in ["llama", "qwen"]:
            rows.append({"group": group_name, "model": model, "n_items": len(group),
                         "accuracy_vs_human_majority": float(group[f"{model}_correct"].mean()),
                         "macro_f1_vs_human_majority": macro_f1(group["human_majority"], group[f"{model}_prediction"])})
    return votes, pd.DataFrame(rows)


def alia_human_summary() -> pd.DataFrame:
    votes = sampled_votes(pd.read_csv(ALIA_DATA))
    return pd.DataFrame([{"dataset": "ALIA civic stance", "n_items": len(votes),
                          "unanimous_items": int((votes["vote_share_majority"] == 1).sum()),
                          "two_of_three_items": int((votes["vote_share_majority"] == 2 / 3).sum()),
                          **vote_baseline(votes)}])


def alia_model_results() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    paths = {
        "llama": ALIA_RUN / "llama3_1_70b_human_votes" / "predictions.csv",
        "qwen": ALIA_RUN / "qwen3_6_27b_human_votes" / "predictions.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    votes = sampled_votes(pd.read_csv(ALIA_DATA, dtype={"source_id": str}))
    votes["human_majority"] = votes["gt_stance"]
    for model, path in paths.items():
        pred = pd.read_csv(path, dtype={"item_id": str}, low_memory=False)
        pred = pred.loc[pred["task"] == "alia_civic_stance_votes", ["item_id", "pred_stance"]]
        votes = votes.merge(pred.rename(columns={"item_id": "source_id", "pred_stance": f"{model}_prediction"}),
                            on="source_id", validate="one_to_one")
        votes[f"{model}_correct"] = votes[f"{model}_prediction"] == votes["human_majority"]
    votes["agreement_bin"] = np.where(votes["vote_share_majority"] == 1, "unanimous", "two of three")
    rows = []
    for group_name, group in [("all", votes), *list(votes.groupby("agreement_bin", sort=True))]:
        for model in ["llama", "qwen"]:
            rows.append({"group": group_name, "model": model, "n_items": len(group),
                         "accuracy_vs_human_majority": float(group[f"{model}_correct"].mean()),
                         "macro_f1_vs_human_majority": macro_f1(group["human_majority"], group[f"{model}_prediction"])})
    return votes, pd.DataFrame(rows)


def paired_accuracy_gaps(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    groups = [("all", frame), *list(frame.groupby("agreement_bin", observed=True, sort=True))]
    for group_name, group in groups:
        paired = group["llama_correct"].astype(int).to_numpy() - group["qwen_correct"].astype(int).to_numpy()
        draws = paired[rng.integers(0, len(paired), size=(10_000, len(paired)))].mean(axis=1)
        rows.append({"dataset": dataset, "agreement_group": str(group_name), "n_items": len(group),
                     "llama_minus_qwen_accuracy": float(paired.mean()),
                     "ci_low": float(np.percentile(draws, 2.5)),
                     "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items, summary = semeval_results()
    items.to_csv(output_dir / "semeval_item_human_agreement.csv", index=False)
    summary.to_csv(output_dir / "semeval_human_model_summary.csv", index=False)
    alia_human_summary().to_csv(output_dir / "alia_human_baseline.csv", index=False)
    gaps = [paired_accuracy_gaps("SemEval stance", items)]
    alia = alia_model_results()
    if alia is not None:
        alia[0].to_csv(output_dir / "alia_item_human_agreement.csv", index=False)
        alia[1].to_csv(output_dir / "alia_human_model_summary.csv", index=False)
        gaps.append(paired_accuracy_gaps("ALIA civic stance", alia[0]))
    pd.concat(gaps, ignore_index=True).to_csv(output_dir / "human_agreement_accuracy_gaps.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
