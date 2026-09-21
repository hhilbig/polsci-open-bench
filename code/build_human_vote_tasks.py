#!/usr/bin/env python3
"""Build local, vote-preserving audit tasks from separately downloaded sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_semeval(text: object) -> str:
    return re.sub(r"\s*#SemST\s*$", "", str(text), flags=re.IGNORECASE).strip()


def build_semeval(current_path: Path, raw_path: Path) -> pd.DataFrame:
    current = pd.read_csv(current_path)
    raw = pd.read_csv(raw_path)
    raw["text_key"] = raw["Tweet"].map(_normalize_semeval)
    raw["vote"] = raw["Stance"].replace({"NEUTRAL": "NONE"})
    counts = raw.groupby(["Target", "text_key", "vote"]).size().unstack(fill_value=0)
    for label in ["FAVOR", "AGAINST", "NONE"]:
        if label not in counts:
            counts[label] = 0
    counts = counts[["FAVOR", "AGAINST", "NONE"]].reset_index()
    current["text_key"] = current["text"].map(_normalize_semeval)
    out = current.merge(counts, left_on=["target", "text_key"], right_on=["Target", "text_key"], how="left", validate="one_to_one")
    out = out.rename(columns={label: f"votes_{label}" for label in ["FAVOR", "AGAINST", "NONE"]})
    out["n_votes"] = out[["votes_FAVOR", "votes_AGAINST", "votes_NONE"]].sum(axis=1, min_count=1)
    out["vote_share_majority"] = out[["votes_FAVOR", "votes_AGAINST", "votes_NONE"]].max(axis=1) / out["n_votes"]
    return out.drop(columns=["Target", "text_key"])


def build_alia(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_json(raw_path, lines=True, dtype={"id": str})
    labels = ["favor", "against", "neutral"]
    for label in labels:
        raw[f"votes_{label.upper() if label != 'neutral' else 'NONE'}"] = raw[["annotation_1", "annotation_2", "annotation_3"]].eq(label).sum(axis=1)
    raw["n_votes"] = 3
    raw["vote_share_majority"] = raw[["votes_FAVOR", "votes_AGAINST", "votes_NONE"]].max(axis=1) / 3
    raw["gt_stance"] = raw["majority_label"].replace(
        {"favor": "FAVOR", "against": "AGAINST", "neutral": "NONE", "": np.nan}
    )
    raw = raw.loc[raw["gt_stance"].notna()].copy()
    raw["source_id"] = "alia_" + raw["id"].astype(str)
    return raw[["source_id", "target", "description", "comment", "annotation_1", "annotation_2", "annotation_3",
                "votes_FAVOR", "votes_AGAINST", "votes_NONE", "n_votes", "vote_share_majority", "gt_stance"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semeval-current", type=Path, required=True)
    parser.add_argument("--semeval-raw", type=Path, required=True)
    parser.add_argument("--alia-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semeval = build_semeval(args.semeval_current, args.semeval_raw)
    if semeval["n_votes"].notna().sum() < len(semeval) - 1:
        raise ValueError("unexpectedly incomplete SemEval raw-vote match")
    alia = build_alia(args.alia_raw)
    if len(alia) != 2850:
        raise ValueError(f"expected 2,850 ALIA majority items, found {len(alia)}")
    semeval.to_csv(args.output_dir / "semeval_stance_votes.csv", index=False)
    alia.to_csv(args.output_dir / "alia_stance_votes.csv", index=False)


if __name__ == "__main__":
    main()
