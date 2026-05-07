#!/usr/bin/env python3
"""
Build a cleaned Bestvater and Monroe Women's March stance task from the public
ground-truth Twitter file in the Harvard Dataverse archive.

The source file exposes tweet text plus binary stance and sentiment labels. This
builder uses the direct stance label, normalizes whitespace, drops exact
duplicate texts with conflicting stance labels, and deduplicates repeated
text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/bestvater_wm_groundtruth.tab")
DEFAULT_OUTPUT = REPO / "data" / "bestvater_wm_stance.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_bestvater_wm_stance_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "stance", "sentiment", "balanced_train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_sentiment": pd.to_numeric(df["sentiment"], errors="coerce"),
            "source_balanced_train": pd.to_numeric(df["balanced_train"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "gt_pro_womens_march": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["gt_pro_womens_march"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_pro_womens_march"] = out["gt_pro_womens_march"].astype(int)

    if not out["gt_pro_womens_march"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_pro_womens_march"].isin([0, 1]), "gt_pro_womens_march"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary stance labels: {bad}")

    conflict_counts = out.groupby("text")["gt_pro_womens_march"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_pro_womens_march"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "wm_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_sentiment",
            "source_balanced_train",
            "text",
            "gt_pro_womens_march",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download WM_tweets_groundtruth.tab "
            "from Harvard Dataverse DOI 10.7910/DVN/MUYYG4 first, or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_bestvater_wm_stance_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_pro_womens_march"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
