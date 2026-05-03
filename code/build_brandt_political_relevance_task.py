#!/usr/bin/env python3
"""
Build the cleaned Brandt et al. political-relevance task file from the public
binary-classification corpus in the Harvard Dataverse archive.

The public file exposes only raw text and a binary label. Positive examples are
politics-relevant news articles, while negative examples are non-political
articles. This builder normalizes whitespace, preserves the original row index
as a stable source identifier, and drops exact duplicate texts that do not
carry conflicting labels.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/raw_bc_data.tab")
DEFAULT_OUTPUT = REPO / "data" / "brandt_political_relevance.csv"


def _clean_text(value):
    return " ".join(str(value).split())


def build_brandt_political_relevance_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "bin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "text": df["text"].map(_clean_text),
            "gt_relevant": pd.to_numeric(df["bin"], errors="raise").astype(int),
        }
    )

    if not out["gt_relevant"].isin([0, 1]).all():
        bad = out.loc[~out["gt_relevant"].isin([0, 1]), "gt_relevant"].unique().tolist()
        raise ValueError(f"{input_path} contains non-binary labels: {bad}")
    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"{input_path} contains empty text rows after cleaning: {bad}")

    conflict_counts = out.groupby("text")["gt_relevant"].nunique()
    conflicts = conflict_counts[conflict_counts > 1]
    if not conflicts.empty:
        sample = conflicts.index[0]
        raise ValueError(
            "Found duplicate texts with conflicting labels. "
            f"Example: {sample!r}"
        )

    return out.drop_duplicates(subset=["text"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Download the Dataverse binary-classification file first and pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_brandt_political_relevance_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_relevant"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
