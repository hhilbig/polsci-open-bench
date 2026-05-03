#!/usr/bin/env python3
"""
Build the cleaned Politicians in the Line of Fire task file from the public
training samples in the Harvard Dataverse archive.

This task combines the human-coded US and Canadian training samples from
Rheault, Rayment, and Musulan (2019). It keeps the public natural-language
tweet text (`clean_text`) and the binary human label (`code`), then removes
exact duplicate texts with no label conflict.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_USA_INPUT = Path("/tmp/usa_training.tab")
DEFAULT_CANADA_INPUT = Path("/tmp/canada_training.tab")
DEFAULT_OUTPUT = REPO / "data" / "rheault_line_of_fire_incivility.csv"


def _clean_text(value):
    return " ".join(str(value).split())


def _load_training_file(path: Path, country: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"code", "clean_text", "text", "maxscore", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_country": country,
            "source_row_id": range(1, len(df) + 1),
            "text": df["clean_text"].map(_clean_text),
            "source_text_processed": df["text"].astype(str),
            "source_maxscore": df["maxscore"].astype(float),
            "source_sentiment": df["sentiment"].astype(float),
            "gt_uncivil": df["code"].astype(int),
        }
    )

    if out["gt_uncivil"].isin([0, 1]).all() is False:
        bad = out.loc[~out["gt_uncivil"].isin([0, 1]), "gt_uncivil"].unique().tolist()
        raise ValueError(f"{path} contains non-binary code labels: {bad}")
    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"{path} contains empty clean_text rows after cleaning: {bad}")

    return out


def build_line_of_fire_task(usa_input: Path, canada_input: Path) -> pd.DataFrame:
    usa = _load_training_file(usa_input, country="USA")
    canada = _load_training_file(canada_input, country="Canada")
    combined = pd.concat([usa, canada], ignore_index=True)

    conflict_counts = combined.groupby("text")["gt_uncivil"].nunique()
    conflicts = conflict_counts[conflict_counts > 1]
    if not conflicts.empty:
        sample = conflicts.index[0]
        raise ValueError(
            "Found duplicate texts with conflicting labels. "
            f"Example: {sample!r}"
        )

    deduped = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return deduped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usa-input", default=str(DEFAULT_USA_INPUT))
    ap.add_argument("--canada-input", default=str(DEFAULT_CANADA_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    usa_input = Path(args.usa_input)
    canada_input = Path(args.canada_input)
    for path in [usa_input, canada_input]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. "
                "Download the Dataverse training samples first and pass them via "
                "--usa-input and --canada-input."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_line_of_fire_task(usa_input, canada_input)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["source_country"].value_counts().sort_index().to_string())
    print(df["gt_uncivil"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
