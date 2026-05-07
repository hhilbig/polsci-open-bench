#!/usr/bin/env python3
"""
Build a cleaned TwitCivility impoliteness task from the public Hugging Face
train/test parquet splits.

The source files expose tweet text plus direct binary labels for impoliteness
and intolerance. This builder combines train and test, keeps the direct
impoliteness label as the benchmark target, normalizes whitespace, checks for
text-label conflicts, and deduplicates exact repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("/tmp/twitcivility_train.parquet")
DEFAULT_TEST = Path("/tmp/twitcivility_test.parquet")
DEFAULT_OUTPUT = REPO / "data" / "twitcivility_impoliteness.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _read_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"text", "impoliteness", "intolerance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "source_split": split,
            "source_row": range(1, len(df) + 1),
            "source_intolerance": pd.to_numeric(df["intolerance"], errors="raise").astype(int),
            "text": df["text"].map(_clean_text),
            "gt_impolite": pd.to_numeric(df["impoliteness"], errors="raise").astype(int),
        }
    )


def build_twitcivility_impoliteness_task(train_path: Path, test_path: Path) -> pd.DataFrame:
    out = pd.concat(
        [
            _read_split(train_path, "train"),
            _read_split(test_path, "test"),
        ],
        ignore_index=True,
    )

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if not out["gt_impolite"].isin([0, 1]).all():
        bad = sorted(out.loc[~out["gt_impolite"].isin([0, 1]), "gt_impolite"].unique().tolist())
        raise ValueError(f"Encountered non-binary impoliteness labels: {bad}")
    if not out["source_intolerance"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["source_intolerance"].isin([0, 1]), "source_intolerance"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary intolerance labels: {bad}")

    conflict_counts = out.groupby("text")["gt_impolite"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_impolite"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = out["source_split"] + "_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_split",
            "source_row",
            "source_intolerance",
            "text",
            "gt_impolite",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download the TwitCivility parquet "
                "splits from https://huggingface.co/datasets/incivility-UOH/TwitCivility "
                "first, or pass explicit --train/--test paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_twitcivility_impoliteness_task(train_path, test_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_impolite"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
