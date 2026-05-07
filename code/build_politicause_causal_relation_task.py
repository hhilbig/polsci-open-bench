#!/usr/bin/env python3
"""
Build a cleaned PolitiCAUSE causal-relation sentence task.

The public repository provides train, validation, and test CSVs with sentence
text plus a direct binary label for whether the sentence contains a causal
relation. This builder combines the splits, removes exact texts with conflicting
labels, and deduplicates exact repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("/tmp/politicause_train.csv")
DEFAULT_VAL = Path("/tmp/politicause_val.csv")
DEFAULT_TEST = Path("/tmp/politicause_test.csv")
DEFAULT_OUTPUT = REPO / "data" / "politicause_causal_relation.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _prepare_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    source_index = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else pd.Series(range(len(df)))
    return pd.DataFrame(
        {
            "source_split": split,
            "source_index": source_index,
            "text": df["text"].map(_clean_text),
            "gt_causal_relation": pd.to_numeric(df["label"], errors="coerce"),
        }
    )


def build_politicause_causal_relation_task(train_path: Path, val_path: Path, test_path: Path) -> pd.DataFrame:
    out = pd.concat(
        [
            _prepare_split(train_path, "train"),
            _prepare_split(val_path, "val"),
            _prepare_split(test_path, "test"),
        ],
        ignore_index=True,
    )
    out = out.dropna(subset=["gt_causal_relation"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_causal_relation"] = out["gt_causal_relation"].astype(int)

    if not out["gt_causal_relation"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_causal_relation"].isin([0, 1]), "gt_causal_relation"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary causal labels: {bad}")

    conflict_counts = out.groupby("text")["gt_causal_relation"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_causal_relation"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "politicause_" + out["source_split"] + "_" + out["source_index"].astype(str)
    return out[["source_id", "source_split", "source_index", "text", "gt_causal_relation"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_paths = [Path(args.train), Path(args.val), Path(args.test)]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download PolitiCAUSE train/val/test CSVs "
                "from https://github.com/pgarco/PolitiCAUSE first, or pass paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_politicause_causal_relation_task(*input_paths)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_causal_relation"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
