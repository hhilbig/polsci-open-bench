#!/usr/bin/env python3
"""
Build a cleaned Trump stance task from Burnham's public replication data.

The replication archive exposes public text plus direct stance labels for
Donald Trump. This builder combines the training and held-out adjudicated test
files, normalizes whitespace, removes exact texts with conflicting labels, and
deduplicates repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("/tmp/burnham_trump_train_data.tab")
DEFAULT_TEST = Path("/tmp/burnham_trump_test_data.tab")
DEFAULT_OUTPUT = REPO / "data" / "burnham_trump_stance.csv"

LABEL_MAP = {
    -1: "Oppose",
    0: "Neutral",
    1: "Support",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _prepare_train(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path, sep="\t", low_memory=False)
    required = {"text", "stance", "dataset", "target_mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{train_path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_split": "train",
            "source_row": range(1, len(df) + 1),
            "source_dataset": df["dataset"].astype(str),
            "source_target_mention": pd.to_numeric(df["target_mention"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "source_stance": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )


def _prepare_test(test_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_path, sep="\t", low_memory=False)
    required = {"text", "adjudicated_label", "dataset", "target_mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{test_path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_split": "test",
            "source_row": range(1, len(df) + 1),
            "source_dataset": df["dataset"].astype(str),
            "source_target_mention": pd.to_numeric(df["target_mention"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "source_stance": pd.to_numeric(df["adjudicated_label"], errors="coerce"),
        }
    )


def build_burnham_trump_stance_task(train_path: Path, test_path: Path) -> pd.DataFrame:
    out = pd.concat([_prepare_train(train_path), _prepare_test(test_path)], ignore_index=True)
    out = out.dropna(subset=["source_stance"]).copy()
    out = out[out["text"].ne("")].copy()
    out["source_stance"] = out["source_stance"].astype(int)

    if not out["source_stance"].isin(LABEL_MAP).all():
        bad = sorted(out.loc[~out["source_stance"].isin(LABEL_MAP), "source_stance"].unique().tolist())
        raise ValueError(f"Encountered unsupported stance labels: {bad}")

    conflict_counts = out.groupby("text")["source_stance"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "source_stance"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["gt_stance_toward_trump"] = out["source_stance"].map(LABEL_MAP)
    out["source_id"] = "trump_" + out["source_split"] + "_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_split",
            "source_row",
            "source_dataset",
            "source_target_mention",
            "text",
            "gt_stance_toward_trump",
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
                f"Input file not found: {path}. Download Burnham's Trump stance files "
                "from Harvard Dataverse DOI 10.7910/DVN/XS0ULP first, or pass paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_burnham_trump_stance_task(train_path, test_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_stance_toward_trump"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
