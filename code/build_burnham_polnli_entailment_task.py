#!/usr/bin/env python3
"""
Build the cleaned Political DEBATE / PolNLI entailment task from the public
Hugging Face test split.

The source file codes entailment as 0 and non-entailment as 1. The benchmark
uses the more natural binary label `gt_entails`, where 1 means the hypothesis
is supported by the premise and 0 means it is not.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/polnli_test.parquet")
DEFAULT_OUTPUT = REPO / "data" / "burnham_polnli_entailment.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_burnham_polnli_entailment_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    required = {"premise", "hypothesis", "entailment", "dataset", "task"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_dataset": df["dataset"].map(_clean_text),
            "source_task": df["task"].map(_clean_text),
            "premise": df["premise"].map(_clean_text),
            "hypothesis": df["hypothesis"].map(_clean_text),
            "source_entailment": pd.to_numeric(df["entailment"], errors="raise").astype(int),
        }
    )
    out["gt_entails"] = out["source_entailment"].map({0: 1, 1: 0})

    if out["premise"].eq("").any():
        bad = out.index[out["premise"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty premise after cleaning at rows {bad}")
    if out["hypothesis"].eq("").any():
        bad = out.index[out["hypothesis"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty hypothesis after cleaning at rows {bad}")
    if out["gt_entails"].isna().any():
        bad = out.loc[out["gt_entails"].isna(), "source_entailment"].drop_duplicates().tolist()
        raise ValueError(f"Encountered unmapped entailment values: {bad}")

    conflict_counts = out.groupby(["premise", "hypothesis"])["gt_entails"].nunique()
    conflict_pairs = set(conflict_counts[conflict_counts > 1].index)
    if conflict_pairs:
        pair_index = pd.MultiIndex.from_frame(out[["premise", "hypothesis"]])
        out = out[~pair_index.isin(conflict_pairs)].copy()

    out = out.drop_duplicates(subset=["premise", "hypothesis", "gt_entails"]).reset_index(
        drop=True
    )
    out["source_id"] = "test_" + out["source_row"].astype(str)

    if out["source_id"].duplicated().any():
        dup = out.loc[out["source_id"].duplicated(), "source_id"].iloc[0]
        raise ValueError(f"Duplicate source_id found: {dup}")
    if out.duplicated(["premise", "hypothesis"]).any():
        dup = out.loc[out.duplicated(["premise", "hypothesis"]), ["premise", "hypothesis"]].iloc[0]
        raise ValueError(f"Duplicate premise/hypothesis pair remained: {dup.to_dict()}")

    return out[
        [
            "source_id",
            "source_row",
            "source_dataset",
            "source_task",
            "premise",
            "hypothesis",
            "source_entailment",
            "gt_entails",
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
            f"Input file not found: {input_path}. Download the public PolNLI test parquet "
            "from Hugging Face first and pass it explicitly if it is not in /tmp."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_burnham_polnli_entailment_task(input_path=input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_entails"].value_counts().sort_index().to_string())
    print(df["source_task"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
