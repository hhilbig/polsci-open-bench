#!/usr/bin/env python3
"""
Build a Political DEBATE / PolNLI event-entailment subset from the cleaned
PolNLI entailment task file.

This task keeps only rows whose source task is event extraction. It
therefore tests event-specific hypothesis entailment while reusing the public
PolNLI labels already normalized by the main PolNLI builder.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = REPO / "data" / "burnham_polnli_entailment.csv"
DEFAULT_OUTPUT = REPO / "data" / "burnham_polnli_event_entailment.csv"


def build_burnham_polnli_event_entailment_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    required = {
        "source_id",
        "source_row",
        "source_dataset",
        "source_task",
        "premise",
        "hypothesis",
        "source_entailment",
        "gt_entails",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = df[df["source_task"].eq("event extraction")].copy()
    if out.empty:
        raise ValueError("No event extraction rows found.")
    if out["premise"].isna().any() or out["premise"].astype(str).str.strip().eq("").any():
        raise ValueError("Encountered empty event premise.")
    if out["hypothesis"].isna().any() or out["hypothesis"].astype(str).str.strip().eq("").any():
        raise ValueError("Encountered empty event hypothesis.")
    if not out["gt_entails"].isin([0, 1]).all():
        bad = sorted(out.loc[~out["gt_entails"].isin([0, 1]), "gt_entails"].unique().tolist())
        raise ValueError(f"Encountered non-binary entailment labels: {bad}")
    if out.duplicated(["premise", "hypothesis"]).any():
        raise ValueError("Duplicate premise/hypothesis pairs found in event subset.")

    out["source_id"] = "event_" + out["source_id"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_dataset",
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
            f"Input file not found: {input_path}. Build burnham_polnli_entailment.csv first, "
            "or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_burnham_polnli_event_entailment_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_entails"].value_counts().sort_index().to_string())
    print(df["source_dataset"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
