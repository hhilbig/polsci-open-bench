#!/usr/bin/env python3
"""
Build a cleaned Erlich et al. Access to Information request-topic task from the
public Mexican ATI human-coded file.

The source file exposes Spanish request text plus multiple binary topic labels.
This builder keeps the seven S8 request-subject indicators, normalizes
whitespace, drops exact duplicate texts with conflicting topic vectors, and
deduplicates repeated text-vector pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/erlich_hc_new.tab")
DEFAULT_OUTPUT = REPO / "data" / "erlich_ati_topics.csv"

TOPIC_COLUMNS = {
    "Activities": "S8_dummy_Activities",
    "Budget": "S8_dummy_Budget",
    "Evaluation": "S8_dummy_Evaluation",
    "External Contracts": "S8_dummy_ExternalContracts",
    "Institutional Structure": "S8_dummy_InstStruc",
    "Other": "S8_dummy_Other",
    "Regulatory": "S8_dummy_Regulatory",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_erlich_ati_topics_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"id", "Text"} | set(TOPIC_COLUMNS.values())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_id": df["id"].astype(str),
            "text": df["Text"].map(_clean_text),
        }
    )
    for label, column in TOPIC_COLUMNS.items():
        out[f"gt_{label}"] = pd.to_numeric(df[column], errors="raise").astype(int)

    out = out[out["text"].ne("")].copy()
    gt_cols = [f"gt_{label}" for label in TOPIC_COLUMNS]
    for column in gt_cols:
        if not out[column].isin([0, 1]).all():
            bad = sorted(out.loc[~out[column].isin([0, 1]), column].unique().tolist())
            raise ValueError(f"Encountered non-binary values in {column}: {bad}")

    conflict_counts = out.groupby("text")[gt_cols].nunique().max(axis=1)
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text"] + gt_cols).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[["source_id", "text"] + gt_cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download hc_new.tab from "
            "Harvard Dataverse DOI 10.7910/DVN/SOVPA4 first, or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_erlich_ati_topics_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    for label in TOPIC_COLUMNS:
        print(f"{label}: {int(df[f'gt_{label}'].sum())}")


if __name__ == "__main__":
    main()
