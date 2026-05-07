#!/usr/bin/env python3
"""
Build a cleaned COVID threat-minimization task from Burnham's public sample.

The source file exposes public text plus direct coding for whether a message
minimizes the COVID-19 threat. This builder uses the primary `threatmin` label,
normalizes whitespace, drops missing labels, removes label conflicts, and
deduplicates the small number of exact repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/burnham_covid_training_sample.tab")
DEFAULT_OUTPUT = REPO / "data" / "burnham_covid_threat_minimization.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_burnham_covid_threat_minimization_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "non_comp", "threatmin", "threatmin2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_non_comp": pd.to_numeric(df["non_comp"], errors="coerce"),
            "source_threatmin": pd.to_numeric(df["threatmin"], errors="coerce"),
            "source_threatmin2": pd.to_numeric(df["threatmin2"], errors="coerce"),
            "text": df["text"].map(_clean_text),
        }
    )
    out = out.dropna(subset=["source_threatmin"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_threat_minimizing"] = out["source_threatmin"].astype(int)

    if not out["gt_threat_minimizing"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_threat_minimizing"].isin([0, 1]), "gt_threat_minimizing"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary threat-minimization labels: {bad}")

    conflict_counts = out.groupby("text")["gt_threat_minimizing"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_threat_minimizing"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "covid_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_non_comp",
            "source_threatmin",
            "source_threatmin2",
            "text",
            "gt_threat_minimizing",
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
            f"Input file not found: {input_path}. Download Burnham's COVID sample "
            "from Harvard Dataverse DOI 10.7910/DVN/XS0ULP first, or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_burnham_covid_threat_minimization_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_threat_minimizing"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
