#!/usr/bin/env python3
"""
Build a cleaned manifesto-sentence populism task from Di Cocco and Monechi.

The public replication archive includes Italian manual manifesto sentences with
direct binary populism labels. This builder normalizes text, validates the
binary label, removes conflicts, and deduplicates exact repeated text-label
pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/dicocco_extract/datasets/IT_manual_sentences.json")
DEFAULT_OUTPUT = REPO / "data" / "dicocco_manifesto_populism.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_dicocco_manifesto_populism_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_json(input_path)
    required = {"year", "party", "text", "orientation", "old_index", "is_populist"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_year": pd.to_numeric(df["year"], errors="coerce"),
            "source_party": df["party"].astype(str),
            "source_orientation": df["orientation"].astype(str),
            "source_old_index": pd.to_numeric(df["old_index"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "gt_populist": pd.to_numeric(df["is_populist"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["gt_populist"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_populist"] = out["gt_populist"].astype(int)

    if not out["gt_populist"].isin([0, 1]).all():
        bad = sorted(out.loc[~out["gt_populist"].isin([0, 1]), "gt_populist"].unique().tolist())
        raise ValueError(f"Encountered non-binary populism labels: {bad}")

    conflict_counts = out.groupby("text")["gt_populist"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_populist"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "dicocco_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_year",
            "source_party",
            "source_orientation",
            "source_old_index",
            "text",
            "gt_populist",
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
            f"Input file not found: {input_path}. Extract datasets/IT_manual_sentences.json "
            "from Harvard Dataverse DOI 10.7910/DVN/BMJYAN first, or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dicocco_manifesto_populism_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_populist"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
