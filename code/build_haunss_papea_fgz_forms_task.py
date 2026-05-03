#!/usr/bin/env python3
"""
Build the cleaned PAPEA protest-form task file from the public FGZ sentence-
level human annotations in the Harvard Dataverse archive.

This task keeps the seven most common original PAPEA form labels directly,
rather than inventing a new bespoke collapse. It removes rows outside that
label set, normalizes whitespace in the text snippets, and drops exact
duplicate snippets that appear with conflicting labels.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/papea_B3_fgz_forms.tab")
DEFAULT_OUTPUT = REPO / "data" / "haunss_papea_fgz_forms.csv"

LABEL_MAP = {
    "demonstration, assembly": "Demonstration / Assembly",
    "petition": "Petition",
    "strike": "Strike",
    "non-verbal protest, cultural event": "Non-verbal Protest / Cultural Event",
    "leaflet, resolution, open letter": "Leaflet / Resolution / Open Letter",
    "attack with damage to property": "Attack with Damage to Property",
    "blockade, sit-in": "Blockade / Sit-in",
}


def _clean_text(value):
    return " ".join(str(value).split())


def build_papea_fgz_forms_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"AN", "FORM", "text", "form_numeric", "form"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_an": df["AN"].astype(str),
            "source_form_codebook": df["FORM"].astype(str),
            "source_form_numeric": pd.to_numeric(df["form_numeric"], errors="coerce"),
            "source_form_raw": df["form"].astype(str),
            "text": df["text"].map(_clean_text),
        }
    )
    out = out[out["source_form_raw"].isin(LABEL_MAP)].copy()
    out["gt_protest_form"] = out["source_form_raw"].map(LABEL_MAP)

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if out["gt_protest_form"].isna().any():
        bad = out.index[out["gt_protest_form"].isna()].tolist()[:5]
        raise ValueError(f"Encountered unmapped form labels at rows {bad}")

    conflict_counts = out.groupby("text")["gt_protest_form"].nunique()
    conflict_texts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflict_texts)].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Download the Dataverse FGZ forms file first and pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_papea_fgz_forms_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_protest_form"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
