#!/usr/bin/env python3
"""
Build an AgoraSpeech criticism-vs-agenda task from human-validated labels.

The public AgoraSpeech CSV exposes Greek campaign-speech paragraphs, English
translations, and both GPT and human-validated labels. This builder uses only
the human criticism-or-agenda label and the English paragraph text.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/agoraspeech.csv")
DEFAULT_OUTPUT = REPO / "data" / "agoraspeech_criticism_agenda.csv"

LABELS = {"criticism", "political agenda"}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_agoraspeech_criticism_agenda_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    required = {
        "elections",
        "speech_id",
        "politician",
        "date (YYYY-MM-DD)",
        "location",
        "paragraph",
        "text",
        "text_el",
        "criticism_or_agenda_human",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_elections": df["elections"].astype(str),
            "source_speech_id": df["speech_id"].astype(str),
            "source_politician": df["politician"].astype(str),
            "source_date": df["date (YYYY-MM-DD)"].astype(str),
            "source_location": df["location"].astype(str),
            "source_paragraph": pd.to_numeric(df["paragraph"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "source_text_el": df["text_el"].map(_clean_text),
            "gt_criticism_or_agenda": df["criticism_or_agenda_human"].map(_clean_text),
        }
    )
    out = out[out["text"].ne("")].copy()
    out = out[out["gt_criticism_or_agenda"].isin(LABELS)].copy()

    conflict_counts = out.groupby("text")["gt_criticism_or_agenda"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_criticism_or_agenda"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = (
        "agora_"
        + out["source_speech_id"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        + "_p"
        + out["source_paragraph"].astype(int).astype(str)
    )
    return out[
        [
            "source_id",
            "source_elections",
            "source_speech_id",
            "source_politician",
            "source_date",
            "source_location",
            "source_paragraph",
            "text",
            "source_text_el",
            "gt_criticism_or_agenda",
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
            f"Input file not found: {input_path}. Download AgoraSpeech.csv "
            "from Zenodo DOI 10.5281/zenodo.13957177 first, or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_agoraspeech_criticism_agenda_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_criticism_or_agenda"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
