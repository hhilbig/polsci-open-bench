#!/usr/bin/env python3
"""
Build the cleaned Spanish protest-toxicity task from the public
toxicity-protests-ES gold-standard file.

The source file exposes two human coder labels for Spanish-language tweets
about protests. This builder keeps only rows where the two coders agree,
normalizes whitespace, maps the agreed label to a binary toxicity target, and
deduplicates exact repeated texts after checking for label conflicts.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/toxicity_protests_es.csv")
DEFAULT_OUTPUT = REPO / "data" / "toxicity_protests_es.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_toxicity_protests_es_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    required = {
        "id_obs",
        "text",
        "coder_1",
        "coder_2",
        "lang",
        "country",
        "type",
        "created_at",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    agreed = df[df["coder_1"].eq(df["coder_2"])].copy()
    if agreed.empty:
        raise ValueError("No rows remain after requiring coder agreement.")

    out = pd.DataFrame(
        {
            "source_id_obs": agreed["id_obs"].astype(str),
            "source_lang": agreed["lang"].astype(str),
            "source_country": agreed["country"].astype(str),
            "source_type": agreed["type"].astype(str),
            "source_created_at": agreed["created_at"].astype(str),
            "source_coder_1": pd.to_numeric(agreed["coder_1"], errors="raise").astype(int),
            "source_coder_2": pd.to_numeric(agreed["coder_2"], errors="raise").astype(int),
            "text": agreed["text"].map(_clean_text),
        }
    )
    out["gt_toxic"] = out["source_coder_1"].astype(int)

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if not out["gt_toxic"].isin([0, 1]).all():
        bad = sorted(out.loc[~out["gt_toxic"].isin([0, 1]), "gt_toxic"].unique().tolist())
        raise ValueError(f"Encountered non-binary toxicity labels: {bad}")

    conflict_counts = out.groupby("text")["gt_toxic"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_toxic"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[
        [
            "source_id_obs",
            "source_lang",
            "source_country",
            "source_type",
            "source_created_at",
            "source_coder_1",
            "source_coder_2",
            "text",
            "gt_toxic",
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
            f"Input file not found: {input_path}. Download goldstd_protests.csv "
            "from https://huggingface.co/datasets/bgonzalezbustamante/toxicity-protests-ES "
            "first, or pass the file path via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_toxicity_protests_es_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_toxic"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
