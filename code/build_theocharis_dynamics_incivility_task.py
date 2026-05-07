#!/usr/bin/env python3
"""
Build the cleaned Theocharis et al. political-incivility task from the public
training data in the incivility-sage-open replication repository.

The source file exposes tweet text and a direct yes/no incivility label. The
builder normalizes whitespace, maps labels to 1/0, drops exact duplicate texts
with conflicting labels, and deduplicates repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/incivility_training_data.csv")
DEFAULT_OUTPUT = REPO / "data" / "theocharis_dynamics_incivility.csv"

LABEL_MAP = {
    "no": 0,
    "yes": 1,
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _clean_tweet_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def build_theocharis_dynamics_incivility_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"uncivil", "id_str", "created_at", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_tweet_id": df["id_str"].map(_clean_tweet_id),
            "source_created_at": df["created_at"].astype(str),
            "source_uncivil_raw": df["uncivil"].astype(str).str.strip().str.lower(),
            "text": df["text"].map(_clean_text),
        }
    )
    out["gt_uncivil"] = out["source_uncivil_raw"].map(LABEL_MAP)

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if out["source_tweet_id"].eq("").any():
        bad = out.index[out["source_tweet_id"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty tweet IDs at rows {bad}")
    if out["gt_uncivil"].isna().any():
        bad = (
            out.loc[out["gt_uncivil"].isna(), "source_uncivil_raw"]
            .drop_duplicates()
            .tolist()[:5]
        )
        raise ValueError(f"Encountered unmapped uncivil labels: {bad}")

    out["gt_uncivil"] = out["gt_uncivil"].astype(int)

    conflict_counts = out.groupby("text")["gt_uncivil"].nunique()
    conflict_texts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflict_texts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_uncivil"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")
    if out["source_tweet_id"].duplicated().any():
        dup = out.loc[out["source_tweet_id"].duplicated(), "source_tweet_id"].iloc[0]
        raise ValueError(f"Duplicate source_tweet_id found: {dup}")

    return out[
        [
            "source_row",
            "source_tweet_id",
            "source_created_at",
            "source_uncivil_raw",
            "text",
            "gt_uncivil",
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
            f"Input file not found: {input_path}. Download data/training-data.csv "
            "from https://github.com/pablobarbera/incivility-sage-open first, "
            "or pass the file path via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_theocharis_dynamics_incivility_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_uncivil"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
