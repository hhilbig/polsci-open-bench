#!/usr/bin/env python3
"""
Build a cleaned PAPEA protest-claim task file from the public FGZ sentence-
level claim annotations in the Harvard Dataverse archive.

The source TSV has malformed quoting, so this builder parses with quoting
disabled, strips literal quote characters, keeps common substantive claim
labels, normalizes whitespace, removes exact duplicate texts with conflicting
labels, and deduplicates repeated text-label pairs.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/fgz_papea_claims.tab")
DEFAULT_OUTPUT = REPO / "data" / "haunss_papea_claims.csv"

LABEL_MAP = {
    "anti-far-right": "Anti-Far-Right",
    "anticapitalist": "Anticapitalist",
    "church": "Church",
    "covid": "COVID",
    "democracy": "Democracy",
    "economy": "Economy",
    "education": "Education",
    "environment (without nuclear)": "Environment",
    "far-right": "Far Right",
    "foreign_rights": "Foreign Rights",
    "gender": "Gender",
    "infrastructure": "Infrastructure",
    "international": "International",
    "labour": "Labour",
    "media": "Media",
    "migration": "Migration",
    "nuclear power": "Nuclear Power",
    "other": "Other",
    "peace": "Peace",
    "peasants": "Peasants",
    "political": "Political System",
    "repression": "Repression",
    "religion": "Religion",
    "rights": "Rights",
    "social": "Social",
    "solidarity": "Solidarity",
    "tolerance": "Tolerance",
    "unclear": "Unclear",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().strip('"').split())


def _clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().strip('"').strip()


def build_haunss_papea_claims_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", engine="python", quoting=csv.QUOTE_NONE)
    required = {"fid", "claim", "claim_txt", "event_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_fid": df["fid"].map(_clean_string),
            "source_event_id": df["event_id"].map(_clean_string),
            "source_claim": pd.to_numeric(df["claim"], errors="coerce"),
            "source_claim_raw": df["claim_txt"].map(_clean_string),
            "text": df["text"].map(_clean_text),
        }
    )
    out = out[out["source_claim_raw"].isin(LABEL_MAP)].copy()
    out["gt_protest_claim"] = out["source_claim_raw"].map(LABEL_MAP)

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if out["gt_protest_claim"].isna().any():
        bad = out.index[out["gt_protest_claim"].isna()].tolist()[:5]
        raise ValueError(f"Encountered unmapped claim labels at rows {bad}")

    conflict_counts = out.groupby("text")["gt_protest_claim"].nunique()
    conflict_texts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflict_texts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_protest_claim"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[
        [
            "source_fid",
            "source_event_id",
            "source_claim",
            "source_claim_raw",
            "text",
            "gt_protest_claim",
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
            f"Input file not found: {input_path}. Download fgz_papea_claims.tab "
            "from Harvard Dataverse DOI 10.7910/DVN/KVP7HA first, or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_haunss_papea_claims_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_protest_claim"].value_counts().to_string())


if __name__ == "__main__":
    main()
