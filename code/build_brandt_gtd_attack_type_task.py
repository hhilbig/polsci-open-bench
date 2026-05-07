#!/usr/bin/env python3
"""
Build the cleaned Brandt et al. GTD attack-type task from the public
multi-label GTD corpus in the Harvard Dataverse archive.

The source file exposes GTD event summaries and the primary attack type. This
builder normalizes whitespace, keeps the original row number as the stable
identifier because the public eventid column is rounded in the released TSV,
drops empty texts, drops exact duplicate texts with conflicting labels, and
deduplicates repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/raw_gtd_multilabel_data.tab")
DEFAULT_OUTPUT = REPO / "data" / "brandt_gtd_attack_type.csv"

ATTACK_TYPE_LABELS = [
    "Assassination",
    "Armed Assault",
    "Bombing/Explosion",
    "Facility/Infrastructure Attack",
    "Hijacking",
    "Hostage Taking (Barricade Incident)",
    "Hostage Taking (Kidnapping)",
    "Unarmed Assault",
    "Unknown",
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _date_string(df: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(df["iyear"], errors="raise").astype(int).astype(str)
    month = pd.to_numeric(df["imonth"], errors="raise").astype(int).astype(str).str.zfill(2)
    day = pd.to_numeric(df["iday"], errors="raise").astype(int).astype(str).str.zfill(2)
    return year + "-" + month + "-" + day


def build_brandt_gtd_attack_type_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "attacktype1_txt", "country_txt", "iyear", "imonth", "iday"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_date": _date_string(df),
            "source_country": df["country_txt"].astype(str),
            "text": df["text"].map(_clean_text),
            "gt_attack_type": df["attacktype1_txt"].astype(str).str.strip(),
        }
    )

    out = out[out["text"].ne("")].copy()
    if out["gt_attack_type"].eq("").any():
        bad = out.index[out["gt_attack_type"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty attack-type labels at rows {bad}")

    unknown_labels = sorted(set(out["gt_attack_type"]) - set(ATTACK_TYPE_LABELS))
    if unknown_labels:
        raise ValueError(f"Encountered unexpected attack-type labels: {unknown_labels}")

    conflict_counts = out.groupby("text")["gt_attack_type"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_attack_type"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[
        [
            "source_row",
            "source_date",
            "source_country",
            "text",
            "gt_attack_type",
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
            f"Input file not found: {input_path}. Download raw_gtd_multilabel_data.tab "
            "from Harvard Dataverse DOI 10.7910/DVN/KDO5AM first, or pass the file "
            "path via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_brandt_gtd_attack_type_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
