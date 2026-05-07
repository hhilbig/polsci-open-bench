#!/usr/bin/env python3
"""
Build a CAP party-platform policy-topic task from public Democratic and
Republican Party Platform CSVs.

The source files expose quasi-statement text plus direct CAP major-topic codes.
This builder combines the two party files, keeps standard CAP major topics,
normalizes whitespace, removes conflicts, and deduplicates exact repeated
text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_DEM = Path("/tmp/cap_dem_platform.csv")
DEFAULT_REP = Path("/tmp/cap_rep_platform.csv")
DEFAULT_OUTPUT = REPO / "data" / "cap_party_platform_policy_topic.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _prepare_party(path: Path, party: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    required = {"year", "id", "majortopic", "subtopic", "description", "words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_party": party,
            "source_year": pd.to_numeric(df["year"], errors="coerce"),
            "source_original_id": df["id"].astype(str),
            "source_subtopic": pd.to_numeric(df["subtopic"], errors="coerce"),
            "source_words": pd.to_numeric(df["words"], errors="coerce"),
            "source_major_topic": pd.to_numeric(df["majortopic"], errors="coerce"),
            "text": df["description"].map(_clean_text),
        }
    )


def build_cap_party_platform_policy_topic_task(dem_path: Path, rep_path: Path) -> pd.DataFrame:
    out = pd.concat(
        [
            _prepare_party(dem_path, "Democratic"),
            _prepare_party(rep_path, "Republican"),
        ],
        ignore_index=True,
    )
    out = out.dropna(subset=["source_major_topic"]).copy()
    out = out[out["text"].ne("")].copy()
    out["source_major_topic"] = out["source_major_topic"].astype(int)
    out = out[out["source_major_topic"].isin(CAP_MAJOR_TOPIC_LABELS)].copy()
    out["gt_policy_topic"] = out["source_major_topic"].map(CAP_MAJOR_TOPIC_LABELS)

    conflict_counts = out.groupby("text")["gt_policy_topic"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_policy_topic"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = (
        "cap_party_"
        + out["source_party"].str.lower()
        + "_"
        + out["source_original_id"].astype(str)
    )
    return out[
        [
            "source_id",
            "source_party",
            "source_year",
            "source_original_id",
            "source_major_topic",
            "source_subtopic",
            "source_words",
            "text",
            "gt_policy_topic",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", default=str(DEFAULT_DEM))
    ap.add_argument("--rep", default=str(DEFAULT_REP))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    dem_path = Path(args.dem)
    rep_path = Path(args.rep)
    for path in [dem_path, rep_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download the CAP Democratic and "
                "Republican Party Platform CSVs first, or pass paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cap_party_platform_policy_topic_task(dem_path, rep_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_topic"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
