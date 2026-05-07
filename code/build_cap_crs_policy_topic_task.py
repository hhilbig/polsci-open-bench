#!/usr/bin/env python3
"""
Build a CAP Congressional Research Service report policy-topic task.

The public CAP CRS file exposes report titles, summaries, and direct CAP
major-topic codes. This builder keeps standard CAP major topics, normalizes
whitespace, removes conflicts, and deduplicates exact repeated text-label pairs.
"""
import argparse
from pathlib import Path

import pandas as pd

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/cap_crs_reports.csv")
DEFAULT_OUTPUT = REPO / "data" / "cap_crs_policy_topic.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_cap_crs_policy_topic_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="latin1", low_memory=False)
    required = {"id", "year", "description", "summary", "crs_category", "majortopic", "subtopic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_id": "cap_crs_" + df["id"].astype(str),
            "source_year": pd.to_numeric(df["year"], errors="coerce"),
            "source_crs_category": df["crs_category"].astype(str),
            "source_major_topic": pd.to_numeric(df["majortopic"], errors="coerce"),
            "source_subtopic": pd.to_numeric(df["subtopic"], errors="coerce"),
            "title": df["description"].map(_clean_text),
            "summary": df["summary"].map(_clean_text),
        }
    )
    out["text"] = (out["title"] + "\n\n" + out["summary"]).str.strip()
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

    return out[
        [
            "source_id",
            "source_year",
            "source_crs_category",
            "source_major_topic",
            "source_subtopic",
            "title",
            "summary",
            "text",
            "gt_policy_topic",
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
            f"Input file not found: {input_path}. Download the CAP CRS Reports CSV "
            "from https://www.comparativeagendas.net/project/us/datasets first, "
            "or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cap_crs_policy_topic_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_topic"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
