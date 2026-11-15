#!/usr/bin/env python3
"""
Build a CAP Congressional Research Service report policy-topic task.

The public CAP CRS file exposes report titles, summaries, and direct CAP
major-topic codes. This builder keeps standard CAP major topics, normalizes
whitespace, removes conflicts, and deduplicates exact repeated text-label pairs.

`--version 2` writes data/cap_crs_policy_topic_v2.csv from the v1 CSV
(data/cap_crs_policy_topic.csv), not from the raw download. The CAP CRS file
has been revised since v1 was built (release 26.1 has 17,204 clean rows under
the v1 rules, renumbered ids, and a UTF-8 byte-order mark the v1 reader does
not handle), so the raw file no longer reproduces v1. Building v2 from the v1
rows keeps the v1 item set and gold labels apart from the two rules below.

1. Semicolon line breaks. The CAP file stores the line breaks of the
   EveryCRSReport title and summary as ";" with no following space; the
   whitespace after real semicolons is dropped too, so the two cannot be told
   apart. Examples, checked against everycrsreport.com: RL33703 "Summary of
   the Pension Protection Act;of 2006" (title has a line break before "of
   2006"); RS21633 summary "fugitives from Northern;Ireland" (line break).
   v2 replaces every ";" followed by a non-space character with a space. Real
   semicolons whose space the source dropped become a space as well.
2. Titles with conflicting gold. After rule 1 the exact-text conflict and
   duplicate rules of v1 are re-applied, and every row whose title carries
   more than one gold label is dropped (different CRS reports with the same
   title coded to different major topics; 38% of rows are shown as the title
   alone, so such rows show the model the same text under different gold).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/cap_crs_reports.csv")
DEFAULT_OUTPUT = REPO / "data" / "cap_crs_policy_topic.csv"
DEFAULT_OUTPUT_V2 = REPO / "data" / "cap_crs_policy_topic_v2.csv"
V2_SEMICOLON_BREAK = re.compile(r";(?=\S)")


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


def _drop_conflicts_and_duplicates(out: pd.DataFrame) -> pd.DataFrame:
    conflict_counts = out.groupby("text")["gt_policy_topic"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    return out.drop_duplicates(subset=["text", "gt_policy_topic"]).reset_index(drop=True)


def build_cap_crs_policy_topic_task_v2(v1_path: Path) -> pd.DataFrame:
    v1 = pd.read_csv(v1_path, low_memory=False)
    out = v1.copy()
    for col in ["title", "summary"]:
        out[col] = (
            out[col]
            .fillna("")
            .astype(str)
            .map(lambda x: " ".join(V2_SEMICOLON_BREAK.sub(" ", x).split()))
        )
    out["text"] = (out["title"] + "\n\n" + out["summary"]).str.strip()
    n0 = len(out)
    out = _drop_conflicts_and_duplicates(out)
    print(f"v2: {n0 - len(out)} rows dropped as exact-text conflicts or duplicates after rule 1")

    title_labels = out.groupby("title")["gt_policy_topic"].nunique()
    conflict_titles = set(title_labels[title_labels > 1].index)
    drop = out["title"].isin(conflict_titles)
    print(
        f"v2: {int(drop.sum())} rows dropped from {len(conflict_titles)} titles "
        "with conflicting gold labels"
    )
    out = out[~drop].reset_index(drop=True)
    if out["text"].duplicated().any():
        raise ValueError("Duplicate text remained after v2 cleaning")
    out["summary"] = out["summary"].replace("", pd.NA)
    return out[v1.columns.tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument(
        "--v1-input",
        default=str(DEFAULT_OUTPUT),
        help="v1 CSV that --version 2 is built from",
    )
    args = ap.parse_args()
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if args.version == 2 else DEFAULT_OUTPUT)

    if args.version == 2:
        df = build_cap_crs_policy_topic_task_v2(Path(args.v1_input))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(df)} rows)")
        print(df["gt_policy_topic"].value_counts().sort_index().to_string())
        return

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
