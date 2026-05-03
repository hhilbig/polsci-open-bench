#!/usr/bin/env python3
"""
Build the cleaned Cross-Domain Topic Classification task file from the public
replication archive.

This task uses the annotated parliamentary-speech target corpus from
Osnabruegge, Ash, and Morelli (2023), collapsed to the eight broad topic
labels used in the paper's cross-domain specification.
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT_ZIP = Path("/tmp/cross_domain_capsule.zip")
DEFAULT_OUTPUT = REPO / "data" / "osnabruegge_cross_domain_topic.csv"

LABEL_MAP = {
    "economy": "Economy",
    "external.relations": "External Relations",
    "fabric.of.society": "Fabric of Society",
    "freedom.and.democracy": "Freedom and Democracy",
    "no.topic": "No Topic",
    "political.system": "Political System",
    "social.groups": "Social Groups",
    "welfare.and.quality.of.life": "Welfare and Quality of Life",
}


def _clean_text(value):
    return " ".join(str(value).split())


def build_cross_domain_task(input_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(input_zip) as zf:
        target = pd.read_csv(zf.open("data/corpora/target_corpus.csv"), low_memory=False)
        topics = pd.read_csv(zf.open("data/files/8topics.csv"), low_memory=False)

    observed_labels = set(target["topic_8"].dropna())
    expected_labels = set(LABEL_MAP)
    if observed_labels != expected_labels:
        raise ValueError(
            "Unexpected 8-topic label set in target_corpus.csv: "
            f"observed={sorted(observed_labels)} expected={sorted(expected_labels)}"
        )

    archive_display_labels = {
        ".".join(str(label).strip().lower().split()): str(label).strip()
        for label in topics["topic"]
    }
    if set(archive_display_labels) != expected_labels:
        raise ValueError(
            "Unexpected 8-topic display label set in 8topics.csv: "
            f"observed={sorted(archive_display_labels)} expected={sorted(expected_labels)}"
        )

    out = pd.DataFrame(
        {
            "source_row_id": range(1, len(target) + 1),
            "text": target["text"].map(_clean_text),
            "source_topic_8": target["topic_8"].astype(str),
            "source_topic_44": target["topic_44"].astype(str),
            "gt_policy_domain": target["topic_8"].map(LABEL_MAP),
        }
    )

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if out["gt_policy_domain"].isna().any():
        bad = out.index[out["gt_policy_domain"].isna()].tolist()[:5]
        raise ValueError(f"Encountered unmapped topic labels at rows {bad}")

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-zip", default=str(DEFAULT_INPUT_ZIP))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_zip = Path(args.input_zip)
    if not input_zip.exists():
        raise FileNotFoundError(
            f"Input archive not found: {input_zip}. "
            "Download the replication ZIP first and pass it via --input-zip."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cross_domain_task(input_zip)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_domain"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
