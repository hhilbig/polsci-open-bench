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

# Harvard Dataverse, doi:10.7910/DVN/CHTWUB, file "capsule-9e05a29c-...zip" (336 MB).
# Direct: https://dataverse.harvard.edu/api/access/datafile/4882383
# Recorded here because the original download left no URL anywhere in the repo,
# which blocked the 2026-09-18 audit from checking this task's construction.
SOURCE_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CHTWUB"
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
            # Three independent validation coders, present on 250 of 4,165 rows.
            # Retained because they establish this task's human ceiling: coders
            # reading the same excerpt match the published topic_8 label only
            # 61.6 / 65.2 / 65.2 percent of the time, and all three agree with
            # each other on 54.8 percent of items (Cohen kappa 0.57-0.65). The
            # archive carries no speech, debate, date or speaker column, so the
            # coders saw exactly what the model is shown. The task is therefore
            # genuinely under-determined by its input rather than missing context.
            "coder1_topic_8": target["topic_8_r1"].map(LABEL_MAP),
            "coder2_topic_8": target["topic_8_r2"].map(LABEL_MAP),
            "coder3_topic_8": target["topic_8_r3"].map(LABEL_MAP),
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
            f"Download the replication capsule from {SOURCE_URL} "
            "(file capsule-9e05a29c-4b3d-457e-a53a-90f4601cda2f.zip) and pass it "
            "via --input-zip."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cross_domain_task(input_zip)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_domain"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
