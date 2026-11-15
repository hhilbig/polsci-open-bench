#!/usr/bin/env python3
"""
Build a Kavanaugh stance task from Bestvater and Monroe's public data.

The source file exposes tweet text plus direct binary stance labels. The exact
same-label repeats are part of this source file and are retained because
deduplicating them would discard most rows. Only exact texts with conflicting
stance labels are removed.

Version 2 (`--version 2`) deduplicates. In v1, 62% of rows repeated another
row's text (mostly retweets; one text appears 78 times), so a sample of 100
items held about 87 unique texts and per-text errors were counted many times.
v2 keeps one row per exact cleaned text (the first source row) and drops every
copy of a text that carries conflicting stance labels. It reads the original
CSV upload (`/api/access/datafile/5374859?format=original`) instead of the
Dataverse-ingested `.tab` export, which turned embedded quotes into literal
`\"` escapes in about 9% of rows; row order is the same, so v2 ids reuse the
v1 source row. v2 also replaces e-mail addresses with "[EMAIL]".
"""
import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/bestvater_kavanaugh_groundtruth.tab")
DEFAULT_OUTPUT = REPO / "data" / "bestvater_kavanaugh_stance.csv"
DEFAULT_INPUT_V2 = Path("/tmp/bestvater_kavanaugh_groundtruth.csv")
DEFAULT_OUTPUT_V2 = REPO / "data" / "bestvater_kavanaugh_stance_v2.csv"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_bestvater_kavanaugh_stance_task(input_path: Path, version: int = 1) -> pd.DataFrame:
    v2 = version == 2
    if v2:
        df = pd.read_csv(input_path, low_memory=False)
    else:
        df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "sentiment", "stance", "fold", "vader_sentiment", "SVM_sentiment", "BERT_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_sentiment": pd.to_numeric(df["sentiment"], errors="coerce"),
            "source_fold": pd.to_numeric(df["fold"], errors="coerce"),
            "source_vader_sentiment": pd.to_numeric(df["vader_sentiment"], errors="coerce"),
            "source_svm_sentiment": pd.to_numeric(df["SVM_sentiment"], errors="coerce"),
            "source_bert_sentiment": pd.to_numeric(df["BERT_sentiment"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "gt_pro_kavanaugh": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["gt_pro_kavanaugh"]).copy()
    if v2:
        out["text"] = out["text"].map(lambda t: EMAIL_RE.sub("[EMAIL]", t))
    out = out[out["text"].ne("")].copy()
    out["gt_pro_kavanaugh"] = out["gt_pro_kavanaugh"].astype(int)

    if not out["gt_pro_kavanaugh"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_pro_kavanaugh"].isin([0, 1]), "gt_pro_kavanaugh"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary Kavanaugh stance labels: {bad}")

    conflict_counts = out.groupby("text")["gt_pro_kavanaugh"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy().reset_index(drop=True)
    if v2:
        out = out.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    if out.groupby("text")["gt_pro_kavanaugh"].nunique().max() != 1:
        raise ValueError("Conflicting duplicate text labels remained after cleaning.")

    out["source_id"] = "kavanaugh_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_sentiment",
            "source_fold",
            "source_vader_sentiment",
            "source_svm_sentiment",
            "source_bert_sentiment",
            "text",
            "gt_pro_kavanaugh",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    v2 = args.version == 2
    if args.input is None:
        args.input = str(DEFAULT_INPUT_V2 if v2 else DEFAULT_INPUT)
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if v2 else DEFAULT_OUTPUT)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download Kavanaugh_tweets_groundtruth.tab "
            "from Harvard Dataverse DOI 10.7910/DVN/MUYYG4 first, or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_bestvater_kavanaugh_stance_task(input_path, version=args.version)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_pro_kavanaugh"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
