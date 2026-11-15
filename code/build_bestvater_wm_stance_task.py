#!/usr/bin/env python3
"""
Build a cleaned Bestvater and Monroe Women's March stance task from the public
ground-truth Twitter file in the Harvard Dataverse archive.

The source file exposes tweet text plus binary stance and sentiment labels. This
builder uses the direct stance label, normalizes whitespace, drops exact
duplicate texts with conflicting stance labels, and deduplicates repeated
text-label pairs.

Version 2 (`--version 2`) makes three changes:

1. It keeps only the rows with `balanced_train == 1`. This is the set the
   authors trained and tested their stance classifier on
   (`1_train_classifiers.py`: `balanced = balanced[balanced['balanced_train']==1]`),
   built by "random undersampling to account for the class imbalance" (paper,
   p. 238). It contains every opposing tweet in the file and a random subset of
   approving tweets. The full file is 87% approving, so always predicting 1
   outscored every model in v1.
2. It reads the original CSV upload
   (`/api/access/datafile/5374866?format=original`, 19,612 rows) instead of the
   Dataverse-ingested `.tab` export (19,957 rows), which splits multiline
   tweets into extra rows and leaves literal `\n` and `\"` escapes.
3. It replaces e-mail addresses with "[EMAIL]", matching the repository's
   redaction of the v1 corpora.

The source file has no flag for tweets whose stance depends on an image or a
parent tweet, so v2 drops no items on that basis.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/bestvater_wm_groundtruth.tab")
DEFAULT_OUTPUT = REPO / "data" / "bestvater_wm_stance.csv"
DEFAULT_INPUT_V2 = Path("/tmp/bestvater_wm_groundtruth.csv")
DEFAULT_OUTPUT_V2 = REPO / "data" / "bestvater_wm_stance_v2.csv"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_bestvater_wm_stance_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "stance", "sentiment", "balanced_train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_sentiment": pd.to_numeric(df["sentiment"], errors="coerce"),
            "source_balanced_train": pd.to_numeric(df["balanced_train"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "gt_pro_womens_march": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["gt_pro_womens_march"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_pro_womens_march"] = out["gt_pro_womens_march"].astype(int)

    if not out["gt_pro_womens_march"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_pro_womens_march"].isin([0, 1]), "gt_pro_womens_march"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary stance labels: {bad}")

    conflict_counts = out.groupby("text")["gt_pro_womens_march"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_pro_womens_march"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "wm_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_sentiment",
            "source_balanced_train",
            "text",
            "gt_pro_womens_march",
        ]
    ].reset_index(drop=True)


def build_bestvater_wm_stance_task_v2(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"text", "stance", "sentiment", "balanced_train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    if len(df) != 19612:
        raise ValueError(
            f"Expected the 19,612-row original CSV, got {len(df)} rows. "
            "Use the original upload (?format=original), not the ingested .tab."
        )

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_sentiment": pd.to_numeric(df["sentiment"], errors="coerce"),
            "source_balanced_train": pd.to_numeric(df["balanced_train"], errors="coerce"),
            "text": df["text"].map(_clean_text).map(lambda t: EMAIL_RE.sub("[EMAIL]", t)),
            "gt_pro_womens_march": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )
    out = out[out["source_balanced_train"].eq(1)].copy()
    out = out.dropna(subset=["gt_pro_womens_march"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_pro_womens_march"] = out["gt_pro_womens_march"].astype(int)
    if not out["gt_pro_womens_march"].isin([0, 1]).all():
        raise ValueError("Encountered non-binary stance labels.")

    conflict_counts = out.groupby("text")["gt_pro_womens_march"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_pro_womens_march"]).reset_index(drop=True)
    if out["text"].duplicated().any():
        raise ValueError("Duplicate text remained after cleaning.")

    out["source_id"] = "wmv2_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_sentiment",
            "source_balanced_train",
            "text",
            "gt_pro_womens_march",
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
            f"Input file not found: {input_path}. Download WM_tweets_groundtruth.tab "
            "from Harvard Dataverse DOI 10.7910/DVN/MUYYG4 first, or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if v2:
        df = build_bestvater_wm_stance_task_v2(input_path)
    else:
        df = build_bestvater_wm_stance_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_pro_womens_march"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
