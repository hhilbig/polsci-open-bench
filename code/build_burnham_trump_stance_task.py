#!/usr/bin/env python3
"""
Build a cleaned Trump stance task from Burnham's public replication data.

The replication archive exposes public text plus direct stance labels for
Donald Trump. This builder combines the training and held-out adjudicated test
files, normalizes whitespace, removes exact texts with conflicting labels, and
deduplicates repeated text-label pairs.

Version 2 (`--version 2`) reads the original CSV uploads
(`/api/access/datafile/10383704?format=original` for train and
`/api/access/datafile/10383678?format=original` for test) instead of the
Dataverse-ingested `.tab` exports. In the `.tab` exports, rows whose text ends
in an escaped quote absorb the following columns into the text field
(26 rows: `dataset` becomes "0", "1" or "0.0", and 12 of them carry the wrong
stance label), and embedded quotes and newlines appear as literal `\"` and
`\n`. The original CSVs parse cleanly with the same row order, so v2 keeps the
v1 item ids. v2 also replaces e-mail addresses with "[EMAIL]".

Texts that do not name Trump (`target_mention == 0`) are kept: they are part
of Burnham's test design (section 3.3 of the paper compares performance on
documents with and without the term "Trump"), and the v2 prompt says the
stance may be implicit.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("/tmp/burnham_trump_train_data.tab")
DEFAULT_TEST = Path("/tmp/burnham_trump_test_data.tab")
DEFAULT_OUTPUT = REPO / "data" / "burnham_trump_stance.csv"
DEFAULT_TRAIN_V2 = Path("/tmp/burnham_trump_train_data.csv")
DEFAULT_TEST_V2 = Path("/tmp/burnham_trump_test_data.csv")
DEFAULT_OUTPUT_V2 = REPO / "data" / "burnham_trump_stance_v2.csv"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KNOWN_DATASETS = {"mtsd", "Kawintiranon", "Titling", "semeval", "constance", "newsletters"}

LABEL_MAP = {
    -1: "Oppose",
    0: "Neutral",
    1: "Support",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _read(path: Path, original_csv: bool) -> pd.DataFrame:
    if original_csv:
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def _prepare_train(train_path: Path, original_csv: bool = False) -> pd.DataFrame:
    df = _read(train_path, original_csv)
    required = {"text", "stance", "dataset", "target_mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{train_path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_split": "train",
            "source_row": range(1, len(df) + 1),
            "source_dataset": df["dataset"].astype(str),
            "source_target_mention": pd.to_numeric(df["target_mention"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "source_stance": pd.to_numeric(df["stance"], errors="coerce"),
        }
    )


def _prepare_test(test_path: Path, original_csv: bool = False) -> pd.DataFrame:
    df = _read(test_path, original_csv)
    required = {"text", "adjudicated_label", "dataset", "target_mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{test_path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_split": "test",
            "source_row": range(1, len(df) + 1),
            "source_dataset": df["dataset"].astype(str),
            "source_target_mention": pd.to_numeric(df["target_mention"], errors="coerce"),
            "text": df["text"].map(_clean_text),
            "source_stance": pd.to_numeric(df["adjudicated_label"], errors="coerce"),
        }
    )


def build_burnham_trump_stance_task(
    train_path: Path, test_path: Path, version: int = 1
) -> pd.DataFrame:
    v2 = version == 2
    out = pd.concat(
        [_prepare_train(train_path, v2), _prepare_test(test_path, v2)], ignore_index=True
    )
    if v2:
        bad = sorted(set(out["source_dataset"]) - KNOWN_DATASETS)
        if bad:
            raise ValueError(f"Unexpected dataset values (column shift?): {bad}")
        out["text"] = out["text"].map(lambda t: EMAIL_RE.sub("[EMAIL]", t))
    out = out.dropna(subset=["source_stance"]).copy()
    out = out[out["text"].ne("")].copy()
    out["source_stance"] = out["source_stance"].astype(int)

    if not out["source_stance"].isin(LABEL_MAP).all():
        bad = sorted(out.loc[~out["source_stance"].isin(LABEL_MAP), "source_stance"].unique().tolist())
        raise ValueError(f"Encountered unsupported stance labels: {bad}")

    conflict_counts = out.groupby("text")["source_stance"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "source_stance"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["gt_stance_toward_trump"] = out["source_stance"].map(LABEL_MAP)
    out["source_id"] = "trump_" + out["source_split"] + "_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_split",
            "source_row",
            "source_dataset",
            "source_target_mention",
            "text",
            "gt_stance_toward_trump",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    v2 = args.version == 2
    if args.train is None:
        args.train = str(DEFAULT_TRAIN_V2 if v2 else DEFAULT_TRAIN)
    if args.test is None:
        args.test = str(DEFAULT_TEST_V2 if v2 else DEFAULT_TEST)
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if v2 else DEFAULT_OUTPUT)

    train_path = Path(args.train)
    test_path = Path(args.test)
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download Burnham's Trump stance files "
                "from Harvard Dataverse DOI 10.7910/DVN/XS0ULP first, or pass paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_burnham_trump_stance_task(train_path, test_path, version=args.version)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_stance_toward_trump"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
