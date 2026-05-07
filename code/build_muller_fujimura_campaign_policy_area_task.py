#!/usr/bin/env python3
"""
Build the cleaned Müller and Fujimura campaign-policy-area task from the
public supervised sentence splits in the Harvard Dataverse replication archive.

The task concatenates the public train/test/eval sentence files, normalizes
whitespace, drops exact duplicate texts with conflicting labels, and then
deduplicates repeated text-label pairs. The resulting benchmark slice keeps the
released supervised classification setup rather than reconstructing a broader
raw corpus.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("/tmp/campaign_data_sentences_train.tab")
DEFAULT_TEST = Path("/tmp/campaign_data_sentences_test.tab")
DEFAULT_EVAL = Path("/tmp/campaign_data_sentences_eval.tab")
DEFAULT_OUTPUT = REPO / "data" / "muller_fujimura_campaign_policy_area.csv"

LABEL_MAP = {
    "Agriculture, Forestry, and Fisheries": "Agriculture, Forestry, and Fisheries",
    "Committees on Cabinet": "Committees on Cabinet",
    "Economy, Trade and Industry": "Economy, Trade and Industry",
    "Education, Culture, Sports, Science, and Technology": (
        "Education, Culture, Sports, Science, and Technology"
    ),
    "Environment": "Environment",
    "Financial Affairs": "Financial Affairs",
    "Foreign Affairs": "Foreign Affairs",
    "Health, Labour, and Welfare": "Health, Labour, and Welfare",
    "Internal Affairs and Communications": "Internal Affairs and Communications",
    "Land, Infrastructure, Transport, and Tourism": (
        "Land, Infrastructure, Transport, and Tourism"
    ),
    "No policy area": "No Policy Area",
    "Security": "Security",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_muller_fujimura_campaign_policy_area_task(
    train_path: Path,
    test_path: Path,
    eval_path: Path,
) -> pd.DataFrame:
    parts = []
    for split_name, path in [("train", train_path), ("test", test_path), ("eval", eval_path)]:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        required = {"policy_area_num", "policy_area", "text", "ntoken"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        part = pd.DataFrame(
            {
                "source_split": split_name,
                "source_row_in_split": range(1, len(df) + 1),
                "source_policy_area_num": pd.to_numeric(
                    df["policy_area_num"], errors="raise"
                ).astype(int),
                "source_policy_area_raw": df["policy_area"].map(_clean_text),
                "source_ntoken": pd.to_numeric(df["ntoken"], errors="raise").astype(int),
                "text": df["text"].map(_clean_text),
            }
        )
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out["gt_policy_area"] = out["source_policy_area_raw"].map(LABEL_MAP)

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    if out["gt_policy_area"].isna().any():
        bad = (
            out.loc[out["gt_policy_area"].isna(), "source_policy_area_raw"]
            .drop_duplicates()
            .tolist()[:5]
        )
        raise ValueError(f"Encountered unmapped policy_area labels: {bad}")

    conflict_counts = out.groupby("text")["gt_policy_area"].nunique()
    conflict_texts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflict_texts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_policy_area"]).reset_index(drop=True)

    out["source_id"] = (
        out["source_split"].astype(str) + "_" + out["source_row_in_split"].astype(str)
    )
    if out["source_id"].duplicated().any():
        dup = out.loc[out["source_id"].duplicated(), "source_id"].iloc[0]
        raise ValueError(f"Duplicate source_id found: {dup}")
    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[
        [
            "source_id",
            "source_split",
            "source_row_in_split",
            "source_policy_area_num",
            "source_policy_area_raw",
            "source_ntoken",
            "text",
            "gt_policy_area",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-input", default=str(DEFAULT_TRAIN))
    ap.add_argument("--test-input", default=str(DEFAULT_TEST))
    ap.add_argument("--eval-input", default=str(DEFAULT_EVAL))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    train_path = Path(args.train_input)
    test_path = Path(args.test_input)
    eval_path = Path(args.eval_input)
    for path in [train_path, test_path, eval_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download the public sentence split files first "
                "and pass them explicitly if they are not in /tmp."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_muller_fujimura_campaign_policy_area_task(
        train_path=train_path,
        test_path=test_path,
        eval_path=eval_path,
    )
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_area"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
