#!/usr/bin/env python3
"""
Build a three-class sentiment task from the English part of ParlaSent 1.0.

Source: Mochtak, Rupnik and Ljubešić (2023), "The ParlaSent multilingual
training dataset for sentiment identification in parliamentary proceedings"
(arXiv:2309.09783). Data: Hugging Face dataset classla/ParlaSent, file
ParlaSent_EN.jsonl at revision 0587c2b6499fbc68a7623439c2af2b24748968dc
(also deposited at http://hdl.handle.net/11356/1868). Licence: CC BY-SA 4.0
(dataset card).

Unit: one sentence from a UK House of Commons or House of Lords speech
(2,600 sentences; paper section 3). Only ParlaSent_EN.jsonl is used.
ParlaSent_EN_test.jsonl is not: it was coded by a single annotator, sampled
without the seed-word step, and its label is that one annotator's code
(paper sections 3.3 and 3.5), so it is a different procedure. The file's
internal train/dev/test `split` column is kept as metadata only.

Gold: the `label` column, which the authors derive from the reconciled
six-level code (`reconciliation`) by keeping the part after the underscore
(paper section 3.4): Positive and M_Positive -> positive; Negative and
M_Negative -> negative; P_Neutral and N_Neutral -> neutral. Two trained
annotators coded every sentence and reconciled disagreements; Krippendorff's
alpha on the English set is 0.656 for the three-class labels and 0.543 for
the six-level codes (paper Table 1).
The builder checks that `label` equals this collapse of `reconciliation` on
every row. Individual annotator codes are kept as metadata columns.

Sampling note: the training sets were drawn from sentences of interquartile
length, stratified by sentiment-lexicon seed words (867 with negative seeds,
867 with positive, 866 with neither; paper section 3.3), so class shares do
not reflect the share of sentiment in parliamentary speech.

Drops, in order: rows with an empty sentence; duplicate sentences (all rows
of a sentence with conflicting labels dropped, one row of a same-label repeat
kept). No rows are dropped in the release.

`source_id` is "parlasent_en_<date as YYYYMMDD>_<document_id>_<sentence_id>",
built from source fields only. The date is needed because the release's
(document_id, sentence_id) pair is not unique: 912 rows share a pair with a
different sentence from a different sitting, so those ids are not global
keys on their own. With the date added, every row has a unique id.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

REVISION = "0587c2b6499fbc68a7623439c2af2b24748968dc"
FILE = "ParlaSent_EN.jsonl"
URL = f"https://huggingface.co/datasets/classla/ParlaSent/resolve/{REVISION}/{FILE}"
SHA256 = "77285ca68109875d4e6b9876ec32aef4bafb745d6c5c4d500d5cbb6ee55eb3e3"
DEFAULT_INPUT_DIR = Path("/tmp/parlasent")
DEFAULT_OUTPUT = REPO / "data" / "parlasent_en_sentiment.csv"
LABELS = ["negative", "neutral", "positive"]
COLLAPSE = {
    "Positive": "positive", "M_Positive": "positive",
    "Negative": "negative", "M_Negative": "negative",
    "P_Neutral": "neutral", "N_Neutral": "neutral",
}


def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / FILE
    if not path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        raise ValueError(f"{path}: sha256 {got} != expected {SHA256}")
    return path


def build(input_dir: Path) -> pd.DataFrame:
    df = pd.read_json(fetch(input_dir), lines=True, dtype={"document_id": str, "sentence_id": str})
    print(f"rows read: {len(df)}")
    assert set(df["reconciliation"]) == set(COLLAPSE), set(df["reconciliation"])
    collapsed = df["reconciliation"].map(COLLAPSE)
    assert collapsed.eq(df["label"].str.lower()).all(), "label is not the collapse of reconciliation"
    df["gt_sentiment"] = collapsed

    df["text"] = df["sentence"].str.strip()
    empty = df["text"].eq("")
    df = df[~empty].copy()
    print(f"empty sentences dropped: {int(empty.sum())}; remaining {len(df)}")
    key = df["text"].str.replace(r"\s+", " ", regex=True)
    conflict = df.groupby(key)["gt_sentiment"].transform("nunique").gt(1)
    n_before = len(df)
    df = df[~conflict].copy()
    df = df[~key[df.index].duplicated()].copy()
    print(f"duplicates: {int(conflict.sum())} conflicting rows dropped; "
          f"{n_before - int(conflict.sum()) - len(df)} same-label repeats dropped; remaining {len(df)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    pair_dup = df.duplicated(["document_id", "sentence_id"], keep=False)
    print(f"rows whose (document_id, sentence_id) pair is shared with another sentence: {int(pair_dup.sum())}")
    df["source_id"] = ("parlasent_en_" + df["date"].str.replace("-", "") + "_"
                       + df["document_id"] + "_" + df["sentence_id"])
    assert df["source_id"].is_unique
    df["language"] = "en"
    df["year"] = df["date"].str[:4].astype(int)
    df = df.rename(columns={"reconciliation": "label6_reconciled", "split": "source_split"})
    cols = ["source_id", "country", "language", "date", "year", "term", "name", "party", "gender", "ruling",
            "source_split", "annotator1", "annotator2", "label6_reconciled", "text", "gt_sentiment"]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the raw JSONL; downloaded from the pinned revision if missing")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_sentiment"].value_counts().reindex(LABELS).to_string())
    print(df["text"].str.split().str.len().describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
