#!/usr/bin/env python3
"""
Build a three-class response-clarity task from QEvasion / CLARITY
(Thomas et al. 2024, "'I Never Said That': A dataset, taxonomy and baselines
on response clarity classification", Findings of EMNLP 2024, arXiv:2409.13879).

EXCEPTION TO THE NO-MODEL-TEXT RULE. The sub-question shown to the model was
written by ChatGPT, not by the interviewer. The authors had ChatGPT split each
multi-part interview question into sub-questions; annotators checked each
sub-question as valid or not and could add missing ones (paper sect. 4,
"Human annotation process"). The clarity label is assigned to the answer
with respect to one sub-question, so the sub-question defines the unit being
labeled and cannot be left out. The model sees the original interview
question, this sub-question and the full original answer. No other
model-generated column (gpt3.5_summary, gpt3.5_prediction, both empty in the
test split) and no annotator column is shown.

Source: Hugging Face dataset ailsntua/QEvasion, file
data/test-00000-of-00001.parquet at revision
3afc18f0b582b3cfdb927822cff57ddc6e871f9c (sha256 below). Licence: CC BY-NC-ND
4.0 (dataset card). Under the no-derivatives term the text and label fields
are kept exactly as released: this builder only selects rows and adds
id and length columns. No whitespace or encoding changes are made.

Unit: one (interview question, sub-question, answer) triple from a US
presidential interview or press conference (2006-2023). Several sub-questions
share one interview question and answer.

Gold: `clarity_label` (Clear Reply, Ambivalent, Clear Non-Reply). Only the
test split is used: its 308 items were labeled by three non-expert
annotators with the nine-way evasion taxonomy (columns annotator1-3), and
`clarity_label` is the majority of the three codes after collapsing them to
the three clarity classes (paper Fig. 3 / Table 4). The builder checks this
on every row. The train split is single-annotator and is not used. Fleiss
kappa on clarity among the non-expert annotators is 0.644 (paper sect. 4,
p. 6).

Drops, in order:
1. Items whose three annotator codes, collapsed to clarity, have no majority
   (each annotator chose a different clarity class). The release labels these
   rows anyway; the label is not a majority, so they are dropped (2 rows).
2. Exact duplicates of (interview question, sub-question, answer): all rows
   of a triple with conflicting labels dropped, one row of same-label
   repeats kept (none in the release).
3. Items above MAX_DOC_TOKENS (10,000 tokens, larger of o200k and cl100k; see
   long_text_budget.py), counted on the three fields together. Dropped, not
   truncated (none expected; the longest answer is about 1,300 words).

`source_id` is "qevasion_" plus the first 12 hex digits of the SHA-1 of the
source url, question_order and sub-question, so it does not depend on row
position. The release's own `index` column is kept as `source_index`.
"""
import argparse
import collections
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

REVISION = "3afc18f0b582b3cfdb927822cff57ddc6e871f9c"
FILE = "data/test-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/ailsntua/QEvasion/resolve/{REVISION}/{FILE}"
SHA256 = "d1de67bd7b7ea68b7f49fcd1f468fd434bc2de3d84146c9217d36713c0546298"
DEFAULT_INPUT_DIR = Path("/tmp/thomas_qevasion")
DEFAULT_OUTPUT = REPO / "data" / "thomas_qevasion_clarity.csv"
LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
# Evasion taxonomy -> clarity (paper Table 4 and Fig. 3).
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}
ANNOTATORS = ["annotator1", "annotator2", "annotator3"]


def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / "test-00000-of-00001.parquet"
    if not path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        raise ValueError(f"{path}: sha256 {got} != expected {SHA256}")
    return path


def majority(row) -> str | None:
    counts = collections.Counter(EVASION_TO_CLARITY[row[a]] for a in ANNOTATORS)
    label, n = counts.most_common(1)[0]
    return label if n >= 2 else None


def build(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"0. test rows {len(df)}")
    print(df["clarity_label"].value_counts().reindex(LABELS).to_string())
    if not set(df["clarity_label"]) <= set(LABELS):
        raise ValueError("unexpected clarity_label")
    for col in ["interview_question", "question", "interview_answer"]:
        if df[col].isna().any() or df[col].str.strip().eq("").any():
            raise ValueError(f"empty {col}")

    df["annotator_majority"] = df.apply(majority, axis=1)
    agree = df["annotator_majority"].notna()
    mismatch = agree & df["annotator_majority"].ne(df["clarity_label"])
    if mismatch.any():
        raise ValueError(f"{int(mismatch.sum())} rows where clarity_label is not the annotator majority")
    dropped = df.loc[~agree, "clarity_label"].value_counts().to_dict()
    df = df[agree].copy()
    print(f"1. drop items with no annotator majority: {int((~agree).sum())} {dropped}; kept {len(df)}")

    key = df["interview_question"] + "\x00" + df["question"] + "\x00" + df["interview_answer"]
    conflict = df.groupby(key)["clarity_label"].transform("nunique").gt(1)
    df = df[~conflict].copy()
    key = key[~conflict]
    before = len(df)
    df = df[~key.duplicated()].copy()
    print(f"2. duplicate triples: dropped {int(conflict.sum())} conflicting rows, "
          f"{before - len(df)} same-label repeats; kept {len(df)}")

    full = df["interview_question"] + "\n" + df["question"] + "\n" + df["interview_answer"]
    df["n_tokens"] = full.map(doc_tokens)
    df["n_words_answer"] = df["interview_answer"].str.split().str.len()
    over = df["n_tokens"].gt(MAX_DOC_TOKENS)
    df = df[~over].copy()
    print(f"3. drop above {MAX_DOC_TOKENS} tokens: {int(over.sum())}; kept {len(df)}")

    ident = df["url"] + "|" + df["question_order"].astype(str) + "|" + df["question"]
    df["source_id"] = "qevasion_" + ident.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12])
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")

    out = pd.DataFrame({
        "source_id": df["source_id"],
        "source_index": df["index"],
        "url": df["url"],
        "question_order": df["question_order"],
        "interview_question": df["interview_question"],
        "sub_question": df["question"],
        "text": df["interview_answer"],
        "n_words_answer": df["n_words_answer"],
        "n_tokens": df["n_tokens"],
        "inaudible": df["inaudible"],
        "multiple_questions": df["multiple_questions"],
        "affirmative_questions": df["affirmative_questions"],
        "evasion_annotator1": df["annotator1"],
        "evasion_annotator2": df["annotator2"],
        "evasion_annotator3": df["annotator3"],
        "gt_clarity": df["clarity_label"],
    })
    return out.sort_values("source_index").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the raw parquet; downloaded and checksummed if missing")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_clarity"].value_counts().reindex(LABELS).to_string())
    print(df["n_words_answer"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())
    print(f"max n_tokens {df['n_tokens'].max()}")


if __name__ == "__main__":
    main()
