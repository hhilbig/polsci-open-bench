#!/usr/bin/env python3
"""
Build a long-text issue-area task from US Supreme Court opinions.

Labels are Supreme Court Database (SCDB; Spaeth et al.) issue areas, as paired
with opinion text in the LexGLUE `scotus` dataset (Chalkidis et al. 2022,
huggingface.co/datasets/coastalcph/lex_glue, CC-BY-4.0 per the dataset card;
the opinions are US government works and SCDB is public). SCDB coders assign
each case one of 278 issues, grouped into 14 issue areas; LexGLUE keeps 13 of
them (Private Action, area 14, does not occur) and stores them as labels 0-12
for areas 1-13.

Input is the LexGLUE test split (1,400 opinions), which is enough for a
100-item benchmark sample and keeps the download at 40 MB.

Steps and drops, in order:
1. Map labels 0-12 to SCDB issue-area names (area = label + 1).
2. Drop Miscellaneous (area 13): it is a residual category with 3 cases in the
   split, too few to sample.
3. Normalize whitespace within lines, drop opinions under 200 words, and drop
   exact duplicate texts.
4. The id is built from the first line of the text (the U.S. Reports citation,
   e.g. "502 U.S. 314"); a running suffix separates the rare cases that share
   a citation line.

5. Drop opinions above MAX_DOC_TOKENS (10,000 tokens under the stricter of two
   tokenizers; see
   long_text_budget.py) so the whole opinion fits the benchmark context.
   Opinions are long (median about 6,900 words in the split), so this keeps
   roughly the shorter half. Opinions are not truncated.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/lex_glue_scotus/test.parquet")
DEFAULT_OUTPUT = REPO / "data" / "scdb_opinion_issue_area.csv"
MIN_WORDS = 200

ISSUE_AREAS = {
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
}
DROP_AREAS = {13}


def _clean(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text).replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if set(df.columns) != {"text", "label"}:
        raise ValueError(f"unexpected columns {df.columns.tolist()}")
    n0 = len(df)
    df["issue_area"] = df["label"].astype(int) + 1
    if not df["issue_area"].isin(ISSUE_AREAS).all():
        raise ValueError("label outside SCDB issue areas 1-13")
    df = df[~df["issue_area"].isin(DROP_AREAS)].copy()
    n1 = len(df)
    df["text"] = df["text"].map(_clean)
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"].ge(MIN_WORDS)].drop_duplicates(subset=["text"]).copy()
    n2 = len(df)
    df["n_tokens"] = df["text"].map(doc_tokens)
    df = df[df["n_tokens"].le(MAX_DOC_TOKENS)].copy()
    print(f"opinions {n0}; without Miscellaneous {n1}; >= {MIN_WORDS} words, deduplicated {n2}; "
          f"<= {MAX_DOC_TOKENS} tokens {len(df)}")

    first = df["text"].str.split("\n").str[0].map(lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40])
    df["source_id"] = "scdb_" + first + "_" + df.groupby(first).cumcount().astype(str)
    decided = df["text"].str.extract(r"Decided\s+[A-Z][a-z]+\.?\s+\d{1,2},\s+(\d{4})")[0]
    df["decided_year"] = pd.to_numeric(decided, errors="coerce")
    df["gt_issue_area"] = df["issue_area"].map(ISSUE_AREAS)
    if df["source_id"].duplicated().any():
        raise ValueError("duplicate source ids")
    return df[["source_id", "decided_year", "issue_area", "n_words", "n_tokens", "text", "gt_issue_area"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="LexGLUE scotus test parquet (refs/convert/parquet/scotus/test/0000.parquet)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_issue_area"].value_counts().to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
