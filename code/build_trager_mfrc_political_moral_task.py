#!/usr/bin/env python3
"""
Build a multi-label moral-foundations task from the political subreddits of
the Moral Foundations Reddit Corpus (MFRC).

Source: Trager et al., "The Moral Foundations Reddit Corpus" (arXiv
2208.05545; LREC 2026, pp. 6383-6407). Data: Hugging Face dataset
USC-MOLA-Lab/MFRC at revision ddc21d2f03e156732fd1ba95a51c4c29f07975be, file
data/train_dedup-00000-of-00001.parquet (SHA-256 below; 53,827 rows). This is
the same file that code/build_external_human_vote_tasks.py reads. It equals
the repository's final_mfrc_data.csv (61,226 rows) after removing its 7,399
exact duplicate rows. Licence: CC BY 4.0 (dataset card, "Licensing
Information").

Unit: one Reddit comment (English, 10-100 words), shown alone. Only the two
political buckets are used: "US Politics" (r/politics, r/Conservative,
r/antiwork) and "French politics" (r/europe, r/worldnews, r/neoliberal,
r/geopolitics, r/Conservative). The "Everyday Morality" bucket is dropped
because it is not political text. The sample is not random: half of each
bucket was drawn from comments with a high dictionary-based moral loading
score (paper sect. 2), so moral content is more common than on Reddit.

The release has one row per (comment, annotator). An annotation is a
comma-separated set of labels: the six foundations (Care, Equality,
Proportionality, Loyalty, Authority, Purity), Thin Morality, or Non-Moral.

Gold per foundation = strict majority: more than half of the comment's
annotators included the foundation. This rule is ours. The authors' tables
use "at least 50% agreement for that label" (paper sect. 4), which also
counts 2-of-4 ties as positive; their rule is kept in `authors_rule_*`
columns and both counts are printed. Thin Morality and Non-Moral are not
labels in this task (the task asks about the six foundations only).

Steps and drops, in order:
1. Read train_dedup (53,827 rows, 17,886 distinct texts).
2. Keep the political buckets (drop Everyday Morality rows).
3. Same-annotator duplicates: some (text, annotator) pairs have more than one
   row. Labels are compared as sets, so "Authority,Equality" and
   "Equality,Authority" are the same vote. Same-set repeats count as one vote.
   If the sets differ, that annotator's vote on that text is dropped (the
   other annotators' votes on the text are kept). Counts are printed.
4. Keep texts with at least 3 remaining annotators (the corpus design is
   "at least three trained annotators"; texts with 2 are dropped).
5. Foundation filter: a foundation is kept as a label only if its
   strict-majority positives are at least 1% of the final items, i.e. a
   random 500-item draw would contain about 5 or more. Purity fails this rule
   (printed) and is left out; its votes stay in `votes_Purity`.
Texts are the unit after step 3, so there are no duplicate texts. Comments
are short, so no token count or length cap is applied.

`source_id` is "mfrc_" + the first 20 hex digits of sha256(text), the same id
as code/build_external_human_vote_tasks.py. Column `in_human_vote_set` marks
the texts that are also in data/human_votes/external/mfrc_votes.csv (the
repo's 1,000-item annotator-disagreement set); they are kept.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

REVISION = "ddc21d2f03e156732fd1ba95a51c4c29f07975be"
FILE = "train_dedup.parquet"
URL = (f"https://huggingface.co/datasets/USC-MOLA-Lab/MFRC/resolve/{REVISION}/"
       "data/train_dedup-00000-of-00001.parquet")
SHA256 = "ba77ffa85cfff34c86d5ee81a9da124eb38c5b36dea275a2c63c089872c6cf6e"
DEFAULT_INPUT_DIR = Path("/tmp/mfrc")
DEFAULT_OUTPUT = REPO / "data" / "trager_mfrc_political_moral.csv"
HUMAN_VOTE_FILE = REPO / "data" / "human_votes" / "external" / "mfrc_votes.csv"
POLITICAL_BUCKETS = ["US Politics", "French politics"]
FOUNDATIONS = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity"]
OTHER = ["Thin Morality", "Non-Moral"]
MIN_ANNOTATORS = 3
MIN_POSITIVE_SHARE = 0.01


def _fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / FILE
    if not path.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "polsci-open-bench/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} does not match {SHA256}")
    return path


def build(input_dir: Path):
    raw = pd.read_parquet(_fetch(input_dir))
    print(f"1. rows read: {len(raw)} ({raw['text'].nunique()} distinct texts)")

    df = raw[raw["bucket"].isin(POLITICAL_BUCKETS)].copy()
    print(f"2. political buckets: {len(df)} rows, {df['text'].nunique()} texts "
          f"(dropped {len(raw) - len(df)} Everyday Morality rows, "
          f"{raw['text'].nunique() - df['text'].nunique()} texts)")
    if df.groupby("text")["subreddit"].nunique().gt(1).any():
        raise ValueError("a text appears in more than one subreddit")

    df["labels"] = df["annotation"].map(lambda s: frozenset(x.strip() for x in s.split(",")))
    unknown = set().union(*df["labels"]) - set(FOUNDATIONS + OTHER)
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    pair = df.groupby(["text", "annotator"])["labels"]
    n_rows = pair.transform("size")
    n_sets = pair.transform(lambda s: len(set(s)))
    conflict = n_sets.gt(1)
    n_conflict_pairs = df[conflict].groupby(["text", "annotator"]).ngroups
    n_repeat_pairs = df[~conflict & n_rows.gt(1)].groupby(["text", "annotator"]).ngroups
    df = df[~conflict].drop_duplicates(["text", "annotator"])
    print(f"3. same-annotator duplicates: {n_repeat_pairs} (text, annotator) pairs repeated with the "
          f"same label set (kept once); {n_conflict_pairs} pairs with different label sets "
          f"({int(conflict.sum())} rows) dropped; {len(df)} votes left")

    rows = []
    for text, g in df.groupby("text", sort=True):
        n = len(g)
        row = {"text": text, "subreddit": g["subreddit"].iloc[0], "bucket": g["bucket"].iloc[0],
               "n_annotators": n}
        for lab in FOUNDATIONS + OTHER:
            row[f"votes_{lab}"] = int(sum(lab in s for s in g["labels"]))
        rows.append(row)
    items = pd.DataFrame(rows)
    keep = items["n_annotators"].ge(MIN_ANNOTATORS)
    print(f"4. texts with >= {MIN_ANNOTATORS} annotators: {int(keep.sum())} "
          f"(dropped {int((~keep).sum())} texts with fewer); "
          f"annotators per text {items.loc[keep, 'n_annotators'].value_counts().sort_index().to_dict()}")
    items = items[keep].copy()

    for lab in FOUNDATIONS:
        items[f"gt_{lab}"] = (2 * items[f"votes_{lab}"] > items["n_annotators"]).astype(int)
        items[f"authors_rule_{lab}"] = (2 * items[f"votes_{lab}"] >= items["n_annotators"]).astype(int)
    labels = []
    for lab in FOUNDATIONS:
        share = items[f"gt_{lab}"].mean()
        ok = share >= MIN_POSITIVE_SHARE
        print(f"5. {lab}: {int(items[f'gt_{lab}'].sum())} positives ({share:.2%}; "
              f"{500 * share:.1f} expected in a random 500) -> {'kept' if ok else 'left out'}")
        if ok:
            labels.append(lab)
        else:
            items = items.drop(columns=[f"gt_{lab}", f"authors_rule_{lab}"])

    items["source_id"] = "mfrc_" + items["text"].map(lambda t: hashlib.sha256(t.encode()).hexdigest()[:20])
    if items["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    used = set(pd.read_csv(HUMAN_VOTE_FILE, usecols=["source_id"])["source_id"])
    items["in_human_vote_set"] = items["source_id"].isin(used)
    print(f"   overlap with {HUMAN_VOTE_FILE.relative_to(REPO)}: {int(items['in_human_vote_set'].sum())} "
          f"of {len(used)} texts (kept, flagged in in_human_vote_set)")
    items["language"] = "en"
    items["n_words"] = items["text"].str.split().str.len()
    cols = (["source_id", "subreddit", "bucket", "language", "n_words", "n_annotators",
             "in_human_vote_set", "text"]
            + [f"votes_{l}" for l in FOUNDATIONS + OTHER]
            + [f"authors_rule_{l}" for l in labels] + [f"gt_{l}" for l in labels])
    return items[cols].sort_values("source_id").reset_index(drop=True), labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory for {FILE} (downloaded from Hugging Face if missing)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df, labels = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    for lab in labels:
        print(f"  {lab}: {int(df[f'gt_{lab}'].sum())} strict-majority yes "
              f"({int(df[f'authors_rule_{lab}'].sum())} under the authors' at-least-50% rule)")
    print(f"  no foundation: {int((df[[f'gt_{l}' for l in labels]].sum(axis=1) == 0).sum())}")
    print(f"  by bucket: {df['bucket'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
