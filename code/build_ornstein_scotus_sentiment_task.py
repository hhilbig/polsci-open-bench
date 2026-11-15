#!/usr/bin/env python3
"""
Build the Ornstein et al. Supreme Court tweet-sentiment task files.

Source: the `scotus_tweets` and `scotus_tweets_examples` datasets shipped with
the promptr R package (github.com/joeornstein/promptr, files
data/scotus_tweets.rda and data/scotus_tweets_examples.rda; last changed in
commits f85d56b and f928291, February 2024). These are the 945 tweets of
Ornstein, Blasingame and Truscott (2025), "How to Train Your Stochastic
Parrot", PSRM 13(2), 264-281, each with three author-coded sentiment scores
(expert1-expert3: -1 negative, 0 neutral, +1 positive), and the 12 tweets the
authors used as few-shot examples in their pre-registered prompts (Table A3).

Download the two .rda files from the promptr repository and pass them with
--input and --examples. Reading them needs the `pyreadr` package.

v1 (data/ornstein_scotus.csv; byte-identical to the file the published release
used):
1. Keep all 945 tweets in source order, with tweet_id, text and case as given.
   `case` is "An identifier denoting which Supreme Court ruling the tweet was
   collected after" (promptr documentation), not a topic code.
2. Replace e-mail addresses in the text with "[EMAIL]" (one tweet).
3. Gold = sign of the mean of the available expert scores (NA scores ignored):
   > 0 Positive, < 0 Negative, 0 Neutral.

v2 (data/ornstein_scotus_v2.csv): v1 minus the 12 tweets that the authors used
as few-shot examples, because the v2 prompt reproduces the authors' prompts
including those examples. Gold is unchanged for every remaining tweet.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_OUTPUT_V1 = REPO / "data" / "ornstein_scotus.csv"
DEFAULT_OUTPUT_V2 = REPO / "data" / "ornstein_scotus_v2.csv"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EXPERTS = ["expert1", "expert2", "expert3"]


def _read_rda(path: Path, name: str) -> pd.DataFrame:
    import pyreadr

    return pyreadr.read_r(str(path))[name]


def build_v1(tweets: pd.DataFrame) -> pd.DataFrame:
    missing = {"tweet_id", "text", "case", *EXPERTS} - set(tweets.columns)
    if missing:
        raise ValueError(f"scotus_tweets is missing columns: {sorted(missing)}")
    score = tweets[EXPERTS].mean(axis=1, skipna=True)
    if score.isna().any():
        raise ValueError("a tweet has no expert score")
    gold = np.where(score > 0, "Positive", np.where(score < 0, "Negative", "Neutral"))
    out = pd.DataFrame(
        {
            "tweet_id": tweets["tweet_id"].astype(str),
            "text": tweets["text"].astype(str).map(lambda t: EMAIL_RE.sub("[EMAIL]", t)),
            "case": tweets["case"].astype(str),
            "gt_sentiment": gold,
        }
    )
    if out["tweet_id"].duplicated().any():
        raise ValueError("duplicate tweet_id in source")
    if out["text"].str.strip().eq("").any():
        raise ValueError("empty tweet text in source")
    return out


def build_v2(v1: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    ex_ids = set(examples["tweet_id"].astype(str))
    missing = ex_ids - set(v1["tweet_id"])
    if missing:
        raise ValueError(f"few-shot example ids not in scotus_tweets: {sorted(missing)}")
    return v1[~v1["tweet_id"].isin(ex_ids)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="promptr data/scotus_tweets.rda")
    ap.add_argument("--examples", required=True, help="promptr data/scotus_tweets_examples.rda")
    ap.add_argument("--output-v1", default=str(DEFAULT_OUTPUT_V1))
    ap.add_argument("--output-v2", default=str(DEFAULT_OUTPUT_V2))
    args = ap.parse_args()

    tweets = _read_rda(Path(args.input), "scotus_tweets")
    examples = _read_rda(Path(args.examples), "scotus_tweets_examples")

    v1 = build_v1(tweets)
    v2 = build_v2(v1, examples)
    for df, path in [(v1, args.output_v1), (v2, args.output_v2)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")
        print(pd.crosstab(df["case"], df["gt_sentiment"]).to_string())


if __name__ == "__main__":
    main()
