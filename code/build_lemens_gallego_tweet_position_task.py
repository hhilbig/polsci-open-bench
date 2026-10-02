#!/usr/bin/env python3
"""
Build a tweet-level left-right position dataset from Le Mens and Gallego (2025).

Le Mens and Gallego, "Positioning Political Texts with Large Language Models by
Asking and Averaging," Political Analysis 33(3): 274-282, use 900 tweets
published by members of the 118th US Congress (450 Democratic, 450
Republican). In November 2023, 597 Prolific participants each rated 30 tweets
by answering: "Where does this text stand on the "left" to "right" wing scale?
If the text does not have political content, select "Not Applicable"." The
rating is a 0 (extremely left) to 100 (extremely right) slider. The paper's
benchmark is the mean of a tweet's non-NA ratings; one tweet has no non-NA
rating and is dropped, leaving 899 tweets.

Inputs, both inside capsule-1286058.zip in the replication archive
(doi:10.7910/DVN/YFM0BW, CC0):
- data/data_tweets_118th_congress/data_raw/test_set_tweet_info.csv
- data/data_tweets_118th_congress/data_analysis/USTLR1_human_positions.csv
"""
import argparse
import html
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TWEETS = Path("/tmp/lemens_gallego/test_set_tweet_info.csv")
DEFAULT_RATINGS = Path("/tmp/lemens_gallego/USTLR1_human_positions.csv")
DEFAULT_OUTPUT = REPO / "data" / "lemens_gallego_tweet_position.csv"


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    # Tweets carry HTML entities such as &amp; from the Twitter API.
    return " ".join(html.unescape(str(value)).split())


def build_lemens_gallego_tweet_position_task(tweets_path: Path, ratings_path: Path) -> pd.DataFrame:
    tweets = pd.read_csv(tweets_path)
    ratings = pd.read_csv(ratings_path)
    for frame, path, required in [
        (tweets, tweets_path, {"id", "category", "text"}),
        (ratings, ratings_path, {"id", "category", "pos_human", "part_id"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if tweets["id"].duplicated().any():
        raise ValueError("Duplicate tweet ids in the tweet file")
    if ratings.duplicated(["id", "part_id"]).any():
        raise ValueError("A participant rated the same tweet more than once")
    valid = ratings["pos_human"].dropna()
    if not valid.between(0, 100).all():
        raise ValueError("Ratings outside the 0-100 slider range")

    stats = ratings.groupby("id")["pos_human"].agg(
        human_position_mean="mean",
        human_position_median="median",
        human_position_sd="std",
        n_position_ratings="count",
    )
    stats["n_raters"] = ratings.groupby("id").size()
    stats["n_not_applicable"] = stats["n_raters"] - stats["n_position_ratings"]

    out = tweets.merge(stats, left_on="id", right_index=True, how="left", validate="one_to_one")
    out["text"] = out["text"].map(_clean_text)
    n_before = len(out)
    out = out[out["n_position_ratings"] > 0].copy()
    print(f"dropped {n_before - len(out)} tweet(s) with no non-NA rating")
    if out["text"].eq("").any():
        raise ValueError("Empty tweet text after cleaning")

    # Three-bin stopgap label: equal thirds of the 0-100 slider. The cutoffs
    # are a benchmark choice, not the authors'; the mean is the source measure.
    out["gt_position3"] = pd.cut(
        out["human_position_mean"], bins=[0, 100 / 3, 200 / 3, 100],
        labels=["left", "center", "right"], include_lowest=True,
    ).astype(str)

    out["source_id"] = "lemens_tweet_" + out["id"].astype(str)
    out = out.rename(columns={"id": "source_tweet_index", "category": "source_party"})
    for col in ["human_position_mean", "human_position_median", "human_position_sd"]:
        out[col] = out[col].round(3)
    return out[[
        "source_id", "source_tweet_index", "source_party", "text",
        "human_position_mean", "human_position_median", "human_position_sd",
        "n_position_ratings", "n_not_applicable", "n_raters", "gt_position3",
    ]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tweets", default=str(DEFAULT_TWEETS))
    ap.add_argument("--ratings", default=str(DEFAULT_RATINGS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    for path in [Path(args.tweets), Path(args.ratings)]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Extract it from capsule-1286058.zip in "
                "Harvard Dataverse doi:10.7910/DVN/YFM0BW, or pass the path."
            )

    df = build_lemens_gallego_tweet_position_task(Path(args.tweets), Path(args.ratings))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"wrote {output} ({len(df)} rows)")
    print(df.groupby("source_party")["human_position_mean"].describe().round(1).to_string())
    print(pd.crosstab(df["source_party"], df["gt_position3"]).to_string())


if __name__ == "__main__":
    main()
