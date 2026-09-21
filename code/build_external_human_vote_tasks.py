#!/usr/bin/env python3
"""Build three frozen 1,000-item tasks from licensed multi-annotator sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

URLS = {
    "hatexplain": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
    "measuring_hate": "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/data/train-00000-of-00001.parquet",
    "mfrc": "https://huggingface.co/datasets/USC-MOLA-Lab/MFRC/resolve/main/data/train_dedup-00000-of-00001.parquet",
}
SEED = 20260822


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "polsci-open-bench/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def select(frame: pd.DataFrame, id_col: str, n: int = 1000) -> pd.DataFrame:
    if frame[id_col].duplicated().any():
        raise ValueError(f"duplicate IDs before sampling: {id_col}")
    ranked = frame.assign(_rank=frame[id_col].map(lambda value: hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()))
    out = ranked.sort_values("_rank").head(n).drop(columns="_rank").reset_index(drop=True)
    if len(out) != n:
        raise ValueError(f"expected {n} sampled rows, found {len(out)}")
    return out


def hatexplain(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    mapping = {"hatespeech": "HATE", "offensive": "OFFENSIVE", "normal": "NORMAL"}
    for key, item in data.items():
        labels = [mapping[a["label"].lower()] for a in item["annotators"]]
        counts = Counter(labels)
        gold, votes = counts.most_common(1)[0]
        if votes <= len(labels) / 2:
            continue
        rows.append({"source_id": f"hatexplain_{key}", "text": " ".join(item["post_tokens"]),
                     "gt_label": gold, "n_votes": len(labels), "vote_share_majority": votes / len(labels),
                     **{f"votes_{label}": counts[label] for label in mapping.values()}})
    return select(pd.DataFrame(rows), "source_id")


def measuring_hate(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["comment_id", "hatespeech", "text"])
    mapping = {0.0: "NOT_HATE", 1.0: "UNCLEAR", 2.0: "HATE"}
    rows = []
    for key, group in frame.dropna(subset=["hatespeech"]).groupby("comment_id", sort=False):
        labels = [mapping[float(v)] for v in group["hatespeech"]]
        counts = Counter(labels)
        gold, votes = counts.most_common(1)[0]
        if len(labels) < 3 or votes <= len(labels) / 2:
            continue
        rows.append({"source_id": f"mhs_{key}", "text": str(group["text"].iloc[0]), "gt_label": gold,
                     "n_votes": len(labels), "vote_share_majority": votes / len(labels),
                     **{f"votes_{label}": counts[label] for label in mapping.values()}})
    return select(pd.DataFrame(rows), "source_id")


def mfrc(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    labels = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity", "Thin Morality", "Non-Moral"]
    per_vote = frame.groupby(["text", "annotator"])["annotation"].agg(lambda x: set(map(str, x)))
    rows = []
    for idx, (text, votes) in enumerate(per_vote.groupby(level=0, sort=False)):
        vote_sets = list(votes)
        if len(vote_sets) < 3:
            continue
        row = {"source_id": f"mfrc_{hashlib.sha256(text.encode()).hexdigest()[:20]}", "text": text,
               "n_votes": len(vote_sets)}
        per_label_shares = []
        for label in labels:
            yes = sum(label in vote for vote in vote_sets)
            row[f"votes_{label}"] = yes
            row[f"gt_{label}"] = int(yes > len(vote_sets) / 2)
            per_label_shares.append(max(yes, len(vote_sets) - yes) / len(vote_sets))
        row["vote_share_majority"] = sum(per_label_shares) / len(per_label_shares)
        rows.append(row)
    return select(pd.DataFrame(rows), "source_id")


def build(output: Path, cache: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    frames = {
        "hatexplain_votes.csv": hatexplain(fetch(URLS["hatexplain"], cache / "hatexplain.json")),
        "measuring_hate_speech_votes.csv": measuring_hate(fetch(URLS["measuring_hate"], cache / "measuring_hate.parquet")),
        "mfrc_votes.csv": mfrc(fetch(URLS["mfrc"], cache / "mfrc.parquet")),
    }
    for name, frame in frames.items():
        if len(frame) != 1000 or frame["source_id"].duplicated().any():
            raise ValueError(f"invalid frozen output: {name}")
        frame.to_csv(output / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(); build(args.output, args.cache)


if __name__ == "__main__": main()
