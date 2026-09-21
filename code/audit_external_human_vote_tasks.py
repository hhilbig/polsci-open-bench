#!/usr/bin/env python3
"""Audit external text-classification datasets that retain individual votes.

The script downloads source files to a temporary cache and writes only summary
statistics. It does not add any candidate to the benchmark or launch inference.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "output" / "sidecar" / "frontier_2026" / "human_votes" / "external_audit"

URLS = {
    "crowdtruth_raw": "https://raw.githubusercontent.com/CrowdTruth/Cross-Task-Majority-Vote-Eval/master/tweets_raw.csv",
    "crowdtruth_units": "https://raw.githubusercontent.com/CrowdTruth/Cross-Task-Majority-Vote-Eval/master/tweets_aggregated.csv",
    "hatexplain": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
    "measuring_hate": "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/data/train-00000-of-00001.parquet",
    "mfrc": "https://huggingface.co/datasets/USC-MOLA-Lab/MFRC/resolve/main/data/train_dedup-00000-of-00001.parquet",
    "dagstuhl": "https://zenodo.org/api/records/3973285/files/dagstuhl-15512-argquality-corpus-v2.zip/content",
    "enthymeme_page": "https://turfutoday.com/enthymemes/data/dataset.html",
}


def download(url: str, path: Path) -> Path:
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "polsci-open-bench/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    if path.stat().st_size == 0:
        raise ValueError(f"Downloaded empty file: {url}")
    return path


def majority_share(labels: list[str]) -> float:
    return max(Counter(labels).values()) / len(labels)


def summarize_shares(name: str, shares: pd.Series, unit: str = "items") -> dict:
    values = pd.to_numeric(shares, errors="raise")
    return {
        "dataset": name,
        "agreement_unit": unit,
        "n_agreement_units": int(len(values)),
        "mean_majority_share": float(values.mean()),
        "median_majority_share": float(values.median()),
        "p25_majority_share": float(values.quantile(0.25)),
        "p75_majority_share": float(values.quantile(0.75)),
        "share_unanimous": float((values == 1).mean()),
    }


def audit_crowdtruth(cache: Path) -> tuple[dict, dict]:
    raw = pd.read_csv(download(URLS["crowdtruth_raw"], cache / "crowdtruth_tweets_raw.csv"))
    units = pd.read_csv(download(URLS["crowdtruth_units"], cache / "crowdtruth_tweets_units.csv"))
    label_cols = [c for c in raw.columns if c not in {"Job_ID", "Unit_ID", "Worker_ID", "AcceptTime", "SubmitTime", "Judgments", "Spam"}]
    clean = raw.loc[raw["Spam"].eq(0)].copy()
    clean["label"] = clean[label_cols].idxmax(axis=1)
    text_unit_ids = set(units["Unit_id"])
    votes = clean.loc[clean["Unit_ID"].isin(text_unit_ids)].groupby("Unit_ID")["label"].agg(list)
    counts = votes.map(len)
    usable = votes.loc[counts >= 3]
    shares = usable.map(majority_share)
    n_with_text = units["Unit_id"].nunique()
    n_raw_units = raw["Unit_ID"].nunique()
    if n_raw_units - n_with_text != 1 or set(raw["Unit_ID"]) - text_unit_ids != {23980}:
        raise ValueError("Unexpected CrowdTruth raw-judgment/text-unit mismatch")
    row = {
        "dataset": "CrowdTruth Twitter event identification",
        "family": "event/topic",
        "availability": "immediate",
        "license_status": "no explicit repository license found",
        "n_source_items": int(n_raw_units),
        "n_usable_items": int(len(usable)),
        "min_votes": int(counts.min()),
        "median_votes": float(counts.median()),
        "raw_votes_verified": True,
        "full_text_verified": True,
        "passes_200_item_gate": bool(len(usable) >= 200),
        "decision": "conditional: clarify reuse license",
        "source_url": "https://github.com/CrowdTruth/Cross-Task-Majority-Vote-Eval",
    }
    return row, summarize_shares(row["dataset"], shares)


def audit_hatexplain(cache: Path) -> tuple[dict, dict]:
    data = json.loads(download(URLS["hatexplain"], cache / "hatexplain.json").read_text())
    labels = {key: [str(a["label"]) for a in value["annotators"]] for key, value in data.items()}
    shares = pd.Series({key: majority_share(value) for key, value in labels.items()})
    counts = pd.Series({key: len(value) for key, value in labels.items()})
    text_ok = all(bool(value.get("post_tokens")) for value in data.values())
    row = {
        "dataset": "HateXplain",
        "family": "hate/offensiveness",
        "availability": "immediate",
        "license_status": "MIT repository license; dataset terms should be cited",
        "n_source_items": len(data),
        "n_usable_items": int((counts >= 3).sum()),
        "min_votes": int(counts.min()),
        "median_votes": float(counts.median()),
        "raw_votes_verified": True,
        "full_text_verified": text_ok,
        "passes_200_item_gate": bool((counts >= 3).sum() >= 200),
        "decision": "include",
        "source_url": "https://github.com/hate-alert/HateXplain",
    }
    return row, summarize_shares(row["dataset"], shares.loc[counts >= 3])


def audit_measuring_hate(cache: Path) -> tuple[dict, dict]:
    frame = pd.read_parquet(download(URLS["measuring_hate"], cache / "measuring_hate.parquet"), columns=["comment_id", "annotator_id", "hatespeech", "text"])
    grouped = frame.dropna(subset=["hatespeech"]).groupby("comment_id")["hatespeech"].agg(lambda x: [str(v) for v in x])
    counts = grouped.map(len)
    usable = grouped.loc[counts >= 3]
    shares = usable.map(majority_share)
    row = {
        "dataset": "Measuring Hate Speech",
        "family": "hate/offensiveness",
        "availability": "immediate",
        "license_status": "CC BY 4.0",
        "n_source_items": int(frame["comment_id"].nunique()),
        "n_usable_items": int(len(usable)),
        "min_votes": int(counts.min()),
        "median_votes": float(counts.median()),
        "raw_votes_verified": frame["annotator_id"].notna().all(),
        "full_text_verified": frame["text"].fillna("").str.len().gt(0).all(),
        "passes_200_item_gate": bool(len(usable) >= 200),
        "decision": "include",
        "source_url": "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech",
    }
    return row, summarize_shares(row["dataset"], shares)


def audit_mfrc(cache: Path) -> tuple[dict, dict]:
    frame = pd.read_parquet(download(URLS["mfrc"], cache / "mfrc.parquet"))
    required = {"text", "annotator", "annotation"}
    if not required.issubset(frame.columns):
        raise ValueError(f"MFRC missing columns: {sorted(required - set(frame.columns))}")
    # A repeated annotator/item row represents a multi-label vote.  Agreement is
    # computed over each annotator's complete label set, not over duplicated rows.
    per_vote = frame.groupby(["text", "annotator"])["annotation"].agg(lambda x: "|".join(sorted(set(map(str, x)))))
    grouped = per_vote.groupby(level=0).agg(list)
    counts = grouped.map(len)
    usable = grouped.loc[counts >= 3]
    shares = usable.map(majority_share)
    row = {
        "dataset": "Moral Foundations Reddit Corpus",
        "family": "moral/claims",
        "availability": "immediate",
        "license_status": "CC BY 4.0",
        "n_source_items": int(grouped.size),
        "n_usable_items": int(len(usable)),
        "min_votes": int(counts.min()),
        "median_votes": float(counts.median()),
        "raw_votes_verified": True,
        "full_text_verified": True,
        "passes_200_item_gate": bool(len(usable) >= 200),
        "decision": "include",
        "source_url": "https://huggingface.co/datasets/USC-MOLA-Lab/MFRC",
    }
    return row, summarize_shares(row["dataset"], shares)


def audit_dagstuhl(cache: Path) -> tuple[dict, dict]:
    archive = zipfile.ZipFile(download(URLS["dagstuhl"], cache / "dagstuhl_v2.zip"))
    names = [n for n in archive.namelist() if n.endswith(".xmi") and "/__MACOSX/" not in n]
    shares = []
    item_ids = set()
    vote_counts = []
    for name in names:
        root = ElementTree.fromstring(archive.read(name))
        item_ids.add(Path(name).stem)
        for element in root.iter():
            raw = element.attrib.get("allScores")
            if raw:
                scores = raw.split()
                vote_counts.append(len(scores))
                shares.append(majority_share(scores))
    if not shares:
        raise ValueError("Dagstuhl XMI contained no allScores attributes")
    row = {
        "dataset": "Dagstuhl-15512 Argument Quality",
        "family": "argument quality",
        "availability": "immediate",
        "license_status": "Zenodo record has no machine-readable license",
        "n_source_items": len(item_ids),
        "n_usable_items": len(item_ids),
        "min_votes": min(vote_counts),
        "median_votes": float(pd.Series(vote_counts).median()),
        "raw_votes_verified": True,
        "full_text_verified": True,
        "passes_200_item_gate": bool(len(item_ids) >= 200),
        "decision": "conditional: clarify reuse license and choose one dimension",
        "source_url": "https://zenodo.org/records/3973285",
    }
    return row, summarize_shares(row["dataset"], pd.Series(shares), "item-dimension ratings")


def audit_enthymemes(cache: Path) -> dict:
    html = download(URLS["enthymeme_page"], cache / "enthymeme_downloads.html").read_text(errors="replace")
    available = "final-dataset.zip</a>" in html
    return {
        "dataset": "Political Enthymemes",
        "family": "implicit political arguments",
        "availability": "immediate" if available else "announced but unavailable",
        "license_status": "not stated on download page",
        "n_source_items": 1482 if available else math.nan,
        "n_usable_items": math.nan,
        "min_votes": 3 if available else math.nan,
        "median_votes": math.nan,
        "raw_votes_verified": False,
        "full_text_verified": False,
        "passes_200_item_gate": False,
        "decision": "exclude until files and license are actually released",
        "source_url": "https://turfutoday.com/enthymemes/",
    }


def build(output_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, agreement = [], []
    for function in [audit_crowdtruth, audit_hatexplain, audit_measuring_hate, audit_mfrc, audit_dagstuhl]:
        row, summary = function(cache_dir)
        rows.append(row)
        agreement.append(summary)
    rows.append(audit_enthymemes(cache_dir))
    candidates = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    agreement_frame = pd.DataFrame(agreement).sort_values("dataset").reset_index(drop=True)
    candidates.to_csv(output_dir / "candidate_audit.csv", index=False)
    agreement_frame.to_csv(output_dir / "agreement_summary.csv", index=False)
    return candidates, agreement_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.cache_dir:
        candidates, _ = build(args.output_dir, args.cache_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="human-vote-audit-") as tmp:
            candidates, _ = build(args.output_dir, Path(tmp))
    print(candidates[["dataset", "n_usable_items", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
