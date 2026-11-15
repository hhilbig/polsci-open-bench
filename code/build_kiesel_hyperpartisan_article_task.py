#!/usr/bin/env python3
"""
Build a long-text hyperpartisan-news task from SemEval-2019 Task 4.

Source: Kiesel et al. (2019), "SemEval-2019 Task 4: Hyperpartisan News
Detection", Zenodo record 5776081 (doi:10.5281/zenodo.1489920), CC-BY-4.0 for
articles and labels. Only the by-article part is used: 1,273 news articles
rated by three crowd workers on a 5-point bias scale, binarized by the
organizers (mean rating 4-5 hyperpartisan, 1-2 not) after dropping articles
with low agreement or a mean of "not sure". Agreement on the released set is
Krippendorff's alpha 0.5 (Kiesel et al. 2019, section 3.1), which limits how
closely any coder, human or model, can be expected to match the labels. The much larger by-publisher part
labels articles by their outlet's overall bias and is not a judgment about the
article, so it is not used.

Files: articles-training-byarticle-20181122.xml (645 articles),
articles-test-byarticle-20181207.xml (628), and the matching ground-truth
files. Article ids are unique only within a file, so ids carry the split.

Text: the title, then the article's <p> paragraphs with markup removed.
Text outside <p> elements is dropped, because in these scrapes it is mostly
advertising and image credits (e.g. "No Pill Can Stop Tinnitus, But This 1
Weird Trick Can"). Paragraphs that consist only of link text are dropped.

Drops, in order: articles without a ground-truth row; articles under 100 words
after cleaning; articles above MAX_DOC_TOKENS (none expected); exact duplicate
texts (all copies dropped if their labels conflict, otherwise one kept).
"""
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT_DIR = Path("/tmp/semeval2019_hyperpartisan")
DEFAULT_OUTPUT = REPO / "data" / "kiesel_hyperpartisan_article.csv"
SPLITS = {
    "train": ("articles-training-byarticle-20181122.xml", "ground-truth-training-byarticle-20181122.xml"),
    "test": ("articles-test-byarticle-20181207.xml", "ground-truth-test-byarticle-20181207.xml"),
}
MIN_WORDS = 100


def _find(input_dir: Path, name: str) -> Path:
    hits = list(input_dir.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name} under {input_dir}, found {len(hits)}")
    return hits[0]


def _paragraphs(article: ET.Element) -> list[str]:
    out = []
    for p in article.iter("p"):
        full = " ".join("".join(p.itertext()).split())
        links = " ".join(" ".join("".join(a.itertext()) for a in p.iter("a")).split())
        if full and full != links:
            out.append(full)
    return out


def build(input_dir: Path) -> pd.DataFrame:
    rows = []
    for split, (art_name, gt_name) in SPLITS.items():
        labels = {
            a.get("id"): a.get("hyperpartisan")
            for a in ET.parse(_find(input_dir, gt_name)).getroot().iter("article")
        }
        for art in ET.parse(_find(input_dir, art_name)).getroot().iter("article"):
            aid = art.get("id")
            title = " ".join((art.get("title") or "").split())
            body = "\n\n".join(_paragraphs(art))
            rows.append({
                "source_id": f"semeval19_{split}_{aid}",
                "split": split,
                "published_at": art.get("published-at"),
                "title": title,
                "body": body,
                "label_raw": labels.get(aid),
            })
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df[df["label_raw"].isin(["true", "false"])].copy()
    n1 = len(df)
    df["text"] = (df["title"] + "\n\n" + df["body"]).str.strip()
    df["n_words"] = df["body"].str.split().str.len()
    df = df[df["n_words"].ge(MIN_WORDS)].copy()
    n2 = len(df)
    df["n_tokens"] = df["text"].map(doc_tokens)
    df = df[df["n_tokens"].le(MAX_DOC_TOKENS)].copy()
    n3 = len(df)
    conflict = df.groupby("text")["label_raw"].transform("nunique").gt(1)
    df = df[~conflict].drop_duplicates(subset=["text"]).copy()
    print(f"articles {n0}; labeled {n1}; >= {MIN_WORDS} words {n2}; <= {MAX_DOC_TOKENS} tokens {n3}; "
          f"deduplicated {len(df)}")
    df["gt_hyperpartisan"] = df["label_raw"].map({"true": 1, "false": 0}).astype(int)
    cols = ["source_id", "split", "published_at", "title", "body", "n_words", "n_tokens", "text", "gt_hyperpartisan"]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory holding the unzipped by-article files from Zenodo record 5776081")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_hyperpartisan"].value_counts().to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
