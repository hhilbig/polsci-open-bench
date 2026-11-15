#!/usr/bin/env python3
"""
Build a long-text policy-topic task from US executive orders.

Labels come from the Policy Agendas Project executive-order dataset
(comparativeagendas.net/us, "Executive Orders"), which gives every order since
1945 one CAP major-topic code. The codebook says the code "corresponds to the
information found in the Description column". That column is, in practice, the
order's title: it matches the Federal Register title at fuzzy ratio >= 90 for
97% of orders (median 9 words). The title is kept at the top of the text the
model sees, so the task gives the model the label's basis plus the full body.
It tests whether a reader of the whole order reaches the topic CAP assigned
from the title, and the body can point elsewhere (a 25-order spot check found
about 15% clearly and 20% plausibly better coded under another topic). It
shares the CAP scheme with `cap_crs_policy_topic` and
`cap_party_platform_policy_topic`.

Text comes from the Federal Register API, which carries executive orders from
EO 12890 (December 1993) on. Orders are matched on EO number. The raw text
files are GPO plain-text renderings; cleaning keeps the order from its
heading for that order's own number ("Executive Order NNNNN of <date>", also
"No." and headings without "of") through the signature block, and drops NUL
bytes, page markers, graphic placeholders, billing codes and the filing line.
Matching the order's own number matters: amendment orders mention the order
they amend on the cover page, and a generic heading pattern starts there.

CAP label corrections (LABEL_FIXES): the CAP file's codes for EO 13673 (Fair Pay
and Safe Workplaces, coded Health) and EO 13674 (quarantinable communicable
diseases, coded Labor) are swapped relative to the orders' content; the builder
swaps them back.

Rows are dropped when (in this order): the CAP major topic is missing or not a
standard topic; the EO number does not match exactly one Federal Register
document; the cleaned text is under 100 words; the text is above
MAX_DOC_TOKENS (10,000 tokens under the stricter of two tokenizers; see
long_text_budget.py). Commemorative orders are kept.

Steps:
1. Download the CAP Executive Orders CSV from
   https://www.comparativeagendas.net/us/datasets (file `eo.csv`, latin-1).
2. `--fetch` downloads the Federal Register listing and raw texts into
   `<input-dir>/fr_texts/` (one request per order, cached; about 1,500
   requests, no key needed).
3. Run without `--fetch` to write data/cap_executive_order_policy_topic.csv.
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS
from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT_DIR = Path("/tmp/cap_executive_orders")
DEFAULT_OUTPUT = REPO / "data" / "cap_executive_order_policy_topic.csv"
FR_API = "https://www.federalregister.gov/api/v1/documents.json"
MIN_WORDS = 100


def fetch_listing(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "fr_listing.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    params = {
        "conditions[presidential_document_type]": "executive_order",
        "per_page": 1000,
        "order": "oldest",
        "fields[]": ["executive_order_number", "signing_date", "raw_text_url",
                     "title", "document_number"],
    }
    rows = []
    resp = requests.get(FR_API, params=params, timeout=60).json()
    rows += resp["results"]
    while resp.get("next_page_url"):
        resp = requests.get(resp["next_page_url"], timeout=60).json()
        rows += resp["results"]
    path.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


def fetch_texts(input_dir: Path, listing: pd.DataFrame) -> None:
    out_dir = input_dir / "fr_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(listing.itertuples()):
        target = out_dir / f"{row.document_number}.txt"
        if target.exists() or not isinstance(row.raw_text_url, str):
            continue
        for attempt in range(5):
            try:
                resp = requests.get(row.raw_text_url, timeout=60)
                resp.raise_for_status()
                target.write_text(resp.text)
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        time.sleep(0.25)
        if i % 100 == 0:
            print(f"fetched through {i}/{len(listing)}", flush=True)


LABEL_FIXES = {13673: 5, 13674: 3}  # EO number -> corrected CAP major topic


def _heading(eo_number: int) -> re.Pattern:
    return re.compile(rf"Executive Order (?:No\.\s*)?{eo_number}\b(?:\s+of\b)?")


def clean_order_text(raw: str, eo_number: int) -> str:
    text = re.sub(r"<[^>]+>", " ", raw).replace("\x00", " ")
    start = _heading(eo_number).search(text)
    if start is None:
        return ""
    text = text[start.start():]
    text = re.split(r"\[?FR Doc\.? [A-Z]?\d", text)[0]
    text = re.sub(r"\[GRAPHIC\]\s*\[TIFF OMITTED\]\s*\S+", " ", text)
    text = re.sub(r"\[\[Page \d+\]\]", " ", text)
    text = re.sub(r"Billing code [\w-]+", " ", text)
    text = text.replace("(Presidential Sig.)", " ")
    text = re.sub(r"&lt;GRAPHIC\(S\) NOT AVAILABLE IN TIFF FORMAT&gt;|GRAPHIC\(S\) NOT AVAILABLE IN TIFF FORMAT", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [" ".join(p.split()) for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p and p not in {">", "<", "<>"})


def build(input_dir: Path) -> pd.DataFrame:
    cap = pd.read_csv(input_dir / "eo.csv", encoding="latin-1")
    required = {"eo_number", "signed_date", "year", "description", "majortopic", "subtopic", "president"}
    missing = required - set(cap.columns)
    if missing:
        raise ValueError(f"eo.csv is missing columns: {sorted(missing)}")
    n0 = len(cap)
    cap = cap[pd.to_numeric(cap["majortopic"], errors="coerce").isin(CAP_MAJOR_TOPIC_LABELS)].copy()
    cap["majortopic"] = cap["majortopic"].astype(int)
    for eo, topic in LABEL_FIXES.items():
        cap.loc[pd.to_numeric(cap["eo_number"], errors="coerce").eq(eo), "majortopic"] = topic
    cap["eo_number"] = pd.to_numeric(cap["eo_number"], errors="coerce")

    listing = fetch_listing(input_dir)
    listing["eo_number"] = pd.to_numeric(listing["executive_order_number"], errors="coerce")
    listing = listing.dropna(subset=["eo_number"])
    counts = listing["eo_number"].value_counts()
    listing = listing[listing["eo_number"].map(counts).eq(1)]

    merged = cap.merge(listing, on="eo_number", how="inner")
    n_matched = len(merged)
    texts = []
    for doc, eo in zip(merged["document_number"], merged["eo_number"].astype(int)):
        path = input_dir / "fr_texts" / f"{doc}.txt"
        texts.append(clean_order_text(path.read_text(), eo) if path.exists() else "")
    merged["text"] = texts
    merged["n_words"] = merged["text"].str.split().str.len().fillna(0).astype(int)
    out = merged[merged["n_words"].ge(MIN_WORDS)].copy()
    n_min = len(out)
    out["n_tokens"] = out["text"].map(doc_tokens)
    out = out[out["n_tokens"].le(MAX_DOC_TOKENS)].copy()
    print(f"CAP rows {n0}; standard topic {len(cap)}; matched to one FR document {n_matched}; "
          f">= {MIN_WORDS} words {n_min}; <= {MAX_DOC_TOKENS} tokens {len(out)}")

    out = pd.DataFrame({
        "source_id": "cap_eo_" + out["eo_number"].astype(int).astype(str),
        "eo_number": out["eo_number"].astype(int),
        "fr_document_number": out["document_number"],
        "signed_date": out["signed_date"],
        "year": out["year"],
        "president": out["president"],
        "cap_description": out["description"].map(lambda v: " ".join(str(v).split())),
        "source_major_topic": out["majortopic"],
        "source_subtopic": out["subtopic"],
        "n_words": out["n_words"],
        "n_tokens": out["n_tokens"],
        "text": out["text"],
        "gt_policy_topic": out["majortopic"].map(CAP_MAJOR_TOPIC_LABELS),
    })
    if out["source_id"].duplicated().any() or out["text"].duplicated().any():
        raise ValueError("duplicate ids or texts after cleaning")
    return out.sort_values("eo_number").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--fetch", action="store_true", help="download Federal Register texts, then stop")
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    if not (input_dir / "eo.csv").exists():
        raise FileNotFoundError(f"{input_dir / 'eo.csv'} not found; download it from comparativeagendas.net first")
    if args.fetch:
        listing = fetch_listing(input_dir)
        fetch_texts(input_dir, listing)
        return
    df = build(input_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())
    print(df["gt_policy_topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
