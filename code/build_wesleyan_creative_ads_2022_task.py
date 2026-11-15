#!/usr/bin/env python3
"""
Build the Wesleyan Media Project / CREATIVE 2022 Meta ad-tone task.

Sources (both public):
- fb_2022_train.xlsx: the WMP human-coded 2022 Meta ads, from
  github.com/Wesleyan-Media-Project/datasets (facebook/fb_2022_train.xlsx).
  For each ad, coders list up to eight candidates (OFFICE1..8, CAND1..8) and
  code how each is referenced (TONE1..8): "In a way to show approval or
  support", "In a way to show disaproval or opposition", or "Unclear whether
  support or opposition".
- fb_2022_adid_text.csv.gz: the CREATIVE raw textual fields, figshare DOI
  10.25438/wes02.26124295.

Version 1 (default) reproduces data/wesleyan_creative_ads_2022.csv: one row per
coded ad with CAND1 as the candidate and TONE1 as the gold tone, and the text
fields concatenated with "[field_name]" markers. E-mail addresses are replaced
with "[EMAIL]" as in the repository-wide redaction.

Version 2 (`--version 2`) writes data/wesleyan_creative_ads_2022_v2.csv:
1. Candidate name. When the first coded candidate is not a federal candidate,
   the coding form does not record a name in CAND1: CAND1 holds the office
   category "Down-ballot candidate/governor", or, for 14 ads coded with an
   earlier form version, OFFICE1 holds the unlabeled codes 1370/1372 and CAND1
   a placeholder ("Pete Ricketts" in 13 of them). The coder typed the name of
   that first non-federal candidate into othercand1_txt. v2 uses
   othercand1_txt as the candidate for these rows. Rows whose first candidate
   is "Unidentified politician", or whose othercand1_txt is blank, are dropped
   because the tone target cannot be named.
2. Text. Field markers are replaced by plain headers, literal "\\n" escapes
   become line breaks, ad_creative_bodies entries that repeat the body are
   removed, and identical repeated fields are kept once.
3. Length. The loader no longer truncates at 4,000 characters; ads above
   MAX_DOC_TOKENS (see long_text_budget.py) are dropped instead.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRAIN = Path("fb_2022_train.xlsx")
DEFAULT_TEXT = Path("fb_2022_adid_text.csv.gz")
OUTPUTS = {
    1: REPO / "data" / "wesleyan_creative_ads_2022.csv",
    2: REPO / "data" / "wesleyan_creative_ads_2022_v2.csv",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

TEXT_FIELDS = [
    "ad_creative_body",
    "ad_creative_bodies",
    "ad_creative_link_title",
    "ad_creative_link_description",
    "google_asr_text",
    "aws_ocr_text_img",
    "aws_ocr_text_vid",
]
V2_HEADERS = {
    "ad_creative_body": "Ad text",
    "ad_creative_bodies": "Other ad text versions",
    "ad_creative_link_title": "Link title",
    "ad_creative_link_description": "Link description",
    "google_asr_text": "Video transcript (automatic)",
    "aws_ocr_text_img": "Text in image (automatic OCR)",
    "aws_ocr_text_vid": "Text in video (automatic OCR)",
}
TONE_MAP = {
    "In a way to show approval or support": "promote",
    "In a way to show disaproval or opposition": "attack",
    "Unclear whether support or opposition": "unclear",
}
NONFEDERAL_OFFICE = {"Down-ballot candidate/governor", "1370", "1372"}


def _present(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def _full_text_v1(row) -> str:
    parts = [f"[{f}] {row[f]}" for f in TEXT_FIELDS if _present(row[f])]
    return "\n".join(parts)


def _clean_field(value) -> str:
    text = str(value).replace("\\n", "\n").replace("\r", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _bodies(value) -> list:
    raw = str(value)
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            items = [raw]
    except (json.JSONDecodeError, ValueError):
        items = [raw]
    return [str(x) for x in items]


def _full_text_v2(row) -> str:
    parts, seen = [], set()
    for field in TEXT_FIELDS:
        if not _present(row[field]):
            continue
        values = _bodies(row[field]) if field == "ad_creative_bodies" else [row[field]]
        kept = []
        for value in values:
            cleaned = _clean_field(value)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                kept.append(cleaned)
        if kept:
            parts.append(f"{V2_HEADERS[field]}:\n" + "\n\n".join(kept))
    return "\n\n".join(parts)


def build(train_path: Path, text_path: Path, version: int) -> pd.DataFrame:
    coded = pd.read_excel(train_path, dtype=str)
    coded = coded[coded["TONE1"].notna() & coded["CAND1"].notna()].copy()
    if coded["adid"].duplicated().any():
        raise ValueError("duplicate adid in coded ads")

    text = pd.read_csv(text_path, usecols=["ad_id", "page_name"] + TEXT_FIELDS, dtype=str)
    text = text[text["ad_id"].isin(coded["adid"])].set_index("ad_id")
    missing = set(coded["adid"]) - set(text.index)
    if missing:
        raise ValueError(f"{len(missing)} coded ads have no text row")

    out = pd.DataFrame({"ad_id": coded["adid"].values})
    out["page_name"] = out["ad_id"].map(text["page_name"])
    out["candidate"] = coded["CAND1"].values
    out["gt_tone"] = coded["TONE1"].map(TONE_MAP).values
    if out["gt_tone"].isna().any():
        raise ValueError("unmapped TONE1 value")
    rows = text.loc[out["ad_id"]]

    if version == 1:
        out["full_text"] = [EMAIL_RE.sub("[EMAIL]", _full_text_v1(r)) for _, r in rows.iterrows()]
        out = out[["ad_id", "page_name", "candidate", "full_text", "gt_tone"]]
        return out.sort_values("ad_id").reset_index(drop=True)

    from long_text_budget import MAX_DOC_TOKENS, doc_tokens

    office1 = coded["OFFICE1"].values
    other1 = coded["othercand1_txt"].map(lambda x: " ".join(str(x).split()) if _present(x) else "")
    other1 = other1.str.replace("\xa0", "").str.strip().values
    nonfed = pd.Series(office1).isin(NONFEDERAL_OFFICE).values
    out.loc[nonfed, "candidate"] = other1[nonfed]
    out["candidate_from_othercand1"] = nonfed.astype(int)
    n0 = len(out)
    drop_unident = pd.Series(office1).eq("Unidentified politician").values
    drop_noname = nonfed & (other1 == "")
    print(f"start: {n0} coded ads")
    print(f"first candidate non-federal, name taken from othercand1_txt: {int(nonfed.sum())}")
    print(f"drop, first candidate 'Unidentified politician': {int(drop_unident.sum())}")
    print(f"drop, non-federal first candidate with blank othercand1_txt: {int(drop_noname.sum())}")

    out["full_text"] = [EMAIL_RE.sub("[EMAIL]", _full_text_v2(r)) for _, r in rows.iterrows()]
    keep = ~(drop_unident | drop_noname)
    out = out[keep].copy()
    empty = out["full_text"].eq("")
    print(f"drop, no ad text: {int(empty.sum())}")
    out = out[~empty].copy()
    out["n_tokens"] = out["full_text"].map(doc_tokens)
    over = out["n_tokens"].gt(MAX_DOC_TOKENS)
    print(f"drop, above {MAX_DOC_TOKENS} tokens: {int(over.sum())}")
    out = out[~over].copy()
    out = out[["ad_id", "page_name", "candidate", "candidate_from_othercand1",
               "n_tokens", "full_text", "gt_tone"]]
    return out.sort_values("ad_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-input", default=str(DEFAULT_TRAIN))
    ap.add_argument("--text-input", default=str(DEFAULT_TEXT))
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument("--output")
    args = ap.parse_args()
    for p in [args.train_input, args.text_input]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{p} not found. Download fb_2022_train.xlsx from "
                "github.com/Wesleyan-Media-Project/datasets and fb_2022_adid_text.csv.gz "
                "from figshare DOI 10.25438/wes02.26124295, or pass the paths."
            )
    output = Path(args.output) if args.output else OUTPUTS[args.version]
    df = build(Path(args.train_input), Path(args.text_input), args.version)
    df.to_csv(output, index=False)
    print(f"wrote {output} ({len(df)} rows)")
    print(df["gt_tone"].value_counts().to_string())


if __name__ == "__main__":
    main()
