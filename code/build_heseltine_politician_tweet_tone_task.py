#!/usr/bin/env python3
"""
Build a three-way tone task from politicians' tweets in four countries.

Source: Heseltine and Clemm von Hohenberg (2024), "Large language models as a
substitute for human experts in annotating political text", Research &
Politics, doi:10.1177/20531680241236239. Replication data: Harvard Dataverse
doi:10.7910/DVN/V2P6YL (CC0 1.0), file Python.rar (Dataverse file id 8138344,
MD5 a5706d733444ee0edf235467c2cd2719), folder Python/ValidationDatasets/.
Coding instructions and coder breakdown: the OSF preprint (osf.io/cx752),
Supplementary Information A (coders) and B (instructions).

Unit and label. One tweet sent by a politician. The four validation sets are
pooled: United States (Neg3way_Val_GPT_USA.csv, English), Germany
(Germany_Val_All.csv), Italy (Italy_Val_All.csv) and Chile
(Chile_Val_All.csv, Spanish). Gold is the expert three-way sentiment code:
`label` in the US file, `neg3_label` in the other three, coded 0 negative,
1 neutral, 2 positive (SI C.5; SI B.3 gives the definitions without numbers).
The mapping agrees with the binary negativity code in the same files:
neg3 = 0 coincides with neg = 1 in all but 16 of 1,995 tweets, which are kept
as released. US tweets were coded by two experts and disagreements were
reconciled; German, Italian and Chilean tweets were coded by one expert each
(SI A).

Columns never read: every GPT* column and every *_Recon / *_Reconciled column
(model outputs). MediaTextSample_GPT_Val.csv (200 news articles) is not used.
The US gold comes from one file only, so no join across files is needed; the
script joins the US binary-negativity file on text (never by row position,
since Ideo_Val_GPT_USA.csv has a different row order) only to count the
neg/neg3 disagreements it reports.

Text repairs, in order (mechanical; counts are printed):
1. Mojibake. Some texts (mostly US) were UTF-8 encoded, then decoded as
   Windows-1252, sometimes twice ("ItÂ’s", "Ã¢â‚¬â„¢"). Every substring that
   has the byte shape of a UTF-8 sequence when read back as Windows-1252
   bytes (lead byte C2-F4 followed by the right number of 80-BF continuation
   bytes; undefined cp1252 positions are taken as the same Latin-1 byte) and
   that decodes as strict UTF-8 is replaced by the decoded character; this
   repeats until nothing changes. Then any remaining C1 control character
   (U+0080-U+009F, left over when the original was itself cp1252 read as
   Latin-1) is replaced by its Windows-1252 character ("\\x92" -> "’"), or
   removed if cp1252 leaves that position undefined. ftfy is not used, so the
   rule is fully stated here.
2. Line endings: runs of "\\r" before "\\n" (US texts carry "\\r\\r\\r\\r\\n") and
   lone "\\r" become "\\n".
3. Twitter's HTML escapes "&amp;", "&lt;", "&gt;" become "&", "<", ">".
4. Leading/trailing whitespace is stripped.

Drops: none expected. Empty texts and exact duplicate texts (after repair)
are handled by the standard rule: rows of a text with conflicting labels are
all dropped, one row of a same-label repeat is kept. Retweets truncated by
Twitter at 140 characters (ending in "…") are kept: the data hold no other
version of these texts, so they are what the coders read.

source_id: "hes_<country>_" + the first 12 hex digits of the SHA-1 of the
text exactly as released (before repair). The source has no tweet ids.
"""
import argparse
import hashlib
import html
import re
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

URL = "https://dataverse.harvard.edu/api/access/datafile/8138344"
RAR_NAME = "Python.rar"
RAR_MD5 = "a5706d733444ee0edf235467c2cd2719"
DEFAULT_INPUT_DIR = Path("/tmp/heseltine_v2p6yl")
DEFAULT_OUTPUT = REPO / "data" / "heseltine_politician_tweet_tone.csv"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

LABELS = ["negative", "neutral", "positive"]
CODE_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# file, country, language, gold column
FILES = [
    ("Neg3way_Val_GPT_USA.csv", "us", "English", "label"),
    ("Germany_Val_All.csv", "de", "German", "neg3_label"),
    ("Italy_Val_All.csv", "it", "Italian", "neg3_label"),
    ("Chile_Val_All.csv", "cl", "Spanish", "neg3_label"),
]
COUNTRY_NAMES = {"us": "United States", "de": "Germany", "it": "Italy", "cl": "Chile"}


# --- mojibake repair -------------------------------------------------------

def _sloppy_byte(ch: str):
    """Byte that `ch` stands for when a byte string was decoded as cp1252.

    Defined cp1252 characters map back to their byte; code points below 256
    (including the C1 positions cp1252 leaves undefined) map to the same byte.
    """
    try:
        return ch.encode("cp1252")[0]
    except UnicodeEncodeError:
        return ord(ch) if ord(ch) < 256 else None


_CANDIDATES = {chr(c) for c in range(0x80, 0x100)} | {
    bytes([b]).decode("cp1252", errors="ignore") for b in range(0x80, 0xA0)} - {""}
# Characters standing for a UTF-8 continuation byte (0x80-0xBF).
_CONT = "".join(sorted(c for c in _CANDIDATES if 0x80 <= (_sloppy_byte(c) or 0) <= 0xBF))
_CONT_CLASS = "[" + re.escape(_CONT) + "]"
_MOJIBAKE = re.compile(
    "[Â-ß]" + _CONT_CLASS
    + "|[à-ï]" + _CONT_CLASS + "{2}"
    + "|[ð-ô]" + _CONT_CLASS + "{3}"
)
_C1 = re.compile("[\u0080-\u009f]")


def _decode_match(m: re.Match) -> str:
    raw = bytes(_sloppy_byte(c) for c in m.group(0))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return m.group(0)


def _c1_to_cp1252(m: re.Match) -> str:
    try:
        return bytes([ord(m.group(0))]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


def repair_mojibake(text: str) -> str:
    for _ in range(5):
        new = _MOJIBAKE.sub(_decode_match, text)
        if new == text:
            break
        text = new
    return _C1.sub(_c1_to_cp1252, text)


def clean(text: str) -> str:
    text = repair_mojibake(text)
    text = re.sub(r"\r+\n", "\n", text).replace("\r", "\n")
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


# --- download --------------------------------------------------------------

def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    rar = input_dir / RAR_NAME
    if not rar.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            rar.write_bytes(resp.read())
    md5 = hashlib.md5(rar.read_bytes()).hexdigest()
    if md5 != RAR_MD5:
        raise ValueError(f"{rar} has MD5 {md5}, expected {RAR_MD5}")
    out = input_dir / "extracted"
    val = out / "Python" / "ValidationDatasets"
    if not val.exists():
        out.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(out)], check=True)
    return val


def build(val_dir: Path) -> pd.DataFrame:
    frames = []
    for fname, country, language, gold in FILES:
        raw = pd.read_csv(val_dir / fname, usecols=["text", gold], keep_default_na=False)
        raw = raw.rename(columns={gold: "code"})
        raw["country_code"] = country
        raw["language"] = language
        raw["source_file"] = fname
        print(f"{fname}: {len(raw)} rows; codes {raw['code'].value_counts().sort_index().to_dict()}")
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True)
    print(f"pooled: {len(df)}")
    if not df["code"].isin(CODE_TO_LABEL).all():
        raise ValueError(f"unexpected codes: {sorted(set(df['code']) - set(CODE_TO_LABEL))}")

    # Diagnostic only: binary negativity vs three-way code.
    us_bin = pd.read_csv(val_dir / "Neg_Val_GPT_USA.csv", usecols=["text", "label"], keep_default_na=False)
    if us_bin["text"].duplicated().any():
        raise ValueError("duplicate texts in Neg_Val_GPT_USA.csv")
    us = df[df["country_code"] == "us"].merge(us_bin.rename(columns={"label": "neg"}), on="text", how="left")
    if us["neg"].isna().any():
        raise ValueError("US texts missing from Neg_Val_GPT_USA.csv")
    others = pd.concat([pd.read_csv(val_dir / f, usecols=["text", "neg_label", "neg3_label"], keep_default_na=False)
                        for f, c, _, _ in FILES if c != "us"])
    n_contra = int(((us["neg"] == 1) != (us["code"] == 0)).sum()
                   + ((others["neg_label"] == 1) != (others["neg3_label"] == 0)).sum())
    print(f"tweets whose binary negativity code disagrees with the three-way code (kept): {n_contra}")

    df["source_id"] = "hes_" + df["country_code"] + "_" + df["text"].map(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:12])
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")

    repaired = df["text"].map(repair_mojibake)
    df["mojibake_repaired"] = repaired.ne(df["text"])
    print(f"texts changed by the mojibake repair: {int(df['mojibake_repaired'].sum())} "
          f"(by country {df.groupby('country_code')['mojibake_repaired'].sum().to_dict()})")
    df["text_released"] = df["text"]
    df["text"] = df["text"].map(clean)
    print(f"texts changed by any repair (mojibake, line endings, HTML escapes, strip): "
          f"{int(df['text'].ne(df['text_released']).sum())}")
    left = df["text"].str.contains("Â|Ã|â€|�", regex=True)
    print(f"texts still containing mojibake markers after repair: {int(left.sum())}")

    empty = df["text"].eq("")
    df = df[~empty].copy()
    print(f"after dropping empty texts: {len(df)} (dropped {int(empty.sum())})")
    n_lab = df.groupby("text")["code"].nunique()
    conflict = df["text"].isin(n_lab[n_lab > 1].index)
    df = df[~conflict].copy()
    print(f"after dropping texts with conflicting labels: {len(df)} (dropped {int(conflict.sum())})")
    dup = df["text"].duplicated()
    df = df[~dup].copy()
    print(f"after keeping one row of same-label repeats: {len(df)} (dropped {int(dup.sum())})")

    df["is_retweet"] = df["text"].str.startswith("RT @")
    df["truncated_retweet"] = df["is_retweet"] & df["text"].str.endswith("…")
    print(f"truncated retweets ending in '…' (kept): {int(df['truncated_retweet'].sum())} "
          f"(by country {df.groupby('country_code')['truncated_retweet'].sum().to_dict()})")
    print(f"any text ending in '…' (kept): {int(df['text'].str.endswith('…').sum())}")

    df["country"] = df["country_code"].map(COUNTRY_NAMES)
    df["coding"] = df["country_code"].map(lambda c: "two experts, reconciled" if c == "us" else "single expert")
    df["gt_tone"] = df["code"].map(CODE_TO_LABEL)
    df["n_words"] = df["text"].str.split().str.len()
    cols = ["source_id", "country", "language", "coding", "source_file", "is_retweet", "truncated_retweet",
            "mojibake_repaired", "n_words", "text", "gt_tone"]
    return df[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory for {RAR_NAME} (downloaded from {URL} if absent) and its extraction")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_tone"].value_counts().reindex(LABELS).to_string())
    print(pd.crosstab(df["country"], df["gt_tone"]).to_string())


if __name__ == "__main__":
    main()
