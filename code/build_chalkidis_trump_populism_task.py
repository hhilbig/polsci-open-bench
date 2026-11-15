#!/usr/bin/env python3
"""
Build a sentence-level populism task from Donald Trump's 2016 campaign speeches.

Source: Chalkidis, Brandl and Aslanidis, "Identifying fine-grained forms of
populism in political discourse: a case study on Donald Trump's
presidential campaigns", EACL 2026 (arXiv:2507.19303). Data: Hugging Face
dataset parisaslanidis/populism-trump-2016, files train.csv and test.csv at
revision 2ce6927ee0ade8a98c649ad3f71501f2788a3717 (MD5 ed43d30e16814cdcae4d3c21c655351e
and 82d7a7da28c387c8687cdc8c2fb34c40). Licence CC BY-NC-ND 4.0.

The licence forbids distributing modified versions, so this builder only
selects rows: the four released codes and the sentence text are kept exactly
as released (no class merging, no text edits, not even whitespace).

Unit and label. One sentence from one of 70 speeches. Four undergraduate
coders, trained on the Degrees of Populism Coding Manual v3.0, each coded a
sentence while reading the full speech (the manual's "holistic reading"); the
released label is one student's code (alpha 0.751 on a subset). Gold is
`pop_code`: 0 no populism, 1 anti-elitism (manual code T), 2 people-centrism
(code U), 3 both (code UT); the mapping is the one in the authors' repository
(coastalcph/populism-llms, audit_llms.py). The model sees the sentence alone.

Steps and drops, in order:
1. Read train.csv and test.csv (15,025 rows; the split is kept as metadata).
   `sentence_position` is the 0-based position of the row within its speech
   in the released file order.
2. Drop sentences under MIN_WORDS whitespace-separated words: fragments such
   as "Thank you." carry almost no information.
3. Duplicate texts, matched after stripping whitespace and lower-casing (the
   released text itself is not changed): drop every row of a text that
   appears with more than one code (the same sentence was coded differently
   in different speeches), and keep the first row (file order) of a text
   repeated with one code.
source_id: speech file stem plus sentence position, e.g. "2016_06_22_trump_s0003".
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

REVISION = "2ce6927ee0ade8a98c649ad3f71501f2788a3717"
BASE_URL = f"https://huggingface.co/datasets/parisaslanidis/populism-trump-2016/resolve/{REVISION}"
FILES = {"train": ("train.csv", "ed43d30e16814cdcae4d3c21c655351e"),
         "test": ("test.csv", "82d7a7da28c387c8687cdc8c2fb34c40")}
DEFAULT_INPUT_DIR = Path("/tmp/chalkidis_populism_trump_2016")
DEFAULT_OUTPUT = REPO / "data" / "chalkidis_trump_populism.csv"
MIN_WORDS = 4

LABELS = ["no populism", "anti-elitism", "people-centrism", "both"]
CODE_TO_LABEL = dict(enumerate(LABELS))


def fetch(input_dir: Path) -> dict:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (name, md5) in FILES.items():
        path = input_dir / name
        if not path.exists():
            req = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "polsci-open-bench-builder"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise ValueError(f"{path} has MD5 {got}, expected {md5}")
        paths[split] = path
    return paths


def build(paths: dict) -> pd.DataFrame:
    frames = []
    for split, path in paths.items():
        x = pd.read_csv(path, keep_default_na=False, dtype={"sentence": str})
        x["split"] = split
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    print(f"released rows: {len(df)}; codes {df['pop_code'].value_counts().sort_index().to_dict()}")
    if not df["pop_code"].isin(CODE_TO_LABEL).all():
        raise ValueError("unexpected pop_code values")
    df["sentence_position"] = df.groupby("speech_id").cumcount()
    df["source_id"] = df["speech_id"].str.replace(r"\.txt$", "", regex=True) + "_s" + \
        df["sentence_position"].map(lambda i: f"{i:04d}")
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    n0 = len(df)

    df["n_words"] = df["sentence"].str.split().str.len()
    short = df["n_words"].lt(MIN_WORDS)
    df = df[~short].copy()
    print(f"after dropping sentences under {MIN_WORDS} words: {len(df)} (dropped {int(short.sum())})")

    key = df["sentence"].str.strip().str.lower()
    n_codes = df.groupby(key)["pop_code"].nunique()
    conflict = key.isin(n_codes[n_codes > 1].index)
    print(f"texts appearing with conflicting codes: {int((n_codes > 1).sum())}; rows dropped: {int(conflict.sum())}")
    df, key = df[~conflict].copy(), key[~conflict]
    dup = key.duplicated()
    df = df[~dup].copy()
    print(f"after keeping one row of same-code repeats: {len(df)} (dropped {int(dup.sum())})")
    print(f"total dropped: {n0 - len(df)} of {n0} ({(n0 - len(df)) / n0:.1%})")

    df["gt_populism"] = df["pop_code"].map(CODE_TO_LABEL)
    df = df.rename(columns={"sentence": "text"})
    cols = ["source_id", "speech_id", "sentence_position", "split", "date", "location", "speaker",
            "n_words", "pop_code", "text", "gt_populism"]
    return df[cols].sort_values(["speech_id", "sentence_position"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for train.csv and test.csv (downloaded from Hugging Face if absent)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_populism"].value_counts().reindex(LABELS).to_string())


if __name__ == "__main__":
    main()
