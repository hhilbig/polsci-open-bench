#!/usr/bin/env python3
"""
Build a binary political-delegitimization task from the PDD dataset.

Source: Rivlin-Angert and Mor-Lan (2025), "The Enemy from Within: A Study of
Political Delegitimization Discourse in Israeli Political Speech", EMNLP 2025
(arXiv:2508.15524). Data: github.com/guymorlan/pdd, files train.csv,
val.csv and test.csv at commit 80a72d5d0c4a368f1fdf9329268c05f1d48eee24.
Licence: CC BY 4.0 (repository README, "License" section; paper Appendix E).

Unit: one Hebrew sentence from a Facebook post by an Israeli politician or
party (2018-2021), a Knesset floor speech (1993-2023), or a Hebrew news
article (2018-2021). The three splits are pooled (10,410 sentences).

Gold: the `delegitimization` column (1 = the sentence contains political
delegitimization discourse, 0 = not), coded by the lead author and two
colleagues (paper section 5 and Appendix F; mean Cohen's kappa 0.82 on a
triple-coded subset). Model input is the Hebrew `text` column only. The
machine-translated English column (`text_en_machine_translated`), the target
span column (`anno_text`) and the second-stage sub-labels (intensity,
incivility, group, person, outgroup, common_good, institute) are not used.

Source correction. The file's `source` values News and Facebook are swapped:
5,969 of the 6,690 rows marked News carry facebook.com URLs and Facebook page
metadata (`Page Name`, `User Name`), no row marked Facebook does except one,
and the README's counts (Facebook 6,690, News 1,216) match the file's counts
with the two names exchanged. The builder therefore maps file News ->
Facebook and file Facebook -> News for all rows (Knesset unchanged) and keeps
the file value in `source_file`. It asserts the evidence above before doing so.

Drops, in order:
1. Sentences whose Hebrew share is below 50%: characters in the Hebrew
   Unicode block (U+0590-U+05FF) divided by all non-whitespace characters.
   These are English or Arabic posts, Arabic posts whose letters were lost in
   the export (only diacritics and punctuation remain), link lists and
   similar residue. The verifier's count was 326 rows, all with gold 0. (A
   letters-only denominator gives 306; the non-whitespace denominator is used
   because it also removes the Arabic rows that kept only a Hebrew header such
   as "עברית בהמשך".)
2. Duplicate texts (identical after collapsing runs of whitespace): all rows
   of a text with conflicting gold labels are dropped; one row of a
   same-label repeat is kept. The full release has 21 such repeats, but all
   of them fall in rows already removed by step 1, so none remain here.
3. Empty texts. (None in the release.)

The source has no row identifier, so `source_id` is "pdd_" plus the first 16
hex digits of the SHA-1 of the Hebrew sentence, which is unique after step 2.
"""
import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

COMMIT = "80a72d5d0c4a368f1fdf9329268c05f1d48eee24"
BASE_URL = f"https://raw.githubusercontent.com/guymorlan/pdd/{COMMIT}"
FILES = {
    "train": ("train.csv", "f2f643a709710daed84a36a2c6d02ea3a8c195894cf174225c8ce3a189c5647e"),
    "val": ("val.csv", "2a91a2e60fa1b0695c5f41ce8688c4127e92f5a1b6acc9cf341770b36f6e289e"),
    "test": ("test.csv", "1176b72073f19f74e916d3fccbb93b4d984829ee680a6140660027f3babab1ad"),
}
DEFAULT_INPUT_DIR = Path("/tmp/rivlin_angert_pdd")
DEFAULT_OUTPUT = REPO / "data" / "rivlin_angert_pdd_delegitimization.csv"
MIN_HEBREW_SHARE = 0.5
SOURCE_FIX = {"News": "Facebook", "Facebook": "News", "Knesset": "Knesset"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, sha) in FILES.items():
        path = input_dir / name
        if not path.exists():
            print(f"downloading {BASE_URL}/{name}")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        got = _sha256(path)
        if got != sha:
            raise ValueError(f"{path}: sha256 {got} != expected {sha}")
        out[split] = path
    return out


def hebrew_share(text: str) -> float:
    chars = re.sub(r"\s", "", text)
    if not chars:
        return 0.0
    return len(re.findall("[֐-׿]", chars)) / len(chars)


def build(input_dir: Path) -> pd.DataFrame:
    paths = fetch(input_dir)
    df = pd.concat(
        [pd.read_csv(p, dtype=str, keep_default_na=False).assign(split=s) for s, p in paths.items()],
        ignore_index=True,
    )
    print(f"rows read: {len(df)} ({df['split'].value_counts().to_dict()})")
    df["gt_delegitimization"] = df["delegitimization"].astype(float).astype(int)
    assert set(df["gt_delegitimization"]) == {0, 1}

    # Source correction, checked against the metadata before applying it.
    fb_meta = df["URL"].str.contains("facebook.com") | df["Page Name"].ne("")
    by_file = fb_meta.groupby(df["source"]).sum().to_dict()
    counts = df["source"].value_counts().to_dict()
    print(f"file source counts {counts}; rows with Facebook URL/page metadata by file source {by_file}")
    assert counts == {"News": 6690, "Knesset": 2504, "Facebook": 1216}, counts
    assert by_file["News"] > 0.85 * counts["News"] and by_file["Facebook"] <= 1 and by_file["Knesset"] == 0
    df["source_file"] = df["source"]
    df["source"] = df["source_file"].map(SOURCE_FIX)
    print(f"corrected source counts {df['source'].value_counts().to_dict()}")

    df["text"] = df["text"].str.strip()
    df["hebrew_share"] = df["text"].map(hebrew_share)
    low = df["hebrew_share"].lt(MIN_HEBREW_SHARE)
    print(f"step 1, Hebrew share < {MIN_HEBREW_SHARE}: drop {int(low.sum())} rows "
          f"(gold {df.loc[low, 'gt_delegitimization'].value_counts().to_dict()}, "
          f"by source {df.loc[low, 'source'].value_counts().to_dict()})")
    df = df[~low].copy()
    print(f"  remaining {len(df)}")

    key = df["text"].str.replace(r"\s+", " ", regex=True)
    conflict = df.groupby(key)["gt_delegitimization"].transform("nunique").gt(1)
    n_before = len(df)
    df = df[~conflict].copy()
    df = df[~key[df.index].duplicated()].copy()
    print(f"step 2, duplicates: {int(conflict.sum())} rows with conflicting labels dropped; "
          f"{n_before - int(conflict.sum()) - len(df)} same-label repeats dropped; remaining {len(df)}")

    empty = df["text"].eq("")
    df = df[~empty].copy()
    print(f"step 3, empty texts: drop {int(empty.sum())}; remaining {len(df)}")

    df["source_id"] = "pdd_" + df["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
    assert df["source_id"].is_unique
    df["language"] = "he"
    df["post_date"] = df["Post Created Date"]
    df["page_name"] = df["Page Name"]
    df["n_words"] = df["text"].str.split().str.len()
    cols = ["source_id", "split", "source", "source_file", "language", "post_date", "page_name",
            "n_words", "hebrew_share", "text", "gt_delegitimization"]
    return df[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the raw CSVs; missing files are downloaded from the pinned commit")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_delegitimization"].value_counts().to_string())
    print(pd.crosstab(df["source"], df["gt_delegitimization"]).to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]).round().to_string())


if __name__ == "__main__":
    main()
