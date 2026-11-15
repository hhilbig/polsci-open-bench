#!/usr/bin/env python3
"""
Build a binary nostalgic-rhetoric task from the PolNos hand-coded manifesto sentences.

Source: Müller and Proksch (2024), "Nostalgia in European Party Politics: A
Text-Based Measurement Approach", British Journal of Political Science 54(3)
(doi:10.1017/S0007123423000571). Data: PolNos, Harvard Dataverse
doi:10.7910/DVN/L198GI, version 1.0, files data_polnos_handcoding.csv
(datafile 7430043, MD5 7967a249cd8130e8622be3f534b5551b) and
data_translated_full.rds (datafile 7430042, MD5 6b9472400a5df0c08e6c22c05d20cb35).
Licence: CC0 1.0 (Dataverse dataset terms). Dataverse refuses the default
Python User-Agent, so the builder sends a browser User-Agent. Reading the .rds
file needs the pyreadr package.

Unit: one manifesto (quasi-)sentence from 24 European countries (50 per
country, 1,200 rows). For the 22 non-English countries the text is the
authors' Google Translate English; UK and Ireland sentences are original
English (`translated` column).

What the coders saw, and so what the model sees. The coding instructions
(SI Section C.1) show the coders the 'target' sentence "highlighted in bold"
with "the sentences before and after the target sentence. Make sure to read
the context too", and ask them to flag non-English sentences whose
translation is inaccurate. The handcoding file has the `translation_coder*`
flags, and the data description (Table 3) describes `text` as "translated for
non-English manifestos". The coders therefore labeled the machine-translated
English text with its two neighbours, and the model is shown the same: the
translated target sentence plus the translated previous and next sentence.
No original-language text is shown (the coders did not see it), and no model
output column is used.

Context reconstruction. The handcoding file has no context columns, so the
builder takes the rows immediately before and after the target sentence within
the same manifesto in data_translated_full.rds, in file order. That file is
used rather than the 6-59-word data_polnos_sentencelevel.rds because 74 of the
1,200 coded sentences fall outside the 6-59-word window and so must have been
drawn from the unfiltered corpus. (For the authors' second coding set,
data_polnos_handcoding_validation.csv, the released text_pre/text_post equal
the adjacent rows of the corpus that set was drawn from in 3,513 of 3,528
matches.) The reconstruction is the builder's, not the authors' release; for
1,118 targets present once in both corpora, both neighbours agree between the
two corpora for 842.

Gold: the authors' rule (paper p. 997; SI Section B.4): nostalgic (1) if at
least three of the four coders labeled the sentence nostalgic
(`nostalgic_at_least_3`), else 0. Four research assistants; Krippendorff's
alpha 0.56 (Table A2).

Sampling (kept as `keyword_selected`): per country the authors drew 25
sentences flagged by their broadest dictionary (base dictionary plus
embeddings, `nostalgia_dictionary_emb` = 1) and 25 not flagged. Almost all
positives come from the flagged half, so a model can score well by spotting
dictionary terms.

Steps and drops, in order:
1. Duplicate target texts (after collapsing whitespace and lower-casing):
   all rows of a text with conflicting gold are dropped; of a text repeated
   with one gold value, the first row in file order is kept.
2. Targets whose context cannot be reconstructed uniquely: the target text
   occurs more than once in its manifesto in data_translated_full.rds with
   different neighbouring sentences (headers such as "The Green Party will:"),
   or does not occur there at all.
   Targets at the start or end of a manifesto keep "(none)" as context.
source_id: "polnos_" plus the release's `doc_id` (manifesto id and sentence
number) with spaces replaced by underscores.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

API = "https://dataverse.harvard.edu/api/access/datafile/{id}?format=original"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
FILES = {
    "handcoding": (7430043, "data_polnos_handcoding.csv", "7967a249cd8130e8622be3f534b5551b"),
    "full": (7430042, "data_translated_full.rds", "6b9472400a5df0c08e6c22c05d20cb35"),
}
DEFAULT_INPUT_DIR = Path("/tmp/muller_proksch_polnos")
DEFAULT_OUTPUT = REPO / "data" / "muller_proksch_polnos_nostalgia.csv"
ENGLISH_COUNTRIES = {"United Kingdom", "Ireland"}
NONE = "(none)"


def fetch(input_dir: Path) -> dict:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (file_id, name, md5) in FILES.items():
        path = input_dir / name
        if not path.exists():
            req = urllib.request.Request(API.format(id=file_id), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as resp:
                path.write_bytes(resp.read())
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise ValueError(f"{path} has MD5 {got}, expected {md5}")
        paths[key] = path
    return paths


def norm(s: pd.Series) -> pd.Series:
    return s.str.split().str.join(" ")


def build(paths: dict) -> pd.DataFrame:
    import pyreadr

    df = pd.read_csv(paths["handcoding"])
    print(f"released rows: {len(df)}; gold (>=3 of 4 coders) {df['nostalgic_at_least_3'].value_counts().sort_index().to_dict()}")
    coder_sum = df[[f"nostalgic_coder{i}" for i in range(1, 5)]].sum(axis=1)
    if not (coder_sum.ge(3).astype(int) == df["nostalgic_at_least_3"]).all():
        raise ValueError("nostalgic_at_least_3 does not equal the 3-of-4 rule")
    if not df["doc_id"].is_unique or df["text"].isna().any():
        raise ValueError("duplicate doc_id or empty text")
    n0 = len(df)

    key = norm(df["text"]).str.lower()
    n_gold = df.groupby(key)["nostalgic_at_least_3"].nunique()
    conflict = key.isin(n_gold[n_gold > 1].index)
    print(f"step 1: texts with conflicting gold: {int((n_gold > 1).sum())}; rows dropped: {int(conflict.sum())}")
    df, key = df[~conflict].copy(), key[~conflict]
    dup = key.duplicated()
    print(f"step 1: same-gold repeats dropped: {int(dup.sum())} "
          f"(of them positive: {int(df.loc[dup, 'nostalgic_at_least_3'].sum())})")
    df = df[~dup].copy()
    print(f"step 1: rows now {len(df)}")

    full = list(pyreadr.read_r(str(paths["full"])).values())[0].reset_index(drop=True)
    full = full[["manifesto_id", "text"]].copy()
    grp = full.groupby("manifesto_id", sort=False)["text"]
    full["text_pre"] = grp.shift(1).fillna(NONE)
    full["text_post"] = grp.shift(-1).fillna(NONE)
    full["key"] = norm(full["text"])
    full = full[full["manifesto_id"].isin(set(df["manifesto_id"]))]
    ctx = (full.groupby(["manifesto_id", "key"])
           .agg(text_pre=("text_pre", "first"), text_post=("text_post", "first"),
                n_pre=("text_pre", "nunique"), n_post=("text_post", "nunique"))
           .reset_index())
    df["key"] = norm(df["text"])
    df = df.merge(ctx, on=["manifesto_id", "key"], how="left", validate="one_to_one")
    bad = df["text_pre"].isna() | df["n_pre"].gt(1) | df["n_post"].gt(1)
    print(f"step 2: targets without a unique context: {int(bad.sum())} "
          f"(not found {int(df['text_pre'].isna().sum())}; of them positive: {int(df.loc[bad, 'nostalgic_at_least_3'].sum())})")
    df = df[~bad].copy()
    print(f"step 2: rows now {len(df)}")
    print(f"total dropped: {n0 - len(df)} of {n0} ({(n0 - len(df)) / n0:.1%})")

    df["source_id"] = "polnos_" + df["doc_id"].str.replace(" ", "_", regex=False)
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    df["gt_nostalgia"] = df["nostalgic_at_least_3"].astype(int)
    df["translated"] = (~df["countryname"].isin(ENGLISH_COUNTRIES)).astype(int)
    df["keyword_selected"] = df["nostalgia_dictionary_emb"].astype(int)
    df["year"] = df["manifesto_id"].str.split().str[1].str[:4].astype(int)
    df["n_coders_nostalgic"] = df["nostalgia_agreement_coders"].astype(int)
    df["n_coders_bad_translation"] = df["translation_agreement_coders"].astype(int)
    cols = ["source_id", "manifesto_id", "countryname", "party", "year", "cmp_code", "translated",
            "keyword_selected", "n_coders_nostalgic", "n_coders_bad_translation",
            "text_pre", "text", "text_post", "gt_nostalgia"]
    return df[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the two Dataverse files (downloaded if absent)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_nostalgia"].value_counts().sort_index().to_string())
    print("positives by sampling half (keyword_selected):")
    print(pd.crosstab(df["keyword_selected"], df["gt_nostalgia"]).to_string())


if __name__ == "__main__":
    main()
