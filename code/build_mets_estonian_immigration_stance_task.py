#!/usr/bin/env python3
"""
Build a three-class immigration-stance task from Estonian news sentences.

Source: Mets, Karjus, Ibrus and Schich (2024), "Automated stance detection in
complex topics and small languages: The challenging case of immigration in
polarizing news media", PLOS ONE 19(4): e0302380
(doi:10.1371/journal.pone.0302380), with its S1 File. Data: GitHub
markmets/immigration-prediction-EST at commit
d910e6548be3b0b42acf6f1face0278d2b821c49, file Annotated_Dataset.csv
(SHA-256 below; 7,435 rows). Licence: the repository LICENSE is GPL-3.0 and
the README says "Data and code are free to use as it is". The sentences are
quoted from articles owned by their publishers (Uued Uudised and Ekspress
Grupp outlets such as Delfi and Eesti Päevaleht); the GPL grant covers the
authors' compilation, not the publishers' copyright in the text.

Unit: one Estonian sentence from a news or opinion article (2015-2021) that
matched an immigration keyword list, shown alone. The annotation sample was
balanced by keyword prevalence and publisher (paper, "Annotations"), so class
shares are not those of the news coverage.

Coding: two Estonian-speaking graduate students (coders J and N) each coded
a disjoint part of the sentences on a 1-5 scale or MH ("nonevaluative /
ambiguous": unintelligible, non-topical, or several stances at once). Each
sentence has one coder. The first author (coder A) later re-coded a 550-item
subset for agreement; the release's `stanceConsolidated` column takes his
consolidation into account on 299 rows (116 changed). The README says this
column was "Used for predictions in the article", so it is the gold here;
`stanceNotConsolidated` and both coder columns are kept as metadata.
Cohen's kappa against the third coder (S1 File, "Inter-rater scores";
both annotators pooled): 0.45 for the 1-5 + NA codes, 0.49 for
against/neutral/supportive/MH (n = 550), 0.57 with MH merged into neutral
(n = 550), 0.68 for against/neutral/supportive on the 222 sentences neither
coder marked MH, and 0.97 for against vs supportive (n = 82).

Gold: the authors' collapse (README; paper, "Annotations"): 1-2 -> against,
3 -> neutral, 4-5 -> supportive. MH rows are excluded, as in the paper, which
leaves the authors' 3,261 sentences.

Steps and drops, in order:
1. Read Annotated_Dataset.csv (7,435 sentences).
2. Drop rows whose consolidated code is MH (the authors' exclusion; 4,174
   rows, 56%, above 20% but set by the authors' rule).
3. Drop rows with an empty sentence (none expected).
4. Exact duplicate sentences: drop all rows of a sentence with conflicting
   labels, keep one row of same-label repeats (none expected; one pair of
   sentences differs only in capitalisation and both are kept).
Sentences are short (at most 152 words), so no token count or length cap is
applied. No model output is used (WrongPredictions_EstRoberta.csv is not read).

`source_id` is "mets_est_" + the first 16 hex digits of sha256(sentence); the
release has no id column.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

COMMIT = "d910e6548be3b0b42acf6f1face0278d2b821c49"
FILE = "Annotated_Dataset.csv"
URL = f"https://raw.githubusercontent.com/markmets/immigration-prediction-EST/{COMMIT}/{FILE}"
SHA256 = "411f34765dd738cf300a1c0c573baa21a695b5104784d5a685c88b8b70371db6"
DEFAULT_INPUT_DIR = Path("/tmp/mets_est")
DEFAULT_OUTPUT = REPO / "data" / "mets_estonian_immigration_stance.csv"
MAPPING = {"1": "against", "2": "against", "3": "neutral", "4": "supportive", "5": "supportive"}
LABELS = ["against", "neutral", "supportive"]


def _fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / FILE
    if not path.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "polsci-open-bench/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} does not match {SHA256}")
    return path


def build(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_fetch(input_dir), dtype=str)
    print(f"1. sentences read: {len(df)}")
    codes = set(df["stanceConsolidated"])
    if codes - set(MAPPING) - {"MH"}:
        raise ValueError(f"unexpected codes: {codes}")

    mh = df["stanceConsolidated"].eq("MH")
    df = df[~mh].copy()
    print(f"2. dropped {int(mh.sum())} MH (nonevaluative/ambiguous) rows "
          f"({mh.mean():.1%}, authors' exclusion); {len(df)} left")

    empty = df["sentence"].fillna("").str.strip().eq("")
    df = df[~empty].copy()
    print(f"3. dropped {int(empty.sum())} empty sentences; {len(df)} left")

    df["gt_stance"] = df["stanceConsolidated"].map(MAPPING)
    conflict = df.groupby("sentence")["gt_stance"].transform("nunique").gt(1)
    df = df[~conflict]
    same = df["sentence"].duplicated(keep="first")
    df = df[~same].copy()
    print(f"4. duplicates: dropped {int(conflict.sum())} rows with conflicting labels, "
          f"{int(same.sum())} same-label repeats; {len(df)} left")

    df["source_id"] = "mets_est_" + df["sentence"].map(lambda t: hashlib.sha256(t.encode()).hexdigest()[:16])
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    df = df.rename(columns={"sentence": "text", "stanceConsolidated": "stance_code",
                            "stanceNotConsolidated": "stance_code_not_consolidated",
                            "coderConsolidated": "coder", "coderNotConsolidated": "coder_original",
                            "publishDate": "publish_date", "publisherGroup": "publisher_group",
                            "keywordGroups": "keyword_groups"})
    df["year"] = df["publish_date"].str[6:10].astype(int)
    df["language"] = "et"
    df["n_words"] = df["text"].str.split().str.len()
    cols = ["source_id", "publisher_group", "publisher", "publish_date", "year", "category",
            "keyword_groups", "language", "n_words", "coder", "coder_original",
            "stance_code", "stance_code_not_consolidated", "text", "gt_stance"]
    return df[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory for {FILE} (downloaded from GitHub if missing)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(f"  classes: {df['gt_stance'].value_counts().reindex(LABELS).to_dict()}")
    print(f"  by publisher group: {df.groupby('publisher_group')['gt_stance'].value_counts().unstack().to_dict('index')}")


if __name__ == "__main__":
    main()
