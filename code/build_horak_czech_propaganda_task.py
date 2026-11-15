#!/usr/bin/env python3
"""
Build a multi-label propaganda-technique task from the MU Propaganda dataset.

Source: Horák, Sabol, Herman and Baisa (2024), "Recognition of propaganda
techniques in newspaper texts: Fusion of content and style analysis", Expert
Systems with Applications 124085 (doi:10.1016/j.eswa.2024.124085). The dataset
is also described in Baisa, Herman and Horák (RANLP 2019) and Horák, Baisa and
Herman (2021, arXiv 2108.11669). Files: Hugging Face MU-NLPC/Propaganda,
revision cde669cb34a89724f14ac3fe8ad476eb79f70390, data/train-...parquet
(7,642 documents) and data/test-...parquet (1,000). Licence CC BY-NC-SA 4.0.
Both splits are used; the split is kept as a column.

Unit: one whole Czech news article (title plus body, as released in `text`)
from four outlets previously investigated for pro-Russian propaganda
(sputnik.cz, parlamentnilisty.cz, ac24.cz, svetkolemnas.info).

Labels (multi_binary): blaming, opinion, labeling, fear-mongering,
demonization. Gold is the dataset's own document-level column for each
attribute, `yes` -> 1, `no` -> 0. Not used: fabulation (judging it requires
fact-checking outside the text) and relativization (Cohen's kappa about 0.2,
the lowest reported, Horák et al. 2021 sect. 5.4.4). The annotator evidence
spans in `ranges` and every other annotation column are never shown to the
model.

Agreement. Horák et al. (2024, ESWA, Table A.7) report Cohen's kappa and
percent agreement per attribute: blaming 0.14 (84.6%), opinion 0.23 (83.0%),
labeling 0.26 (87.2%), fear-mongering 0.20 (90.7%), demonization 0.16
(95.3%); relativization 0.11 and fabulation 0.15. Agreement on the `yes`
value alone is 17.8-33.4% for the five kept attributes. The ESWA paper says
the second phase had "at least two annotators per article"; the 2021 report
says three with a 2-of-3 rule. The release does not record the count.

Gold procedure and the main drop. Horák et al. (2021, sect. 5.4.4) state that
the 2016 documents were coded by one annotator each, and the 2017 and 2018
documents by three annotators, a value entering the dataset when at least two
agreed. (RANLP 2019 said the 2017 part had one annotator; the 2021 text is
later and is the one followed here.) The release has no year column. The
2017-18 documents were coded in an editor that required evidence spans for
every significant value (RANLP 2019, p. 78), so step 2 keeps documents with at
least one span in `ranges`. Check on ids that carry dates (Sputnik ids start
with YYYYMMDD): all 1,239 dated 2016 have no span; 1,476 of 1,517 dated
2017-18 have spans. Parlamentní listy article numbers fall in two separate
ranges (about 424k-465k without spans, 510k-530k with spans). The builder
prints this check. The rule drops about 64% of documents; the drop follows
from the gold procedure (single-coded documents have no majority gold), not
from the labels.

Steps and drops, in order:
1. Read both parquet files (8,642 documents; ids unique).
2. Keep documents with at least one evidence span (see above).
3. Drop documents whose label for any of the five attributes is not yes/no
   (none expected).
4. Exact duplicate texts: drop all copies if their five labels differ,
   otherwise keep the first by source_id.
5. Drop documents above MAX_DOC_TOKENS (long_text_budget.py); not truncated.
"""
import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

REVISION = "cde669cb34a89724f14ac3fe8ad476eb79f70390"
BASE_URL = f"https://huggingface.co/datasets/MU-NLPC/Propaganda/resolve/{REVISION}/data/"
FILES = {
    "train": ("train-00000-of-00001-0fd4bf47f4aed0e1.parquet",
              "d8f0fe3b2e4e6366692efb01d76cfc01536c420063442aec5c432061eba0e214"),
    "test": ("test-00000-of-00001-1978a383c6422834.parquet",
             "bc56e9bddb6c443eee24d111749b5d40f9f7b5b9883ded1c64772a29494dc086"),
}
DEFAULT_INPUT_DIR = Path("/tmp/mu_nlpc_propaganda")
DEFAULT_OUTPUT = REPO / "data" / "horak_czech_propaganda.csv"
LABELS = ["blaming", "opinion", "labeling", "fear-mongering", "demonization"]


def gt_col(label: str) -> str:
    return "gt_" + label.replace("-", "_")


def _fetch(input_dir: Path) -> dict:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (name, sha) in FILES.items():
        path = input_dir / name
        if not path.exists():
            print(f"downloading {BASE_URL + name}")
            req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                path.write_bytes(r.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            raise ValueError(f"{name}: sha256 {digest} does not match {sha}")
        paths[split] = path
    return paths


def _year_from_id(source_id: str):
    m = re.match(r"^(20[12]\d)(\d{2})(\d{2})\d+-", source_id)
    return int(m.group(1)) if m else pd.NA


def build(input_dir: Path) -> pd.DataFrame:
    paths = _fetch(input_dir)
    df = pd.concat([pd.read_parquet(p).assign(split=s) for s, p in paths.items()], ignore_index=True)
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids")
    print(f"1. documents read: {len(df)}")

    df["n_spans"] = df["ranges"].map(len)
    df["year_from_id"] = df["id"].map(_year_from_id).astype("Int64")
    check = pd.crosstab(df["year_from_id"].fillna(0), df["n_spans"].gt(0))
    print("   span check on ids with a date (0 = no date in id); columns = has span:")
    print(check.to_string())
    has_span = df["n_spans"].gt(0)
    n0 = len(df)
    df = df[has_span].copy()
    print(f"2. with at least one evidence span (triple-coded 2017-18): {len(df)} "
          f"(dropped {n0 - len(df)}, {100 * (n0 - len(df)) / n0:.1f}%)")

    ok = df[LABELS].isin(["yes", "no"]).all(axis=1)
    df = df[ok].copy()
    print(f"3. all five labels yes/no: {len(df)} (dropped {int((~ok).sum())})")
    for lab in LABELS:
        df[gt_col(lab)] = df[lab].eq("yes").astype(int)
    gts = [gt_col(l) for l in LABELS]

    df = df.rename(columns={"id": "source_id"}).sort_values("source_id")
    key = df[gts].astype(str).agg("".join, axis=1)
    n_label_sets = key.groupby(df["text"]).transform("nunique")
    conflict = n_label_sets.gt(1)
    df = df[~conflict]
    same = df["text"].duplicated(keep="first")
    df = df[~same].copy()
    print(f"4. duplicates: dropped {int(conflict.sum())} rows with conflicting labels, "
          f"{int(same.sum())} same-label repeats; {len(df)} left")

    df["n_words"] = df["text"].str.split().str.len()
    df["n_tokens"] = df["text"].map(doc_tokens)
    over = df["n_tokens"].gt(MAX_DOC_TOKENS)
    df = df[~over].copy()
    print(f"5. <= {MAX_DOC_TOKENS} tokens: {len(df)} (dropped {int(over.sum())})")

    df["language"] = "cs"
    cols = ["source_id", "split", "year_from_id", "genre", "topic", "language",
            "n_spans", "n_words", "n_tokens", "text"] + gts
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the two parquet files (downloaded if missing)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    for lab in LABELS:
        print(f"  {lab}: {int(df[gt_col(lab)].sum())} yes")
    print(f"  no label: {int((df[[gt_col(l) for l in LABELS]].sum(axis=1) == 0).sum())}")
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
