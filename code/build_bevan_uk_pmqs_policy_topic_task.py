#!/usr/bin/env python3
"""
Build a CAP major-topic task from UK Prime Minister's Questions, 1997-2008
(UK Policy Agendas Project; Bevan and John 2016, "Policy Representation by
Party Leaders and Followers: What drives UK Prime Minister's Questions?",
Government and Opposition 51(1): 59-83).

Source: Comparative Agendas Project, UK "Prime Minister's Questions" dataset,
https://minio.la.utexas.edu/compagendas/datasetfiles/uk_pmqs.csv (sha256
below; 9,062 rows). Codebook: "UK PMQs: Master Codebook Release 1997-2008
(v 1.0)", Shaun Bevan, 15 September 2016. Licence: CC BY-NC-SA 4.0 for the
CAP codes (comparativeagendas.net Copyright-and-Legal page); the question
text is Hansard (Open Parliament Licence).

Unit: one question put to the Prime Minister, including the interruptions
Hansard records within it (codebook, `question`). The `debate` column (the
Prime Minister's answer) is not shown and not kept. Column `minister` is the
MP asking the question, not a minister.

Gold: `majortopic`, the CAP major topic "associated with the question asked
ignoring other rhetoric and asides. This coding is based on the question and
not the debate therefore ignoring any answers that attempt to change the
topic at hand" (codebook). Named with the current CAP master-codebook labels
(code/cap_topic_labels.py, v2 names: topic 5 is Labor). No coder-agreement
figure for this dataset was found (see the task report).

Steps, in order:
1. Read the CSV; check ids are unique and every code is a CAP major topic.
2. Text repairs (labels untouched): the file's non-ASCII characters were
   decoded with the wrong Windows code page (cp1250 instead of cp1252:
   "Ł" for "£", "ľ" for "¾"); each text is re-encoded as cp1250 and decoded as
   cp1252, which is lossless on this file. Hansard page markers
   ("21 Apr 1999 : Column 906") are removed, lines are stripped, runs of
   spaces collapsed and blank-line runs shortened.
3. Drop engagement questions: every question containing the open
   "official engagements" formula ("If he will list his official engagements
   for Wednesday ...", including the variants with "I" lost, "To ask the
   Prime Minister" and "his engagements"). These are the procedural Q1 that
   opens each session, almost all coded Government Operations. The rule is
   mechanical: it also drops 2 rows where the formula is followed by further
   text (id 1264, the formula plus the Prime Minister's answer, coded 20; id
   7297, the formula plus a supplementary on MRSA, coded 3).
4. Drop questions under 10 words (after repair), too short to carry a topic.
5. Exact duplicate texts: drop all rows of a text with conflicting codes,
   keep one row of same-code repeats.
No item is near 1,000 words (maximum about 360), so no token cap is applied.

`source_id` is "uk_pmqs_<id>", from the release's `id` counting variable.
"""
import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS_V2


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

URL = "https://minio.la.utexas.edu/compagendas/datasetfiles/uk_pmqs.csv"
SHA256 = "de5c1ddd056eff09438ea5943beb54fb2fd22e84418a372aa0074f1575482962"
DEFAULT_INPUT_DIR = Path("/tmp/bevan_uk_pmqs")
DEFAULT_OUTPUT = REPO / "data" / "bevan_uk_pmqs_policy_topic.csv"
MIN_WORDS = 10
ENGAGEMENT_RE = re.compile(
    r"(?:to ask the prime minister,? )?i?f (?:he|she) will list (?:his|her) "
    r"(?:official )?engagements for", re.I)
COLUMN_MARK_RE = re.compile(r"(?:\d{1,2} )?[A-Z][a-z]{2,8} \d{4}\s*:\s*Column \d+")


def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / "uk_pmqs.csv"
    if not path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        raise ValueError(f"{path}: sha256 {got} != expected {SHA256}")
    return path


def repair(text: str) -> str:
    text = text.encode("cp1250").decode("cp1252")
    text = COLUMN_MARK_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def party(row) -> str:
    if row.filter_labid == 1:
        return "Labour"
    if row.filter_conid == 1:
        return "Conservative"
    if row.filter_ldid == 1:
        return "Liberal Democrat"
    return "other"


def build(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"1. rows {len(df)}")
    if not df["id"].is_unique:
        raise ValueError("id not unique")
    if not df["majortopic"].isin(CAP_MAJOR_TOPIC_LABELS_V2).all():
        raise ValueError("non-CAP major topic code")
    if df["question"].isna().any():
        raise ValueError("missing question")

    df["text"] = df["question"].map(repair)
    changed = df["text"].ne(df["question"].str.strip()).sum()
    marks = df["question"].str.count(COLUMN_MARK_RE.pattern).sum()
    print(f"2. text repaired in {changed} rows ({int(marks)} Hansard column markers removed); "
          f"'£' now in {int(df['text'].str.contains('£').sum())} rows, 'Ł' in "
          f"{int(df['text'].str.contains('Ł').sum())}")

    eng = df["text"].str.contains(ENGAGEMENT_RE)
    print(f"3. drop engagement questions: {int(eng.sum())} "
          f"{df.loc[eng, 'majortopic'].value_counts().to_dict()}; kept {int((~eng).sum())}")
    df = df[~eng].copy()

    df["n_words"] = df["text"].str.split().str.len()
    short = df["n_words"].lt(MIN_WORDS)
    print(f"4. drop under {MIN_WORDS} words: {int(short.sum())}; kept {int((~short).sum())}")
    df = df[~short].copy()

    conflict = df.groupby("text")["majortopic"].transform("nunique").gt(1)
    df = df[~conflict].copy()
    before = len(df)
    df = df.drop_duplicates("text").copy()
    print(f"5. duplicate texts: dropped {int(conflict.sum())} conflicting rows, "
          f"{before - len(df)} same-code repeats; kept {len(df)}")

    out = pd.DataFrame({
        "source_id": "uk_pmqs_" + df["id"].astype(str),
        "year": df["year"],
        "date": df["datereworking"],
        "question_number": df["number"].astype("Int64"),
        "asker": df["minister"].map(lambda v: " ".join(str(v).split())),
        "asker_party": [party(r) for r in df.itertuples()],
        "conservative_leader": df["filter_conleader"],
        "libdem_leader": df["filter_libdemleader"],
        "n_words": df["n_words"],
        "source_major_topic": df["majortopic"],
        "text": df["text"],
        "gt_policy_topic": df["majortopic"].map(CAP_MAJOR_TOPIC_LABELS_V2),
    })
    if out["source_id"].duplicated().any() or out["text"].str.len().eq(0).any():
        raise ValueError("duplicate id or empty text")
    return out.sort_values("year", kind="stable").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the raw CSV; downloaded and checksummed if missing")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_policy_topic"].value_counts().reindex(list(CAP_MAJOR_TOPIC_LABELS_V2.values())).to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
