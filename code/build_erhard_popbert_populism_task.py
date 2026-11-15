#!/usr/bin/env python3
"""
Build a multi-label populism task from the PopBERT training data.

Source: Erhard, Hanke, Remer, Falenska and Heiberger (2025), "PopBERT.
Detecting Populism and Its Host Ideologies in the German Bundestag",
Political Analysis 33(1) (doi:10.1017/pan.2024.12; arXiv 2309.14355).
Data: Harvard Dataverse doi:10.7910/DVN/HZMSUR, version 2.0, file
`labelled_data.tab` (file id 7675659), downloaded in its original CSV format
(`?format=original`, MD5 cfcc9e7226cb68563d13ef0c31e96ee2). Licence CC0 1.0.

Unit: one sentence from a German Bundestag speech (18th and 19th legislative
periods, 2013-2021), shown alone, in German. The sample is not random: it was
drawn with a populism-dictionary stratum and then seven rounds of active
learning aimed at edge cases and rare categories (paper sect. 4.2), so
populist sentences are far more common than in the Bundestag record.

Labels (multi_binary): anti_elitism, people_centrism, left_wing, right_wing.
Each sentence was coded by the same five trained coders, whose votes per
label are stored as JSON ({"coder_0": true, ...}). Gold per label = at least
3 of the 5 coders. This majority rule is ours, chosen so that each item has a
single gold label. The authors train on a different rule, "at least one
coder" (paper sect. 4.3), and argue against majority aggregation. The vote
counts and the authors' any-coder flags are kept as columns for analysis.
Fleiss' kappa per label is 0.41 / 0.24 / 0.36 / 0.36 (paper Table 1).

Host ideology: the codebook (Supplementary Appendix 2.2) tells coders to code
the populism dimensions first and to code left- or right-wing host ideology
only for populist statements. Under a per-label majority rule, a few sentences
have a host-ideology majority without an anti-elitism or people-centrism
majority (40 in the released file). The builder applies the codebook's order:
a host-ideology label is 1 only if the sentence also has a populism majority.
This sets those 40 host labels to 0 (the count is printed); the raw votes stay
in the `votes_*` columns.

Steps and drops, in order:
1. Read labelled_data (8,795 sentences, 5 coders each; ids unique).
2. Drop sentences whose vote JSON does not hold exactly five coders (none
   expected).
3. Exact duplicate texts (the same sentence in several speeches): drop all
   copies if their majority labels differ (none expected), otherwise keep the
   copy with the lowest id.
Sentences are short, so no token count or length cap is applied.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

URL = "https://dataverse.harvard.edu/api/access/datafile/7675659?format=original"
FILE_NAME = "labelled_data.csv"
MD5 = "cfcc9e7226cb68563d13ef0c31e96ee2"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
DEFAULT_INPUT_DIR = Path("/tmp/popbert")
DEFAULT_OUTPUT = REPO / "data" / "erhard_popbert_populism.csv"
LABELS = ["anti_elitism", "people_centrism", "left_wing", "right_wing"]
N_CODERS = 5
MAJORITY = 3


def _fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / FILE_NAME
    if not path.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r:
            path.write_bytes(r.read())
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != MD5:
        raise ValueError(f"{path}: md5 {digest} does not match {MD5}")
    return path


def build(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(_fetch(input_dir))
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids")
    print(f"1. sentences read: {len(df)}")

    ok = pd.Series(True, index=df.index)
    for lab in LABELS:
        votes = df[lab].map(json.loads)
        ok &= votes.map(len).eq(N_CODERS)
        df[f"votes_{lab}"] = votes.map(lambda v: sum(bool(x) for x in v.values()))
    df = df[ok].copy()
    print(f"2. with {N_CODERS} coder votes on every label: {len(df)} (dropped {int((~ok).sum())})")

    for lab in LABELS:
        df[f"gt_{lab}"] = df[f"votes_{lab}"].ge(MAJORITY).astype(int)
        df[f"any_coder_{lab}"] = df[f"votes_{lab}"].ge(1).astype(int)
    gts = [f"gt_{lab}" for lab in LABELS]

    df = df.sort_values("id")
    key = df[gts].astype(str).agg("".join, axis=1)
    conflict = key.groupby(df["text"]).transform("nunique").gt(1)
    df = df[~conflict]
    same = df["text"].duplicated(keep="first")
    df = df[~same].copy()
    print(f"3. duplicates: dropped {int(conflict.sum())} rows with conflicting majority labels, "
          f"{int(same.sum())} same-label repeats; {len(df)} left")

    populist = df["gt_anti_elitism"].eq(1) | df["gt_people_centrism"].eq(1)
    host_only = (df["gt_left_wing"].eq(1) | df["gt_right_wing"].eq(1)) & ~populist
    for lab in ("left_wing", "right_wing"):
        df.loc[~populist, f"gt_{lab}"] = 0
    print(f"   host-ideology majority without a populism majority: {int(host_only.sum())} "
          "(host labels set to 0, per codebook Appendix 2.2)")

    df["source_id"] = "popbert_" + df["id"].astype(str)
    df["language"] = "de"
    df["n_words"] = df["text"].str.split().str.len()
    cols = (["source_id", "id", "speech_id", "speech_date", "sentence_no", "faction",
             "speaker_first_name", "speaker_last_name", "language", "n_words", "text"]
            + [f"votes_{l}" for l in LABELS] + [f"any_coder_{l}" for l in LABELS] + gts)
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory for {FILE_NAME} (downloaded from Dataverse if missing)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    for lab in LABELS:
        print(f"  {lab}: {int(df[f'gt_{lab}'].sum())} majority yes "
              f"({int(df[f'any_coder_{lab}'].sum())} under the authors' any-coder rule)")
    print(f"  no label: {int((df[[f'gt_{l}' for l in LABELS]].sum(axis=1) == 0).sum())}")


if __name__ == "__main__":
    main()
