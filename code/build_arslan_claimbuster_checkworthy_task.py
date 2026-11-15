#!/usr/bin/env python3
"""
Build a check-worthy factual claim task from the ClaimBuster dataset.

Source: Arslan, Hassan, Li and Tremayne, "A Benchmark Dataset of Check-worthy
Factual Claims", ICWSM 2020 (arXiv:2004.14425). Zenodo record 3836810,
ClaimBuster_Datasets.zip (md5 06ca00d0705e0a7fe9fb9a23a539ca97), CC BY 4.0.

Unit and label. One sentence spoken by a candidate or moderator in a US
general-election presidential debate (1960-2016). Gold `claim_type` is the
`Verdict` column of datasets/crowdsourced.csv (22,501 rows):
  -1 -> non-factual            (NFS)
   0 -> unimportant factual    (UFS)
   1 -> check-worthy factual   (CFS)
The bundled README swaps the two file names: the 22,501-row file is
crowdsourced.csv, labelled by the paid crowd participants with the best
screening scores, and groundtruth.csv (1,032 rows) was labelled by a
different, expert procedure. Only crowdsourced.csv is used; the two are not
merged. The `Sentiment` column is a computed score and is never shown.
Annotators could ask to see the four preceding sentences (about 14% of the
time); items are shown without context.

Steps and drops, in order:
1. Read crowdsourced.csv; strip surrounding whitespace from Text; drop empty
   texts.
2. Exact duplicate texts: drop every row of a text that carries conflicting
   verdicts; keep the first row (lowest Sentence_id) of same-label repeats.
3. Drop sentences whose text is identical to one of the example sentences the
   prompt quotes from the paper (Section 4.1 and Table 1), so no item's answer
   is shown in the prompt.
"""
import argparse
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUTPUT = REPO / "data" / "arslan_claimbuster_checkworthy.csv"
DEFAULT_INPUT_DIR = Path("/tmp/arslan_claimbuster")
ZIP_NAME = "ClaimBuster_Datasets.zip"
ZIP_URL = "https://zenodo.org/api/records/3836810/files/ClaimBuster_Datasets.zip/content"
ZIP_MD5 = "06ca00d0705e0a7fe9fb9a23a539ca97"
MEMBER = "ClaimBuster_Datasets/datasets/crowdsourced.csv"
VERDICT = {-1: "non-factual", 0: "unimportant factual", 1: "check-worthy factual"}
LABELS = list(VERDICT.values())


def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / ZIP_NAME
    if not path.exists():
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(path), ZIP_URL], check=True)
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != ZIP_MD5:
        path.unlink()
        raise ValueError(f"{ZIP_NAME}: md5 {got} != {ZIP_MD5} (file removed)")
    return path


def _norm(s: str) -> str:
    return " ".join(s.replace("’", "'").split()).lower()


# Example sentences quoted in prompts/arslan_claimbuster_checkworthy.txt
# (paper Section 4.1 and Table 1).
PROMPT_EXAMPLES = [
    "In the last month, we've had a net loss of one hundred and sixty-three thousand jobs.",
    "We've spent $4.7 billion a year in the State of Texas for uninsured people.",
    "When they tried to reduce taxes, he voted against that 127 times.",
    "China and India are graduating more graduates in technology and science than we are.",
    "My opponent opposed the missile defenses.",
    "I am a son of a Methodist minister.",
    "Just yesterday, I was in Toledo shaking some hands in a line.",
    "Well, the Vice President and I came to the Congress together 1946; we both served in the Labor Committee.",
    "And I've got two daughters and I want to make sure that they have the same opportunities that anybody's sons have.",
    "The worst thing we could do in this economic climate is to raise people's taxes.",
    "I think the Head Start program is a great program.",
    "We need to cut the business tax rate in America.",
    "I'll get America and North America energy independent.",
    "Well, you know, nailing down Senator Obama's various tax proposals is like nailing Jell-O to the wall.",
    "I'm simply not going to do that.",
    "In addition to that, we've suffered because we haven't had leadership in this administration.",
    "I was Governor of Georgia for four years.",
    "In Puerto Rico this year, I met with six of the leading industrial nations' heads of state to meet the problem of inflation so we would be able to solve it before it got out of hand.",
    "But first of all, this is a nation of immigrants.",
    "I think everybody understands at this point that we are experiencing the worst financial crisis since the Great Depression.",
    "Government spending has gone completely out of control; $10 trillion dollar debt we're giving to our kids, a half-a-trillion dollars we owe China.",
    "In the first place I've never suggested that Cuba was lost except for the present.",
]


def build(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        df = pd.read_csv(io.BytesIO(archive.read(MEMBER)))
    print(f"1. rows {len(df)}; Sentence_id unique: {df['Sentence_id'].is_unique}; "
          f"verdicts {df['Verdict'].value_counts().sort_index().to_dict()}")
    df["text"] = df["Text"].astype(str).str.strip()
    empty = df["Text"].isna() | df["text"].eq("")
    df = df[~empty].sort_values("Sentence_id").copy()
    print(f"   empty texts dropped: {int(empty.sum())}")

    n_labels = df.groupby("text")["Verdict"].transform("nunique")
    conflict = n_labels.gt(1)
    print(f"2. conflicting duplicates: {int(conflict.sum())} rows "
          f"({df.loc[conflict, 'text'].nunique()} texts) dropped")
    df = df[~conflict].copy()
    before = len(df)
    df = df.drop_duplicates("text", keep="first").copy()
    print(f"   same-label repeats dropped: {before - len(df)}; kept {len(df)}")

    examples = {_norm(e) for e in PROMPT_EXAMPLES}
    in_prompt = df["text"].map(_norm).isin(examples)
    print(f"3. prompt example sentences found among items: {int(in_prompt.sum())} "
          f"(of {len(examples)} examples) dropped; kept {int((~in_prompt).sum())}")
    df = df[~in_prompt].copy()

    df["gt_claim_type"] = df["Verdict"].map(VERDICT)
    if df["gt_claim_type"].isna().any():
        raise ValueError("unexpected Verdict value")
    df["source_id"] = "claimbuster_" + df["Sentence_id"].astype(str)
    df["debate_date"] = df["File_id"].str.replace(".txt", "", regex=False)
    df["year"] = df["debate_date"].str[:4].astype(int)
    df = df.rename(columns={"Speaker": "speaker", "Speaker_title": "speaker_title",
                            "Speaker_party": "speaker_party", "Line_number": "line_number",
                            "Length": "n_words"})
    cols = ["source_id", "debate_date", "year", "speaker", "speaker_title", "speaker_party",
            "line_number", "n_words", "text", "gt_claim_type"]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory for {ZIP_NAME}; downloaded from Zenodo and checksummed if missing")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_claim_type"].value_counts().reindex(LABELS).to_string())


if __name__ == "__main__":
    main()
