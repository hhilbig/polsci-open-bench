#!/usr/bin/env python3
"""
Build a 9-class issue-frame task from the Gun Violence Frame Corpus (GVFC).

Source: Liu, Guo, Mays, Betke and Wijaya (2019), "Detecting Frames in News
Headlines and Its Application to Analyzing News Framing Trends Surrounding
U.S. Gun Violence", CoNLL 2019 (aclanthology.org/K19-1047). Data: GVFC.zip,
linked from derrywijaya.github.io/GVFC.html as Google Drive file
1JQQT3I-7xNCiFtw9HIeNP9nr0znQYExu (MD5 475315a31074f314d024695882176eb7).
The builder reads GVFC/GVFC_headlines_and_annotations.xlsx from the zip; the
2021 multimodal extension in the same zip is not used.

Licence: none stated; headlines are copyrighted news text; needs sign-off
before public release.

Unit: one headline of a 2018 news article about gun violence from 30 top U.S.
news websites (2,990 headlines). Coders saw the headline only, as the model
does (codebook: "take a look at the headline and then answer the following
questions").

Gold: `Q3 Theme1`, the first of up to two dominant frames (codes 1-9 in the
codebook order, 99 = none). Two communication graduate students reached
reliability on a 100-headline sample (frame A: 94.10% agreement, Krippendorff's
alpha 0.90; frame B: 96.04%, 0.82; paper section 3.2); one of them coded the
remaining 2,790 headlines alone. The episodic/thematic label (`Q2 Focus`) has
no reported agreement and is not used.

Steps and drops, in order:
1. Duplicate headlines, matched after collapsing whitespace and lower-casing
   (the released text is not changed), checked on the full file: every row of
   a headline whose relevance-and-frame annotation (Q1, Theme1, Theme2)
   differs across rows is dropped; of a headline repeated with one
   annotation, the first row in file order is kept. Q2 is ignored here since
   it is not used.
2. Headlines not relevant to U.S. gun violence (Q1 = 2): no frame was coded.
3. Relevant headlines with no frame (Theme1 = 99). Flagged: about half the
   relevant headlines; the task asks for a frame, and "none" is not offered.
4. Headlines with two frames (Theme2 != 99), so that each item has exactly one
   coded frame and the 9-class answer is well defined.
source_id: "gvfc_" plus the first 16 hex digits of the SHA-1 of the headline
(the release has no identifier); unique after step 1.
"""
import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DRIVE_ID = "1JQQT3I-7xNCiFtw9HIeNP9nr0znQYExu"
URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
ZIP_MD5 = "475315a31074f314d024695882176eb7"
MEMBER = "GVFC/GVFC_headlines_and_annotations.xlsx"
DEFAULT_INPUT_DIR = Path("/tmp/liu_gvfc")
DEFAULT_OUTPUT = REPO / "data" / "liu_gvfc_frame.csv"

# Codebook Q3 order (GVFC_codebook.docx).
LABELS = [
    "gun rights",
    "gun control",
    "politics",
    "mental health",
    "school or public space safety",
    "race/ethnicity",
    "public opinion",
    "society/culture",
    "economic consequences",
]
CODE_TO_LABEL = {i + 1: lab for i, lab in enumerate(LABELS)}


def fetch(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / "GVFC.zip"
    if not path.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != ZIP_MD5:
        raise ValueError(f"{path} has MD5 {got}, expected {ZIP_MD5}")
    return path


def build(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_excel(io.BytesIO(z.read(MEMBER)))
    df = df.rename(columns={"news_title": "text", "Q1 Relevant": "q1_relevant", "Q2 Focus": "q2_focus",
                            "Q3 Theme1": "q3_theme1", "Q3 Theme2": "q3_theme2"})
    print(f"released rows: {len(df)}")
    if df["text"].isna().any() or not df["q3_theme1"].isin(list(CODE_TO_LABEL) + [99]).all():
        raise ValueError("unexpected empty headline or Theme1 code")
    n0 = len(df)

    key = df["text"].str.split().str.join(" ").str.lower()
    ann = df["q1_relevant"].astype(str) + "|" + df["q3_theme1"].astype(str) + "|" + df["q3_theme2"].astype(str)
    n_ann = ann.groupby(key).nunique()
    conflict = key.isin(n_ann[n_ann > 1].index)
    print(f"step 1: headlines with conflicting annotations: {int((n_ann > 1).sum())}; rows dropped: {int(conflict.sum())}")
    df, key = df[~conflict].copy(), key[~conflict]
    dup = key.duplicated()
    df = df[~dup].copy()
    print(f"step 1: after keeping one row of same-annotation repeats: {len(df)} (dropped {int(dup.sum())})")

    irrelevant = df["q1_relevant"].ne(1)
    df = df[~irrelevant].copy()
    print(f"step 2: after dropping headlines not relevant to U.S. gun violence: {len(df)} (dropped {int(irrelevant.sum())})")

    n1 = len(df)
    no_frame = df["q3_theme1"].eq(99)
    df = df[~no_frame].copy()
    print(f"step 3: after dropping relevant headlines with no frame: {len(df)} "
          f"(dropped {int(no_frame.sum())}, {no_frame.sum() / n1:.1%} of relevant rows; FLAG >20%)")

    two = df["q3_theme2"].ne(99)
    df = df[~two].copy()
    print(f"step 4: after dropping headlines with two frames: {len(df)} (dropped {int(two.sum())})")
    print(f"total dropped: {n0 - len(df)} of {n0} ({(n0 - len(df)) / n0:.1%})")

    df["source_id"] = "gvfc_" + df["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
    if df["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    df["gt_frame"] = df["q3_theme1"].map(CODE_TO_LABEL)
    df["focus"] = df["q2_focus"].map({1: "episodic", 2: "thematic"})  # metadata only, no reported agreement
    df["year"] = 2018
    cols = ["source_id", "text", "gt_frame", "q3_theme1", "focus", "year"]
    return df[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for GVFC.zip (downloaded from Google Drive if absent)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(fetch(Path(args.input_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_frame"].value_counts().reindex(LABELS).to_string())


if __name__ == "__main__":
    main()
