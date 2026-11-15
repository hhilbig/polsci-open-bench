#!/usr/bin/env python3
"""
Build a long-text populism task from the Global Populism Database (GPD v2.1).

Source: Hawkins, Aguilar, Castanho Silva et al., Global Populism Database
v2.1 (Team Populism), Harvard Dataverse doi:10.7910/DVN/LFTQEZ, CC0 1.0.
The GPD grades whole speeches by chief executives of 77 countries with
holistic grading: trained native-speaker coders read each speech in its
original language and assign a grade on a 0-2 rubric, in tenths. Where two
coders graded a speech, differing grades are averaged. The codebook reports
Krippendorff's alpha (interval) of 0.835 across the speeches with two or more
coders.

Unit and label. A speech is one text file, identified by `merging_variable`
and `leader` together (two Argentine presidents share file names, below). Gold
is the mean of the non-missing coder grades (`rubricgrade`) for that file,
rounded to the rubric's integer categories with the codebook's own rule
(0.5 rounds to 1 and 1.5 rounds to 2):
  [0, 0.5)   -> not populist      (rubric category 0)
  [0.5, 1.5) -> somewhat populist (rubric category 1)
  [1.5, 2]   -> very populist     (rubric category 2)
The GPD's `averagerubric` column is not used: for Donald Trump's first term it
pools the grades of two different texts per speech slot (for example the
inaugural address, `US_Trump_Famous_1`, and the 2018 State of the Union,
`US_Trump_Famous_1.2`), which changes three of the eight labels.

Steps and drops, in order:
1. Read GPD_v2.1_20251120.tab (one row per speech file and coder); drop rows
   with no `merging_variable` and speeches with no non-missing grade.
2. Match each speech to its text file in speeches_20251120.zip:
   - Cristina Fernández (2008, 2011) and Alberto Fernández (2020) share the
     label names `Argentina_Fernandez_<type>_1.txt`; the archive keeps
     Alberto's speeches under the accented `Argentina_Fernández_...` names, so
     his rows are mapped to those files.
   - File-name typos in the archive are corrected (FILE_NAME_FIXES).
   - Otherwise names match after folding accents to ASCII and removing spaces
     before ".txt". A folded name that matches two files is an error.
3. Decode text as UTF-8, falling back to cp1252; normalize line endings.
   Remove research-assistant metadata that can give away the speech type
   (see strip_metadata) and every line containing a URL.
4. Drop speeches under MIN_TOKENS tokens. A token minimum is used instead of a
   word count because Japanese and Chinese texts have no spaces between words.
5. Drop every speech whose text is identical to another speech's text. The
   archive stores the same file under two speech names in a few cases (e.g.
   Poland_Kwasniewski_Ribbon_2 and _Campaign_2), often with different grades,
   so neither label can be trusted to belong to the text.
6. Drop speeches above MAX_DOC_TOKENS (10,000 tokens under the stricter of two
   tokenizers; see long_text_budget.py) so the whole speech fits the context.
   This removes very populist speeches and non-Latin scripts (Central Asia,
   Israel, Georgia, Armenia) more often than others; the builder prints the
   drop shares by class.
Speeches are kept in their original languages, as the coders read them.
"""
import argparse
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT_DIR = Path("/tmp/gpd_v2_1")
LABEL_FILE = "GPD_v2.1_20251120.tab"
SPEECH_ZIP = "speeches_20251120.zip"
DEFAULT_OUTPUT = REPO / "data" / "gpd_speech_populism.csv"
MIN_TOKENS = 300
LABELS = ["not populist", "somewhat populist", "very populist"]

# Archive file name (ASCII-folded) -> label-file name.
FILE_NAME_FIXES = {
    "Bolovia_Anez_Ribbon_1.txt": "Bolivia_Anez_Ribbon_1.txt",
    "Israel_Begin_Famous_1txt.txt": "Israel_Begin_Famous_1.txt",
    "Israel_Bennet_Campaign_1.txt": "Israel_Bennett_Campaign_1.txt",
    "Israel_Bennet_Famous_1.txt": "Israel_Bennett_Famous_1.txt",
    "Israel_Bennet_Interntational_1.txt": "Israel_Bennett_International_1.txt",
    "Israel_Bennet_Ribbon_1.txt": "Israel_Bennett_Ribbon_1.txt",
}

_META_KEY = re.compile(
    r"^\s*(category|type|speech type|type of speech|source|link|url|note|notes)\s*:", re.IGNORECASE
)
_TYPE_WORDS = re.compile(
    r"\b(campaign|famous|ribbon[- ]?cutting|ribboncutting|international)\b.{0,20}\bspeech\b"
    r"|\bspeech\b.{0,20}\b(campaign|famous|ribbon[- ]?cutting|international)\b"
    r"|^\s*(campaign|famous|ribbon[- ]?cutting|ribboncutting|international)\s*(speech)?\s*$",
    re.IGNORECASE,
)
_INLINE_TYPE = re.compile(
    r"^.{0,40}?\b(campaign|famous|ribbon[- ]?cutting|ribboncutting|international)\s+speech\b[.:]?\s*",
    re.IGNORECASE,
)
_SHORT_TYPE = re.compile(r"\b(campaign|famous|ribbon|ribbon[- ]?cutting|international)\b", re.IGNORECASE)
_NOTE = re.compile(r"transcript|youtube|brief description|survey of the speeches|identified to be the|other year than", re.IGNORECASE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
HEADER_LINES = 15


def _fold(name: str) -> str:
    """Accents folded to ASCII, spaces before .txt removed."""
    name = unicodedata.normalize("NFKD", name.split("/")[-1]).encode("ascii", "ignore").decode()
    return re.sub(r"\s+\.txt$", ".txt", name).strip()


def _decode(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in text.split("\n"))


def strip_metadata(text: str) -> str:
    """Remove research-assistant headers and URL lines.

    In the first HEADER_LINES non-empty lines, drop lines that are metadata
    fields that can reveal the speech type or its sourcing ("Category: Famous
    Speech", "Type: RibbonCutting", "Source: ..."), lines naming the GPD speech
    type ("Howard 3rd Term Famous Speech"), and transcription notes ("Not the
    best transcript: here is Youtube link"). Anywhere in the text, drop lines
    containing a URL.
    """
    out, seen = [], 0
    for line in text.split("\n"):
        if line and _URL.search(line):
            continue
        if line:
            seen += 1
            if seen <= HEADER_LINES and (
                (len(line) < 120 and (_META_KEY.search(line) or _TYPE_WORDS.search(line)))
                or (len(line) < 400 and _NOTE.search(line))
            ):
                continue
        out.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    # Second pass on the first three lines: remove inline GPD speech-type labels
    # ("Howard 3rd Term International Speech. Address to ...") and short label
    # lines ("Ribbon-cutting.", "Dzurinda 2 new government- Ribbon").
    lines = text.split("\n")
    for i in range(min(3, len(lines))):
        line = _INLINE_TYPE.sub("", lines[i]).strip()
        if len(line) < 40 and _SHORT_TYPE.search(line):
            line = ""
        lines[i] = line
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def grade_to_label(avg: float) -> str:
    if avg < 0.5:
        return LABELS[0]
    if avg < 1.5:
        return LABELS[1]
    return LABELS[2]


def _file_index(archive: zipfile.ZipFile) -> tuple[dict, dict]:
    exact, folded = {}, {}
    for name in archive.namelist():
        if not name.endswith("txt") or name.endswith("/"):
            continue
        base = unicodedata.normalize("NFC", name.split("/")[-1])
        exact[base] = name
        key = FILE_NAME_FIXES.get(_fold(base), _fold(base))
        folded.setdefault(key, []).append(name)
    return exact, folded


def _resolve(row, exact: dict, folded: dict):
    name = unicodedata.normalize("NFC", row.merging_variable)
    if str(row.leader).startswith("Alberto Fern") and name.startswith("Argentina_Fernandez_"):
        return exact.get(name.replace("Argentina_Fernandez_", "Argentina_Fernández_"))
    if str(row.leader).startswith("Cristina Fern") and name in exact:
        return exact[name]
    hits = folded.get(_fold(name), [])
    if len(hits) > 1:
        raise ValueError(f"{name} matches several archive files: {hits}")
    return hits[0] if hits else None


def build(input_dir: Path) -> pd.DataFrame:
    grades = pd.read_csv(input_dir / LABEL_FILE, sep="\t", low_memory=False)
    grades = grades.dropna(subset=["merging_variable"])
    per_speech = grades.groupby(["merging_variable", "leader"], dropna=False).agg(
        average_grade=("rubricgrade", "mean"),
        n_coders=("rubricgrade", "count"),
        gpd_averagerubric=("averagerubric", "first"),
        country=("country", "first"),
        term=("term", "first"),
        speech_type=("speechtype", "first"),
        year_begin=("yearbegin", "first"),
        region=("region", "first"),
    ).reset_index()
    n_speeches = len(per_speech)
    per_speech = per_speech[per_speech["n_coders"].gt(0)].copy()

    archive = zipfile.ZipFile(input_dir / SPEECH_ZIP)
    exact, folded = _file_index(archive)
    per_speech["zip_name"] = [_resolve(r, exact, folded) for r in per_speech.itertuples()]
    matched = per_speech.dropna(subset=["zip_name"]).copy()
    if matched["zip_name"].duplicated().any():
        raise ValueError(f"archive files used twice: {matched.loc[matched['zip_name'].duplicated(), 'zip_name'].tolist()}")
    matched["text"] = matched["zip_name"].map(lambda n: strip_metadata(_decode(archive.read(n))))
    matched["n_tokens"] = matched["text"].map(doc_tokens)
    matched["n_words"] = matched["text"].str.split().str.len()
    out = matched[matched["n_tokens"].ge(MIN_TOKENS)].copy()
    n_min = len(out)
    dup = out["text"].duplicated(keep=False)
    out = out[~dup].copy()
    n_dedup = len(out)
    out["gt_populism"] = out["average_grade"].map(grade_to_label)
    over = out["n_tokens"].gt(MAX_DOC_TOKENS)
    drop_share = over.groupby(out["gt_populism"]).mean().round(2).to_dict()
    out = out[~over].copy()
    print(f"speeches {n_speeches}; graded {len(per_speech)}; matched to text {len(matched)}; "
          f">= {MIN_TOKENS} tokens {n_min}; without duplicate texts {n_dedup} (dropped {int(dup.sum())}); "
          f"<= {MAX_DOC_TOKENS} tokens {len(out)}")
    print(f"share dropped by the token cap, by class: {drop_share}")

    out["source_id"] = "gpd_" + out["zip_name"].map(lambda v: re.sub(r"[^A-Za-z0-9]+", "_", _fold(v)[:-4]))
    # The two Argentine Fernandez presidents fold to the same file name.
    clash = out["source_id"].duplicated(keep=False)
    out.loc[clash, "source_id"] += "_" + out.loc[clash, "leader"].astype(str).str.split().str[0].map(_fold)
    if out["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    out.loc[out["leader"].astype(str).str.startswith("Alberto Fern"), "leader"] = "Alberto Fernández"
    cols = ["source_id", "merging_variable", "zip_name", "country", "leader", "term", "speech_type",
            "year_begin", "region", "n_coders", "average_grade", "gpd_averagerubric", "n_words",
            "n_tokens", "text", "gt_populism"]
    return out[cols].sort_values("source_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"directory holding {LABEL_FILE} and {SPEECH_ZIP} from doi:10.7910/DVN/LFTQEZ")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    df = build(Path(args.input_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_populism"].value_counts().to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
