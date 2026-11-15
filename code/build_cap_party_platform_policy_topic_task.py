#!/usr/bin/env python3
"""
Build a CAP party-platform policy-topic task from public Democratic and
Republican Party Platform CSVs.

The source files expose quasi-statement text plus direct CAP major-topic codes.
This builder combines the two party files, keeps standard CAP major topics,
normalizes whitespace, removes conflicts, and deduplicates exact repeated
text-label pairs.

`--version 2` writes data/cap_party_platform_policy_topic_v2.csv. v1 output is
unchanged (it reads the source as latin-1, which double-encodes every
non-ASCII character). v2 differs in four ways:

1. The source CSVs are valid UTF-8 and are read as UTF-8.
2. Encoding residue is repaired with V2_MOJIBAKE_REPLACEMENTS. The source
   descriptions already carry layered mojibake for curly quotes, apostrophes,
   dashes, ellipses and list bullets (for example "President√≠¬¥s" for
   "President's"; "√≠_" and "¬å¬´" are orphan bytes and list-bullet
   markers). The original byte was partly replaced by "_" in the source, so a
   codec round trip cannot recover it; the table maps each observed sequence
   to the punctuation its contexts show.
3. Page markers such as "[p.86]" are removed and the year is written as an
   integer (v1 rendered "Year: 1968.0").
4. Bare section headers are dropped by rule (_is_header_v2): at
   most 8 words, no final punctuation, and either no lowercase letter or every
   word of 4+ letters capitalized.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_DEM = Path("/tmp/cap_dem_platform.csv")
DEFAULT_REP = Path("/tmp/cap_rep_platform.csv")
DEFAULT_OUTPUT = REPO / "data" / "cap_party_platform_policy_topic.csv"
DEFAULT_OUTPUT_V2 = REPO / "data" / "cap_party_platform_policy_topic_v2.csv"

# Ordered: longer sequences first. Contexts for each are in the source CSVs
# (Rep 19.3, Dem 20.2), e.g. "abortions √≠√Ø a procedure" (dash),
# "the mid√≠¬ê1960s" (hyphen), "√≠√çunder God√≠√±" (quotes),
# "be√≠¬ädeprived of life" (ellipsis), "a √íBridge to Work√ì" (quotes).
V2_MOJIBAKE_REPLACEMENTS = [
    ("√≠¬¢√§¬â√•¬å_", ""),
    ("Mart√≠¬Ä¬å_", "Martí"),
    ("Mart√§√≥¬ª", "Martí"),
    ("communiqu¬Å_s", "communiqués"),
    ("¬å¬´", " "),  # list bullet
    ("√≠_", ""),  # orphan byte, replaced by "_" in the source
    ("√≠√Ø", "—"),
    ("√≠¬¥", "'"),
    ("√≠¬ê", "-"),
    ("√≠√±", '"'),
    ("√≠√ç", '"'),
    ("√≠√Æ", '"'),
    ("√≠¬ä", "..."),
    ("√í", '"'),
    ("√ì", '"'),
    ("√ë", "—"),
    ("√î", "'"),
    ("¬Ñ", "—"),
    ("¬á", "á"),
    ("¬é", "é"),
    ("¬É", "..."),
]
V2_MOJIBAKE_CHARS = re.compile("[√≠¬]")
V2_PAGE_MARKER = re.compile(r"\[p\.\s*\d+\]")
V2_HEADER_MAX_WORDS = 8


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _clean_text_v2(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    for old, new in V2_MOJIBAKE_REPLACEMENTS:
        text = text.replace(old, new)
    text = V2_PAGE_MARKER.sub(" ", text)
    return " ".join(text.split())


def _is_header_v2(text: str) -> bool:
    """A bare section heading: short, unpunctuated, all caps or title case."""
    words = text.split()
    if not words or len(words) > V2_HEADER_MAX_WORDS:
        return False
    if re.search(r"[.!?;:,\"')]$", text):
        return False
    if not re.search("[a-z]", text):
        return True
    long_words = [w for w in words if len(re.sub("[^A-Za-z]", "", w)) >= 4]
    return bool(long_words) and all(
        re.sub("^[^A-Za-z]+", "", w)[:1].isupper() for w in long_words
    )


def _prepare_party(path: Path, party: str, version: int = 1) -> pd.DataFrame:
    if version == 2:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        clean = _clean_text_v2
    else:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        clean = _clean_text
    required = {"year", "id", "majortopic", "subtopic", "description", "words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return pd.DataFrame(
        {
            "source_party": party,
            "source_year": pd.to_numeric(df["year"], errors="coerce"),
            "source_original_id": df["id"].astype(str),
            "source_subtopic": pd.to_numeric(df["subtopic"], errors="coerce"),
            "source_words": pd.to_numeric(df["words"], errors="coerce"),
            "source_major_topic": pd.to_numeric(df["majortopic"], errors="coerce"),
            "text": df["description"].map(clean),
        }
    )


def build_cap_party_platform_policy_topic_task(
    dem_path: Path, rep_path: Path, version: int = 1
) -> pd.DataFrame:
    out = pd.concat(
        [
            _prepare_party(dem_path, "Democratic", version),
            _prepare_party(rep_path, "Republican", version),
        ],
        ignore_index=True,
    )
    out = out.dropna(subset=["source_major_topic"]).copy()
    out = out[out["text"].ne("")].copy()
    if version == 2:
        n_before = len(out)
        headers = out["text"].map(_is_header_v2)
        out = out[~headers].copy()
        print(f"v2: dropped {int(headers.sum())} of {n_before} rows as bare section headers")
        residue = out["text"].str.contains(V2_MOJIBAKE_CHARS)
        print(f"v2: {int(residue.sum())} rows still contain mojibake characters")
        out["source_year"] = out["source_year"].astype("Int64")
    out["source_major_topic"] = out["source_major_topic"].astype(int)
    out = out[out["source_major_topic"].isin(CAP_MAJOR_TOPIC_LABELS)].copy()
    out["gt_policy_topic"] = out["source_major_topic"].map(CAP_MAJOR_TOPIC_LABELS)

    conflict_counts = out.groupby("text")["gt_policy_topic"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_policy_topic"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = (
        "cap_party_"
        + out["source_party"].str.lower()
        + "_"
        + out["source_original_id"].astype(str)
    )
    return out[
        [
            "source_id",
            "source_party",
            "source_year",
            "source_original_id",
            "source_major_topic",
            "source_subtopic",
            "source_words",
            "text",
            "gt_policy_topic",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", default=str(DEFAULT_DEM))
    ap.add_argument("--rep", default=str(DEFAULT_REP))
    ap.add_argument("--output", default=None)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if args.version == 2 else DEFAULT_OUTPUT)

    dem_path = Path(args.dem)
    rep_path = Path(args.rep)
    for path in [dem_path, rep_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download the CAP Democratic and "
                "Republican Party Platform CSVs first, or pass paths."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cap_party_platform_policy_topic_task(dem_path, rep_path, args.version)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_policy_topic"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
