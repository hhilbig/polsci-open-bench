#!/usr/bin/env python3
"""
Build the Ballard et al. congressional-tweet incivility task.

Source: Harvard Dataverse doi:10.7910/DVN/2ET6VL (CC0), "Replication Data for:
Dynamics of Polarizing Rhetoric in Congressional Tweets" (Ballard, DeTamble,
Dorsey, Heseltine and Johnson, Legislative Studies Quarterly 2022/2023). File
`training_final` (datafile 5747924, described as "Training data oversampled for
negative biased tweets"), original format:
https://dataverse.harvard.edu/api/access/datafile/5747924?format=original

The file holds 2,682 hand-coded tweets by members of Congress with two coder
columns (`uncivil`, `uncivil_2`) and the resolved label `uncivil_final`, which
the authors' training script (`train_wandering_replication.py`) uses as the
classifier label. Every row has `divisive` = 1; the file does not document what
that column records. `tweet_id` is 0 or a float rounded to three significant
digits, so real tweet ids cannot be recovered; `id` is the source's own row id.

v1 (default) reproduces data/ballard_incivility.csv byte for byte: `id` as
`status_id`, the text with surrounding whitespace stripped, and `uncivil_final`.

`--version 2` writes data/ballard_incivility_v2.csv and
- unescapes HTML entities (`&amp;` and similar);
- repairs two mechanical encoding errors in the source text: UTF-8 bytes
  decoded as Latin-1 (`â\\x80\\x99` for a right quote) and Windows-1252
  punctuation stored as C1 control characters (`\\x92` for an apostrophe,
  `\\x85` for an ellipsis); characters the source already replaced by `?` are
  left as they are, since the original character cannot be recovered;
- collapses the doubled quotes (`""`) left by repeated CSV escaping;
- normalizes line breaks to `\\n`;
- drops retweets that Twitter truncated (text starts with `RT @` and ends in
  the truncation ellipsis, or in `?` standing for it at the 140-character
  limit), because the coded tweet is cut off mid-sentence;
- drops tweets that appear more than once (same text after removing URLs,
  case and punctuation): all copies when the copies carry different labels,
  otherwise every copy after the first;
- renames the id column to `source_id`.
"""
import argparse
import html
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/ballard_training_final.csv")
DEFAULT_OUTPUT = REPO / "data" / "ballard_incivility.csv"
DEFAULT_OUTPUT_V2 = REPO / "data" / "ballard_incivility_v2.csv"

# Runs of Latin-1 characters that are really UTF-8 byte sequences.
MOJIBAKE_RE = re.compile(r"[\u00c2-\u00f4][\u0080-\u00bf]{1,3}")
C1_RE = re.compile(r"[\u0080-\u009f]")
TRUNCATION_MARKS = ("\u2026", "\u0085")
MULTI_QUOTE_RE = re.compile(r'"{2,}')
URL_RE = re.compile(r"https?://\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
# Old-API retweets were cut at 140 characters; the shortest "?"-terminated
# retweet in the file is 137 characters long.
TRUNCATED_QMARK_MIN_LEN = 130


def _fix_utf8_mojibake(text: str) -> str:
    def repl(match):
        chunk = match.group(0)
        try:
            return chunk.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            return chunk

    return MOJIBAKE_RE.sub(repl, text)


def _fix_c1_controls(text: str) -> str:
    def repl(match):
        try:
            return match.group(0).encode("latin-1").decode("cp1252")
        except UnicodeDecodeError:
            return ""

    return C1_RE.sub(repl, text)


def _unescape_html(text: str) -> str:
    # Some entities are escaped twice (`&amp;#039;`).
    for _ in range(3):
        new = html.unescape(text)
        if new == text:
            break
        text = new
    return text


def _dedup_key(text: str) -> str:
    """Text with URLs removed, lowercased, letters and digits only.

    Catches copies of one tweet that differ only in how the source encoded
    punctuation (e.g. `You’re` in one row, `You?re` in another).
    """
    return NON_ALNUM_RE.sub("", URL_RE.sub("", text).lower())


def _clean_text_v2(text: str) -> str:
    text = _unescape_html(text)
    text = _fix_utf8_mojibake(text)
    text = _fix_c1_controls(text)
    text = MULTI_QUOTE_RE.sub('"', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _is_truncated_retweet(raw: str) -> bool:
    body = raw.strip(" \t\r\n")
    if not body.startswith("RT @"):
        return False
    if body.endswith(TRUNCATION_MARKS):
        return True
    return body.endswith("?") and len(body) >= TRUNCATED_QMARK_MIN_LEN


def build_ballard_incivility_task(input_path: Path, version: int = 1) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    required = {"id", "text", "uncivil_final", "uncivil", "uncivil_2", "divisive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    if not df["uncivil_final"].isin([0, 1]).all():
        raise ValueError("uncivil_final contains values other than 0/1")
    if df["id"].duplicated().any():
        raise ValueError("source id column is not unique")

    if version == 1:
        return pd.DataFrame(
            {
                "status_id": df["id"],
                "text": df["text"].astype(str).str.strip(),
                "gt_uncivil": df["uncivil_final"].astype(int),
            }
        )

    raw = df["text"].astype(str)
    truncated = raw.map(_is_truncated_retweet)
    print(
        f"v2: dropping {truncated.sum()} of {len(df)} truncated retweets (labels: "
        f"{df.loc[truncated, 'uncivil_final'].value_counts().sort_index().to_dict()})"
    )
    keep = df[~truncated].copy()
    out = pd.DataFrame(
        {
            "source_id": keep["id"].astype(int),
            "source_coder_a": keep["uncivil"],
            "source_coder_b": keep["uncivil_2"],
            "text": keep["text"].astype(str).map(_clean_text_v2),
            "gt_uncivil": keep["uncivil_final"].astype(int),
        }
    ).reset_index(drop=True)

    if out["text"].eq("").any():
        raise ValueError("empty text after cleaning")
    key = out["text"].map(_dedup_key)
    conflict_counts = out.groupby(key)["gt_uncivil"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    in_conflict = key.isin(conflicts)
    print(f"v2: dropping {in_conflict.sum()} rows that duplicate another row's text with a different label")
    out, key = out[~in_conflict], key[~in_conflict]
    dup = key.duplicated()
    print(f"v2: dropping {dup.sum()} repeated copies of a tweet with the same label")
    return out[~dup].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if args.version == 2 else DEFAULT_OUTPUT)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download training_final from "
            "https://dataverse.harvard.edu/api/access/datafile/5747924?format=original "
            "first, or pass the file path via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_ballard_incivility_task(input_path, version=args.version)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_uncivil"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
