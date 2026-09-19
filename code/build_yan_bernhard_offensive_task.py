#!/usr/bin/env python3
"""
Build the cleaned offensive-reply task from the Yan and Bernhard (2024)
replication archive.

The source is a field experiment on gendered harassment of volunteer canvassers.
Volunteers sent canvassing text messages; recipients replied; those replies were
then rated for offensiveness by Prolific respondents, ten texts per respondent.

Label. The archive ships `coder1.tab`, `coder2.tab` and `coder3.tab`. These are
not three named expert coders: they are the first, second and third rating slot
for each text, so the same slot number is a different respondent from row to row.
That is why slot 3 agrees far less with the others (Cohen's kappa 0.27-0.33
against 0.73 between slots 1 and 2). Gold here is the majority of the three
slots, which is the same rule the benchmark already uses for
`toxicity_protests_es`.

The per-slot value is the authors' own binary offensiveness flag. The instrument
respondents actually saw was a five-point ordinal scale -- "Non-offensive",
"Slightly offensive", "Moderately offensive", "Fairly offensive", "Very
offensive", plus "Not displaying/unreadable" -- and the binary is the authors'
collapse of it. The exact cut point is not recoverable from the released rating
file alone, because the raw wide-format survey export cannot be joined one-to-one
to the cleaned per-text rows. Mapping raw categories onto the binary does show a
monotone gradient in the expected direction, so the collapse tracks the scale.
The benchmark therefore treats the authors' binary as the direct source label and
does not re-derive it.

Source: Yan and Bernhard (2024), "The Silenced Text: Field Experiments on
Gendered Experiences of Political Participation".
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SOURCE_URL = (
    "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UYKE7T"
)
CODER_FILES = ("coder1.tab", "coder2.tab", "coder3.tab")
DEFAULT_INPUT_DIR = Path("/tmp/yan_bernhard_silenced")
DEFAULT_OUTPUT = REPO / "data" / "yan_bernhard_offensive.csv"

# The coder files carry the survey question as the column header. Despite the
# .tab extension they are comma-separated.
EXPECTED_COLUMNS = 3


def _read_coder(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path} parsed to {frame.shape[1]} columns, expected {EXPECTED_COLUMNS}. "
            "These files are comma-separated despite the .tab extension."
        )
    frame.columns = ["text", "offensive", "discourage"]
    return frame


def build_offensive_task(input_dir: Path) -> pd.DataFrame:
    frames = [_read_coder(input_dir / name) for name in CODER_FILES]

    lengths = {len(f) for f in frames}
    if len(lengths) != 1:
        raise ValueError(f"coder files disagree on row count: {sorted(lengths)}")
    if not all(frames[0]["text"].equals(f["text"]) for f in frames[1:]):
        raise ValueError(
            "coder files are not row-aligned on text; the majority vote assumes "
            "slot i of row n rates the same text across all three files"
        )

    merged = pd.DataFrame(
        {
            "text": frames[0]["text"].astype(str).str.strip(),
            "rating1": frames[0]["offensive"],
            "rating2": frames[1]["offensive"],
            "rating3": frames[2]["offensive"],
        }
    )

    before = len(merged)
    merged = merged.dropna(subset=["rating1", "rating2", "rating3"])
    for column in ("rating1", "rating2", "rating3"):
        merged[column] = merged[column].astype(int)
        if not merged[column].isin({0, 1}).all():
            raise ValueError(f"{column} holds values outside 0/1")

    merged["n_offensive_ratings"] = merged[["rating1", "rating2", "rating3"]].sum(axis=1)
    merged["gt_offensive"] = (merged["n_offensive_ratings"] >= 2).astype(int)
    merged["unanimous"] = merged["n_offensive_ratings"].isin({0, 3}).astype(int)

    merged = merged[merged["text"].str.len() > 0]

    # Entry-threshold rules from TODO.md: drop texts whose duplicates disagree on
    # gold, then keep one row per remaining text.
    conflicts = merged.groupby("text")["gt_offensive"].nunique()
    conflicted = set(conflicts[conflicts > 1].index)
    merged = merged[~merged["text"].isin(conflicted)]
    merged = merged.drop_duplicates("text").reset_index(drop=True)

    dropped = 1 - len(merged) / before
    if dropped > 0.20:
        raise ValueError(
            f"cleaning dropped {dropped:.1%} of rows, above the 20% threshold that "
            "TODO.md requires to be flagged and justified"
        )

    merged.insert(0, "source_row_id", range(1, len(merged) + 1))
    return merged[
        [
            "source_row_id",
            "text",
            "rating1",
            "rating2",
            "rating3",
            "n_offensive_ratings",
            "unanimous",
            "gt_offensive",
        ]
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    missing = [name for name in CODER_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {input_dir}. Download them from {SOURCE_URL} "
            "and pass the containing directory via --input-dir."
        )

    frame = build_offensive_task(input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"wrote {output} ({len(frame)} rows)")
    print(f"positive rate: {frame['gt_offensive'].mean():.4f} "
          f"({int(frame['gt_offensive'].sum())} offensive)")
    print(f"unanimous across the three rating slots: {frame['unanimous'].mean():.3f}")


if __name__ == "__main__":
    main()
