#!/usr/bin/env python3
"""
Build a cleaned COVID threat-minimization task from Burnham's public sample.

The source file exposes public text plus direct coding for whether a message
minimizes the COVID-19 threat. This builder uses the primary `threatmin` label,
normalizes whitespace, drops missing labels, removes label conflicts, and
deduplicates the small number of exact repeated text-label pairs.

Version 2 (`--version 2`) fixes two errors in the v1 build:

1. Gold column. In Burnham's replication script `sect_4_analysis.R`,
   `non_comp` is the original hand-coded label from Block et al. (2022) and
   `threatmin` / `threatmin2` are the predictions of his two NLI hypothesis
   sets ("Test set MCC Between original labels and hypothesis set 1:
   mcc(training$non_comp, training$threatmin)"). v1 scored models against
   the NLI predictions. v2 uses the human label `non_comp`.
2. Parse. The Dataverse-ingested `.tab` export of this file splits multiline
   tweets into extra rows (319 rows instead of 300), so v1 attached labels to
   tweet fragments. v2 reads the original CSV upload
   (`/api/access/datafile/10383707?format=original`, 300 rows), which parses
   cleanly.

v2 also replaces e-mail addresses with "[EMAIL]", matching the repository's
redaction of the v1 corpora.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/burnham_covid_training_sample.tab")
DEFAULT_OUTPUT = REPO / "data" / "burnham_covid_threat_minimization.csv"
DEFAULT_INPUT_V2 = Path("/tmp/burnham_covid_training_sample.csv")
DEFAULT_OUTPUT_V2 = REPO / "data" / "burnham_covid_threat_minimization_v2.csv"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_burnham_covid_threat_minimization_task(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {"text", "non_comp", "threatmin", "threatmin2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_non_comp": pd.to_numeric(df["non_comp"], errors="coerce"),
            "source_threatmin": pd.to_numeric(df["threatmin"], errors="coerce"),
            "source_threatmin2": pd.to_numeric(df["threatmin2"], errors="coerce"),
            "text": df["text"].map(_clean_text),
        }
    )
    out = out.dropna(subset=["source_threatmin"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_threat_minimizing"] = out["source_threatmin"].astype(int)

    if not out["gt_threat_minimizing"].isin([0, 1]).all():
        bad = sorted(
            out.loc[~out["gt_threat_minimizing"].isin([0, 1]), "gt_threat_minimizing"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Encountered non-binary threat-minimization labels: {bad}")

    conflict_counts = out.groupby("text")["gt_threat_minimizing"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_threat_minimizing"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    out["source_id"] = "covid_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_non_comp",
            "source_threatmin",
            "source_threatmin2",
            "text",
            "gt_threat_minimizing",
        ]
    ].reset_index(drop=True)


def build_burnham_covid_threat_minimization_task_v2(input_path: Path) -> pd.DataFrame:
    """Build v2 from the original CSV upload, with the human `non_comp` label as gold."""
    df = pd.read_csv(input_path)
    required = {"text", "non_comp", "threatmin", "threatmin2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    if len(df) != 300:
        raise ValueError(
            f"Expected the 300-row original CSV, got {len(df)} rows. "
            "Use the original upload (?format=original), not the ingested .tab."
        )

    out = pd.DataFrame(
        {
            "source_row": range(1, len(df) + 1),
            "source_non_comp": pd.to_numeric(df["non_comp"], errors="coerce"),
            "source_nli_threatmin1": pd.to_numeric(df["threatmin"], errors="coerce"),
            "source_nli_threatmin2": pd.to_numeric(df["threatmin2"], errors="coerce"),
            "text": df["text"].map(_clean_text).map(lambda t: EMAIL_RE.sub("[EMAIL]", t)),
        }
    )
    out = out.dropna(subset=["source_non_comp"]).copy()
    out = out[out["text"].ne("")].copy()
    out["gt_threat_minimizing"] = out["source_non_comp"].astype(int)
    if not out["gt_threat_minimizing"].isin([0, 1]).all():
        raise ValueError("Encountered non-binary non_comp labels.")

    conflict_counts = out.groupby("text")["gt_threat_minimizing"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_threat_minimizing"]).reset_index(drop=True)
    if out["text"].duplicated().any():
        raise ValueError("Duplicate text remained after cleaning.")

    out["source_id"] = "covidv2_" + out["source_row"].astype(str)
    return out[
        [
            "source_id",
            "source_row",
            "source_non_comp",
            "source_nli_threatmin1",
            "source_nli_threatmin2",
            "text",
            "gt_threat_minimizing",
        ]
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    v2 = args.version == 2
    if args.input is None:
        args.input = str(DEFAULT_INPUT_V2 if v2 else DEFAULT_INPUT)
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if v2 else DEFAULT_OUTPUT)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download Burnham's COVID sample "
            "from Harvard Dataverse DOI 10.7910/DVN/XS0ULP first (for --version 2, "
            "the original CSV via ?format=original), or pass --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if v2:
        df = build_burnham_covid_threat_minimization_task_v2(input_path)
    else:
        df = build_burnham_covid_threat_minimization_task(input_path)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_threat_minimizing"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
