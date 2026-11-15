#!/usr/bin/env python3
"""
Build a cleaned PLOVER CAMEO event-type task from the public gold-standard
record file.

The source file is a JSON list of event examples. This builder removes the
documentation record, keeps event text and high-level CAMEO/PLOVER event type,
normalizes whitespace, drops exact duplicate texts with conflicting event
labels, and deduplicates repeated text-label pairs.

`--version 2` writes data/plover_cameo_event_v2.csv: the same rows in the same
order, but `source_id` is "plover_" plus the first 12 hex digits of the SHA-256
of the item text, because the source ids (e.g. PLOVER-GSR01-AGREE-0014) contain
the gold label. The source id is kept as `source_original_id` for tracing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_INPUT = Path("/tmp/plover_gsr_cameo.txt")
DEFAULT_OUTPUT = REPO / "data" / "plover_cameo_event.csv"
DEFAULT_OUTPUT_V2 = REPO / "data" / "plover_cameo_event_v2.csv"

EVENT_LABELS = [
    "AGREE",
    "AID",
    "ASSAULT",
    "COERCE",
    "CONCEDE",
    "CONSULT",
    "COOPERATE",
    "DEMAND",
    "DISAPPROVE",
    "FIGHT",
    "INVESTIGATE",
    "MOBILIZE",
    "PROTEST",
    "REJECT",
    "RETREAT",
    "SANCTION",
    "SUPPORT",
    "THREATEN",
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_plover_cameo_event_task(input_path: Path) -> pd.DataFrame:
    records = json.loads(input_path.read_text())
    out = pd.DataFrame(
        {
            "source_id": [r.get("id", "") for r in records if r.get("event") != "DOCUMENT"],
            "source_event_text": [
                _clean_text(r.get("eventText", "")) for r in records if r.get("event") != "DOCUMENT"
            ],
            "text": [_clean_text(r.get("text", "")) for r in records if r.get("event") != "DOCUMENT"],
            "gt_event_type": [r.get("event", "") for r in records if r.get("event") != "DOCUMENT"],
        }
    )

    if out["text"].eq("").any():
        bad = out.index[out["text"].eq("")].tolist()[:5]
        raise ValueError(f"Encountered empty text after cleaning at rows {bad}")
    unknown = sorted(set(out["gt_event_type"]) - set(EVENT_LABELS))
    if unknown:
        raise ValueError(f"Encountered unexpected event labels: {unknown}")

    conflict_counts = out.groupby("text")["gt_event_type"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    out = out[~out["text"].isin(conflicts)].copy()
    out = out.drop_duplicates(subset=["text", "gt_event_type"]).reset_index(drop=True)

    if out["text"].duplicated().any():
        dup = out.loc[out["text"].duplicated(), "text"].iloc[0]
        raise ValueError(f"Duplicate text remained after cleaning: {dup!r}")

    return out[["source_id", "source_event_text", "text", "gt_event_type"]].reset_index(
        drop=True
    )


def to_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"source_id": "source_original_id"})
    out.insert(
        0,
        "source_id",
        out["text"].map(lambda t: "plover_" + hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]),
    )
    if out["source_id"].duplicated().any():
        raise ValueError("Hashed ids collided")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_V2 if args.version == 2 else DEFAULT_OUTPUT)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download PLOVER_GSR_CAMEO.txt first, "
            "or pass it via --input."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_plover_cameo_event_task(input_path)
    if args.version == 2:
        df = to_v2(df)
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_event_type"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
