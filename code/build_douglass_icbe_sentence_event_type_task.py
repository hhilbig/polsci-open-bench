#!/usr/bin/env python3
"""
Build the cleaned ICBe sentence-level event-type task from the public agreed
event annotations and aligned sentence corpus.

The resulting task keeps every public sentence and assigns one of five labels:
`No Event`, `Action`, `Speech`, `Thought`, or `Mixed`. The `Mixed` label
captures sentences with multiple agreed event types, while `No Event` captures
sentences with no agreed event type in the public agreed-event file.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pyreadr


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_EVENTS_INPUT = Path("/tmp/ICBe_V1.1_events_agreed.Rds")
DEFAULT_SENTENCES_INPUT = Path("/tmp/icb_long_crisis_sentence_unique.Rds")
DEFAULT_TITLES_INPUT = Path("/tmp/icb_corpus_V1.0_May_16_2022.Rds")
DEFAULT_OUTPUT = REPO / "data" / "douglass_icbe_sentence_event_type.csv"

BASE_TYPES = {"action", "speech", "thought"}
LABEL_MAP = {
    "action": "Action",
    "speech": "Speech",
    "thought": "Thought",
}
ALLOWED_LABELS = {"No Event", "Action", "Speech", "Thought", "Mixed"}


def _read_rds(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    if None not in result:
        raise ValueError(f"{path} did not contain a default R object")
    df = result[None]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not decode to a pandas DataFrame")
    return df.copy()


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _clean_crisis_title(value) -> str:
    text = _clean_text(value)
    text = re.sub(r"^CRISNO\s+\d+\s+", "", text).strip()
    return text


def _label_from_types(type_string: str) -> str:
    if type_string == "":
        return "No Event"
    if "|" in type_string:
        return "Mixed"
    return LABEL_MAP[type_string]


def build_douglass_icbe_sentence_event_type_task(
    events_path: Path,
    sentences_path: Path,
    titles_path: Path,
) -> pd.DataFrame:
    events = _read_rds(events_path)
    sentences = _read_rds(sentences_path)
    titles = _read_rds(titles_path)

    required_events = {"crisno", "sentence_number_int_aligned", "event_type"}
    required_sentences = {"crisno", "sentence_clean"}
    required_titles = {"crisno", "crisis_title"}
    missing_events = required_events - set(events.columns)
    missing_sentences = required_sentences - set(sentences.columns)
    missing_titles = required_titles - set(titles.columns)
    if missing_events:
        raise ValueError(f"{events_path} is missing required columns: {sorted(missing_events)}")
    if missing_sentences:
        raise ValueError(
            f"{sentences_path} is missing required columns: {sorted(missing_sentences)}"
        )
    if missing_titles:
        raise ValueError(f"{titles_path} is missing required columns: {sorted(missing_titles)}")

    events = events.copy()
    events["crisno"] = pd.to_numeric(events["crisno"], errors="raise").astype(int)
    events["sentence_number_int_aligned"] = pd.to_numeric(
        events["sentence_number_int_aligned"], errors="raise"
    ).astype(int)
    events["event_type"] = events["event_type"].fillna("").map(_clean_text).str.lower()

    labels = (
        events.groupby(["crisno", "sentence_number_int_aligned"], as_index=False)["event_type"]
        .agg(lambda s: "|".join(sorted({value for value in s if value in BASE_TYPES})))
        .rename(columns={"event_type": "source_type_string"})
    )
    labels["gt_event_type"] = labels["source_type_string"].map(_label_from_types)

    sentences = sentences.copy()
    sentences["crisno"] = pd.to_numeric(sentences["crisno"], errors="raise").astype(int)
    sentences["text"] = sentences["sentence_clean"].map(_clean_text)
    sentences["source_sentence_number"] = sentences.groupby("crisno").cumcount() + 1
    sentences = sentences[["crisno", "source_sentence_number", "text"]]
    sentences = sentences[sentences["text"].ne("")].reset_index(drop=True)

    titles = titles.copy()
    titles["crisno"] = pd.to_numeric(titles["crisno"], errors="raise").astype(int)
    titles["source_crisis_title_raw"] = titles["crisis_title"].map(_clean_text)
    titles["crisis_title"] = titles["crisis_title"].map(_clean_crisis_title)
    titles = titles[["crisno", "source_crisis_title_raw", "crisis_title"]]

    out = sentences.merge(titles, on="crisno", how="left", validate="many_to_one")
    out = out.merge(
        labels[["crisno", "sentence_number_int_aligned", "source_type_string", "gt_event_type"]],
        left_on=["crisno", "source_sentence_number"],
        right_on=["crisno", "sentence_number_int_aligned"],
        how="left",
        validate="one_to_one",
    )

    out["gt_event_type"] = out["gt_event_type"].fillna("No Event")
    out["source_type_string"] = out["source_type_string"].fillna("")
    out["source_id"] = (
        out["crisno"].astype(str) + "_" + out["source_sentence_number"].astype(str)
    )

    final = out[
        [
            "source_id",
            "crisno",
            "source_sentence_number",
            "source_crisis_title_raw",
            "crisis_title",
            "text",
            "source_type_string",
            "gt_event_type",
        ]
    ].rename(columns={"crisno": "source_crisno"})

    if final["crisis_title"].isna().any() or final["crisis_title"].eq("").any():
        bad = final.index[final["crisis_title"].isna() | final["crisis_title"].eq("")].tolist()[:5]
        raise ValueError(f"Missing crisis_title values after join/cleaning at rows {bad}")
    if final["text"].isna().any() or final["text"].eq("").any():
        bad = final.index[final["text"].isna() | final["text"].eq("")].tolist()[:5]
        raise ValueError(f"Missing sentence text after cleaning at rows {bad}")
    if final["source_id"].duplicated().any():
        dup = final.loc[final["source_id"].duplicated(), "source_id"].iloc[0]
        raise ValueError(f"Duplicate source_id found: {dup}")
    if not set(final["gt_event_type"]).issubset(ALLOWED_LABELS):
        bad = sorted(set(final["gt_event_type"]) - ALLOWED_LABELS)
        raise ValueError(f"Unexpected labels found: {bad}")

    task_text = "Crisis title: " + final["crisis_title"] + "\nSentence: " + final["text"]
    if task_text.duplicated().any():
        dup = task_text[task_text.duplicated()].iloc[0]
        raise ValueError(f"Duplicate task text found after title+sentence construction: {dup!r}")

    return final.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-input", default=str(DEFAULT_EVENTS_INPUT))
    ap.add_argument("--sentences-input", default=str(DEFAULT_SENTENCES_INPUT))
    ap.add_argument("--titles-input", default=str(DEFAULT_TITLES_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    events_path = Path(args.events_input)
    sentences_path = Path(args.sentences_input)
    titles_path = Path(args.titles_input)
    for path in [events_path, sentences_path, titles_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. Download the public ICBe RDS files first "
                "and pass them in explicitly if they are not in /tmp."
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_douglass_icbe_sentence_event_type_task(
        events_path=events_path,
        sentences_path=sentences_path,
        titles_path=titles_path,
    )
    df.to_csv(output_path, index=False)

    print(f"wrote {output_path} ({len(df)} rows)")
    print(df["gt_event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
