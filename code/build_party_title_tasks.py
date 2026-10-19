#!/usr/bin/env python3
"""
Build the "manifesto title shown vs hidden" tasks for the party-names experiment.

Question: does telling a model which party wrote a manifesto sentence move its
left-right code, and by how much compared with people? Benoit et al. (2016)
recorded, for every human coding, whether the coder saw the manifesto's title
(`title_shown`). Two human comparisons exist in their data:

1. Experts coded the corpus twice: first in natural order with the title shown,
   about a year later in random order without it. Same coder, same sentence.
2. Semi-experts were randomly shown or not shown the title (their Online
   Appendix, Section 5), on sentences from four manifestos.

This script takes every Conservative or Labour sentence that people coded in a
domain both with and without the title, in either comparison, and writes one
data file per domain plus two task specs per domain that differ only in one
line above the passage: "Source: Conservative Party manifesto, 1992" (shown) or
nothing (hidden). Prompt, context and response format are those of the existing
benoit_manifesto_*_position tasks. Every row is used (no 250+250 draw).

gt_code is the rounded mean of all title-hidden human codes in the domain. It
only lets the benchmark runner compute its usual metrics; the analysis compares
model and human shifts directly.

Inputs (github.com/kbenoit/CSTA-APSR, "Data - Created/"): master.sentences.csv
and coding_all_long_2014-03-14.csv from coding_all_long.zip.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from build_benoit_manifesto_position_tasks import LABELS, load_sentences  # noqa: E402
import task_registry  # noqa: E402

OUT_TASKS = REPO / "tasks_party_title"
PARTY_NAME = {"Conservatives": "Conservative Party", "Labour": "Labour Party"}
DOMAINS = {"Economic": "economic", "Social": "social"}


def human_pairs(codes: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Sentences coded in `domain` with and without the title, and by whom."""
    d = codes[codes["scale"].eq(domain)]
    experts = d[d["source"].eq("Experts")]
    per_coder = experts.groupby(["coderid", "sentenceid"])["title_shown"].nunique()
    expert_ids = set(per_coder[per_coder.eq(2)].reset_index()["sentenceid"])
    semi = d[d["source"].eq("SemiExperts")]
    per_sentence = semi.groupby("sentenceid")["title_shown"].nunique()
    semi_ids = set(per_sentence[per_sentence.eq(2)].index)
    ids = sorted(expert_ids | semi_ids)
    hidden = d[d["title_shown"].eq("No") & d["sentenceid"].isin(ids)]
    gt = hidden.groupby("sentenceid")["code"].mean().round().astype(int).clip(-2, 2)
    return pd.DataFrame({
        "sentenceid": ids,
        "in_expert_pairs": [i in expert_ids for i in ids],
        "in_semi_expert_experiment": [i in semi_ids for i in ids],
        "gt_code": [gt.get(i) for i in ids],
    })


def write_task(domain_key: str, condition: str, data_file: Path, order: int):
    base = yaml.safe_load((REPO / "tasks_ext" / f"benoit_manifesto_{domain_key}_position.yaml").read_text())
    spec = dict(base)
    spec["name"] = f"benoit_title_{domain_key}_position_{condition}"
    spec["order"] = order
    spec["data_file"] = f"../data/{data_file.name}"
    passage = base["text"]["template"]
    if condition == "shown":
        passage = "Source: {title_line}\n" + passage
    spec["text"] = {"template": passage}
    spec["sampling"] = {"n_v1": 100000, "n_v2_new": 0}
    out = OUT_TASKS / f"{spec['name']}.yaml"
    header = ("# Manifesto title shown vs hidden (party-names experiment); built by\n"
              "# code/build_party_title_tasks.py. Uses every row of the data file.\n")
    out.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--codings", required=True)
    args = ap.parse_args()

    sentences = load_sentences(Path(args.sentences))
    codes = pd.read_csv(args.codings, low_memory=False,
                        usecols=["coderid", "sentenceid", "code", "title_shown", "scale", "party", "source"])
    codes = codes[codes["party"].isin(PARTY_NAME) & codes["code"].isin([-2, -1, 0, 1, 2])]

    OUT_TASKS.mkdir(exist_ok=True)
    for i, (domain, key) in enumerate(DOMAINS.items()):
        pairs = human_pairs(codes, domain)
        df = pairs.merge(sentences, on="sentenceid", how="left", validate="one_to_one")
        if df["sentence_text"].isna().any() or df["sentence_text"].eq("").any():
            raise ValueError(f"{domain}: some sentences have no text")
        df["source_id"] = "benoit_" + df["sentenceid"].astype(str)
        df["title_line"] = df["party"].map(PARTY_NAME) + " manifesto, " + df["year"].astype(str)
        df["gt_position"] = df["gt_code"].map(LABELS[domain])
        df = df.rename(columns={"sentence_text": "text", "manifestoid": "source_manifesto",
                                "party": "source_party", "year": "source_year"})
        cols = ["source_id", "sentenceid", "source_manifesto", "source_party", "source_year", "title_line",
                "context_before", "text", "context_after", "in_expert_pairs", "in_semi_expert_experiment",
                "gt_code", "gt_position"]
        data_file = REPO / "data" / f"benoit_title_{key}_position.csv"
        df[cols].to_csv(data_file, index=False)
        print(f"{domain}: {len(df)} sentences ({int(df.in_expert_pairs.sum())} expert pairs, "
              f"{int(df.in_semi_expert_experiment.sum())} semi-expert experiment), "
              f"{df.source_party.value_counts().to_dict()}")

        loaded = {}
        for j, condition in enumerate(["hidden", "shown"]):
            path = write_task(key, condition, data_file, order=400 + 2 * i + j)
            loaded[condition] = task_registry.load_task_definitions(task_manifest=path)[0]["loader"]()
        h, s = loaded["hidden"], loaded["shown"]
        assert len(h) == len(s) == len(df), "a task does not use every row"
        assert [x["item_id"] for x in h] == [x["item_id"] for x in s]
        orig = (REPO / "tasks_ext" / f"benoit_manifesto_{key}_position.yaml")
        assert h[0]["user_content"].startswith("Manifesto section:"), "hidden passage changed"
        for x, y in zip(h, s):
            assert y["user_content"].startswith("Source: ") and y["user_content"].split("\n", 1)[1] == x["user_content"]
        print(f"  tasks: hidden and shown, {len(h)} items each, identical except the source line "
              f"(template from {orig.name})")


if __name__ == "__main__":
    main()
