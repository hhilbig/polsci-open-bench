#!/usr/bin/env python3
"""
Build two sentence-level position tasks from Benoit et al. (2016, APSR).

Benoit, Conway, Lauderdale, Laver and Mikhaylov coded 18,263 sentences from
British Conservative, Labour and Liberal Democrat manifestos (1987-2010). Each
coder first assigned a sentence to economic policy, social policy, or neither,
then placed economic sentences on a five-point left-right scale and social
sentences on a five-point liberal-conservative scale. Coders saw the target
sentence with two sentences of context on either side.

Ground truth uses the expert codings, the paper's reference coding:

1. Keep one coding per expert and sentence (the earliest, since some experts
   coded the same sentence in more than one stage).
2. A sentence enters the economic (social) task if a strict majority of its
   experts assigned it to economic (social) policy.
3. The label is the most common five-point code among the experts who placed
   the sentence in that domain. Sentences with a tied most common code are
   dropped.

Inputs:
- "Data - Created/master.sentences.csv" from github.com/kbenoit/CSTA-APSR
  (sentence text plus the context sentences shown to coders).
- coding_all_long_2014-03-14.csv inside "Data - Created/coding_all_long.zip"
  from the same repository (one row per individual coding). The Ornstein,
  Blasingame and Truscott (2025, PSRM) archive, doi:10.7910/DVN/DZZ0OM,
  redistributes a byte-identical copy as application3-benoit-sentence-estimates.tab.
"""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_SENTENCES = Path("/tmp/benoit_csta/master.sentences.csv")
DEFAULT_CODINGS = Path("/tmp/benoit_csta/coding_all_long_2014-03-14.csv")
DEFAULT_ECON_OUTPUT = REPO / "data" / "benoit_manifesto_economic_position.csv"
DEFAULT_SOCIAL_OUTPUT = REPO / "data" / "benoit_manifesto_social_position.csv"

LABELS = {
    "Economic": {
        -2: "very left",
        -1: "somewhat left",
        0: "neither left nor right",
        1: "somewhat right",
        2: "very right",
    },
    "Social": {
        -2: "very liberal",
        -1: "somewhat liberal",
        0: "neither liberal nor conservative",
        1: "somewhat conservative",
        2: "very conservative",
    },
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_sentences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sentenceid", "manifestoid", "party", "year", "sentence_text",
                "pre_sentence", "post_sentence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[list(required)].copy()
    for col in ["sentence_text", "pre_sentence", "post_sentence"]:
        df[col] = df[col].map(_clean_text)
    if df["sentenceid"].duplicated().any():
        raise ValueError("Duplicate sentenceid values in the sentence list")
    # pre_sentence and post_sentence are the two-sentence context the coders saw.
    return df.rename(columns={"pre_sentence": "context_before", "post_sentence": "context_after"})


def load_expert_codings(path: Path) -> pd.DataFrame:
    codes = pd.read_csv(path, low_memory=False)
    required = {"coderid", "sentenceid", "code", "scale", "source", "coding_timestamp"}
    missing = required - set(codes.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    experts = codes[codes["source"].eq("Experts")].copy()
    experts["timestamp"] = pd.to_datetime(
        experts["coding_timestamp"], format="%d%b%Y %H:%M:%S", errors="coerce"
    )
    experts = experts.sort_values(["timestamp", "coderid", "sentenceid"], na_position="last")
    experts = experts.drop_duplicates(["coderid", "sentenceid"], keep="first")
    experts["domain"] = experts["scale"].fillna("None")
    bad = experts.loc[experts["domain"].ne("None") & ~experts["code"].isin([-2, -1, 0, 1, 2])]
    if len(bad):
        raise ValueError(f"{len(bad)} expert codings have a domain but no valid -2..2 code")
    return experts


def build_domain_task(sentences: pd.DataFrame, experts: pd.DataFrame, domain: str):
    shares = experts.groupby("sentenceid")["domain"].value_counts(normalize=True).unstack(fill_value=0)
    n_experts = experts.groupby("sentenceid").size()
    in_domain = shares.index[shares.get(domain, 0) > 0.5]

    coded = experts[experts["sentenceid"].isin(in_domain) & experts["domain"].eq(domain)]
    counts = coded.groupby("sentenceid")["code"].value_counts().unstack(fill_value=0)
    top = counts.max(axis=1)
    tied = counts.eq(top, axis=0).sum(axis=1) > 1
    mode = counts.idxmax(axis=1)[~tied].astype(int)

    out = pd.DataFrame({"sentenceid": mode.index, "gt_code": mode.values})
    out["gt_position"] = out["gt_code"].map(LABELS[domain])
    out["n_experts"] = out["sentenceid"].map(n_experts).astype(int)
    out["n_experts_in_domain"] = out["sentenceid"].map(coded.groupby("sentenceid").size()).astype(int)
    out["mean_code_in_domain"] = out["sentenceid"].map(coded.groupby("sentenceid")["code"].mean()).round(3)
    out = out.merge(sentences, on="sentenceid", how="left", validate="one_to_one")
    if out["sentence_text"].isna().any() or out["sentence_text"].eq("").any():
        raise ValueError(f"{domain}: some coded sentences have no text in the sentence list")

    # Same text with different labels is a conflict; same text with the same
    # label is a repeat. Context differs across repeats, so keep the first.
    conflicts = out.groupby("sentence_text")["gt_position"].nunique()
    n_conflict_rows = out["sentence_text"].isin(conflicts[conflicts > 1].index).sum()
    out = out[~out["sentence_text"].isin(conflicts[conflicts > 1].index)]
    n_before_dedup = len(out)
    out = out.drop_duplicates("sentence_text").sort_values("sentenceid").reset_index(drop=True)

    stats = {
        "domain_majority": len(in_domain),
        "tied_mode_dropped": int(tied.sum()),
        "conflict_rows_dropped": int(n_conflict_rows),
        "duplicate_rows_dropped": n_before_dedup - len(out),
        "final": len(out),
    }

    out["source_id"] = "benoit_" + out["sentenceid"].astype(str)
    out = out.rename(columns={"sentence_text": "text", "manifestoid": "source_manifesto",
                              "party": "source_party", "year": "source_year"})
    cols = ["source_id", "sentenceid", "source_manifesto", "source_party", "source_year",
            "context_before", "text", "context_after",
            "n_experts", "n_experts_in_domain", "mean_code_in_domain", "gt_code", "gt_position"]
    return out[cols], stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--codings", default=str(DEFAULT_CODINGS))
    ap.add_argument("--econ-output", default=str(DEFAULT_ECON_OUTPUT))
    ap.add_argument("--social-output", default=str(DEFAULT_SOCIAL_OUTPUT))
    args = ap.parse_args()

    for path, hint in [
        (Path(args.sentences), "Data - Created/master.sentences.csv from github.com/kbenoit/CSTA-APSR"),
        (Path(args.codings), "coding_all_long_2014-03-14.csv from Data - Created/coding_all_long.zip in github.com/kbenoit/CSTA-APSR"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}. Download {hint}, or pass the path.")

    sentences = load_sentences(Path(args.sentences))
    experts = load_expert_codings(Path(args.codings))
    print(f"sentences: {len(sentences)}; expert codings after per-coder dedup: {len(experts)}")

    for domain, output in [("Economic", args.econ_output), ("Social", args.social_output)]:
        df, stats = build_domain_task(sentences, experts, domain)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n{domain}: wrote {output} ({len(df)} rows)")
        print("  " + ", ".join(f"{k}={v}" for k, v in stats.items()))
        print(df["gt_position"].value_counts().reindex(LABELS[domain].values()).to_string())


if __name__ == "__main__":
    main()
