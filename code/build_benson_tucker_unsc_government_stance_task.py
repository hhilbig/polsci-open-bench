#!/usr/bin/env python3
"""
Build a long-text task on UN Security Council resolutions' stance toward the
government side of a civil conflict (Benson and Tucker, UNCR).

Sources (all public, no registration):
- Labels: Benson and Tucker, UN Security Council Resolutions and Civil
  Conflict (UNCR), Harvard Dataverse doi:10.7910/DVN/EUYIW0, version 2.0,
  CC0 1.0. File "UNCR (Resolution).tab" (Dataverse file id 6411763, archival
  tab-separated version, md5 372f8ecad3b08f3309478b28b7f932d9), one row per
  conflict-resolution pair. Codebook "UNCR Codebook (Conflict-Resolution).pdf"
  (file id 5150252). Resolutions were paired and coded by a graduate student
  and a faculty member; a third coder re-coded a random sample of conflicts
  (95% agreement on the coded variables, pooled; no chance-corrected figure).
- Text: Fobbe, Gasbarri and Ridi, Corpus of Resolutions: UN Security Council
  (CR-UNSC), version 2024-05-19, Zenodo doi:10.5281/zenodo.11212056, CC0 1.0.
  File CR-UNSC_2024-05-19_EN_TXT_BEST.zip (md5 e8be280bf3714ab67ff10a243b31935f):
  expert-revised OCR text for resolutions 1-899 (and some others) and
  born-digital text for 900 and later.
- Conflict sides: UCDP/PRIO Armed Conflict Dataset v4-2016
  (https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-4-2016.csv, sha256
  ec4ef7668d1ca4c4252abbc2002d14ed2869e9f83ad240c0bf89aba18435720f). The UNCR
  codebook says its conflicts are those of ACD v4-2016, and its
  `conflictidold` matches that version's ConflictId ("1-<conflictidold>").

Unit and label. One resolution, paired with one UCDP/PRIO intrastate
conflict. Gold `government_stance` comes from the codebook's government
sentiment variables, which are mutually exclusive in the file (checked):
  progov = 1 -> supportive   (praises the government faction only)
  antigov = 1 -> condemning  (condemns the government faction only)
  mixgov = 1 -> mixed        (both praises and condemns the government)
  all three 0 -> none
The label is relative to the government side of that specific conflict, so the
item names both sides as UCDP/PRIO lists them for the resolution's year.

Steps and drops, in order:
1. Read the 1,132 conflict-resolution rows.
2. Drop every resolution that appears in more than one conflict row (UNCR's
   "umbrella" resolutions; the check asserts this equals umbrella == 1). Their
   codes are per conflict, and the text alone cannot tell which conflict a
   code refers to. This drops about 36% of rows and 17% of resolutions.
3. Attach UCDP/PRIO side names: SideA (government side) and SideB (opposition
   side) from the ACD v4-2016 row for the conflict and the resolution's year.
   UNCR keeps resolutions up to six months after a conflict's last active year
   and ACD has rows only for active years, so when that year is missing the
   nearest earlier active year is used, else the nearest later one. Secondary
   supporters (SideA2nd, SideB2nd) are not shown.
4. Attach the resolution text from EN_TXT_BEST (file S_RES_<nnnn>_<year>_EN*.txt);
   resolutions without a text file are dropped. Whitespace is normalized
   (lines stripped, runs of spaces collapsed, blank-line runs shortened). Page
   furniture (document symbols, page numbers, barcodes) is left in place.
5. Drop all rows of any text shared by resolutions with conflicting labels;
   keep one row of same-label repeats (none expected).
6. Drop resolutions above MAX_DOC_TOKENS (10,000 tokens, larger of o200k and
   cl100k; see long_text_budget.py). Long texts are dropped, not truncated.
Do not build tasks from UNCR's demand or Chapter VII variables.
"""
import argparse
import hashlib
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from long_text_budget import MAX_DOC_TOKENS, doc_tokens


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUTPUT = REPO / "data" / "benson_tucker_unsc_government_stance.csv"
DEFAULT_INPUT_DIR = Path("/tmp/benson_tucker_unsc")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

FILES = {
    "uncr_resolution.tab": (
        "https://dataverse.harvard.edu/api/access/datafile/6411763",
        "md5", "372f8ecad3b08f3309478b28b7f932d9"),
    "ucdp-prio-acd-4-2016.csv": (
        "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-4-2016.csv",
        "sha256", "ec4ef7668d1ca4c4252abbc2002d14ed2869e9f83ad240c0bf89aba18435720f"),
    "CR-UNSC_2024-05-19_EN_TXT_BEST.zip": (
        "https://zenodo.org/api/records/11212056/files/CR-UNSC_2024-05-19_EN_TXT_BEST.zip/content",
        "md5", "e8be280bf3714ab67ff10a243b31935f"),
}
LABELS = ["none", "supportive", "condemning", "mixed"]
_FILE_RE = re.compile(r"S_RES_(\d{4})_(\d{4})_EN(?:_GOLD)?\.txt$")


def fetch(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, algo, digest) in FILES.items():
        path = input_dir / name
        if not path.exists():
            # Dataverse refuses curl's default user agent; Zenodo refuses browser ones.
            agent = ["-A", UA] if "dataverse" in url else []
            subprocess.run(["curl", "-sSL", "--fail", *agent, "-o", str(path), url], check=True)
        got = hashlib.new(algo, path.read_bytes()).hexdigest()
        if got != digest:
            path.unlink()
            raise ValueError(f"{name}: {algo} {got} != {digest} (file removed)")


def normalize(text: str) -> str:
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def gold(row) -> str:
    if row.mixgov == 1:
        return "mixed"
    if row.progov == 1:
        return "supportive"
    if row.antigov == 1:
        return "condemning"
    return "none"


def sides(acd: pd.DataFrame, cid: int, year: int):
    rows = acd[acd["cid"] == cid]
    if rows.empty:
        return None
    earlier = rows[rows["Year"] <= year]
    pick = earlier.loc[earlier["Year"].idxmax()] if len(earlier) else rows.loc[rows["Year"].idxmin()]
    return pick["SideA"].strip(), pick["SideB"].strip(), int(pick["Year"])


def build(input_dir: Path) -> pd.DataFrame:
    uncr = pd.read_csv(input_dir / "uncr_resolution.tab", sep="\t")
    print(f"1. conflict-resolution rows {len(uncr)}; resolutions {uncr['resolution'].nunique()}; "
          f"conflicts {uncr['conflictid'].nunique()}")
    exclusive = uncr[["progov", "antigov", "mixgov"]].sum(axis=1)
    if exclusive.max() > 1:
        raise ValueError("progov/antigov/mixgov are not mutually exclusive")

    counts = uncr["resolution"].value_counts()
    multi = uncr["resolution"].isin(counts[counts > 1].index)
    if not multi.equals(uncr["umbrella"].eq(1)):
        raise ValueError("multi-conflict resolutions differ from umbrella == 1")
    single = uncr[~multi].copy()
    print(f"2. drop resolutions in more than one conflict row: {int(multi.sum())} rows "
          f"({multi.mean():.1%}) and {int((counts > 1).sum())} resolutions "
          f"({(counts > 1).mean():.1%}); kept {len(single)} resolutions")

    acd = pd.read_csv(input_dir / "ucdp-prio-acd-4-2016.csv")
    acd["cid"] = acd["ConflictId"].str.split("-").str[1].astype(int)
    if acd.duplicated(["cid", "Year"]).any():
        raise ValueError("ACD conflict-year not unique")
    found = [sides(acd, int(r.conflictidold), int(r.year)) for r in single.itertuples()]
    single["government_side"] = [f[0] if f else None for f in found]
    single["opposition_side"] = [f[1] if f else None for f in found]
    single["acd_year"] = [f[2] if f else None for f in found]
    missing = single["government_side"].isna()
    single = single[~missing].copy()
    single["acd_year"] = single["acd_year"].astype(int)
    exact = single["acd_year"].eq(single["year"]).sum()
    print(f"3. UCDP/PRIO sides attached: {len(single)} (no ACD conflict: {int(missing.sum())}); "
          f"same year {exact}, earlier year {int((single['acd_year'] < single['year']).sum())}, "
          f"later year {int((single['acd_year'] > single['year']).sum())}")

    archive = zipfile.ZipFile(input_dir / "CR-UNSC_2024-05-19_EN_TXT_BEST.zip")
    index = {}
    for name in archive.namelist():
        m = _FILE_RE.search(name)
        if m:
            num = int(m.group(1))
            if num in index:
                raise ValueError(f"two text files for resolution {num}")
            index[num] = (name, int(m.group(2)))
    single["text_file"] = single["resolution"].map(lambda n: index.get(int(n), (None, None))[0])
    single["text_year"] = single["resolution"].map(lambda n: index.get(int(n), (None, None))[1])
    no_text = single["text_file"].isna()
    single = single[~no_text].copy()
    year_gap = (single["text_year"] - single["year"]).abs()
    print(f"4. text attached: {len(single)} (no text file: {int(no_text.sum())}); "
          f"UNCR year differs from CR-UNSC year for {int(year_gap.gt(0).sum())} (max gap {int(year_gap.max())})")
    single["text"] = single["text_file"].map(lambda n: normalize(archive.read(n).decode("utf-8")))
    if single["text"].str.len().eq(0).any():
        raise ValueError("empty text")

    single["gt_government_stance"] = [gold(r) for r in single.itertuples()]
    n_labels = single.groupby("text")["gt_government_stance"].transform("nunique")
    conflict = n_labels.gt(1)
    single = single[~conflict].copy()
    before = len(single)
    single = single.drop_duplicates("text").copy()
    print(f"5. duplicate texts: dropped {int(conflict.sum())} rows with conflicting labels, "
          f"{before - len(single)} same-label repeats; kept {len(single)}")

    single["n_tokens"] = single["text"].map(doc_tokens)
    single["n_words"] = single["text"].str.split().str.len()
    over = single["n_tokens"].gt(MAX_DOC_TOKENS)
    by_class = single.loc[over, "gt_government_stance"].value_counts().to_dict()
    single = single[~over].copy()
    print(f"6. drop above {MAX_DOC_TOKENS} tokens: {int(over.sum())} {by_class}; kept {len(single)}")

    single["source_id"] = single["resolution"].map(lambda n: f"unsc_res_{int(n):04d}")
    if single["source_id"].duplicated().any():
        raise ValueError("source_id collision")
    single = single.rename(columns={"country": "conflict_country"})
    cols = ["source_id", "resolution", "year", "month", "day", "conflictid", "conflictidold",
            "conflict_country", "acd_year", "government_side", "opposition_side", "text_file",
            "n_words", "n_tokens", "text", "gt_government_stance"]
    return single[cols].sort_values("resolution").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="directory for the three raw files; missing files are downloaded and checksummed")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    fetch(Path(args.input_dir))
    df = build(Path(args.input_dir))
    if not set(df["gt_government_stance"]) <= set(LABELS):
        raise ValueError("unexpected label")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")
    print(df["gt_government_stance"].value_counts().reindex(LABELS).to_string())
    print(df["n_words"].describe(percentiles=[0.1, 0.5, 0.9]).round().to_string())


if __name__ == "__main__":
    main()
