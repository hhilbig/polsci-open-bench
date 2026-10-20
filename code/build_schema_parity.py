#!/usr/bin/env python3
"""
Score the constrained-decoding experiment from its item-level predictions.

The main release gave API models server-side JSON-schema enforcement and local
models none. Four open-weight checkpoints were therefore run twice on Hive over
the same 16,425 items: once with schema-constrained decoding (`guided`) and once
with the prompt alone (`plain`). Model, GPU, prompt, items, seed and temperature
are identical across the two arms; the configs are
`experiments/schema_parity_20260918_{guided,plain}.yaml`.

Input is `output/sidecar/schema_parity_20260918/predictions.csv.gz`, a compact
copy of the Hive outputs (labels and parse errors only, no texts or raw
responses). Scores use `scoring.headline_f1` with unparseable rows dropped, the
main-release rule, so the result is comparable with `output/summary.csv`.

Usage:
  python3 code/build_schema_parity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import headline_f1, mask_unparsed
from task_registry import load_task_definitions

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "output/sidecar/schema_parity_20260918"
SEED = 20260918
DRAWS = 2000


def score_cells(predictions: pd.DataFrame, tasks: dict) -> pd.DataFrame:
    rows = []
    for (arm, model, task), g in predictions.groupby(["arm", "model", "task"]):
        n = len(g)
        ok = mask_unparsed(g)
        rows.append({"arm": arm, "model": model, "task": task, "n": n,
                     "parse_errors": n - len(ok), "headline_f1": headline_f1(tasks[task], ok)})
    return pd.DataFrame(rows)


def paired_difference(cells: pd.DataFrame, seed: int = SEED, draws: int = DRAWS) -> dict:
    """Guided minus plain, averaged over checkpoint-task cells, with a cell bootstrap."""
    wide = cells.pivot_table(index=["model", "task"], columns="arm", values="headline_f1")
    diff = (wide["guided"] - wide["plain"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(draws)])
    return {"cells": len(diff), "mean_diff": diff.mean(),
            "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
            "guided_better": int((diff > 0).sum()), "ties": int((diff == 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", default=str(DIR / "predictions.csv.gz"))
    parser.add_argument("--output", default=str(DIR / "task_scores.csv"))
    args = parser.parse_args()
    tasks = {t["name"]: t for t in load_task_definitions()}
    predictions = pd.read_csv(args.predictions, low_memory=False)
    cells = score_cells(predictions, tasks)
    cells.to_csv(args.output, index=False)
    summary = paired_difference(cells)
    print(f"wrote {args.output}")
    print(f"guided minus plain over {summary['cells']} cells: {summary['mean_diff']:+.4f} F1 "
          f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}]; guided better in "
          f"{summary['guided_better']}, tied in {summary['ties']}")
    print(cells.groupby("arm")[["n", "parse_errors"]].sum().to_string())


if __name__ == "__main__":
    main()
