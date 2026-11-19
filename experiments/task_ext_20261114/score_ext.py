#!/usr/bin/env python3
"""Score the 2026-11-14 open-model run on the 47-task pool, with intervals.

Uses the release scorer (malformed decisions wrong, macro F1 over labels with
gold support, headline_exclude_labels honoured) and the release bootstrap
helper, but draws 500 items per task instead of the release's fixed 100.
Gold comes from the prediction rows, so item ids are never joined to the
task loader (theocharis ids are float strings whose last digit can differ).

The numeric-score task (lemens_gallego_tweet_position) is reported separately
as a Pearson correlation and is not part of the F1 mean.

Run from the repository root:
    python3 experiments/task_ext_20261114/score_ext.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from build_frontier_2026 import _score_malformed_as_incorrect  # noqa: E402
from build_refresh_release import SEED, headline_replicates, interval  # noqa: E402
from scoring import score_metrics  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

RUN = ROOT / "output/sidecar/task_ext_20261114_r2"
OUT = ROOT / "output/sidecar/task_ext_20261114_r2_scored"
RELEASE = ROOT / "output/sidecar/refresh_20261106_release_33"
ITERATIONS = 2000
BASELINE = "qwen3_6_27b_fp8"


def load_tasks() -> dict:
    return {t["name"]: t for t in load_task_definitions(tasks_dir=ROOT / "tasks_ext_v2_20261114")}


def main() -> None:
    frames = []
    for path in sorted(RUN.glob("*/predictions.csv")):
        frame = pd.read_csv(path, low_memory=False, dtype={"item_id": str})
        if frame.duplicated(["task", "item_id"]).any():
            raise ValueError(f"duplicate task/item key in {path}")
        frames.append(frame)
    score(pd.concat(frames, ignore_index=True), OUT, BASELINE)


def score(predictions: pd.DataFrame, out: Path, baseline: str) -> pd.DataFrame:
    """Score every model in predictions (same items per model) and write CSVs to out."""
    tasks = load_tasks()
    released = set(pd.read_csv(RELEASE / "tasks.csv").task)
    models = sorted(predictions.model.unique())
    names = sorted(set(predictions.task))
    if len(names) != 47 or set(names) != set(tasks):
        raise ValueError("task pool mismatch")
    f1_tasks = [n for n in names if tasks[n]["label_kind"] != "score"]
    score_tasks = [n for n in names if tasks[n]["label_kind"] == "score"]

    rng = np.random.default_rng(SEED)
    sizes = {n: int((predictions.task == n).sum() / len(models)) for n in f1_tasks}
    weights = {n: rng.multinomial(sizes[n], np.full(sizes[n], 1 / sizes[n]), size=ITERATIONS).astype(float)
               for n in f1_tasks}
    task_draws = rng.integers(len(f1_tasks), size=(ITERATIONS, len(f1_tasks)))

    per_task, aggregates = [], []
    item_samples, task_samples, points = {}, {}, {}
    for model in models:
        data = predictions[predictions.model == model]
        matrix, estimates = [], []
        for name in f1_tasks:
            g = data[data.task == name].sort_values("item_id").reset_index(drop=True)
            if len(g) != sizes[name]:
                raise ValueError(f"{model}/{name}: {len(g)} rows, expected {sizes[name]}")
            task = tasks[name]
            reference = _score_malformed_as_incorrect(g, task, support_only=True)["headline_f1"]
            estimate = float(headline_replicates(g, task, np.ones((1, len(g))), True)[0])
            if not np.isclose(estimate, reference, atol=1e-12):
                raise ValueError(f"{model}/{name}: fast F1 {estimate} differs from scorer {reference}")
            draws = headline_replicates(g, task, weights[name], True)
            matrix.append(draws)
            estimates.append(estimate)
            lo, hi = interval(draws)
            per_task.append(dict(model=model, task=name, kind=task["label_kind"], in_release=name in released,
                                 n=len(g), headline_f1=estimate, item_ci_low=lo, item_ci_high=hi))
        for name in score_tasks:
            g = data[data.task == name]
            ok = g.parse_error.isna() | g.parse_error.astype(str).eq("not_applicable")
            s = score_metrics(tasks[name], g[ok])
            per_task.append(dict(model=model, task=name, kind="score", in_release=name in released, n=len(g),
                                 pearson_r=s["pearson_r"], mae=s["mae"],
                                 not_applicable_rate=float(g.parse_error.astype(str).eq("not_applicable").mean())))
        matrix, estimates = np.asarray(matrix), np.asarray(estimates)
        item_samples[model] = matrix.mean(axis=0)
        task_samples[model] = estimates[task_draws].mean(axis=1)
        points[model] = float(estimates.mean())
        in_rel = np.array([n in released for n in f1_tasks])
        aggregates.append(dict(
            model=model, tasks=len(f1_tasks), mean_task_f1=points[model],
            item_ci_low=interval(item_samples[model])[0], item_ci_high=interval(item_samples[model])[1],
            task_ci_low=interval(task_samples[model])[0], task_ci_high=interval(task_samples[model])[1],
            mean_f1_tasks_in_release=float(estimates[in_rel].mean()),
            mean_f1_tasks_not_in_release=float(estimates[~in_rel].mean()),
            malformed=int((data.parse_error.notna() & ~data.parse_error.astype(str).eq("not_applicable")).sum())))
    pairs = []
    for model, ref in itertools.permutations(models, 2):
        pairs.append(dict(model=model, reference=ref, difference=points[model] - points[ref],
                          item_ci_low=interval(item_samples[model] - item_samples[ref])[0],
                          item_ci_high=interval(item_samples[model] - item_samples[ref])[1],
                          task_ci_low=interval(task_samples[model] - task_samples[ref])[0],
                          task_ci_high=interval(task_samples[model] - task_samples[ref])[1]))
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(aggregates).sort_values("mean_task_f1", ascending=False)
    lemens = pd.DataFrame(per_task).query("kind == 'score'").set_index("model").pearson_r
    summary["lemens_position_pearson"] = summary.model.map(lemens)
    summary.to_csv(out / "model_summary.csv", index=False)
    pd.DataFrame(per_task).to_csv(out / "per_task.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "pairs.csv", index=False)
    print(summary.round(3).to_string(index=False))
    vs = pd.DataFrame(pairs).query("reference == @baseline").sort_values("difference", ascending=False)
    print(f"\nDifference from {baseline}:")
    print(vs[["model", "difference", "item_ci_low", "item_ci_high", "task_ci_low", "task_ci_high"]].round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
