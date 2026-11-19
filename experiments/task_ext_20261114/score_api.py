#!/usr/bin/env python3
"""Score the API models and the open models on the 4,400-item API draw.

The API models ran on a stratified draw of the items the open models ran
(output/sidecar/task_ext_20261114_api/draw.json: 100 items per task, 50 for the
six longest tasks). This script puts both groups on exactly the same items:
API predictions are the pilot plus remainder stage, and open-model predictions
are the rows of the full open run whose item id is the draw row's open_item_id.

The items OpenRouter's moderation kept out of the Anthropic batches
(moderation_excluded.json, 2 items) are dropped for every model, so all models
are scored on 4,398 items. Scoring and intervals use score_ext.score (release
scorer, malformed answers wrong, 2,000 bootstrap draws).

Run from the repository root:
    python3 experiments/task_ext_20261114/score_api.py            # all models must be complete
    python3 experiments/task_ext_20261114/score_api.py --partial  # skip incomplete API models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_ext  # noqa: E402

ROOT = score_ext.ROOT
API = ROOT / "output/sidecar/task_ext_20261114_api"
OUT = ROOT / "output/sidecar/task_ext_20261114_api_scored"
API_MODELS = ["claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5", "gpt-6.1-sol", "claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "gemini-3.1-flash-lite", "deepseek-v4.1-flash-1017",
              "gpt-6-luna", "gemini-4-argon", "glm-5.3"]


def main(partial: bool) -> None:
    predictions, keep = load_predictions(partial)
    print(f"{predictions.model.nunique()} models x {len(keep)} items")
    score_ext.score(predictions, OUT, score_ext.BASELINE)


def load_predictions(partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every model's predictions on the draw items, and the draw items themselves."""
    draw = json.loads((API / "draw.json").read_text())["rows"]
    excluded = set(json.loads((API / "moderation_excluded.json").read_text()))
    keep = pd.DataFrame([dict(custom_id=f"x{i:05d}", task=r["task"], item_id=str(r["open_item_id"]))
                         for i, r in enumerate(draw) if f"x{i:05d}" not in excluded])

    frames = []
    for model in API_MODELS:
        paths = [API / model / stage / "predictions.csv" for stage in ("pilot", "remainder")]
        if not all(p.exists() for p in paths):
            if partial:
                print(f"skipping {model}: not collected", file=sys.stderr)
                continue
            raise FileNotFoundError(f"{model}: missing predictions")
        frame = pd.concat([pd.read_csv(p, low_memory=False, dtype={"item_id": str}) for p in paths])
        frame = frame[~frame.custom_id.isin(excluded)]
        missing = set(keep.custom_id) - set(frame.custom_id)
        # A request the provider failed counts as a wrong answer, but a model
        # with many unanswered rows still has batches that have not run.
        # Gemini rejected the score task's schema (all 100 requests failed), which
        # leaves its Pearson correlation missing but does not touch the F1 tasks.
        score_tasks = {n for n, t in score_ext.load_tasks().items() if t["label_kind"] == "score"}
        unanswered = int(frame[~frame.task.isin(score_tasks)].stop_reason.isna().sum())
        if missing or unanswered > 0.01 * len(frame):
            message = f"{model}: {len(missing)} draw items missing, {unanswered} without an answer"
            if partial:
                print(f"skipping {message}", file=sys.stderr)
                continue
            raise ValueError(message)
        if unanswered:
            print(f"{model}: {unanswered} request(s) failed at the provider; scored as wrong", file=sys.stderr)
        frame = frame.drop(columns="item_id").merge(keep, on=["custom_id", "task"], validate="one_to_one")
        frames.append(frame)

    for path in sorted(score_ext.RUN.glob("*/predictions.csv")):
        frame = pd.read_csv(path, low_memory=False, dtype={"item_id": str})
        frame = frame.merge(keep, on=["task", "item_id"], validate="one_to_one")
        if len(frame) != len(keep):
            raise ValueError(f"{path}: {len(frame)} of {len(keep)} draw items found")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), keep


if __name__ == "__main__":
    main("--partial" in sys.argv)
