#!/usr/bin/env python3
"""Diagnose why Qwen3.6 gains and losses differ across coding tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from task_registry import load_task_definitions  # noqa: E402

ROOT = REPO / "output" / "sidecar" / "frontier_2026" / "broad18"
LLAMA = ROOT / "holdout16" / "llama3_1_70b_instruct_fp8_dynamic_full34" / "predictions.csv"
QWEN = REPO / "output" / "sidecar" / "hive_model_bakeoff_20260804" / "qwen3_6_27b_fp8" / "predictions.csv"
SINGLE_USER = ROOT / "targeted_ablation" / "qwen3_6_27b_fp8_single_user" / "predictions.csv"
DELTA_FILE = ROOT / "holdout_task_deltas.csv"
MODEL_TASK = ROOT / "model_by_task.csv"
SEED = 20260820


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"item_id": str}, low_memory=False)
    if frame.duplicated(["task", "item_id"]).any():
        raise ValueError(f"duplicate task/item keys in {path}")
    return frame


def _f1(gold: pd.Series, pred: pd.Series, label: object) -> float:
    valid = gold.notna() & pred.notna()
    g, p = gold.loc[valid] == label, pred.loc[valid] == label
    tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def _score(group: pd.DataFrame, task: dict) -> float:
    if task["label_kind"] == "binary":
        key = task["label_key"]
        return _f1(pd.to_numeric(group[f"gt_{key}"]), pd.to_numeric(group[f"pred_{key}"]), 1)
    if task["label_kind"] == "categorical":
        key = task["label_key"]
        return float(np.mean([_f1(group[f"gt_{key}"], group[f"pred_{key}"], label) for label in task["labels"]]))
    return float(np.mean([
        _f1(pd.to_numeric(group[f"gt_{label}"]), pd.to_numeric(group[f"pred_{label}"]), 1)
        for label in task["labels"]
    ]))


def _row_correct(frame: pd.DataFrame, task: dict, prefix: str) -> pd.Series:
    keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
    matches = []
    for key in keys:
        if task["label_kind"] in {"binary", "multi_binary"}:
            matches.append(pd.to_numeric(frame[f"{prefix}_{key}"]) == pd.to_numeric(frame[f"gt_{key}"]))
        else:
            matches.append(frame[f"{prefix}_{key}"].astype(str) == frame[f"gt_{key}"].astype(str))
    return pd.concat(matches, axis=1).all(axis=1)


def confusion_decomposition(llama: pd.DataFrame, qwen: pd.DataFrame, tasks: dict[str, dict]) -> pd.DataFrame:
    merged = llama.merge(qwen, on=["task", "item_id"], suffixes=("_llama", "_qwen"), validate="one_to_one")
    rows = []
    for name, group in merged.groupby("task", sort=True):
        task = tasks[name]
        keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
        decisions = []
        for key in keys:
            decisions.append(pd.DataFrame({
                "gold": group[f"gt_{key}_llama"],
                "llama": group[f"pred_{key}_llama"],
                "qwen": group[f"pred_{key}_qwen"],
            }))
        decision = pd.concat(decisions, ignore_index=True)
        llama_correct = _row_correct(group.rename(columns={c: c.removesuffix("_llama") for c in group if c.endswith("_llama")}), task, "pred")
        qwen_correct = _row_correct(group.rename(columns={c: c.removesuffix("_qwen") for c in group if c.endswith("_qwen")}), task, "pred")
        row = {
            "task": name, "label_kind": task["label_kind"], "n_items": len(group),
            "llama_only_correct": int((llama_correct & ~qwen_correct).sum()),
            "qwen_only_correct": int((qwen_correct & ~llama_correct).sum()),
            "both_correct": int((qwen_correct & llama_correct).sum()),
            "both_wrong": int((~qwen_correct & ~llama_correct).sum()),
            "llama_headline_f1": _score(group.rename(columns={c: c.removesuffix("_llama") for c in group if c.endswith("_llama")}), task),
            "qwen_headline_f1": _score(group.rename(columns={c: c.removesuffix("_qwen") for c in group if c.endswith("_qwen")}), task),
        }
        if task["label_kind"] in {"binary", "multi_binary"}:
            for model in ["llama", "qwen"]:
                g = pd.to_numeric(decision["gold"]) == 1
                p = pd.to_numeric(decision[model]) == 1
                tp, fp, fn, tn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum()), int((~g & ~p).sum())
                row.update({
                    f"{model}_positive_rate": float(p.mean()),
                    f"{model}_precision": tp / (tp + fp) if tp + fp else np.nan,
                    f"{model}_recall": tp / (tp + fn) if tp + fn else np.nan,
                    f"{model}_specificity": tn / (tn + fp) if tn + fp else np.nan,
                })
            row["gold_positive_rate"] = float((pd.to_numeric(decision["gold"]) == 1).mean())
        else:
            neutral = {str(x).lower() for x in task["labels"]} & {"none", "neutral", "unclear"}
            for model in ["llama", "qwen"]:
                row[f"{model}_neutral_rate"] = float(decision[model].astype(str).str.lower().isin(neutral).mean()) if neutral else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["delta_f1"] = out["qwen_headline_f1"] - out["llama_headline_f1"]
    if len(out) != 34:
        raise ValueError(f"expected 34 task decompositions, found {len(out)}")
    return out


def prompt_sensitivity(qwen: pd.DataFrame, ablation: pd.DataFrame, tasks: dict[str, dict]) -> pd.DataFrame:
    keys = ablation[["task", "item_id"]]
    baseline = qwen.merge(keys, on=["task", "item_id"], validate="one_to_one")
    if len(baseline) != len(ablation) or len(ablation) != 3600:
        raise ValueError("Qwen single-user comparison lacks exact 3,600-row paired coverage")
    rows = []
    for name, alternate in ablation.groupby("task", sort=True):
        base = baseline.loc[baseline["task"] == name]
        rows.append({"task": name, "baseline_f1": _score(base, tasks[name]),
                     "single_user_f1": _score(alternate, tasks[name])})
    out = pd.DataFrame(rows)
    out["single_user_minus_baseline"] = out["single_user_f1"] - out["baseline_f1"]
    return out


def task_trajectories(deltas: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_csv(MODEL_TASK)
    scores = scores.loc[(scores["series"] == "open_hive") & scores["headline_f1"].notna()].copy()
    scores["year"] = pd.to_datetime(scores["plot_date"]).dt.year + (pd.to_datetime(scores["plot_date"]).dt.dayofyear - 1) / 365.25
    family = deltas[["task", "family"]].drop_duplicates()
    rows = []
    for task, group in scores.groupby("task", sort=True):
        slope = np.polyfit(group["year"], group["headline_f1"], 1)[0]
        rows.append({"task": task, "models": len(group), "f1_points_per_year": slope * 100})
    out = pd.DataFrame(rows).merge(family, on="task", how="left", validate="one_to_one")
    if out["family"].isna().any():
        raise ValueError("missing family for trajectory task")
    return out


def source_timing(decomposition: pd.DataFrame, tasks: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for row in decomposition.itertuples(index=False):
        source = str(tasks[row.task]["source"])
        years = re.findall(r"(?:19|20)\d{2}", source)
        year = int(years[-1]) if years else np.nan
        rows.append({
            "task": row.task,
            "source": source,
            "source_publication_year": year,
            "publication_group": "2025–2026" if pd.notna(year) and year >= 2025 else "through 2024",
            "qwen_minus_llama_f1": row.delta_f1,
        })
    return pd.DataFrame(rows)


def disagreement_audit(llama: pd.DataFrame, qwen: pd.DataFrame, tasks: dict[str, dict], deltas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ranked = deltas.sort_values("delta_f1")
    selected = list(ranked.head(3)["task"]) + list(ranked.tail(3)["task"])
    merged = llama.merge(qwen, on=["task", "item_id"], suffixes=("_A", "_B"), validate="one_to_one")
    rng = np.random.default_rng(SEED)
    rows = []
    for name in selected:
        group = merged.loc[merged["task"] == name].copy()
        task = tasks[name]
        a = _row_correct(group.rename(columns={c: c.removesuffix("_A") for c in group if c.endswith("_A")}), task, "pred")
        b = _row_correct(group.rename(columns={c: c.removesuffix("_B") for c in group if c.endswith("_B")}), task, "pred")
        group = group.loc[a != b]
        take = min(20, len(group))
        group = group.iloc[np.sort(rng.choice(len(group), take, replace=False))]
        item_text = {str(item["item_id"]): item["user_content"] for item in task["loader"]()}
        keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
        for _, row in group.iterrows():
            rows.append({
                "task": name, "item_id": row["item_id"], "text": item_text[str(row["item_id"])],
                "gold": json.dumps({key: row[f"gt_{key}_A"] for key in keys}, ensure_ascii=False),
                "prediction_A": json.dumps({key: row[f"pred_{key}_A"] for key in keys}, ensure_ascii=False),
                "prediction_B": json.dumps({key: row[f"pred_{key}_B"] for key in keys}, ensure_ascii=False),
                "implicit_or_pragmatic": "", "codebook_ambiguous": "", "explicit_decisive_cue": "", "coder_note": "",
            })
    return pd.DataFrame(rows), {"prediction_A": "Llama 3.1", "prediction_B": "Qwen3.6", "seed": SEED, "tasks": selected}


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {task["name"]: task for task in load_task_definitions()}
    llama, qwen = _read_predictions(LLAMA), _read_predictions(QWEN)
    deltas = pd.read_csv(DELTA_FILE)
    decomposition = confusion_decomposition(llama, qwen, tasks)
    enriched = decomposition.merge(deltas[["task", "set", "family"]], on="task", validate="one_to_one")
    enriched.to_csv(output_dir / "confusion_decomposition.csv", index=False)
    prompt_sensitivity(qwen, _read_predictions(SINGLE_USER), tasks).merge(
        deltas[["task", "family"]], on="task", validate="one_to_one"
    ).to_csv(output_dir / "prompt_sensitivity.csv", index=False)
    task_trajectories(deltas).to_csv(output_dir / "task_trajectories.csv", index=False)
    source_timing(decomposition, tasks).to_csv(output_dir / "source_timing.csv", index=False)
    audit, key = disagreement_audit(llama, qwen, tasks, deltas)
    audit.to_csv(output_dir / "blinded_disagreement_audit.csv", index=False)
    (output_dir / "blinding_key.json").write_text(json.dumps(key, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mechanism_tests")
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
