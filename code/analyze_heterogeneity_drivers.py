#!/usr/bin/env python3
"""Test locked exploratory hypotheses about heterogeneous benchmark progress."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from build_broad_robustness import TASK_TO_SOURCE  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

DEFAULT_SPEC = REPO / "experiments/heterogeneity_drivers_20260824.yaml"
DEFAULT_OUTPUT = REPO / "output/sidecar/frontier_2026/heterogeneity_drivers"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path: Path = DEFAULT_SPEC) -> dict:
    spec = yaml.safe_load(path.read_text())
    if spec.get("status") != "exploratory_locked_after_outcome_inspection":
        raise ValueError("heterogeneity analysis must retain its exploratory status")
    for entry in spec["inputs"].values():
        target = REPO / entry["path"]
        if not target.exists() or _sha256(target) != entry["sha256"]:
            raise ValueError(f"frozen input identity failure: {entry['path']}")
    return spec


def _rank_result(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    result = spearmanr(np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float))
    return float(result.statistic), float(result.pvalue)


def _permutation_rank(
    moderator: np.ndarray,
    outcome: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    observed, _ = _rank_result(moderator, outcome)
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for index in range(draws):
        null[index], _ = _rank_result(moderator, rng.permutation(outcome))
    one_sided = (1 + int((null >= observed).sum())) / (draws + 1)
    return observed, float(one_sided)


def _leave_one_out_rank(frame: pd.DataFrame, moderator: str, outcome: str) -> tuple[float, float]:
    values = []
    for index in frame.index:
        rho, _ = _rank_result(frame.drop(index)[moderator], frame.drop(index)[outcome])
        values.append(rho)
    return float(min(values)), float(max(values))


def _permutation_mean_gap(
    values: np.ndarray,
    binary: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Test the locked prediction that binary-task change is lower."""
    observed = float(values[binary].mean() - values[~binary].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for index in range(draws):
        shuffled = rng.permutation(binary)
        null[index] = values[shuffled].mean() - values[~shuffled].mean()
    p_value = (1 + int((null <= observed).sum())) / (draws + 1)
    return observed, float(p_value)


def task_metadata(spec: dict) -> pd.DataFrame:
    path = REPO / spec["inputs"]["task_metadata"]["path"]
    data = pd.read_csv(path)
    required = {
        "task", "label_kind", "label_count", "effective_label_count",
        "prompt_words", "item_words_median",
    }
    if not required.issubset(data.columns) or data["task"].duplicated().any():
        raise ValueError("task metadata is incomplete or duplicated")
    data = data.copy()
    data["log_effective_label_count"] = np.log(data["effective_label_count"])
    data["binary_output_format"] = data["label_kind"].eq("binary")
    return data


def endpoint_tests(spec: dict, metadata: pd.DataFrame) -> pd.DataFrame:
    deltas = pd.read_csv(REPO / spec["inputs"]["endpoint_deltas"]["path"])
    if len(deltas) != 34 or deltas["task"].nunique() != 34:
        raise ValueError("endpoint analysis requires exactly 34 tasks")
    data = deltas.merge(
        metadata[["task", "log_effective_label_count", "effective_label_count"]],
        on="task", validate="one_to_one",
    )
    data["delta_f1_points"] = 100 * data["delta_f1"]
    rows = []
    scopes = [("all_34", data)] + [
        (name.lower().replace(" ", "_"), group) for name, group in data.groupby("set", sort=False)
    ]
    for scope, group in scopes:
        rho, p_value = _rank_result(group["log_effective_label_count"], group["delta_f1_points"])
        rows.append({
            "scope": scope, "unit": "task", "units": len(group),
            "spearman_rho": rho, "p_two_sided": p_value,
            "p_positive_permutation": np.nan,
            "loo_min_rho": np.nan, "loo_max_rho": np.nan,
        })

    broad = data.loc[data["set"].eq("Broad18")].copy()
    broad["source_cluster_id"] = broad["task"].map(TASK_TO_SOURCE)
    if broad["source_cluster_id"].isna().any() or broad["source_cluster_id"].nunique() != 13:
        raise ValueError("Broad18 source mapping drifted")
    sources = broad.groupby("source_cluster_id", as_index=False).agg(
        log_effective_label_count=("log_effective_label_count", "mean"),
        delta_f1_points=("delta_f1_points", "mean"),
    )
    rho, p_perm = _permutation_rank(
        sources.log_effective_label_count.to_numpy(), sources.delta_f1_points.to_numpy(),
        draws=int(spec["permutation_draws"]), seed=int(spec["seed"]),
    )
    loo_low, loo_high = _leave_one_out_rank(
        sources, "log_effective_label_count", "delta_f1_points"
    )
    rows.append({
        "scope": "broad18_source_equal", "unit": "source_family", "units": len(sources),
        "spearman_rho": rho, "p_two_sided": np.nan,
        "p_positive_permutation": p_perm,
        "loo_min_rho": loo_low, "loo_max_rho": loo_high,
    })
    return pd.DataFrame(rows)


def _task_slopes(scores: pd.DataFrame, checkpoint_ids: list[str] | None) -> pd.DataFrame:
    work = scores if checkpoint_ids is None else scores.loc[scores["checkpoint_id"].isin(checkpoint_ids)]
    dates = pd.to_datetime(work["plot_date"])
    work = work.assign(_years=(dates - dates.min()).dt.days / 365.25)
    rows = []
    for task, group in work.groupby("task", sort=True):
        if group["checkpoint_id"].nunique() < 3:
            raise ValueError(f"too few checkpoints for task slope: {task}")
        slope = np.polyfit(group["_years"], group["headline_f1"], 1)[0] * 100
        rows.append({"task": task, "f1_points_per_year": float(slope)})
    return pd.DataFrame(rows)


def longitudinal_tests(spec: dict, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(REPO / spec["inputs"]["longitudinal_scores"]["path"])
    scores = scores.loc[scores["series"].eq("open_hive") & scores["headline_f1"].notna()].copy()
    if scores["task"].nunique() != 18 or scores["checkpoint_id"].nunique() != 21:
        raise ValueError("longitudinal analysis requires the 18-task/21-checkpoint panel")
    scopes: list[tuple[str, list[str] | None]] = [("all_21", None)]
    scopes.extend((name, list(ids)) for name, ids in spec["lineages"].items())
    task_rows, result_rows = [], []
    for scope_index, (scope, checkpoint_ids) in enumerate(scopes):
        slopes = _task_slopes(scores, checkpoint_ids).merge(
            metadata[["task", "log_effective_label_count", "effective_label_count"]],
            on="task", validate="one_to_one",
        )
        slopes["source_cluster_id"] = slopes["task"].map(TASK_TO_SOURCE)
        slopes["scope"] = scope
        task_rows.append(slopes)
        task_rho, task_p = _rank_result(
            slopes.log_effective_label_count, slopes.f1_points_per_year
        )
        result_rows.append({
            "scope": scope, "unit": "task", "units": len(slopes),
            "checkpoints": scores["checkpoint_id"].nunique() if checkpoint_ids is None else len(checkpoint_ids),
            "spearman_rho": task_rho, "p_two_sided": task_p,
            "p_positive_permutation": np.nan,
            "loo_min_rho": np.nan, "loo_max_rho": np.nan,
        })
        sources = slopes.groupby("source_cluster_id", as_index=False).agg(
            log_effective_label_count=("log_effective_label_count", "mean"),
            f1_points_per_year=("f1_points_per_year", "mean"),
        )
        rho, p_perm = _permutation_rank(
            sources.log_effective_label_count.to_numpy(), sources.f1_points_per_year.to_numpy(),
            draws=int(spec["permutation_draws"]), seed=int(spec["seed"]) + scope_index,
        )
        loo_low, loo_high = _leave_one_out_rank(
            sources, "log_effective_label_count", "f1_points_per_year"
        )
        result_rows.append({
            "scope": scope, "unit": "source_family", "units": len(sources),
            "checkpoints": scores["checkpoint_id"].nunique() if checkpoint_ids is None else len(checkpoint_ids),
            "spearman_rho": rho, "p_two_sided": np.nan,
            "p_positive_permutation": p_perm,
            "loo_min_rho": loo_low, "loo_max_rho": loo_high,
        })
    return pd.concat(task_rows, ignore_index=True), pd.DataFrame(result_rows)


def _as_values(series: pd.Series, numeric: bool) -> np.ndarray:
    if numeric:
        return pd.to_numeric(series, errors="coerce").fillna(-999999).to_numpy()
    return series.fillna("__MALFORMED__").astype(str).to_numpy()


def _f1(gold: np.ndarray, prediction: np.ndarray, label: object) -> float:
    gold_label = gold == label
    pred_label = prediction == label
    tp = int((gold_label & pred_label).sum())
    fp = int((~gold_label & pred_label).sum())
    fn = int((gold_label & ~pred_label).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _score_task(group: pd.DataFrame, task: dict, model: str, *, symmetric: bool) -> float:
    suffix = f"_{model}"
    if task["label_kind"] == "categorical":
        key = task["label_key"]
        gold = _as_values(group[f"gt_{key}_llama"], numeric=False)
        pred = _as_values(group[f"pred_{key}{suffix}"], numeric=False)
        return float(np.mean([_f1(gold, pred, str(label)) for label in task["labels"]]))
    keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
    values = []
    for key in keys:
        gold = _as_values(group[f"gt_{key}_llama"], numeric=True)
        pred = _as_values(group[f"pred_{key}{suffix}"], numeric=True)
        labels = [0, 1] if symmetric else [1]
        values.extend(_f1(gold, pred, label) for label in labels)
    return float(np.mean(values))


def calibration_tests(spec: dict, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = {task["name"]: task for task in load_task_definitions()}
    paths = {
        "llama": REPO / spec["inputs"]["llama_predictions"]["path"],
        "qwen": REPO / spec["inputs"]["qwen_predictions"]["path"],
    }
    frames = {
        name: pd.read_csv(path, dtype={"item_id": str}, low_memory=False)
        for name, path in paths.items()
    }
    for name, frame in frames.items():
        if frame.duplicated(["task", "item_id"]).any() or frame["task"].nunique() != 34:
            raise ValueError(f"{name} prediction panel is incomplete or duplicated")
    merged = frames["llama"].merge(
        frames["qwen"], on=["task", "item_id"], suffixes=("_llama", "_qwen"),
        validate="one_to_one",
    )
    set_map = pd.read_csv(REPO / spec["inputs"]["endpoint_deltas"]["path"]).set_index("task")["set"]
    rows = []
    for task_name, group in merged.groupby("task", sort=True):
        task = tasks[task_name]
        keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
        for key in keys:
            left = _as_values(group[f"gt_{key}_llama"], task["label_kind"] != "categorical")
            right = _as_values(group[f"gt_{key}_qwen"], task["label_kind"] != "categorical")
            if not np.array_equal(left, right):
                raise ValueError(f"gold mismatch for {task_name}/{key}")
        headline = {
            model: _score_task(group, task, model, symmetric=False) for model in ("llama", "qwen")
        }
        symmetric = {
            model: _score_task(group, task, model, symmetric=True) for model in ("llama", "qwen")
        }
        row = {
            "task": task_name, "set": set_map[task_name], "label_kind": task["label_kind"],
            "binary_output_format": task["label_kind"] == "binary",
            "headline_delta_f1_points": 100 * (headline["qwen"] - headline["llama"]),
            "symmetric_delta_f1_points": 100 * (symmetric["qwen"] - symmetric["llama"]),
        }
        if task["label_kind"] == "binary":
            key = task["label_key"]
            gold = _as_values(group[f"gt_{key}_llama"], numeric=True)
            for model in ("llama", "qwen"):
                pred = _as_values(group[f"pred_{key}_{model}"], numeric=True)
                positive = pred == 1
                gold_positive = gold == 1
                tp = int((positive & gold_positive).sum())
                fp = int((positive & ~gold_positive).sum())
                fn = int((~positive & gold_positive).sum())
                tn = int((~positive & ~gold_positive).sum())
                row[f"{model}_positive_rate"] = float(positive.mean())
                row[f"{model}_precision"] = tp / (tp + fp) if tp + fp else np.nan
                row[f"{model}_recall"] = tp / (tp + fn) if tp + fn else np.nan
                row[f"{model}_specificity"] = tn / (tn + fp) if tn + fp else np.nan
            row["gold_positive_rate"] = float((gold == 1).mean())
            for metric in ("positive_rate", "precision", "recall", "specificity"):
                row[f"delta_{metric}"] = row[f"qwen_{metric}"] - row[f"llama_{metric}"]
        rows.append(row)
    detail = pd.DataFrame(rows).merge(
        metadata[["task", "effective_label_count"]], on="task", validate="one_to_one"
    )
    binary = detail.loc[detail["binary_output_format"]]
    lower = int((binary["delta_positive_rate"] < 0).sum())
    nonzero = int((binary["delta_positive_rate"] != 0).sum())
    summary_rows = [{
        "scope": "all_34", "result": "binary_tasks_lower_positive_rate",
        "estimate": lower, "units": nonzero,
        "p_value": binomtest(lower, nonzero, .5, alternative="greater").pvalue,
    }, {
        "scope": "all_34", "result": "binary_mean_delta_precision",
        "estimate": binary.delta_precision.mean(), "units": len(binary), "p_value": np.nan,
    }, {
        "scope": "all_34", "result": "binary_mean_delta_recall",
        "estimate": binary.delta_recall.mean(), "units": len(binary), "p_value": np.nan,
    }]
    calibration_scopes = [("all_34", detail)] + [
        (name.lower().replace(" ", "_"), group)
        for name, group in detail.groupby("set", sort=False)
    ]
    for scope_index, (scope, group) in enumerate(calibration_scopes):
        is_binary = group["binary_output_format"].to_numpy(bool)
        binary_group = group.loc[is_binary]
        nonbinary_group = group.loc[~is_binary]
        headline_gap, headline_p = _permutation_mean_gap(
            group.headline_delta_f1_points.to_numpy(), is_binary,
            draws=int(spec["permutation_draws"]), seed=int(spec["seed"]) + 100 + scope_index,
        )
        symmetric_gap, symmetric_p = _permutation_mean_gap(
            group.symmetric_delta_f1_points.to_numpy(), is_binary,
            draws=int(spec["permutation_draws"]), seed=int(spec["seed"]) + 200 + scope_index,
        )
        summary_rows.extend([
            {"scope": scope, "result": "binary_headline_mean_delta",
             "estimate": binary_group.headline_delta_f1_points.mean(), "units": len(binary_group), "p_value": np.nan},
            {"scope": scope, "result": "nonbinary_headline_mean_delta",
             "estimate": nonbinary_group.headline_delta_f1_points.mean(), "units": len(nonbinary_group), "p_value": np.nan},
            {"scope": scope, "result": "binary_minus_nonbinary_headline",
             "estimate": headline_gap, "units": len(group), "p_value": headline_p},
            {"scope": scope, "result": "binary_symmetric_mean_delta",
             "estimate": binary_group.symmetric_delta_f1_points.mean(), "units": len(binary_group), "p_value": np.nan},
            {"scope": scope, "result": "nonbinary_symmetric_mean_delta",
             "estimate": nonbinary_group.symmetric_delta_f1_points.mean(), "units": len(nonbinary_group), "p_value": np.nan},
            {"scope": scope, "result": "binary_minus_nonbinary_symmetric",
             "estimate": symmetric_gap, "units": len(group), "p_value": symmetric_p},
            {"scope": scope, "result": "binary_gap_attenuation_from_rescoring",
             "estimate": symmetric_gap - headline_gap, "units": len(group), "p_value": np.nan},
        ])
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def build(spec_path: Path = DEFAULT_SPEC, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, pd.DataFrame]:
    spec = load_spec(spec_path)
    metadata = task_metadata(spec)
    endpoint = endpoint_tests(spec, metadata)
    task_slopes, longitudinal = longitudinal_tests(spec, metadata)
    calibration, calibration_summary = calibration_tests(spec, metadata)
    outputs = {
        "endpoint_moderator_tests": endpoint,
        "task_progress_slopes": task_slopes,
        "longitudinal_moderator_tests": longitudinal,
        "calibration_task_metrics": calibration,
        "calibration_summary": calibration_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.spec, args.output_dir)


if __name__ == "__main__":
    main()
