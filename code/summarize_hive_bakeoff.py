#!/usr/bin/env python3
"""Audit and summarize the revision-pinned Hive model bake-off.

The script requires exact paired coverage for every configured model before it
writes results. Model averages give every political-science task equal weight;
they do not pool items across tasks. The report keeps structural failures
separate from wrong but parseable predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_summary import _metrics_for_group
from hive_vllm_benchmark import (
    BakeoffError,
    assert_benchmark_commit,
    config_sha256,
    file_sha256,
    load_config,
    load_task_items,
    require_sidecar_output_dir,
    selected_tasks,
    task_checkpoint_complete,
    task_fingerprint,
    write_csv_atomic,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_ITERS = 10000


def _prediction_columns(task: dict[str, Any]) -> list[str]:
    if task["label_kind"] == "multi_binary":
        return [f"pred_{label}" for label in task["labels"]]
    return [f"pred_{task['label_key']}"]


def _gold_columns(task: dict[str, Any]) -> list[str]:
    if task["label_kind"] == "multi_binary":
        return [f"gt_{label}" for label in task["labels"]]
    return [f"gt_{task['label_key']}"]


def _usable_mask(group: pd.DataFrame, task: dict[str, Any]) -> pd.Series:
    pred_cols = _prediction_columns(task)
    missing = [column for column in pred_cols if column not in group.columns]
    if missing:
        raise BakeoffError(f"{task['name']}: missing prediction columns {missing}")
    parse_ok = group["parse_error"].isna() | (group["parse_error"].astype(str).str.strip() == "")
    predictions_present = ~group[pred_cols].isna().any(axis=1)
    return parse_ok & predictions_present


def _usable_rows(group: pd.DataFrame, task: dict[str, Any]) -> pd.DataFrame:
    return group[_usable_mask(group, task)]


def class_recall(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: Any,
    usable_mask: pd.Series | None = None,
) -> float:
    if len(y_true) != len(y_pred):
        raise BakeoffError("recall inputs have different lengths")
    if usable_mask is None:
        usable = [True] * len(y_true)
    else:
        if len(usable_mask) != len(y_true):
            raise BakeoffError("recall mask has a different length")
        usable = [bool(value) for value in usable_mask.tolist()]
    truth = y_true.tolist()
    predictions = y_pred.tolist()
    support = sum(value == label for value in truth)
    if support == 0:
        return np.nan
    true_positives = sum(
        true_value == label
        and is_usable
        and not pd.isna(predicted_value)
        and predicted_value == label
        for true_value, predicted_value, is_usable in zip(truth, predictions, usable)
    )
    return float(true_positives / support)


def rare_class_recall(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[Any],
    usable_mask: pd.Series | None = None,
) -> float:
    if len(y_true) == 0:
        return np.nan
    counts = Counter(y_true.tolist())
    present = [label for label in labels if counts.get(label, 0) > 0]
    if not present:
        return np.nan
    minimum_support = min(counts[label] for label in present)
    rare_labels = [label for label in present if counts[label] == minimum_support]
    recalls = [class_recall(y_true, y_pred, label, usable_mask) for label in rare_labels]
    return float(np.nanmean(recalls))


def _categorical_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value) if not pd.isna(value) else None)


def task_rare_class_recall(group: pd.DataFrame, task: dict[str, Any]) -> float:
    usable_mask = _usable_mask(group, task)
    if task["label_kind"] == "multi_binary":
        recalls = []
        for label in task["labels"]:
            recalls.append(
                rare_class_recall(
                    pd.to_numeric(group[f"gt_{label}"], errors="raise").astype(int),
                    pd.to_numeric(group[f"pred_{label}"], errors="coerce"),
                    [0, 1],
                    usable_mask,
                )
            )
        return float(np.nanmean(recalls)) if recalls else np.nan
    key = task["label_key"]
    if task["label_kind"] == "binary":
        return rare_class_recall(
            pd.to_numeric(group[f"gt_{key}"], errors="raise").astype(int),
            pd.to_numeric(group[f"pred_{key}"], errors="coerce"),
            [0, 1],
            usable_mask,
        )
    return rare_class_recall(
        _categorical_series(group[f"gt_{key}"]),
        _categorical_series(group[f"pred_{key}"]),
        task["labels"],
        usable_mask,
    )


def per_class_recall_rows(
    frames: dict[str, pd.DataFrame], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for model_key, frame in frames.items():
        for task in tasks:
            group = frame[frame["task"] == task["name"]].copy()
            usable_mask = _usable_mask(group, task)
            if task["label_kind"] == "multi_binary":
                fields = [(label, [0, 1]) for label in task["labels"]]
            else:
                values = [0, 1] if task["label_kind"] == "binary" else task["labels"]
                fields = [(task["label_key"], values)]
            for field, values in fields:
                if task["label_kind"] in {"binary", "multi_binary"}:
                    truth = pd.to_numeric(group[f"gt_{field}"], errors="raise").astype(int)
                    predictions = pd.to_numeric(group[f"pred_{field}"], errors="coerce")
                else:
                    truth = _categorical_series(group[f"gt_{field}"])
                    predictions = _categorical_series(group[f"pred_{field}"])
                counts = Counter(truth.tolist())
                present = [value for value in values if counts.get(value, 0) > 0]
                minimum_support = min((counts[value] for value in present), default=None)
                for value in values:
                    rows.append(
                        {
                            "task": task["name"],
                            "model": model_key,
                            "label_field": field,
                            "label_value": value,
                            "support": int(counts.get(value, 0)),
                            "recall": class_recall(truth, predictions, value, usable_mask),
                            "is_observed_rarest_class": (
                                minimum_support is not None
                                and counts.get(value, 0) == minimum_support
                            ),
                        }
                    )
    return rows


def class_support_rows(task: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if task["label_kind"] == "multi_binary":
        for label in task["labels"]:
            counts = frame[f"gt_{label}"].astype(int).value_counts()
            for value in [0, 1]:
                rows.append(
                    {
                        "task": task["name"],
                        "label_field": label,
                        "label_value": value,
                        "support": int(counts.get(value, 0)),
                    }
                )
        return rows
    key = task["label_key"]
    values = [0, 1] if task["label_kind"] == "binary" else task["labels"]
    counts = frame[f"gt_{key}"].value_counts()
    for value in values:
        rows.append(
            {
                "task": task["name"],
                "label_field": key,
                "label_value": value,
                "support": int(counts.get(value, 0)),
            }
        )
    return rows


def _load_model_predictions(run_root: Path, model_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions_path = run_root / model_key / "predictions.csv"
    metadata_path = run_root / model_key / "run_metadata.json"
    if not predictions_path.exists():
        raise BakeoffError(f"missing predictions: {predictions_path}")
    if not metadata_path.exists():
        raise BakeoffError(f"missing run metadata: {metadata_path}")
    frame = pd.read_csv(predictions_path, low_memory=False)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "completed":
        raise BakeoffError(f"{model_key}: run metadata status is {metadata.get('status')!r}")
    return frame, metadata


def normalized_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["item_id"] = normalized["item_id"].astype(str)
    return (
        normalized.sort_values(["task", "model", "item_id"])
        .reset_index(drop=True)
        .sort_index(axis=1)
    )


def validate_task_generation_seconds(
    checkpoint_frame: pd.DataFrame,
    task_result: dict[str, Any],
    model_key: str,
    task_name: str,
) -> float:
    """Verify that promotion-gate timing metadata matches the prediction rows."""
    latencies = pd.to_numeric(checkpoint_frame["latency_s"], errors="coerce")
    if latencies.isna().any() or not np.isfinite(latencies).all() or (latencies < 0).any():
        raise BakeoffError(f"{model_key}/{task_name}: invalid checkpoint latency values")
    try:
        reported = float(task_result["generation_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BakeoffError(
            f"{model_key}/{task_name}: missing or invalid generation_seconds metadata"
        ) from exc
    observed = float(latencies.sum())
    if not np.isfinite(reported) or reported < 0:
        raise BakeoffError(f"{model_key}/{task_name}: invalid generation_seconds metadata")
    if not np.isclose(reported, observed, rtol=1e-9, atol=1e-6):
        raise BakeoffError(
            f"{model_key}/{task_name}: generation_seconds={reported} does not match "
            f"checkpoint latency total {observed}"
        )
    return reported


def _gpu_signature(run_metadata: dict[str, Any], model_key: str) -> tuple[tuple[str, int], ...]:
    gpus = run_metadata.get("gpu_after_load", {}).get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise BakeoffError(f"{model_key}: missing GPU identity metadata")
    try:
        return tuple(
            sorted((str(gpu["name"]), int(gpu["memory_total_mib"])) for gpu in gpus)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BakeoffError(f"{model_key}: invalid GPU identity metadata") from exc


def validate_comparable_runtime(metadata: dict[str, dict[str, Any]]) -> None:
    """Require identical software locks and GPU types for operational contrasts."""
    if not metadata:
        raise BakeoffError("no model runtime metadata supplied")
    baseline_key = next(iter(metadata))
    baseline = metadata[baseline_key]
    baseline_versions = baseline.get("runtime_versions")
    if not isinstance(baseline_versions, dict) or not baseline_versions:
        raise BakeoffError(f"{baseline_key}: missing resolved runtime versions")
    baseline_gpu = _gpu_signature(baseline, baseline_key)
    for model_key, run_metadata in metadata.items():
        if run_metadata.get("runtime_versions") != baseline_versions:
            raise BakeoffError(
                f"{model_key}: resolved runtime differs from {baseline_key}; "
                "throughput is not comparable"
            )
        if _gpu_signature(run_metadata, model_key) != baseline_gpu:
            raise BakeoffError(
                f"{model_key}: GPU type differs from {baseline_key}; throughput is not comparable"
            )


def validate_model_coverage(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    model_key: str,
    model: dict[str, Any],
    tasks: list[dict[str, Any]],
    model_dir: Path,
    frozen_config_hash: str,
    generation: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    required = {
        "task",
        "model",
        "model_id",
        "model_revision",
        "item_id",
        "parse_error",
        "latency_s",
    }
    missing = required - set(frame.columns)
    if missing:
        raise BakeoffError(f"{model_key}: predictions missing columns {sorted(missing)}")
    if frame.duplicated(["task", "model", "item_id"]).any():
        raise BakeoffError(f"{model_key}: duplicate task/model/item_id keys")
    if set(frame["model"].astype(str)) != {model_key}:
        raise BakeoffError(f"{model_key}: model column does not match the configured key")
    if set(frame["model_id"].astype(str)) != {str(model["model_id"])}:
        raise BakeoffError(f"{model_key}: model_id does not match the frozen config")
    if set(frame["model_revision"].astype(str)) != {str(model["revision"])}:
        raise BakeoffError(f"{model_key}: model_revision does not match the frozen config")
    expected_metadata = {
        "model_key": model_key,
        "model_id": model["model_id"],
        "revision": model["revision"],
        "config_sha256": frozen_config_hash,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise BakeoffError(
                f"{model_key}: metadata {field}={metadata.get(field)!r}, expected {expected!r}"
            )
    if metadata.get("generation") != generation or metadata.get("runtime") != runtime:
        raise BakeoffError(f"{model_key}: metadata runtime or generation settings drifted")
    predictions_path = model_dir / "predictions.csv"
    if metadata.get("predictions_sha256") != file_sha256(predictions_path):
        raise BakeoffError(f"{model_key}: predictions.csv does not match its recorded SHA-256")
    if int(metadata.get("rows_written", -1)) != len(frame):
        raise BakeoffError(f"{model_key}: metadata row count does not match predictions.csv")
    observed_tasks = set(frame["task"].astype(str))
    expected_tasks = {task["name"] for task in tasks}
    if observed_tasks != expected_tasks:
        raise BakeoffError(
            f"{model_key}: task coverage mismatch; "
            f"missing={sorted(expected_tasks - observed_tasks)}, "
            f"extra={sorted(observed_tasks - expected_tasks)}"
        )
    task_results = metadata.get("task_results", {})
    if set(task_results) != expected_tasks:
        raise BakeoffError(f"{model_key}: task manifests do not cover the frozen task set")
    checkpoint_frames = []
    for task in tasks:
        items = load_task_items(task)
        expected_ids = {str(item["item_id"]) for item in items}
        observed_ids = set(
            frame.loc[frame["task"] == task["name"], "item_id"].astype(str)
        )
        if observed_ids != expected_ids:
            raise BakeoffError(
                f"{model_key}/{task['name']}: item coverage mismatch; "
                f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}"
            )
        fingerprint = task_fingerprint(
            task,
            items,
            model_alias=model_key,
            model=model,
            config_hash=frozen_config_hash,
            generation=generation,
            runtime=runtime,
            runtime_versions=metadata["runtime_versions"],
        )
        task_path = model_dir / "tasks" / f"{task['name']}.csv"
        if not task_checkpoint_complete(
            task_path,
            task["name"],
            model_key,
            sorted(expected_ids),
            expected_fingerprint=fingerprint,
        ):
            raise BakeoffError(f"{model_key}/{task['name']}: stale or incomplete checkpoint")
        result = task_results[task["name"]]
        if result.get("task_fingerprint") != fingerprint:
            raise BakeoffError(f"{model_key}/{task['name']}: metadata fingerprint mismatch")
        if result.get("checkpoint_sha256") != file_sha256(task_path):
            raise BakeoffError(f"{model_key}/{task['name']}: metadata checkpoint hash mismatch")
        checkpoint_frame = pd.read_csv(task_path, low_memory=False)
        validate_task_generation_seconds(
            checkpoint_frame,
            result,
            model_key,
            task["name"],
        )
        checkpoint_frames.append(checkpoint_frame)

    checkpoint_frame = pd.concat(checkpoint_frames, ignore_index=True, sort=False)
    expected_frame = normalized_prediction_frame(checkpoint_frame)
    observed_frame = normalized_prediction_frame(frame)
    try:
        pd.testing.assert_frame_equal(expected_frame, observed_frame, check_dtype=False)
    except AssertionError as exc:
        raise BakeoffError(
            f"{model_key}: predictions.csv differs from the verified task checkpoints"
        ) from exc


def validate_paired_gold(
    frames: dict[str, pd.DataFrame], tasks: list[dict[str, Any]]
) -> None:
    baseline_key = next(iter(frames))
    baseline = frames[baseline_key]
    for task in tasks:
        key_cols = ["task", "item_id"]
        gold_cols = _gold_columns(task)
        base = (
            baseline.loc[baseline["task"] == task["name"], key_cols + gold_cols]
            .sort_values(key_cols)
            .reset_index(drop=True)
        )
        for model_key, frame in frames.items():
            candidate = (
                frame.loc[frame["task"] == task["name"], key_cols + gold_cols]
                .sort_values(key_cols)
                .reset_index(drop=True)
            )
            try:
                pd.testing.assert_frame_equal(base, candidate, check_dtype=False)
            except AssertionError as exc:
                raise BakeoffError(
                    f"gold labels or paired IDs differ for {task['name']} between "
                    f"{baseline_key} and {model_key}"
                ) from exc


def task_metrics(
    frames: dict[str, pd.DataFrame], tasks: list[dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for model_key, frame in frames.items():
        for task in tasks:
            group = frame[frame["task"] == task["name"]].copy()
            # support_only=False pins the LEGACY metric on purpose: this panel's
            # stored scores were computed by averaging macro F1 over every manifest
            # label, including labels with no gold support. code/scoring.py is the
            # corrected implementation and is now the default in build_summary, but
            # flipping it here would silently invalidate the frozen frontier and
            # refresh panels. Migrate deliberately, rebuilding the panels, rather
            # than by inheriting a changed default.
            base_metrics = _metrics_for_group(task, group, support_only=False)
            usable = _usable_rows(group, task)
            rows.append(
                {
                    "task": task["name"],
                    "model": model_key,
                    **base_metrics,
                    "unusable_output_rate": 1 - len(usable) / len(group),
                    "rare_class_recall": task_rare_class_recall(group, task),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "task"]).reset_index(drop=True)


def model_metrics(
    task_metric_frame: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for model_key, group in task_metric_frame.groupby("model"):
        frame = frames[model_key]
        task_results = metadata[model_key].get("task_results", {})
        generation_seconds = sum(
            float(result.get("generation_seconds", 0.0)) for result in task_results.values()
        )
        rows.append(
            {
                "model": model_key,
                "tasks": int(group["task"].nunique()),
                "items": int(len(frame)),
                "mean_task_f1": float(group["headline_f1"].mean()),
                "median_task_f1": float(group["headline_f1"].median()),
                "mean_task_rare_class_recall": float(group["rare_class_recall"].mean()),
                "unusable_output_rate": float(group["unusable_output_rate"].mean()),
                "parse_error_rate": float(group["parse_err_rate"].mean()),
                "generation_seconds": generation_seconds,
                "items_per_second": len(frame) / generation_seconds if generation_seconds else np.nan,
                "startup_seconds": metadata[model_key].get("startup_seconds"),
                "observed_peak_gpu_memory_used_mib": metadata[model_key].get(
                    "observed_peak_gpu_memory_used_mib"
                ),
                "observed_peak_gpu_memory_delta_mib": metadata[model_key].get(
                    "observed_peak_gpu_memory_delta_mib"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_task_f1", ascending=False).reset_index(drop=True)


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_ITERS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrasts_vs_baseline(
    task_metric_frame: pd.DataFrame,
    model_metric_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gate = config["promotion_gate"]
    baseline_key = gate["baseline_model_key"]
    baseline_tasks = task_metric_frame[task_metric_frame["model"] == baseline_key].set_index("task")
    baseline_model = model_metric_frame.set_index("model").loc[baseline_key]
    rows = []
    for model_key in config["models"]:
        if model_key == baseline_key:
            continue
        candidate_tasks = task_metric_frame[task_metric_frame["model"] == model_key].set_index("task")
        common = sorted(set(baseline_tasks.index) & set(candidate_tasks.index))
        deltas = (
            candidate_tasks.loc[common, "headline_f1"].to_numpy()
            - baseline_tasks.loc[common, "headline_f1"].to_numpy()
        )
        low, high = bootstrap_mean_interval(deltas)
        candidate_model = model_metric_frame.set_index("model").loc[model_key]
        parse_increase = (
            candidate_model["parse_error_rate"] - baseline_model["parse_error_rate"]
        )
        throughput_ratio = (
            candidate_model["items_per_second"] / baseline_model["items_per_second"]
        )
        accuracy_gate = (
            float(np.mean(deltas)) >= float(gate["minimum_mean_f1_gain"])
            and parse_increase <= float(gate["maximum_parse_error_increase"])
        )
        efficiency_gate = (
            float(np.mean(deltas)) >= -float(gate["equivalence_tolerance"])
            and throughput_ratio >= float(gate["material_throughput_ratio"])
            and parse_increase <= float(gate["maximum_parse_error_increase"])
        )
        rows.append(
            {
                "model": model_key,
                "baseline": baseline_key,
                "tasks": len(common),
                "mean_f1_delta": float(np.mean(deltas)),
                "mean_f1_delta_ci_low": low,
                "mean_f1_delta_ci_high": high,
                "positive_task_count": int(np.sum(deltas > 0)),
                "equal_task_count": int(np.sum(deltas == 0)),
                "parse_error_rate_increase": float(parse_increase),
                "throughput_ratio": float(throughput_ratio),
                "accuracy_promotion_gate": bool(accuracy_gate),
                "efficiency_promotion_gate": bool(efficiency_gate),
                "promotion_gate": bool(accuracy_gate or efficiency_gate),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_f1_delta", ascending=False).reset_index(drop=True)


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prediction_json(row: pd.Series, task: dict[str, Any]) -> str:
    parse_ok = pd.isna(row["parse_error"]) or str(row["parse_error"]).strip() == ""
    pred_cols = _prediction_columns(task)
    predictions_present = all(not pd.isna(row[column]) for column in pred_cols)
    if not parse_ok or not predictions_present:
        return "__UNUSABLE__"
    if task["label_kind"] == "multi_binary":
        payload = {label: _json_scalar(row[f"pred_{label}"]) for label in task["labels"]}
    else:
        key = task["label_key"]
        payload = {key: _json_scalar(row[f"pred_{key}"])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def random_blinding_aliases(model_keys: list[str]) -> dict[str, str]:
    shuffled = list(model_keys)
    secrets.SystemRandom().shuffle(shuffled)
    return {model_key: f"model_{index + 1}" for index, model_key in enumerate(shuffled)}


def ordered_blinded_predictions(
    predictions: dict[str, str], aliases: dict[str, str]
) -> dict[str, str]:
    return {
        alias: predictions[model_key]
        for alias, model_key in sorted((alias, model_key) for model_key, alias in aliases.items())
    }


def blinded_disagreement_rows(
    frames: dict[str, pd.DataFrame], tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    model_keys = list(frames)
    aliases = random_blinding_aliases(model_keys)
    blinding_key = {alias: model_key for model_key, alias in aliases.items()}
    rows = []
    for task in tasks:
        items = load_task_items(task)
        item_lookup = {str(item["item_id"]): item for item in items}
        model_rows = {}
        for model_key, frame in frames.items():
            group = frame[frame["task"] == task["name"]].copy()
            group["item_id"] = group["item_id"].astype(str)
            model_rows[model_key] = group.set_index("item_id", verify_integrity=True)
        for item_id, item in item_lookup.items():
            predictions = {
                model_key: _prediction_json(model_rows[model_key].loc[item_id], task)
                for model_key in model_keys
            }
            if len(set(predictions.values())) <= 1:
                continue
            row = {
                "task": task["name"],
                "item_id": item_id,
                "input_text": str(item["user_content"]),
                "gold": json.dumps(item["gt"], ensure_ascii=False, sort_keys=True),
                "distinct_prediction_count": len(set(predictions.values())),
            }
            row.update(ordered_blinded_predictions(predictions, aliases))
            rows.append(row)
    return rows, blinding_key


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def display(value: Any) -> str:
        if pd.isna(value):
            rendered = ""
        elif isinstance(value, (bool, np.bool_)):
            rendered = "true" if bool(value) else "false"
        elif isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.4f}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", "<br>")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    return "\n".join(lines)


def render_report(
    path: Path,
    model_frame: pd.DataFrame,
    contrast_frame: pd.DataFrame,
    support_frame: pd.DataFrame,
    config: dict[str, Any],
    config_path: Path | None = None,
) -> None:
    passing = contrast_frame[contrast_frame["promotion_gate"]]
    if len(passing):
        best = passing.sort_values(["mean_f1_delta", "throughput_ratio"], ascending=False).iloc[0]
        verdict = (
            f"{best['model']} passes the frozen automated promotion gate against "
            f"{best['baseline']}. Review the blinded disagreement file before changing the default."
        )
    else:
        verdict = "No candidate passes the frozen promotion gate; retain the baseline."
    low_support = int((support_frame["support"] < 30).sum())
    zero_support = int((support_frame["support"] == 0).sum())
    if config_path is None:
        config_label = "the supplied frozen configuration"
    else:
        try:
            config_label = f"`{config_path.resolve().relative_to(REPO)}`"
        except ValueError:
            config_label = f"`{config_path.resolve()}`"
    lines = [
        "# Hive model bake-off",
        "",
        verdict,
        "",
        "## What is compared",
        "",
        (
            f"The benchmark compares {len(config['models'])} revision-pinned open-weight "
            "models on the same "
            "16,425 labeled texts from 34 political-science classification tasks. Every "
            "model receives the same task prompt, input text, JSON schema, temperature-zero "
            "decoding rule, and output-token limit."
        ),
        "",
        "## Model-level results",
        "",
        markdown_table(model_frame),
        "",
        "## Paired contrasts against the baseline",
        "",
        markdown_table(contrast_frame),
        "",
        "The F1 contrast gives every task equal weight. Its 95% interval resamples the 34 tasks.",
        "",
        "## Assumptions and limitations",
        "",
        (
            "1. The benchmark's fixed release sample is deterministic but not stratified by "
            "class. The class-support table contains "
            f"{low_support} task-label cells with fewer than 30 examples, including "
            f"{zero_support} configured classes absent from the sample. Categorical headline "
            "F1 follows the benchmark definition and retains the full configured label set."
        ),
        "2. JSON-schema compliance measures structural validity, not classification accuracy.",
        (
            "3. Throughput is measured within each task-sized offline vLLM batch. Startup time "
            "and observed GPU-memory use are reported separately."
        ),
        (
            "4. The promotion decision follows the thresholds frozen in "
            f"{config_label}."
        ),
        "5. A passing automated gate remains provisional until the blinded disagreements are audited.",
        "",
        "## Output files",
        "",
        "- `task_metrics.csv`: task-model accuracy and reliability metrics.",
        "- `model_summary.csv`: equal-task model averages and operational measurements.",
        "- `contrasts_vs_baseline.csv`: paired promotion-gate comparisons.",
        "- `class_support.csv`: observed gold-label support by task.",
        "- `per_class_recall.csv`: recall for every observed class; unusable outputs count as misses.",
        "- `disagreements_blinded.csv`: item-level model disagreements under blinded aliases.",
        "- `blinding_key.json`: alias-to-model mapping, kept separate from the review file.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "experiments" / "hive_model_bakeoff_20260804.yaml",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO / "output" / "sidecar" / "hive_model_bakeoff_20260804",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    assert_benchmark_commit(config)
    frozen_config_hash = config_sha256(config_path)
    run_root = require_sidecar_output_dir(args.run_root.resolve())
    output_dir = require_sidecar_output_dir(
        (args.output_dir or (run_root / "comparison")).resolve()
    )
    tasks = selected_tasks(config)

    frames = {}
    metadata = {}
    for model_key, model in config["models"].items():
        frame, model_metadata = _load_model_predictions(run_root, model_key)
        validate_model_coverage(
            frame=frame,
            metadata=model_metadata,
            model_key=model_key,
            model=model,
            tasks=tasks,
            model_dir=run_root / model_key,
            frozen_config_hash=frozen_config_hash,
            generation=config["generation"],
            runtime=config["runtime"],
        )
        frames[model_key] = frame
        metadata[model_key] = model_metadata
    validate_comparable_runtime(metadata)
    validate_paired_gold(frames, tasks)

    support_rows = []
    baseline_frame = frames[config["promotion_gate"]["baseline_model_key"]]
    for task in tasks:
        support_rows.extend(
            class_support_rows(task, baseline_frame[baseline_frame["task"] == task["name"]])
        )
    support_frame = pd.DataFrame(support_rows)
    task_frame = task_metrics(frames, tasks)
    per_class_frame = pd.DataFrame(per_class_recall_rows(frames, tasks))
    model_frame = model_metrics(task_frame, frames, metadata)
    contrast_frame = contrasts_vs_baseline(task_frame, model_frame, config)
    disagreement_rows, blinding_key = blinded_disagreement_rows(frames, tasks)
    disagreement_frame = pd.DataFrame(disagreement_rows)

    write_csv_atomic(output_dir / "class_support.csv", support_frame)
    write_csv_atomic(output_dir / "task_metrics.csv", task_frame)
    write_csv_atomic(output_dir / "per_class_recall.csv", per_class_frame)
    write_csv_atomic(output_dir / "model_summary.csv", model_frame)
    write_csv_atomic(output_dir / "contrasts_vs_baseline.csv", contrast_frame)
    write_csv_atomic(output_dir / "disagreements_blinded.csv", disagreement_frame)
    write_json_atomic(output_dir / "blinding_key.json", blinding_key)
    render_report(
        output_dir / "benchmark_report.md",
        model_frame=model_frame,
        contrast_frame=contrast_frame,
        support_frame=support_frame,
        config=config,
        config_path=config_path,
    )
    print(f"wrote audited comparison to {output_dir}")
    print(model_frame.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
