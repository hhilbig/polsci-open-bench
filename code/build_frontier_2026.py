#!/usr/bin/env python3
"""Audit checkpoint evidence and build the partial 2025--2026 frontier view.

Only complete, audited 34-task results can set a plotted frontier. Calibrated
18-task panel scores can trigger a full-suite promotion, but never enter a
frontier envelope or a historical lag calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from build_summary import _metrics_for_group

from hive_vllm_benchmark import task_fingerprint
from panel_manifest import PanelSelection, load_panel_manifest
from task_registry import load_task_definitions


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_PANEL = REPO / "experiments" / "frontier_panel_18.yaml"
DEFAULT_REGISTRY = REPO / "experiments" / "frontier_checkpoints_2026.yaml"
DEFAULT_OUTPUT_DIR = REPO / "output" / "sidecar" / "frontier_2026"
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_ITERATIONS = 10000
Z_95 = 1.959963984540054
PROMOTION_DISTANCE = 0.01

SERIES_ORDER = ("hive_98gb", "api")
SERIES_LABELS = {
    "hive_98gb": "Best tested open checkpoint on one 98 GB-class Hive GPU",
    "api": "Best tested available API checkpoint",
}
SERIES_COLORS = {
    "hive_98gb": "#0072B2",
    "api": "#D55E00",
}


class FrontierError(ValueError):
    """Raised when registry evidence is incomplete or internally inconsistent."""


@dataclass
class Evidence:
    row: dict[str, Any]
    task_scores: dict[str, float]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _iso_date(value: Any, context: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise FrontierError(f"{context} must be an ISO date") from exc


def load_checkpoint_registry(path: Path, panel: PanelSelection) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 1:
        raise FrontierError("checkpoint registry schema_version must be 1")
    if str(raw.get("benchmark_commit")) != panel.benchmark_commit:
        raise FrontierError("checkpoint registry and panel benchmark commits differ")
    panel_path = resolve_repo_path(str(raw.get("panel_manifest", ""))).resolve()
    if panel_path != panel.manifest_path:
        raise FrontierError("checkpoint registry points to a different panel manifest")

    availability_path = resolve_repo_path(
        str(raw.get("api_availability_manifest", ""))
    )
    availability_raw = yaml.safe_load(availability_path.read_text())
    if not isinstance(availability_raw, dict) or availability_raw.get("schema_version") != 1:
        raise FrontierError("API availability manifest schema_version must be 1")
    availability = availability_raw.get("checkpoints")
    if not isinstance(availability, dict):
        raise FrontierError("API availability manifest needs checkpoints")

    tiers = raw.get("hardware_tiers")
    checkpoints = raw.get("checkpoints")
    if not isinstance(tiers, dict) or not isinstance(checkpoints, list) or not checkpoints:
        raise FrontierError("checkpoint registry needs hardware_tiers and checkpoints")
    if set(tiers) != set(SERIES_ORDER):
        raise FrontierError(
            "checkpoint registry must contain exactly the hive_98gb and api series"
        )
    seen: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise FrontierError(f"checkpoints[{index}] must be a mapping")
        required = {
            "checkpoint_id",
            "display_name",
            "family",
            "provider",
            "model_id",
            "release_date",
            "release_source",
            "quantization",
            "hardware_tier",
            "hardware_qualification",
            "immutable",
            "historical_lag_eligible",
            "reasoning",
            "result_status",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise FrontierError(f"checkpoints[{index}] missing fields: {missing}")
        checkpoint_id = str(checkpoint["checkpoint_id"])
        if checkpoint_id in seen:
            raise FrontierError(f"duplicate checkpoint_id: {checkpoint_id}")
        seen.add(checkpoint_id)
        checkpoint["release_date"] = _iso_date(
            checkpoint["release_date"], f"{checkpoint_id}.release_date"
        )
        if checkpoint["hardware_tier"] not in tiers:
            raise FrontierError(f"{checkpoint_id}: unknown hardware tier")
        if checkpoint["family"] not in {"open_weight", "api"}:
            raise FrontierError(f"{checkpoint_id}: invalid family")
        expected_tier = "hive_98gb" if checkpoint["family"] == "open_weight" else "api"
        if checkpoint["hardware_tier"] != expected_tier:
            raise FrontierError(
                f"{checkpoint_id}: {checkpoint['family']} checkpoints must use {expected_tier}"
            )
        if checkpoint["family"] == "open_weight":
            for field in ("runtime_profile", "observed_peak_memory_mib"):
                if field not in checkpoint:
                    raise FrontierError(f"{checkpoint_id}: missing {field}")
        else:
            api_state = availability.get(checkpoint_id)
            if not isinstance(api_state, dict):
                raise FrontierError(f"{checkpoint_id}: missing API availability record")
            if str(api_state.get("model_id")) != str(checkpoint["model_id"]):
                raise FrontierError(f"{checkpoint_id}: API availability model mismatch")
            checkpoint["access_status"] = str(api_state.get("access_status") or "")
            checkpoint["lifecycle_status"] = str(api_state.get("lifecycle_status") or "")
            checkpoint["identity_class"] = str(api_state.get("identity_class") or "")
            checkpoint["availability_checked_at"] = str(
                availability_raw.get("checked_at") or ""
            )
            checkpoint.setdefault("runtime_profile", "provider_api_runtime_to_be_pinned")
            checkpoint.setdefault("observed_peak_memory_mib", None)
        if checkpoint["runtime_profile"] not in raw.get("runtime_profiles", {}):
            raise FrontierError(f"{checkpoint_id}: unknown runtime profile")
        tier = tiers[checkpoint["hardware_tier"]]
        checkpoint.setdefault("hardware_capacity_mib", tier.get("ceiling_mib"))
        if not str(checkpoint["release_source"]).startswith("https://"):
            raise FrontierError(f"{checkpoint_id}: release_source must be HTTPS")
        status = str(checkpoint["result_status"])
        if status in {"full34_confirmed", "panel_only"} and not isinstance(
            checkpoint.get("result"), dict
        ):
            raise FrontierError(f"{checkpoint_id}: result metadata is required")
        if checkpoint["family"] == "open_weight" and bool(checkpoint["immutable"]):
            revision = str(checkpoint.get("revision") or "")
            if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
                raise FrontierError(f"{checkpoint_id}: immutable open artifact lacks revision")
    return raw


def _task_context(panel: PanelSelection) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    tasks = load_task_definitions(tasks_dir=REPO / "tasks")
    by_name = {str(task["name"]): task for task in tasks}
    if len(by_name) != 34:
        raise FrontierError(f"full benchmark resolves to {len(by_name)} tasks, expected 34")
    counts = {name: len(task["loader"]()) for name, task in by_name.items()}
    if sum(counts.values()) != 16425:
        raise FrontierError(
            f"full benchmark resolves to {sum(counts.values())} items, expected 16425"
        )
    if set(panel.task_names) - set(by_name):
        raise FrontierError("panel contains a task outside the full benchmark")
    return by_name, counts


def _prediction_cache_reader() -> Any:
    cache: dict[Path, pd.DataFrame] = {}

    def read(path: Path) -> pd.DataFrame:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = pd.read_csv(resolved, low_memory=False)
        return cache[resolved]

    return read


def _metric_cache_reader() -> Any:
    cache: dict[Path, pd.DataFrame] = {}

    def read(path: Path) -> pd.DataFrame:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = pd.read_csv(resolved, low_memory=False)
        return cache[resolved]

    return read


def _validate_gold(
    frame: pd.DataFrame,
    tasks: Mapping[str, dict[str, Any]],
    task_names: set[str],
    checkpoint_id: str,
) -> None:
    for task_name in sorted(task_names):
        task = tasks[task_name]
        expected_items = task["loader"]()
        group = frame.loc[frame["task"] == task_name].copy()
        group["item_id"] = group["item_id"].astype(str)
        observed = group.set_index("item_id", verify_integrity=True)
        for item in expected_items:
            item_id = str(item["item_id"])
            for label, expected_value in item["gt"].items():
                column = f"gt_{label}"
                if column not in observed.columns:
                    raise FrontierError(f"{checkpoint_id}: missing {column}")
                observed_value = observed.at[item_id, column]
                if isinstance(expected_value, (int, np.integer)):
                    try:
                        equal = int(float(observed_value)) == int(expected_value)
                    except (TypeError, ValueError):
                        equal = False
                else:
                    equal = str(observed_value) == str(expected_value)
                if not equal:
                    raise FrontierError(
                        f"{checkpoint_id}/{task_name}/{item_id}: gold label drift"
                    )


def _validate_prediction_keys(
    frame: pd.DataFrame,
    tasks: Mapping[str, dict[str, Any]],
    expected_names: set[str],
    checkpoint_id: str,
) -> None:
    required = {"task", "model", "item_id", "parse_error"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FrontierError(f"{checkpoint_id}: predictions missing columns {missing}")
    if frame.duplicated(["task", "model", "item_id"]).any():
        raise FrontierError(f"{checkpoint_id}: duplicate task/model/item_id keys")
    observed_names = set(frame["task"].astype(str))
    if observed_names != expected_names:
        raise FrontierError(
            f"{checkpoint_id}: task coverage mismatch; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for task_name in sorted(expected_names):
        expected_ids = {str(item["item_id"]) for item in tasks[task_name]["loader"]()}
        observed_ids = set(
            frame.loc[frame["task"] == task_name, "item_id"].astype(str)
        )
        if observed_ids != expected_ids:
            raise FrontierError(
                f"{checkpoint_id}/{task_name}: item-key coverage mismatch; "
                f"missing={len(expected_ids - observed_ids)}, "
                f"extra={len(observed_ids - expected_ids)}"
            )
    _validate_gold(frame, tasks, expected_names, checkpoint_id)


def _validate_hive_metadata(
    checkpoint: Mapping[str, Any],
    result: Mapping[str, Any],
    predictions_path: Path,
    tasks: Mapping[str, dict[str, Any]],
    task_counts: Mapping[str, int],
    panel: PanelSelection,
) -> tuple[dict[str, Any], str]:
    metadata_path = resolve_repo_path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    checkpoint_id = str(checkpoint["checkpoint_id"])
    full_mode = str(checkpoint["result_status"]) == "full34_confirmed"
    expected_task_names = set(tasks) if full_mode else set(panel.task_names)
    expected_task_counts = (
        dict(task_counts) if full_mode else dict(panel.task_counts)
    )
    expected_total_items = sum(expected_task_counts.values())
    expected = {
        "status": "completed",
        "benchmark_commit": panel.benchmark_commit,
        "model_key": result["model_key"],
        "model_id": checkpoint["model_id"],
        "revision": checkpoint["revision"],
        "total_tasks": len(expected_task_names),
        "total_items": expected_total_items,
        "rows_written": expected_total_items,
    }
    if not full_mode:
        expected.update(
            {
                "panel_id": panel.panel_id,
                "panel_sha256": panel.panel_sha256,
            }
        )
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise FrontierError(
                f"{checkpoint_id}: metadata {field}={metadata.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if metadata.get("task_counts") != expected_task_counts:
        raise FrontierError(f"{checkpoint_id}: metadata task counts drifted")
    observed_predictions_sha = file_sha256(predictions_path)
    if metadata.get("predictions_sha256") != observed_predictions_sha:
        raise FrontierError(f"{checkpoint_id}: predictions SHA-256 mismatch")

    model = {"model_id": checkpoint["model_id"], "revision": checkpoint["revision"]}
    task_results = metadata.get("task_results")
    if not isinstance(task_results, dict) or set(task_results) != expected_task_names:
        raise FrontierError(f"{checkpoint_id}: task fingerprint ledger is incomplete")
    for task_name in sorted(expected_task_names):
        task = tasks[task_name]
        panel_sha256 = None if full_mode else panel.panel_sha256
        panel_task_fingerprint = (
            None if full_mode else panel.task_fingerprints[task_name]
        )
        observed = task_fingerprint(
            task,
            task["loader"](),
            model_alias=str(result["model_key"]),
            model=model,
            config_hash=str(metadata["config_sha256"]),
            generation=dict(metadata["generation"]),
            runtime=dict(metadata["runtime"]),
            runtime_versions=dict(metadata["runtime_versions"]),
            panel_sha256=panel_sha256,
            panel_task_fingerprint=panel_task_fingerprint,
        )
        if task_results[task_name].get("task_fingerprint") != observed:
            raise FrontierError(
                f"{checkpoint_id}/{task_name}: exact input/config fingerprint mismatch"
            )
        if not full_mode:
            task_result = task_results[task_name]
            if task_result.get("panel_sha256") != panel.panel_sha256:
                raise FrontierError(
                    f"{checkpoint_id}/{task_name}: task ledger panel hash mismatch"
                )
            if (
                task_result.get("panel_task_fingerprint")
                != panel.task_fingerprints[task_name]
            ):
                raise FrontierError(
                    f"{checkpoint_id}/{task_name}: task ledger panel fingerprint mismatch"
                )
    return metadata, file_sha256(metadata_path)


def _bootstrap_mean(values: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise FrontierError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    return values[indices].mean(axis=1)


def mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float, float]:
    draws = _bootstrap_mean(values, iterations, seed)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(draws.std(ddof=1)),
    )


def calibrated_panel_ci(
    values: np.ndarray,
    calibration: Mapping[str, Any],
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    intercept = float(calibration["intercept"])
    slope = float(calibration["slope"])
    residuals = np.asarray(
        calibration["held_out_validation"].get("residuals", []), dtype=float
    )
    if len(residuals) != int(calibration["held_out_validation"]["models"]):
        raise FrontierError("calibration must contain one residual per held-out model")
    if not np.isfinite(residuals).all():
        raise FrontierError("calibration residuals must be finite")
    residuals = residuals - residuals.mean()
    panel_mean = float(values.mean())
    estimate = intercept + slope * panel_mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    panel_draws = values[indices].mean(axis=1)
    residual_draws = rng.choice(residuals, size=iterations, replace=True)
    draws = intercept + slope * panel_draws + residual_draws
    combined_se = float(draws.std(ddof=1))
    return (
        estimate,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        combined_se,
    )


def _base_row(checkpoint: Mapping[str, Any], panel: PanelSelection) -> dict[str, Any]:
    result = checkpoint.get("result") or {}
    immutable = bool(checkpoint["immutable"])
    family = str(checkpoint["family"])
    base_release_date = str(checkpoint.get("base_model_release_date") or checkpoint["release_date"])
    artifact_release_date = str(checkpoint.get("artifact_release_date") or "")
    if not artifact_release_date and immutable:
        artifact_release_date = str(checkpoint["release_date"])
    release_date = artifact_release_date or base_release_date
    evaluation_date = str(
        result.get("evaluated_at") or checkpoint.get("evaluation_date") or ""
    )
    if family == "api" and not immutable and evaluation_date:
        plot_date = evaluation_date
        date_basis = "evaluation_date_mutable_endpoint"
    elif family == "api":
        plot_date = release_date
        date_basis = "immutable_api_snapshot_release_date"
    elif artifact_release_date:
        plot_date = artifact_release_date
        date_basis = "artifact_release_date"
    else:
        plot_date = base_release_date
        date_basis = "base_model_release_date_unqualified_artifact"
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "display_name": checkpoint["display_name"],
        "family": family,
        "provider": checkpoint["provider"],
        "model_id": checkpoint["model_id"],
        "revision": checkpoint.get("revision") or "",
        "base_model_release_date": base_release_date,
        "artifact_release_date": artifact_release_date,
        "release_date": release_date,
        "evaluation_date": evaluation_date,
        "plot_date": plot_date,
        "date_basis": date_basis,
        "release_source": checkpoint["release_source"],
        "quantization": checkpoint["quantization"],
        "hardware_tier": checkpoint["hardware_tier"],
        "hardware_qualification": checkpoint["hardware_qualification"],
        "access_status": checkpoint.get("access_status", "not_applicable"),
        "lifecycle_status": checkpoint.get("lifecycle_status", "not_applicable"),
        "identity_class": checkpoint.get("identity_class", "huggingface_revision"),
        "availability_checked_at": checkpoint.get("availability_checked_at", ""),
        "immutable": immutable,
        "historical_lag_eligible": bool(checkpoint["historical_lag_eligible"]),
        "reasoning": checkpoint["reasoning"],
        "runtime_profile": checkpoint["runtime_profile"],
        "result_status": checkpoint["result_status"],
        "result_source_kind": result.get("source_kind", "")
        or checkpoint.get("evidence_source_kind", ""),
        "panel_id": panel.panel_id,
        "panel_sha256": panel.panel_sha256,
        "panel_tasks": np.nan,
        "panel_items": np.nan,
        "panel_coverage_audit": "not_run",
        "full34_tasks": np.nan,
        "full34_items": np.nan,
        "full34_coverage_audit": "not_run",
        "full34_input_state": "not_run",
        "fingerprint_audit": "not_run",
        "raw_panel_f1": np.nan,
        "calibrated_full34_f1": np.nan,
        "calibrated_ci_low": np.nan,
        "calibrated_ci_high": np.nan,
        "calibrated_combined_se": np.nan,
        "calibration_supported": np.nan,
        "full34_f1": np.nan,
        "full34_ci_low": np.nan,
        "full34_ci_high": np.nan,
        "full34_bootstrap_se": np.nan,
        "plot_f1": np.nan,
        "plot_ci_low": np.nan,
        "plot_ci_high": np.nan,
        "parse_error_rate": checkpoint.get("pilot_malformed_rate", np.nan),
        "reliability_pass": checkpoint.get("pilot_reliability_pass", np.nan),
        "observed_peak_memory_mib": checkpoint["observed_peak_memory_mib"],
        "observed_gpu_capacity_mib": checkpoint.get("hardware_capacity_mib", np.nan),
        "predictions_sha256": checkpoint.get("pilot_predictions_sha256", ""),
        "task_metrics_sha256": "",
        "metadata_sha256": checkpoint.get("pilot_metadata_sha256", ""),
        "result_provenance_sha256": "",
        "model_identity_verified": bool(
            checkpoint.get("pilot_model_identity_verified", False)
        ),
        "promotion_triggered": False,
        "promotion_required": False,
        "promotion_status": "not_evaluated",
        "promotion_reasons": "",
        "preceding_confirmed_frontier_f1": np.nan,
        "frontier_eligible": False,
        "frontier_setter": False,
        "frontier_score_after_checkpoint": np.nan,
    }


def _score_malformed_as_incorrect(
    group: pd.DataFrame, task: Mapping[str, Any], support_only: bool = False
) -> dict[str, Any]:
    """Compute frontier metrics over every row, forcing malformed decisions wrong.

    `support_only` defaults to False so the frozen frontier panels keep resolving
    to the legacy metric they were built on. The refresh release passes True when
    rebuilding onto the current task set and metric.
    """
    scored = group.copy()
    malformed = ~(
        scored["parse_error"].isna()
        | (scored["parse_error"].astype(str).str.strip() == "")
    )
    kind = str(task["label_kind"])
    if kind == "multi_binary":
        for label in task["labels"]:
            gold = pd.to_numeric(scored[f"gt_{label}"], errors="raise").astype(int)
            scored.loc[malformed, f"pred_{label}"] = 1 - gold.loc[malformed]
    elif kind == "binary":
        key = str(task["label_key"])
        gold = pd.to_numeric(scored[f"gt_{key}"], errors="raise").astype(int)
        scored.loc[malformed, f"pred_{key}"] = 1 - gold.loc[malformed]
    elif kind == "categorical":
        key = str(task["label_key"])
        scored.loc[malformed, f"pred_{key}"] = "__MALFORMED_INCORRECT__"
    else:
        raise FrontierError(f"{task['name']}: unknown label kind {kind}")
    scored["parse_error"] = np.nan
    if "latency_s" not in scored or scored["latency_s"].isna().all():
        scored["latency_s"] = 0.0
    # support_only=False pins the LEGACY metric on purpose: this panel's
    # stored scores were computed by averaging macro F1 over every manifest
    # label, including labels with no gold support. code/scoring.py is the
    # corrected implementation and is now the default in build_summary, but
    # flipping it here would silently invalidate the frozen frontier and
    # refresh panels. Migrate deliberately, rebuilding the panels, rather
    # than by inheriting a changed default.
    metrics = _metrics_for_group(task, scored, support_only=support_only)
    metrics["parse_ok"] = int((~malformed).sum())
    metrics["parse_err_rate"] = float(malformed.mean()) if len(scored) else np.nan
    return metrics


def passes_reliability_gate(parse_rate: float, panel: PanelSelection) -> bool:
    return float(parse_rate) <= float(panel.reliability_gate["max_malformed_rate"])


def analyze_checkpoint(
    checkpoint: Mapping[str, Any],
    panel: PanelSelection,
    tasks: Mapping[str, dict[str, Any]],
    task_counts: Mapping[str, int],
    read_predictions: Any,
    read_metrics: Any,
    iterations: int,
    seed: int,
) -> Evidence:
    row = _base_row(checkpoint, panel)
    result = checkpoint.get("result")
    status = str(checkpoint["result_status"])
    if not isinstance(result, dict):
        return Evidence(row=row, task_scores={})

    checkpoint_id = str(checkpoint["checkpoint_id"])
    metrics_path = resolve_repo_path(result["task_metrics_path"])
    predictions_path = resolve_repo_path(result["predictions_path"])
    for path in (metrics_path, predictions_path):
        if not path.exists():
            raise FrontierError(f"{checkpoint_id}: missing evidence file {path}")
    model_key = str(result["model_key"])
    metric_source = read_metrics(metrics_path)
    prediction_source = read_predictions(predictions_path)
    metrics = metric_source.loc[metric_source["model"].astype(str) == model_key].copy()
    predictions = prediction_source.loc[
        prediction_source["model"].astype(str) == model_key
    ].copy()
    if metrics.empty or predictions.empty:
        raise FrontierError(f"{checkpoint_id}: model key is absent from result files")
    required_metrics = {"task", "model", "n", "parse_err_rate", "headline_f1"}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise FrontierError(f"{checkpoint_id}: task metrics missing {missing_metrics}")
    if metrics.duplicated(["task", "model"]).any():
        raise FrontierError(f"{checkpoint_id}: duplicate task-metric rows")

    full_mode = status == "full34_confirmed"
    expected_names = set(tasks) if full_mode else set(panel.task_names)
    observed_metric_names = set(metrics["task"].astype(str))
    if observed_metric_names != expected_names:
        raise FrontierError(
            f"{checkpoint_id}: metric task coverage differs from declared result status"
        )
    expected_count_map = task_counts if full_mode else panel.task_counts
    observed_counts = {
        str(record.task): int(record.n)
        for record in metrics[["task", "n"]].itertuples(index=False)
    }
    if observed_counts != {name: int(expected_count_map[name]) for name in expected_names}:
        raise FrontierError(f"{checkpoint_id}: task metric item counts drifted")

    predictions_sha = file_sha256(predictions_path)
    declared_sha = result.get("predictions_sha256")
    metadata: dict[str, Any] | None = None
    metadata_sha = ""
    if result["source_kind"] == "hive_task_metrics":
        metadata, metadata_sha = _validate_hive_metadata(
            checkpoint,
            result,
            predictions_path,
            tasks,
            task_counts,
            panel,
        )
        row["fingerprint_audit"] = "passed_exact_hive_task_fingerprints"
        row["full34_input_state"] = "exact_frozen_commit_inputs"
        row["model_identity_verified"] = True
    elif declared_sha and predictions_sha != str(declared_sha):
        raise FrontierError(f"{checkpoint_id}: canonical predictions SHA-256 mismatch")
    else:
        row["fingerprint_audit"] = "passed_panel_manifest_plus_canonical_keys"
        if full_mode:
            row["full34_input_state"] = "legacy_one_nonpanel_wesleyan_redaction"
        if checkpoint["family"] == "open_weight":
            row["model_identity_verified"] = bool(declared_sha)
        else:
            returned_identity = str(result.get("returned_model_identity") or "")
            row["model_identity_verified"] = (
                bool(checkpoint["immutable"])
                and returned_identity == str(checkpoint["model_id"])
            )

    _validate_prediction_keys(predictions, tasks, expected_names, checkpoint_id)
    panel_predictions = predictions.loc[predictions["task"].isin(panel.task_names)]
    if len(panel_predictions) != panel.expected_items:
        raise FrontierError(f"{checkpoint_id}: panel prediction count is not 8,793")
    row["panel_tasks"] = panel.expected_tasks
    row["panel_items"] = panel.expected_items
    row["panel_coverage_audit"] = "passed_exact_task_item_gold_keys"
    if full_mode:
        row["full34_tasks"] = len(tasks)
        row["full34_items"] = sum(task_counts.values())
        row["full34_coverage_audit"] = "passed_exact_task_item_gold_keys"

    rescored_records = []
    for task_name in sorted(expected_names):
        group = predictions.loc[predictions["task"].astype(str) == task_name]
        rescored_records.append(
            {"task": task_name, **_score_malformed_as_incorrect(group, tasks[task_name])}
        )
    metrics = pd.DataFrame(rescored_records)
    if not np.isfinite(pd.to_numeric(metrics["headline_f1"], errors="raise")).all():
        raise FrontierError(f"{checkpoint_id}: non-finite rescored task F1")
    panel_metrics = metrics.loc[metrics["task"].isin(panel.task_names)].copy()
    panel_metrics = panel_metrics.set_index("task").loc[list(panel.task_names)].reset_index()
    panel_values = panel_metrics["headline_f1"].to_numpy(dtype=float)
    raw_panel_f1 = float(panel_values.mean())
    calibrated, calibrated_low, calibrated_high, calibrated_se = calibrated_panel_ci(
        panel_values,
        panel.calibration,
        iterations,
        seed,
    )
    support = panel.calibration["supported_panel_f1"]
    parse_rate = float(
        np.average(metrics["parse_err_rate"], weights=pd.to_numeric(metrics["n"]))
    )
    reliability_pass = passes_reliability_gate(parse_rate, panel)
    row.update(
        {
            "raw_panel_f1": raw_panel_f1,
            "calibrated_full34_f1": calibrated,
            "calibrated_ci_low": calibrated_low,
            "calibrated_ci_high": calibrated_high,
            "calibrated_combined_se": calibrated_se,
            "calibration_supported": (
                float(support["min"]) <= raw_panel_f1 <= float(support["max"])
            ),
            "parse_error_rate": parse_rate,
            "reliability_pass": reliability_pass,
            "predictions_sha256": predictions_sha,
            "task_metrics_sha256": file_sha256(metrics_path),
            "metadata_sha256": metadata_sha,
        }
    )

    task_scores = {
        str(record.task): float(record.headline_f1)
        for record in metrics[["task", "headline_f1"]].itertuples(index=False)
    }
    if full_mode:
        full_values = metrics.sort_values("task")["headline_f1"].to_numpy(dtype=float)
        full_mean, full_low, full_high, full_se = mean_ci(
            full_values,
            iterations,
            seed + 1,
        )
        row.update(
            {
                "full34_f1": full_mean,
                "full34_ci_low": full_low,
                "full34_ci_high": full_high,
                "full34_bootstrap_se": full_se,
                "plot_f1": full_mean,
                "plot_ci_low": full_low,
                "plot_ci_high": full_high,
            }
        )
    elif status == "panel_only":
        row.update(
            {
                "plot_f1": calibrated,
                "plot_ci_low": calibrated_low,
                "plot_ci_high": calibrated_high,
            }
        )
    if metadata is not None:
        metadata_peak = metadata.get("observed_peak_gpu_memory_used_mib", np.nan)
        declared_peak = checkpoint["observed_peak_memory_mib"]
        if declared_peak is None or int(declared_peak) != int(metadata_peak):
            raise FrontierError(
                f"{checkpoint_id}: registry and Hive peak-memory evidence differ"
            )
        row["observed_peak_memory_mib"] = metadata_peak
        metadata_capacity = metadata.get(
            "observed_gpu_capacity_mib", metadata.get("gpu_memory_total_mib", np.nan)
        )
        if pd.notna(metadata_capacity):
            row["observed_gpu_capacity_mib"] = metadata_capacity
    row["result_provenance_sha256"] = canonical_sha256(
        {
            "checkpoint_id": checkpoint_id,
            "panel_sha256": panel.panel_sha256,
            "predictions_sha256": row["predictions_sha256"],
            "task_metrics_sha256": row["task_metrics_sha256"],
            "metadata_sha256": row["metadata_sha256"],
            "model_id": checkpoint["model_id"],
            "revision": checkpoint.get("revision"),
        }
    )
    return Evidence(row=row, task_scores=task_scores)


def apply_promotion_and_frontiers(frame: pd.DataFrame, panel: PanelSelection) -> pd.DataFrame:
    result = frame.copy()
    promotion = panel.promotion
    distance = float(promotion.get("distance_f1", PROMOTION_DISTANCE))
    gap_threshold = float(promotion.get("substantive_gap_threshold_f1", 0.01))
    for tier in SERIES_ORDER:
        indices = result.index[result["hardware_tier"] == tier].tolist()
        indices.sort(key=lambda idx: (result.at[idx, "plot_date"], result.at[idx, "checkpoint_id"]))
        frontier_score: float | None = None
        for idx in indices:
            status = str(result.at[idx, "result_status"])
            has_panel = pd.notna(result.at[idx, "raw_panel_f1"])
            previous = frontier_score
            if previous is not None:
                result.at[idx, "preceding_confirmed_frontier_f1"] = previous
            reasons: list[str] = []
            if has_panel:
                estimate = float(result.at[idx, "calibrated_full34_f1"])
                ci_low = float(result.at[idx, "calibrated_ci_low"])
                ci_high = float(result.at[idx, "calibrated_ci_high"])
                if previous is None:
                    reasons.append("first_tested_checkpoint_requires_confirmation")
                elif ci_high - previous >= -distance - 1e-12:
                    reasons.append("upper_difference_bound_at_least_minus_0.01")
                if not bool(result.at[idx, "calibration_supported"]):
                    reasons.append("outside_calibration_support")
                if previous is not None and ci_low - previous <= 0 <= ci_high - previous:
                    reasons.append("ordering_interval_crosses_zero")
                if previous is not None:
                    delta_low, delta_high = ci_low - previous, ci_high - previous
                    if delta_low <= gap_threshold <= delta_high or delta_low <= -gap_threshold <= delta_high:
                        reasons.append("reported_gap_threshold_crossed")
            triggered = bool(reasons)
            result.at[idx, "promotion_triggered"] = triggered
            result.at[idx, "promotion_reasons"] = ";".join(reasons)
            if status == "panel_only":
                result.at[idx, "promotion_required"] = triggered
                result.at[idx, "promotion_status"] = (
                    "requires_full34" if triggered else "screened_no_promotion"
                )
            elif status == "full34_confirmed":
                result.at[idx, "promotion_status"] = "full34_already_available"
            elif status.startswith("pending"):
                result.at[idx, "promotion_status"] = "not_evaluated"

            eligible = (
                status == "full34_confirmed"
                and result.at[idx, "hardware_qualification"] == "confirmed"
                and bool(result.at[idx, "reliability_pass"])
                and result.at[idx, "full34_coverage_audit"]
                == "passed_exact_task_item_gold_keys"
                and bool(result.at[idx, "immutable"])
                and bool(result.at[idx, "historical_lag_eligible"])
                and bool(result.at[idx, "model_identity_verified"])
            )
            if eligible and tier == "hive_98gb":
                peak = result.at[idx, "observed_peak_memory_mib"]
                capacity = result.at[idx, "observed_gpu_capacity_mib"]
                eligible = pd.notna(peak) and pd.notna(capacity) and float(peak) <= float(capacity)
            result.at[idx, "frontier_eligible"] = eligible
            if eligible:
                score = float(result.at[idx, "full34_f1"])
                if frontier_score is None or score > frontier_score:
                    frontier_score = score
                    result.at[idx, "frontier_setter"] = True
            if frontier_score is not None:
                result.at[idx, "frontier_score_after_checkpoint"] = frontier_score
    return result


def paired_difference_ci(
    api_scores: Mapping[str, float],
    open_scores: Mapping[str, float],
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    common = sorted(set(api_scores) & set(open_scores))
    if len(common) != 34:
        raise FrontierError(f"paired lag comparison has {len(common)} tasks, expected 34")
    differences = np.array(
        [float(api_scores[task]) - float(open_scores[task]) for task in common],
        dtype=float,
    )
    draws = _bootstrap_mean(differences, iterations, seed)
    return (
        float(differences.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def build_lag_table(
    scores: pd.DataFrame,
    evidence_by_id: Mapping[str, Evidence],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    columns = [
        "open_checkpoint_id",
        "open_display_name",
        "hardware_tier",
        "open_release_date",
        "open_full34_f1",
        "earliest_point_estimate_api_checkpoint_id",
        "earliest_point_estimate_api_date",
        "api_full34_f1",
        "api_minus_open_f1",
        "paired_difference_ci_low",
        "paired_difference_ci_high",
        "lag_months",
        "lag_status",
        "lag_note",
    ]
    open_rows = scores.loc[
        scores["frontier_setter"] & (scores["hardware_tier"] == "hive_98gb")
    ].sort_values("plot_date")
    api_rows = scores.loc[
        scores["frontier_setter"]
        & (scores["hardware_tier"] == "api")
        & scores["immutable"]
        & scores["historical_lag_eligible"]
    ].sort_values("plot_date")
    rows: list[dict[str, Any]] = []
    for _, open_row in open_rows.iterrows():
        base = {
            "open_checkpoint_id": open_row["checkpoint_id"],
            "open_display_name": open_row["display_name"],
            "hardware_tier": open_row["hardware_tier"],
            "open_release_date": open_row["release_date"],
            "open_full34_f1": open_row["full34_f1"],
            "earliest_point_estimate_api_checkpoint_id": "",
            "earliest_point_estimate_api_date": "",
            "api_full34_f1": np.nan,
            "api_minus_open_f1": np.nan,
            "paired_difference_ci_low": np.nan,
            "paired_difference_ci_high": np.nan,
            "lag_months": np.nan,
            "lag_status": "",
            "lag_note": "",
        }
        if not bool(open_row["immutable"]) or not bool(open_row["historical_lag_eligible"]):
            base["lag_status"] = "excluded_mutable_open_artifact"
            base["lag_note"] = "The evaluated open artifact lacks immutable provenance."
            rows.append(base)
            continue
        prior_api = api_rows.loc[api_rows["plot_date"] <= open_row["release_date"]]
        if prior_api.empty:
            base["lag_status"] = "no_prior_api_coverage"
            base["lag_note"] = "No immutable tested API checkpoint predates this open release."
            rows.append(base)
            continue
        reached = prior_api.loc[prior_api["full34_f1"] >= float(open_row["full34_f1"])]
        if reached.empty:
            base["lag_status"] = "not_reached_by_point_estimate"
            base["lag_note"] = "No earlier immutable API point estimate reaches the open score."
            rows.append(base)
            continue
        api_row = reached.iloc[0]
        delta, low, high = paired_difference_ci(
            evidence_by_id[str(api_row["checkpoint_id"])].task_scores,
            evidence_by_id[str(open_row["checkpoint_id"])].task_scores,
            iterations,
            seed + len(rows),
        )
        base.update(
            {
                "earliest_point_estimate_api_checkpoint_id": api_row["checkpoint_id"],
                "earliest_point_estimate_api_date": api_row["plot_date"],
                "api_full34_f1": api_row["full34_f1"],
                "api_minus_open_f1": delta,
                "paired_difference_ci_low": low,
                "paired_difference_ci_high": high,
            }
        )
        if low > 0:
            days = (
                date.fromisoformat(str(open_row["release_date"]))
                - date.fromisoformat(str(api_row["plot_date"]))
            ).days
            base["lag_months"] = days / 30.4375
            base["lag_status"] = "confirmed"
            base["lag_note"] = "Earlier API score is higher under the paired-task bootstrap."
        else:
            base["lag_status"] = "ambiguous_overlapping_interval"
            base["lag_note"] = (
                "Point estimates imply a lag, but the paired-task interval includes zero; "
                "no precise horizontal lag is reported."
            )
        rows.append(base)
    return pd.DataFrame(rows, columns=columns)


def _step_coordinates(group: pd.DataFrame, end_date: pd.Timestamp) -> tuple[list[pd.Timestamp], list[float]]:
    setters = group.loc[group["frontier_setter"]].sort_values("plot_date")
    dates = list(pd.to_datetime(setters["plot_date"]))
    scores = [float(value) for value in setters["full34_f1"]]
    if dates:
        dates.append(end_date)
        scores.append(scores[-1])
    return dates, scores


def render_figure(scores: pd.DataFrame, output_dir: Path, registry: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    window_start = pd.Timestamp(registry["window"]["start"])
    window_end = pd.Timestamp(registry["window"]["end"])
    measured = int((scores["result_status"] == "full34_confirmed").sum())
    total = len(scores)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.10, right=0.80, top=0.80, bottom=0.22)

    for tier in SERIES_ORDER:
        color = SERIES_COLORS[tier]
        group = scores.loc[scores["hardware_tier"] == tier].copy()
        dates, values = _step_coordinates(group, window_end)
        if dates:
            ax.step(dates, values, where="post", color=color, linewidth=2.2, zorder=3)
            final_value = values[-1]
            ax.text(
                window_end + pd.Timedelta(days=5),
                final_value,
                f"{SERIES_LABELS[tier]}  {final_value:.3f}",
                color=color,
                fontsize=9.5,
                va="center",
                clip_on=False,
            )
        measured_group = group.loc[group["result_status"] == "full34_confirmed"]
        for _, point in measured_group.iterrows():
            face = color if bool(point["immutable"]) else "white"
            ax.scatter(
                pd.Timestamp(point["plot_date"]),
                float(point["full34_f1"]),
                s=42,
                marker="o",
                facecolor=face,
                edgecolor=color,
                linewidth=1.4,
                zorder=5,
            )
        panel_only = group.loc[group["result_status"] == "panel_only"]
        if not panel_only.empty:
            ax.scatter(
                pd.to_datetime(panel_only["plot_date"]),
                panel_only["calibrated_full34_f1"],
                s=44,
                marker="^",
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )

    api_setters = scores.loc[
        (scores["hardware_tier"] == "api") & scores["frontier_setter"]
    ].sort_values("plot_date")
    if not api_setters.empty:
        first = api_setters.iloc[0]
        first_date = pd.Timestamp(first["plot_date"])
        first_score = float(first["full34_f1"])
        ax.plot(
            [window_start, first_date],
            [first_score, first_score],
            color="#888888",
            linewidth=1.1,
            linestyle=(0, (2, 3)),
            zorder=1,
        )
        ax.text(
            window_start + (first_date - window_start) / 2,
            first_score + 0.003,
            "API history not measured",
            color="#777777",
            fontsize=8.5,
            ha="center",
        )

    rug_y = {"hive_98gb": 0.5615, "api": 0.558}
    for tier in SERIES_ORDER:
        pending = scores.loc[
            (scores["hardware_tier"] == tier)
            & scores["result_status"].astype(str).str.startswith("pending")
        ]
        if not pending.empty:
            ax.scatter(
                pd.to_datetime(pending["release_date"]),
                [rug_y[tier]] * len(pending),
                marker="|",
                s=60,
                linewidth=1,
                color=SERIES_COLORS[tier],
                alpha=0.35,
                clip_on=False,
                zorder=2,
            )

    ax.set_xlim(window_start, window_end + pd.Timedelta(days=3))
    ax.set_ylim(0.555, 0.695)
    ax.set_ylabel("Mean task-level F1 (axis starts at 0.555)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", colors="#444444", labelsize=9)

    fig.suptitle(
        "The historical comparison remains incomplete pending Hive and API evidence",
        x=0.10,
        y=0.94,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.10,
        0.865,
        f"Political-science text classification; {measured} of {total} planned artifact/API checkpoints have complete 34-task results",
        ha="left",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(
        0.10,
        0.075,
        (
            "Each task receives equal weight. Filled points have immutable model identity; hollow points have an unresolved or mutable identity. "
            "Small ticks are planned, unscored checkpoints. The dashed API segment is missing coverage, not an estimate."
        ),
        ha="left",
        va="top",
        fontsize=8.3,
        color="#555555",
        wrap=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "frontier_staircase.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "frontier_staircase.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(tmp, index=False, float_format="%.10f")
    os.replace(tmp, path)


def build_outputs(
    panel_path: Path = DEFAULT_PANEL,
    registry_path: Path = DEFAULT_REGISTRY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    render_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if iterations < 100:
        raise FrontierError("bootstrap iterations must be at least 100")
    panel = load_panel_manifest(panel_path)
    registry = load_checkpoint_registry(registry_path, panel)
    tasks, task_counts = _task_context(panel)
    read_predictions = _prediction_cache_reader()
    read_metrics = _metric_cache_reader()
    evidence: list[Evidence] = []
    for index, checkpoint in enumerate(registry["checkpoints"]):
        evidence.append(
            analyze_checkpoint(
                checkpoint,
                panel,
                tasks,
                task_counts,
                read_predictions,
                read_metrics,
                iterations,
                seed + index * 17,
            )
        )
    scores = pd.DataFrame([item.row for item in evidence])
    scores = apply_promotion_and_frontiers(scores, panel)
    scores = scores.sort_values(["plot_date", "hardware_tier", "checkpoint_id"]).reset_index(
        drop=True
    )
    evidence_by_id = {str(item.row["checkpoint_id"]): item for item in evidence}
    lags = build_lag_table(scores, evidence_by_id, iterations, seed + 10000)
    _write_csv_atomic(scores, output_dir / "frontier_scores.csv")
    _write_csv_atomic(lags, output_dir / "frontier_lags.csv")
    if render_plot:
        render_figure(scores, output_dir, registry)
    return scores, lags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--checkpoint-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    scores, lags = build_outputs(
        panel_path=args.panel_manifest,
        registry_path=args.checkpoint_registry,
        output_dir=args.output_dir,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        render_plot=not args.no_figure,
    )
    measured = int((scores["result_status"] == "full34_confirmed").sum())
    print(
        f"Wrote {len(scores)} checkpoint rows ({measured} complete full-suite results) "
        f"and {len(lags)} open-frontier lag rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
