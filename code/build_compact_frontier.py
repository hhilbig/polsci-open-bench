#!/usr/bin/env python3
"""Audit and plot the compact two-year open-versus-API benchmark.

The compact protocol scores one frozen eight-task benchmark directly.  It has
no calibration, screening, promotion, or larger confirmation suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from build_frontier_2026 import _score_malformed_as_incorrect
from hive_vllm_benchmark import task_fingerprint
from panel_manifest import PanelSelection, load_panel_manifest
from task_registry import load_task_definitions


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = REPO / "experiments" / "frontier_compact_8.yaml"
DEFAULT_REGISTRY = REPO / "experiments" / "frontier_compact_checkpoints_2026.yaml"
DEFAULT_OUTPUT_DIR = REPO / "output" / "sidecar" / "frontier_2026" / "compact8"
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_ITERATIONS = 10_000
SERIES_ORDER = ("open_hive", "api")
SERIES_LABELS = {
    "open_hive": "Best tested open checkpoint on one 98 GB-class Hive GPU",
    "api": "Best tested, still-accessible API checkpoint",
}
SERIES_COLORS = {"open_hive": "#0072B2", "api": "#D55E00"}
PUBLIC_LABEL = (
    "Best tested checkpoints on a frozen eight-task political-science "
    "classification benchmark."
)
PUBLIC_CAVEAT = "Compact and non-exhaustive; not a claim about the global model frontier."
BROAD_CAVEAT = (
    "Broader task coverage, but non-exhaustive; not a claim about the global model frontier."
)
HIVE_GPU_NAME_FRAGMENT = "RTX PRO 6000 Blackwell"
HIVE_GPU_CAPACITY_MIB = 97_887


class CompactFrontierError(ValueError):
    """Raised when compact evidence or registry metadata fails closed."""


def public_label(expected_tasks: int) -> str:
    task_count = "eight" if expected_tasks == 8 else str(expected_tasks)
    return (
        f"Best tested checkpoints on a frozen {task_count}-task "
        "political-science classification benchmark."
    )


def public_caveat(expected_tasks: int) -> str:
    return PUBLIC_CAVEAT if expected_tasks == 8 else BROAD_CAVEAT


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
        raise CompactFrontierError(f"{context} must be an ISO date") from exc


def load_compact_registry(path: Path, benchmark: PanelSelection) -> dict[str, Any]:
    """Load the compact registry and validate its public semantics."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 1:
        raise CompactFrontierError("compact registry schema_version must be 1")
    if str(raw.get("benchmark_commit")) != benchmark.benchmark_commit:
        raise CompactFrontierError("registry and compact benchmark commits differ")
    manifest_path = resolve_repo_path(str(raw.get("compact_manifest", ""))).resolve()
    if manifest_path != benchmark.manifest_path:
        raise CompactFrontierError("registry points to a different compact manifest")
    if str(raw.get("public_label")) != public_label(benchmark.expected_tasks):
        raise CompactFrontierError("registry public label drifted")
    if str(raw.get("public_caveat")) != public_caveat(benchmark.expected_tasks):
        raise CompactFrontierError("registry public caveat drifted")

    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise CompactFrontierError("compact registry needs checkpoints")
    required = {
        "checkpoint_id",
        "display_name",
        "series",
        "provider",
        "model_id",
        "identity",
        "plot_date",
        "date_basis",
        "release_source",
        "quantization",
        "immutable",
        "result_status",
    }
    ids: set[str] = set()
    counts = {series: 0 for series in SERIES_ORDER}
    dates = {series: [] for series in SERIES_ORDER}
    allowed_status = {"pending", "completed", "failed", "unavailable"}
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise CompactFrontierError(f"checkpoints[{index}] must be a mapping")
        missing = sorted(required - set(checkpoint))
        if missing:
            raise CompactFrontierError(f"checkpoints[{index}] missing fields {missing}")
        checkpoint_id = str(checkpoint["checkpoint_id"])
        if checkpoint_id in ids:
            raise CompactFrontierError(f"duplicate checkpoint_id {checkpoint_id}")
        ids.add(checkpoint_id)
        series = str(checkpoint["series"])
        if series not in SERIES_ORDER:
            raise CompactFrontierError(f"{checkpoint_id}: invalid series {series}")
        counts[series] += 1
        plot_date = _iso_date(checkpoint["plot_date"], f"{checkpoint_id}.plot_date")
        dates[series].append(plot_date)
        if not bool(checkpoint["immutable"]):
            raise CompactFrontierError(f"{checkpoint_id}: compact checkpoints must be immutable")
        identity = str(checkpoint["identity"])
        if not identity:
            raise CompactFrontierError(f"{checkpoint_id}: immutable identity is required")
        if series == "open_hive":
            if len(identity) != 40 or any(ch not in "0123456789abcdef" for ch in identity.lower()):
                raise CompactFrontierError(
                    f"{checkpoint_id}: open identity must be a full revision hash"
                )
            if checkpoint["date_basis"] != "artifact_publication_date":
                raise CompactFrontierError(
                    f"{checkpoint_id}: open plot date must use artifact publication"
                )
        elif checkpoint["date_basis"] != "immutable_snapshot_date":
            raise CompactFrontierError(
                f"{checkpoint_id}: API plot date must use immutable snapshot publication"
            )
        if checkpoint["result_status"] not in allowed_status:
            raise CompactFrontierError(
                f"{checkpoint_id}: invalid result_status {checkpoint['result_status']}"
            )
        if checkpoint["result_status"] == "completed" and not isinstance(
            checkpoint.get("result"), dict
        ):
            raise CompactFrontierError(f"{checkpoint_id}: completed row needs result evidence")
    expected_open = 22 if benchmark.expected_tasks == 18 else 17
    expected_counts = {"open_hive": expected_open, "api": 7}
    if counts != expected_counts:
        raise CompactFrontierError(
            f"registry needs the frozen {expected_open}-open/7-API roster: {counts}"
        )
    for series, observed in dates.items():
        if observed != sorted(observed):
            raise CompactFrontierError(f"{series} checkpoints are not chronological")
    return raw


def _validate_gold(
    frame: pd.DataFrame,
    tasks: Mapping[str, Mapping[str, Any]],
    checkpoint_id: str,
) -> None:
    for task_name, task in tasks.items():
        observed = (
            frame.loc[frame["task"].astype(str) == task_name]
            .assign(item_id=lambda x: x["item_id"].astype(str))
            .set_index("item_id", verify_integrity=True)
        )
        for item in task["loader"]():
            item_id = str(item["item_id"])
            for label, expected in item["gt"].items():
                column = f"gt_{label}"
                if column not in observed:
                    raise CompactFrontierError(f"{checkpoint_id}: missing {column}")
                value = observed.at[item_id, column]
                if isinstance(expected, (int, np.integer)):
                    try:
                        equal = int(float(value)) == int(expected)
                    except (TypeError, ValueError):
                        equal = False
                else:
                    equal = str(value) == str(expected)
                if not equal:
                    raise CompactFrontierError(
                        f"{checkpoint_id}/{task_name}/{item_id}: gold-label drift"
                    )


def _validate_compact_rows(
    frame: pd.DataFrame,
    benchmark: PanelSelection,
    checkpoint: Mapping[str, Any],
) -> pd.DataFrame:
    checkpoint_id = str(checkpoint["checkpoint_id"])
    required = {"task", "model", "item_id", "parse_error"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CompactFrontierError(f"{checkpoint_id}: predictions missing columns {missing}")
    expected_keys = {
        (str(task["name"]), str(item["item_id"]))
        for task in benchmark.tasks
        for item in task["loader"]()
    }
    frame_keys = list(
        zip(frame["task"].astype(str), frame["item_id"].astype(str))
    )
    selected = frame.loc[[key in expected_keys for key in frame_keys]].copy()
    selected["task"] = selected["task"].astype(str)
    selected["item_id"] = selected["item_id"].astype(str)
    if len(selected) != benchmark.expected_items:
        raise CompactFrontierError(
            f"{checkpoint_id}: observed {len(selected)} compact rows, "
            f"expected {benchmark.expected_items}"
        )
    if selected.duplicated(["task", "model", "item_id"]).any():
        raise CompactFrontierError(f"{checkpoint_id}: duplicate task/model/item keys")
    if selected["model"].astype(str).nunique() != 1:
        raise CompactFrontierError(f"{checkpoint_id}: predictions contain multiple model aliases")
    if "model_id" in selected and set(selected["model_id"].astype(str)) != {
        str(checkpoint["model_id"])
    }:
        raise CompactFrontierError(f"{checkpoint_id}: model_id mismatch")
    if "model_revision" in selected and set(selected["model_revision"].astype(str)) != {
        str(checkpoint["identity"])
    }:
        raise CompactFrontierError(f"{checkpoint_id}: model revision mismatch")

    tasks = {str(task["name"]): task for task in benchmark.tasks}
    observed_tasks = set(selected["task"])
    if observed_tasks != set(benchmark.task_names):
        raise CompactFrontierError(f"{checkpoint_id}: compact task coverage mismatch")
    canonical_groups: list[pd.DataFrame] = []
    for task_name in benchmark.task_names:
        task = tasks[task_name]
        expected_ids = [str(item["item_id"]) for item in task["loader"]()]
        observed = selected.loc[selected["task"] == task_name].copy()
        observed_ids = observed["item_id"].tolist()
        if set(observed_ids) != set(expected_ids) or len(observed_ids) != len(expected_ids):
            raise CompactFrontierError(
                f"{checkpoint_id}/{task_name}: exact item-key coverage mismatch"
            )
        canonical_groups.append(
            observed.set_index("item_id", verify_integrity=True)
            .loc[expected_ids]
            .reset_index()
        )
    selected = pd.concat(canonical_groups, ignore_index=True)
    _validate_gold(selected, tasks, checkpoint_id)
    return selected


def _measured_hive_gpu(
    metadata: Mapping[str, Any], checkpoint_id: str
) -> tuple[int, int]:
    """Validate and return peak/capacity from actual one-GPU measurements."""
    measured: list[Mapping[str, Any]] = []
    for snapshot_name in ("gpu_before_load", "gpu_after_load"):
        snapshot = metadata.get(snapshot_name)
        if not isinstance(snapshot, Mapping):
            raise CompactFrontierError(
                f"{checkpoint_id}: missing measured {snapshot_name} GPU record"
            )
        gpus = snapshot.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 1 or not isinstance(gpus[0], Mapping):
            raise CompactFrontierError(
                f"{checkpoint_id}: {snapshot_name} must record exactly one GPU"
            )
        gpu = gpus[0]
        name = str(gpu.get("name") or "")
        if HIVE_GPU_NAME_FRAGMENT.lower() not in name.lower():
            raise CompactFrontierError(
                f"{checkpoint_id}: measured GPU is not the required Blackwell type"
            )
        try:
            capacity = int(gpu["memory_total_mib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompactFrontierError(
                f"{checkpoint_id}: measured GPU capacity is missing or invalid"
            ) from exc
        if capacity != HIVE_GPU_CAPACITY_MIB:
            raise CompactFrontierError(
                f"{checkpoint_id}: measured GPU capacity is {capacity}, "
                f"expected {HIVE_GPU_CAPACITY_MIB} MiB"
            )
        measured.append(gpu)

    before, after = measured
    for identity_field in ("name", "uuid"):
        before_value = str(before.get(identity_field) or "")
        after_value = str(after.get(identity_field) or "")
        if not before_value or before_value != after_value:
            raise CompactFrontierError(
                f"{checkpoint_id}: measured GPU {identity_field} changed during the run"
            )
    try:
        peak = int(metadata["observed_peak_gpu_memory_used_mib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompactFrontierError(
            f"{checkpoint_id}: measured peak GPU memory is missing or invalid"
        ) from exc
    if peak < 0 or peak > HIVE_GPU_CAPACITY_MIB:
        raise CompactFrontierError(f"{checkpoint_id}: Hive capacity gate failed")
    return peak, HIVE_GPU_CAPACITY_MIB


def _validate_compact_hive_qualification(
    metadata: Mapping[str, Any], checkpoint_id: str, revision: str
) -> None:
    """Require the compact runner's pilot, offline, OOM, and final gates."""
    if metadata.get("execution_protocol") != "compact_direct":
        raise CompactFrontierError(
            f"{checkpoint_id}: compact Hive execution protocol is missing"
        )
    snapshot_path = str(metadata.get("model_snapshot_path") or "")
    if (
        metadata.get("provenance_status") != "qualified"
        or not snapshot_path
        or Path(snapshot_path).name.lower() != revision.lower()
    ):
        raise CompactFrontierError(
            f"{checkpoint_id}: compact Hive offline snapshot provenance is unqualified"
        )
    pilot = metadata.get("pilot")
    pilot_checks = pilot.get("checks") if isinstance(pilot, Mapping) else None
    required_pilot_checks = {
        "completed",
        "exact_model_id",
        "exact_revision",
        "exact_row_count",
        "one_typed_gpu",
        "exact_capacity",
        "peak_within_capacity",
        "oom_free",
        "malformed_rate_within_limit",
    }
    if (
        not isinstance(pilot, Mapping)
        or pilot.get("passed") is not True
        or not isinstance(pilot_checks, Mapping)
        or not all(pilot_checks.get(field) is True for field in required_pilot_checks)
    ):
        raise CompactFrontierError(
            f"{checkpoint_id}: compact Hive pilot qualification is incomplete"
        )
    qualification = metadata.get("qualification")
    required_final_checks = {
        "exact_model_identity",
        "exact_coverage",
        "within_gpu_capacity",
        "oom_free",
    }
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("passed") is not True
        or not all(qualification.get(field) is True for field in required_final_checks)
        or metadata.get("evidence_stage") != "terminal_qualified"
    ):
        raise CompactFrontierError(
            f"{checkpoint_id}: compact Hive final qualification is incomplete"
        )
    reliability = qualification.get("reliability")
    if not isinstance(reliability, Mapping) or reliability.get("passed") is not True:
        raise CompactFrontierError(
            f"{checkpoint_id}: compact Hive reliability qualification is incomplete"
        )


def _validate_hive_evidence(
    checkpoint: Mapping[str, Any],
    result: Mapping[str, Any],
    predictions_path: Path,
    benchmark: PanelSelection,
) -> tuple[dict[str, Any], str]:
    checkpoint_id = str(checkpoint["checkpoint_id"])
    metadata_path = resolve_repo_path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    source_protocol = str(result.get("source_protocol", "compact8"))
    if source_protocol not in {"compact8", "broad18", "legacy_panel18", "legacy_full34"}:
        raise CompactFrontierError(f"{checkpoint_id}: invalid source_protocol")
    expected = {
        "status": "completed",
        "benchmark_commit": benchmark.benchmark_commit,
        "model_key": result["model_key"],
        "model_id": checkpoint["model_id"],
        "revision": checkpoint["identity"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise CompactFrontierError(
                f"{checkpoint_id}: metadata {field}={metadata.get(field)!r}, expected {value!r}"
            )
    if metadata.get("predictions_sha256") != file_sha256(predictions_path):
        raise CompactFrontierError(f"{checkpoint_id}: predictions SHA-256 mismatch")

    if source_protocol in {"compact8", "broad18"}:
        compact_expected = {
            "panel_id": benchmark.panel_id,
            "panel_sha256": benchmark.panel_sha256,
            "total_tasks": benchmark.expected_tasks,
            "total_items": benchmark.expected_items,
            "rows_written": benchmark.expected_items,
        }
        for field, value in compact_expected.items():
            if metadata.get(field) != value:
                raise CompactFrontierError(
                    f"{checkpoint_id}: compact metadata {field} mismatch"
                )
        _validate_compact_hive_qualification(
            metadata, checkpoint_id, str(checkpoint["identity"])
        )
    elif source_protocol == "legacy_full34":
        if int(metadata.get("total_tasks", -1)) != 34 or int(
            metadata.get("total_items", -1)
        ) != int(metadata.get("rows_written", -2)):
            raise CompactFrontierError(f"{checkpoint_id}: archived full evidence is incomplete")
    else:
        if int(metadata.get("total_tasks", -1)) != 18 or int(
            metadata.get("total_items", -1)
        ) != int(metadata.get("rows_written", -2)):
            raise CompactFrontierError(f"{checkpoint_id}: archived panel evidence is incomplete")

    model = {"model_id": checkpoint["model_id"], "revision": checkpoint["identity"]}
    task_results = metadata.get("task_results")
    if not isinstance(task_results, dict):
        raise CompactFrontierError(f"{checkpoint_id}: missing task fingerprint ledger")
    config_hash = str(metadata.get("config_sha256") or "")
    if not config_hash:
        raise CompactFrontierError(f"{checkpoint_id}: missing config hash")
    task_specs = {spec.name: spec for spec in benchmark.task_specs}
    source_tasks = {str(task["name"]): task for task in load_task_definitions()}
    for task in benchmark.tasks:
        task_name = str(task["name"])
        spec = task_specs[task_name]
        if task_name not in task_results:
            raise CompactFrontierError(f"{checkpoint_id}: missing {task_name} task ledger")
        if source_protocol in {"compact8", "broad18"}:
            panel_sha = benchmark.panel_sha256
            panel_task_sha = benchmark.task_fingerprints[task_name]
        elif source_protocol == "legacy_panel18":
            panel_sha = str(metadata.get("panel_sha256") or "")
            panel_task_sha = (
                spec.source_fingerprint_sha256
                or benchmark.task_fingerprints[task_name]
            )
            if not panel_sha:
                raise CompactFrontierError(
                    f"{checkpoint_id}: archived panel hash is missing"
                )
            if task_results[task_name].get("panel_task_fingerprint") != panel_task_sha:
                raise CompactFrontierError(
                    f"{checkpoint_id}/{task_name}: archived panel input fingerprint "
                    "does not match the compact benchmark"
                )
        else:
            panel_sha = None
            panel_task_sha = None
        fingerprint_task = (
            source_tasks[task_name]
            if source_protocol in {"legacy_panel18", "legacy_full34"}
            and spec.source_fingerprint_sha256
            else task
        )
        observed = task_fingerprint(
            fingerprint_task,
            fingerprint_task["loader"](),
            model_alias=str(result["model_key"]),
            model=model,
            config_hash=config_hash,
            generation=dict(metadata.get("generation") or {}),
            runtime=dict(metadata.get("runtime") or {}),
            runtime_versions=dict(metadata.get("runtime_versions") or {}),
            panel_sha256=panel_sha,
            panel_task_fingerprint=panel_task_sha,
        )
        if task_results[task_name].get("task_fingerprint") != observed:
            raise CompactFrontierError(
                f"{checkpoint_id}/{task_name}: prompt/schema/input/config fingerprint mismatch"
            )
    _measured_hive_gpu(metadata, checkpoint_id)
    return metadata, file_sha256(metadata_path)


def _validate_api_evidence(
    checkpoint: Mapping[str, Any],
    result: Mapping[str, Any],
    predictions_path: Path,
    benchmark: PanelSelection,
) -> tuple[dict[str, Any], str]:
    checkpoint_id = str(checkpoint["checkpoint_id"])
    metadata_path = resolve_repo_path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    required = {
        "evidence_type": "compact8_api_pilot_remainder_final",
        "status": "completed",
        "benchmark_commit": benchmark.benchmark_commit,
        "panel_id": benchmark.panel_id,
        "panel_sha256": benchmark.panel_sha256,
        "requested_model": checkpoint["model_id"],
        "returned_model_identity": checkpoint["identity"],
        "expected_items": benchmark.expected_items,
        "rows_written": benchmark.expected_items,
        "pilot_items": 16,
        "remainder_items": benchmark.expected_items - 16,
        "pilot_reliability_pass": True,
        "reliability_pass": True,
    }
    for field, value in required.items():
        if metadata.get(field) != value:
            raise CompactFrontierError(f"{checkpoint_id}: API metadata {field} mismatch")
    for field in ("pilot_reliability_pass", "reliability_pass"):
        if metadata.get(field) is not True:
            raise CompactFrontierError(
                f"{checkpoint_id}: API metadata {field} must be true"
            )
    try:
        pilot_malformed_count = int(metadata["pilot_malformed_count"])
        pilot_malformed_rate = float(metadata["pilot_malformed_rate"])
        pilot_malformed_limit = float(metadata["pilot_malformed_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompactFrontierError(
            f"{checkpoint_id}: API pilot malformed evidence is invalid"
        ) from exc
    if (
        pilot_malformed_count < 0
        or pilot_malformed_count > 16
        or pilot_malformed_limit != 0.05
        or pilot_malformed_rate != pilot_malformed_count / 16
        or pilot_malformed_rate > pilot_malformed_limit
    ):
        raise CompactFrontierError(
            f"{checkpoint_id}: API pilot malformed gate failed"
        )
    source_stages = metadata.get("source_stages")
    if not isinstance(source_stages, Mapping):
        raise CompactFrontierError(
            f"{checkpoint_id}: API source-stage evidence is missing"
        )
    for stage, expected_count in (("pilot", 16), ("remainder", benchmark.expected_items - 16)):
        evidence = source_stages.get(stage)
        if not isinstance(evidence, Mapping) or evidence.get("request_count") != expected_count:
            raise CompactFrontierError(
                f"{checkpoint_id}: API {stage} source-stage count mismatch"
            )
    if metadata.get("predictions_sha256") != file_sha256(predictions_path):
        raise CompactFrontierError(f"{checkpoint_id}: API predictions hash mismatch")
    manifest_path = resolve_repo_path(result["requests_manifest_path"])
    if file_sha256(manifest_path) != str(metadata.get("requests_manifest_sha256")):
        raise CompactFrontierError(f"{checkpoint_id}: API request-manifest hash mismatch")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required_columns = {
        "task",
        "item_id",
        "panel_id",
        "panel_sha256",
        "panel_task_fingerprint",
        "requested_model",
        "expected_response_model",
        "request_sha256",
    }
    if not required_columns.issubset(manifest.columns):
        raise CompactFrontierError(f"{checkpoint_id}: API request manifest is incomplete")
    if len(manifest) != benchmark.expected_items:
        raise CompactFrontierError(f"{checkpoint_id}: API request count mismatch")
    if manifest.duplicated(["task", "item_id"]).any():
        raise CompactFrontierError(f"{checkpoint_id}: duplicate API logical requests")
    if set(manifest["panel_id"]) != {benchmark.panel_id} or set(
        manifest["panel_sha256"]
    ) != {benchmark.panel_sha256}:
        raise CompactFrontierError(f"{checkpoint_id}: API compact-manifest identity mismatch")
    if set(manifest["requested_model"]) != {str(checkpoint["model_id"])} or set(
        manifest["expected_response_model"]
    ) != {str(checkpoint["identity"])}:
        raise CompactFrontierError(f"{checkpoint_id}: API immutable identity mismatch")
    for task_name, fingerprint in benchmark.task_fingerprints.items():
        observed = set(
            manifest.loc[manifest["task"] == task_name, "panel_task_fingerprint"]
        )
        if observed != {fingerprint}:
            raise CompactFrontierError(
                f"{checkpoint_id}/{task_name}: API task fingerprint mismatch"
            )
    return metadata, file_sha256(metadata_path)


def passes_reliability_gate(malformed_rate: float, benchmark: PanelSelection) -> bool:
    """Exactly the configured threshold passes; any value above it fails."""
    return float(malformed_rate) <= float(
        benchmark.reliability_gate["max_malformed_rate"]
    )


def _score_task_with_malformed_predictions(
    group: pd.DataFrame, task: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    """Count parse failures and invalid predictions as malformed, then score wrong."""
    scored = group.copy()
    scored["parse_error"] = scored["parse_error"].astype(object)
    parse_error = ~(
        scored["parse_error"].isna()
        | (scored["parse_error"].astype(str).str.strip() == "")
    )
    malformed = parse_error.copy()
    kind = str(task["label_kind"])
    if kind == "multi_binary":
        prediction_columns = [f"pred_{label}" for label in task["labels"]]
        missing = sorted(set(prediction_columns) - set(scored.columns))
        if missing:
            raise CompactFrontierError(f"{task['name']}: missing predictions {missing}")
        for column in prediction_columns:
            valid = pd.to_numeric(scored[column], errors="coerce").isin([0, 1])
            malformed |= ~valid
    elif kind == "binary":
        key = str(task["label_key"])
        column = f"pred_{key}"
        if column not in scored:
            raise CompactFrontierError(f"{task['name']}: missing prediction {column}")
        valid = pd.to_numeric(scored[column], errors="coerce").isin([0, 1])
        malformed |= ~valid
    elif kind == "categorical":
        key = str(task["label_key"])
        column = f"pred_{key}"
        if column not in scored:
            raise CompactFrontierError(f"{task['name']}: missing prediction {column}")
        valid = scored[column].astype(str).isin([str(label) for label in task["labels"]])
        malformed |= ~valid
    else:
        raise CompactFrontierError(f"{task['name']}: unknown label kind {kind}")

    scored.loc[malformed & ~parse_error, "parse_error"] = "invalid_prediction_value"
    metrics = _score_malformed_as_incorrect(scored, task)

    if kind == "multi_binary":
        for label in task["labels"]:
            gold = pd.to_numeric(scored[f"gt_{label}"], errors="raise").astype(int)
            scored.loc[malformed, f"pred_{label}"] = 1 - gold.loc[malformed]
    elif kind == "binary":
        key = str(task["label_key"])
        gold = pd.to_numeric(scored[f"gt_{key}"], errors="raise").astype(int)
        scored.loc[malformed, f"pred_{key}"] = 1 - gold.loc[malformed]
    else:
        key = str(task["label_key"])
        scored.loc[malformed, f"pred_{key}"] = "__MALFORMED_INCORRECT__"
    return metrics, scored, malformed


def _headline_f1_draws(
    scored: pd.DataFrame,
    task: Mapping[str, Any],
    indices: np.ndarray,
) -> np.ndarray:
    """Vectorized item bootstrap for one task's configured headline F1."""
    kind = str(task["label_kind"])
    labels = list(task["labels"])

    def binary_f1(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        sampled_gt = gt[indices]
        sampled_pred = pred[indices]
        tp = ((sampled_gt == 1) & (sampled_pred == 1)).sum(axis=1)
        fp = ((sampled_gt == 0) & (sampled_pred == 1)).sum(axis=1)
        fn = ((sampled_gt == 1) & (sampled_pred == 0)).sum(axis=1)
        denominator = 2 * tp + fp + fn
        return np.divide(
            2 * tp,
            denominator,
            out=np.zeros(len(indices), dtype=float),
            where=denominator != 0,
        )

    if kind == "binary":
        key = str(task["label_key"])
        gt = pd.to_numeric(scored[f"gt_{key}"], errors="raise").to_numpy(dtype=np.int8)
        pred = pd.to_numeric(scored[f"pred_{key}"], errors="raise").to_numpy(dtype=np.int8)
        return binary_f1(gt, pred)
    if kind == "multi_binary":
        per_label = []
        for label in labels:
            gt = pd.to_numeric(scored[f"gt_{label}"], errors="raise").to_numpy(
                dtype=np.int8
            )
            pred = pd.to_numeric(scored[f"pred_{label}"], errors="raise").to_numpy(
                dtype=np.int8
            )
            per_label.append(binary_f1(gt, pred))
        return np.mean(np.stack(per_label, axis=1), axis=1)
    if kind == "categorical":
        key = str(task["label_key"])
        gt = scored[f"gt_{key}"].astype(str).to_numpy()
        pred = scored[f"pred_{key}"].astype(str).to_numpy()
        sampled_gt = gt[indices]
        sampled_pred = pred[indices]
        per_label = []
        for label in labels:
            label = str(label)
            tp = ((sampled_gt == label) & (sampled_pred == label)).sum(axis=1)
            fp = ((sampled_gt != label) & (sampled_pred == label)).sum(axis=1)
            fn = ((sampled_gt == label) & (sampled_pred != label)).sum(axis=1)
            denominator = 2 * tp + fp + fn
            per_label.append(
                np.divide(
                    2 * tp,
                    denominator,
                    out=np.zeros(len(indices), dtype=float),
                    where=denominator != 0,
                )
            )
        return np.mean(np.stack(per_label, axis=1), axis=1)
    raise CompactFrontierError(f"unknown label kind {kind}")


def bootstrap_indices(
    benchmark: PanelSelection,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    if iterations < 100:
        raise CompactFrontierError("bootstrap iterations must be at least 100")
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for task in benchmark.tasks:
        name = str(task["name"])
        n = len(task["loader"]())
        dtype = np.uint16 if n <= np.iinfo(np.uint16).max else np.uint32
        draws[name] = rng.integers(0, n, size=(iterations, n), dtype=dtype)
    return draws


def _empty_summary_row(
    checkpoint: Mapping[str, Any], benchmark: PanelSelection
) -> dict[str, Any]:
    result = checkpoint.get("result") or {}
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "display_name": checkpoint["display_name"],
        "series": checkpoint["series"],
        "series_label": SERIES_LABELS[str(checkpoint["series"])],
        "provider": checkpoint["provider"],
        "model_id": checkpoint["model_id"],
        "immutable_identity": checkpoint["identity"],
        "plot_date": checkpoint["plot_date"],
        "date_basis": checkpoint["date_basis"],
        "release_source": checkpoint["release_source"],
        "quantization": checkpoint["quantization"],
        "reasoning": checkpoint.get("reasoning", "disabled"),
        "result_status": checkpoint["result_status"],
        "failure_reason": checkpoint.get("failure_reason", ""),
        "evidence_source": result.get("source_protocol", ""),
        "benchmark_commit": benchmark.benchmark_commit,
        "manifest_id": benchmark.panel_id,
        "manifest_sha256": benchmark.panel_sha256,
        "scorer_version": benchmark.scorer_version,
        "expected_tasks": benchmark.expected_tasks,
        "expected_items": benchmark.expected_items,
        "observed_tasks": np.nan,
        "observed_items": np.nan,
        "malformed_count": np.nan,
        "malformed_rate": np.nan,
        "reliability_pass": False,
        "mean_f1": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "bootstrap_se": np.nan,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "observed_peak_gpu_memory_mib": np.nan,
        "observed_gpu_capacity_mib": np.nan,
        "identity_verified": False,
        "coverage_audit": "not_run",
        "provenance_audit": "not_run",
        "predictions_sha256": "",
        "metadata_sha256": "",
        "evidence_sha256": "",
        "frontier_eligible": False,
        "frontier_setter": False,
        "frontier_score_after_checkpoint": np.nan,
    }


def analyze_checkpoint(
    checkpoint: Mapping[str, Any],
    benchmark: PanelSelection,
    shared_indices: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    row = _empty_summary_row(checkpoint, benchmark)
    if checkpoint["result_status"] != "completed":
        return row, [], None

    result = checkpoint["result"]
    predictions_path = resolve_repo_path(result["predictions_path"])
    predictions = pd.read_csv(predictions_path, low_memory=False)
    selected = _validate_compact_rows(predictions, benchmark, checkpoint)
    if checkpoint["series"] == "open_hive":
        metadata, metadata_sha = _validate_hive_evidence(
            checkpoint, result, predictions_path, benchmark
        )
    else:
        metadata, metadata_sha = _validate_api_evidence(
            checkpoint, result, predictions_path, benchmark
        )

    task_rows: list[dict[str, Any]] = []
    task_draws: list[np.ndarray] = []
    malformed_total = 0
    tasks = {str(task["name"]): task for task in benchmark.tasks}
    for task_name in benchmark.task_names:
        task = tasks[task_name]
        group = selected.loc[selected["task"].astype(str) == task_name].copy()
        metrics, scored, malformed = _score_task_with_malformed_predictions(group, task)
        malformed_count = int(malformed.sum())
        malformed_total += malformed_count
        draws = _headline_f1_draws(scored, task, shared_indices[task_name])
        task_draws.append(draws)
        task_rows.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "display_name": checkpoint["display_name"],
                "series": checkpoint["series"],
                "plot_date": checkpoint["plot_date"],
                "task": task_name,
                "n": int(metrics["n"]),
                "malformed_count": malformed_count,
                "malformed_rate": malformed_count / len(group),
                "headline_f1": float(metrics["headline_f1"]),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "manifest_sha256": benchmark.panel_sha256,
                "scorer_version": benchmark.scorer_version,
            }
        )

    overall_draws = np.mean(np.stack(task_draws, axis=1), axis=1)
    mean_f1 = float(np.mean([task_row["headline_f1"] for task_row in task_rows]))
    malformed_rate = malformed_total / benchmark.expected_items
    reliability_pass = passes_reliability_gate(malformed_rate, benchmark)
    if checkpoint["series"] == "open_hive":
        peak, capacity = _measured_hive_gpu(metadata, str(checkpoint["checkpoint_id"]))
    else:
        peak, capacity = np.nan, np.nan
    row.update(
        {
            "observed_tasks": benchmark.expected_tasks,
            "observed_items": benchmark.expected_items,
            "malformed_count": malformed_total,
            "malformed_rate": malformed_rate,
            "reliability_pass": reliability_pass,
            "mean_f1": mean_f1,
            "ci_low": float(np.quantile(overall_draws, 0.025)),
            "ci_high": float(np.quantile(overall_draws, 0.975)),
            "bootstrap_se": float(np.std(overall_draws, ddof=1)),
            "observed_peak_gpu_memory_mib": peak,
            "observed_gpu_capacity_mib": capacity,
            "identity_verified": True,
            "coverage_audit": (
                f"passed_exact_{benchmark.expected_tasks}_task_"
                f"{benchmark.expected_items}_item_keys_and_gold"
            ),
            "provenance_audit": "passed_prompt_schema_input_model_and_hash_audit",
            "predictions_sha256": file_sha256(predictions_path),
            "metadata_sha256": metadata_sha,
        }
    )
    hardware_pass = checkpoint["series"] == "api" or (
        pd.notna(peak)
        and int(peak) <= HIVE_GPU_CAPACITY_MIB
        and int(capacity) == HIVE_GPU_CAPACITY_MIB
    )
    row["frontier_eligible"] = bool(
        reliability_pass and row["identity_verified"] and hardware_pass
    )
    row["evidence_sha256"] = canonical_sha256(
        {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "identity": checkpoint["identity"],
            "manifest_sha256": benchmark.panel_sha256,
            "predictions_sha256": row["predictions_sha256"],
            "metadata_sha256": metadata_sha,
            "mean_f1": mean_f1,
        }
    )
    return row, task_rows, overall_draws


def apply_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for series in SERIES_ORDER:
        frontier: float | None = None
        group = result.loc[result["series"] == series].sort_values("plot_date")
        for index in group.index:
            if bool(result.at[index, "frontier_eligible"]):
                score = float(result.at[index, "mean_f1"])
                if frontier is None or score > frontier:
                    frontier = score
                    result.at[index, "frontier_setter"] = True
            if frontier is not None:
                result.at[index, "frontier_score_after_checkpoint"] = frontier
    return result.sort_values(["plot_date", "series", "checkpoint_id"]).reset_index(drop=True)


def task_sensitivity_frame(
    summary: pd.DataFrame,
    by_task: pd.DataFrame,
    item_draws: Mapping[str, np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Compare each checkpoint with its series baseline across tasks and items."""
    columns = [
        "series",
        "checkpoint_id",
        "baseline_checkpoint_id",
        "task_count",
        "mean_f1_delta",
        "paired_item_ci_low",
        "paired_item_ci_high",
        "paired_item_probability_positive",
        "task_bootstrap_ci_low",
        "task_bootstrap_ci_high",
        "task_bootstrap_probability_positive",
        "leave_one_task_out_min_delta",
        "leave_one_task_out_max_delta",
        "tasks_better",
        "tasks_tied",
        "tasks_worse",
    ]
    if by_task.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    eligible = summary.loc[summary["frontier_eligible"]].copy()
    for series in SERIES_ORDER:
        group = eligible.loc[eligible["series"] == series].sort_values("plot_date")
        if group.empty:
            continue
        baseline_id = str(group.iloc[0]["checkpoint_id"])
        baseline_tasks = (
            by_task.loc[by_task["checkpoint_id"] == baseline_id]
            .set_index("task")["headline_f1"]
            .sort_index()
        )
        for checkpoint_id in group["checkpoint_id"].astype(str):
            candidate_tasks = (
                by_task.loc[by_task["checkpoint_id"] == checkpoint_id]
                .set_index("task")["headline_f1"]
                .sort_index()
            )
            if not candidate_tasks.index.equals(baseline_tasks.index):
                raise CompactFrontierError(
                    f"{checkpoint_id}: task sensitivity coverage differs from baseline"
                )
            differences = (candidate_tasks - baseline_tasks).to_numpy(dtype=float)
            n_tasks = len(differences)
            sampled = rng.integers(0, n_tasks, size=(iterations, n_tasks))
            task_bootstrap = differences[sampled].mean(axis=1)
            if n_tasks > 1:
                leave_one_out = np.array(
                    [np.delete(differences, index).mean() for index in range(n_tasks)]
                )
            else:
                leave_one_out = differences.copy()
            paired_item = item_draws[checkpoint_id] - item_draws[baseline_id]
            rows.append(
                {
                    "series": series,
                    "checkpoint_id": checkpoint_id,
                    "baseline_checkpoint_id": baseline_id,
                    "task_count": n_tasks,
                    "mean_f1_delta": float(differences.mean()),
                    "paired_item_ci_low": float(np.quantile(paired_item, 0.025)),
                    "paired_item_ci_high": float(np.quantile(paired_item, 0.975)),
                    "paired_item_probability_positive": float(np.mean(paired_item > 0)),
                    "task_bootstrap_ci_low": float(np.quantile(task_bootstrap, 0.025)),
                    "task_bootstrap_ci_high": float(np.quantile(task_bootstrap, 0.975)),
                    "task_bootstrap_probability_positive": float(
                        np.mean(task_bootstrap > 0)
                    ),
                    "leave_one_task_out_min_delta": float(leave_one_out.min()),
                    "leave_one_task_out_max_delta": float(leave_one_out.max()),
                    "tasks_better": int((differences > 0).sum()),
                    "tasks_tied": int((differences == 0).sum()),
                    "tasks_worse": int((differences < 0).sum()),
                }
            )
    # Broad18 retains the pre-extension Llama 3.1 anchor comparison used by
    # the robustness analysis. Replaying the original post-August roster order
    # preserves its frozen task-bootstrap draws after adding earlier models.
    if int(summary["expected_tasks"].dropna().iloc[0]) == 18:
        anchor_id = "llama3_1_70b_instruct_fp8_dynamic_hive"
        candidate_id = "qwen3_6_27b_fp8_hive"
        open_group = eligible.loc[eligible["series"] == "open_hive"].sort_values("plot_date")
        historical_extension_ids = {
            "qwen1_5_72b_chat_awq_hive", "qwen1_5_32b_chat_hive", "qwen2_72b_instruct_awq_hive",
            "llama3_70b_instruct_fp8_hive", "llama3_1_nemotron_70b_fp8_dynamic_hive",
        }
        ids = [
            checkpoint_id
            for checkpoint_id in open_group["checkpoint_id"].astype(str)
            if checkpoint_id not in historical_extension_ids
        ]
        if anchor_id in ids and candidate_id in ids:
            anchor_pos, candidate_pos = ids.index(anchor_id), ids.index(candidate_id)
            anchor_tasks = by_task.loc[by_task["checkpoint_id"] == anchor_id].set_index("task")["headline_f1"].sort_index()
            special_rng = np.random.default_rng(seed)
            special = None
            differences = None
            for checkpoint_id in ids[anchor_pos : candidate_pos + 1]:
                candidate_tasks = by_task.loc[by_task["checkpoint_id"] == checkpoint_id].set_index("task")["headline_f1"].sort_index()
                differences = (candidate_tasks - anchor_tasks).to_numpy(dtype=float)
                sampled = special_rng.integers(0, len(differences), size=(iterations, len(differences)))
                special = differences[sampled].mean(axis=1)
            assert special is not None and differences is not None
            leave_one_out = np.array([np.delete(differences, i).mean() for i in range(len(differences))])
            paired_item = item_draws[candidate_id] - item_draws[anchor_id]
            rows.append({
                "series": "open_hive", "checkpoint_id": candidate_id,
                "baseline_checkpoint_id": anchor_id, "task_count": len(differences),
                "mean_f1_delta": float(differences.mean()),
                "paired_item_ci_low": float(np.quantile(paired_item, 0.025)),
                "paired_item_ci_high": float(np.quantile(paired_item, 0.975)),
                "paired_item_probability_positive": float(np.mean(paired_item > 0)),
                "task_bootstrap_ci_low": float(np.quantile(special, 0.025)),
                "task_bootstrap_ci_high": float(np.quantile(special, 0.975)),
                "task_bootstrap_probability_positive": float(np.mean(special > 0)),
                "leave_one_task_out_min_delta": float(leave_one_out.min()),
                "leave_one_task_out_max_delta": float(leave_one_out.max()),
                "tasks_better": int((differences > 0).sum()),
                "tasks_tied": int((differences == 0).sum()),
                "tasks_worse": int((differences < 0).sum()),
            })
    return pd.DataFrame(rows, columns=columns)


def _step_coordinates(
    group: pd.DataFrame, end_date: pd.Timestamp
) -> tuple[list[pd.Timestamp], list[float]]:
    setters = group.loc[group["frontier_setter"]].sort_values("plot_date")
    if setters.empty:
        return [], []
    dates = list(pd.to_datetime(setters["plot_date"]))
    values = [float(value) for value in setters["mean_f1"]]
    return dates + [end_date], values + [values[-1]]


def render_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    registry: Mapping[str, Any],
    *,
    iterations: int,
    seed: int,
) -> None:
    start = pd.Timestamp(registry["window"]["start"])
    end = pd.Timestamp(registry["window"]["end"])
    measured = summary.loc[summary["frontier_eligible"]]
    if measured.empty:
        y_min, y_max = 0.45, 0.75
    else:
        y_min = max(0.0, float(measured["ci_low"].min()) - 0.04)
        y_max = min(1.0, float(measured["ci_high"].max()) + 0.04)
        if y_max - y_min < 0.18:
            midpoint = (y_min + y_max) / 2
            y_min, y_max = max(0, midpoint - 0.09), min(1, midpoint + 0.09)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    for series in SERIES_ORDER:
        group = summary.loc[summary["series"] == series].sort_values("plot_date")
        dates, values = _step_coordinates(group, end)
        if dates:
            ax.step(
                dates,
                values,
                where="post",
                linewidth=2.5,
                color=SERIES_COLORS[series],
                label=SERIES_LABELS[series],
                zorder=2,
            )
        else:
            ax.plot([], [], linewidth=2.5, color=SERIES_COLORS[series], label=SERIES_LABELS[series])
        completed = group.loc[group["frontier_eligible"]]
        if not completed.empty:
            ax.errorbar(
                pd.to_datetime(completed["plot_date"]),
                completed["mean_f1"],
                yerr=np.vstack(
                    [
                        completed["mean_f1"].to_numpy() - completed["ci_low"].to_numpy(),
                        completed["ci_high"].to_numpy() - completed["mean_f1"].to_numpy(),
                    ]
                ),
                fmt="o",
                color=SERIES_COLORS[series],
                ecolor=SERIES_COLORS[series],
                capsize=3,
                markersize=6,
                zorder=3,
            )
        if completed.empty:
            ax.text(
                end,
                y_min + (0.03 if series == "api" else 0.06),
                f"{SERIES_LABELS[series]}: pending",
                ha="right",
                va="bottom",
                color=SERIES_COLORS[series],
                fontsize=9,
            )

    ax.set_xlim(start, end)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel(f"Mean F1 across {int(summary['expected_tasks'].max())} equally weighted tasks")
    ax.set_xlabel("Artifact or immutable snapshot publication date")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(str(registry["public_label"]), x=0.07, y=0.98, ha="left", fontsize=15, fontweight="bold")
    ax.set_title(str(registry["public_caveat"]), loc="left", fontsize=10, color="#555555", pad=12)
    fig.text(
        0.07,
        0.01,
        f"Points show percentile 95% item-bootstrap intervals ({iterations:,} draws; "
        f"seed {seed}), conditional on the fixed tasks.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.99, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {})):
        final = output_dir / f"frontier_staircase.{suffix}"
        temporary = output_dir / f".frontier_staircase.{os.getpid()}.{suffix}"
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, final)
    plt.close(fig)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_outputs(
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    iterations: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = load_panel_manifest(manifest_path)
    if iterations is None:
        iterations = int(benchmark.bootstrap["iterations"])
    if seed is None:
        seed = int(benchmark.bootstrap["seed"])
    registry = load_compact_registry(registry_path, benchmark)
    indices = bootstrap_indices(benchmark, iterations=iterations, seed=seed)
    summary_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    item_draws: dict[str, np.ndarray] = {}
    for checkpoint in registry["checkpoints"]:
        summary, tasks, draws = analyze_checkpoint(checkpoint, benchmark, indices)
        summary["bootstrap_iterations"] = iterations
        summary["bootstrap_seed"] = seed
        summary_rows.append(summary)
        task_rows.extend(tasks)
        if draws is not None:
            item_draws[str(checkpoint["checkpoint_id"])] = draws
    summary = apply_frontier(pd.DataFrame(summary_rows))
    by_task = pd.DataFrame(task_rows)
    if not by_task.empty:
        by_task = by_task.sort_values(["plot_date", "series", "checkpoint_id", "task"])
    _write_csv_atomic(summary, output_dir / "checkpoint_summary.csv")
    _write_csv_atomic(by_task, output_dir / "model_by_task.csv")
    sensitivity = task_sensitivity_frame(
        summary,
        by_task,
        item_draws,
        iterations=iterations,
        seed=seed,
    )
    _write_csv_atomic(sensitivity, output_dir / "task_sensitivity.csv")
    render_figure(summary, output_dir, registry, iterations=iterations, seed=seed)
    return summary, by_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    args = parser.parse_args()
    summary, by_task = build_outputs(
        args.compact_manifest,
        args.checkpoint_registry,
        args.output_dir,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    measured = int(summary["frontier_eligible"].sum())
    print(
        f"wrote {len(summary)} checkpoints, {len(by_task)} model-task scores, "
        f"and {measured} eligible results to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
