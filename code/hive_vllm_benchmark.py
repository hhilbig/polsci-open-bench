#!/usr/bin/env python3
"""Run one revision-pinned benchmark model with offline vLLM on Hive.

The runner keeps the public benchmark outputs untouched. It writes one atomic
CSV per task under a run-specific sidecar directory and skips a task on restart
only when its exact item coverage is already complete. The merged predictions
file follows the schema used by ``code/build_summary.py``.

Examples
--------
Plan without importing vLLM::

    python3 code/hive_vllm_benchmark.py \
      --config experiments/hive_model_bakeoff_20260804.yaml \
      --model-key qwen3_6_35b_a3b_fp8 --plan-only

Run a structural pilot::

    python3 code/hive_vllm_benchmark.py \
      --config experiments/hive_model_bakeoff_20260804.yaml \
      --model-key qwen3_6_35b_a3b_fp8 \
      --only-task brandt_political_relevance --limit-items 16 \
      --output-dir output/sidecar/hive_model_bakeoff_20260804_pilot/qwen3_6_35b_a3b_fp8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import yaml

from benchmark import parse_content
from panel_manifest import (
    PanelSelection,
    assert_panel_checkout,
    assert_panel_commit,
    benchmark_requests,
    load_panel_manifest,
)
from task_registry import load_task_definitions


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

ALLOWED_MODEL_LLM_KWARGS = {
    "attention_backend",
    "block_size",
    "enforce_eager",
    "kernel_config",
    "linear_backend",
    "max_num_batched_tokens",
    "max_num_seqs",
    "moe_backend",
    "tensor_parallel_size",
    "tokenizer_mode",
}

ALLOWED_MODEL_MOE_BACKENDS = {
    "auto",
    "cutlass",
    "flashinfer_b12x",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "triton",
}

ALLOWED_MODEL_LINEAR_BACKENDS = {
    "cutlass",
}

ALLOWED_REASONING_EFFORTS = {
    "none",
    "low",
    "medium",
    "high",
}

ALLOWED_INFERENCE_INTERFACES = {
    "offline_chat",
    "vllm_responses_harmony",
}


def _f1_for_label(gold: pd.Series, predicted: pd.Series, label: Any) -> float:
    """Return sklearn-compatible one-vs-rest F1 without a runtime dependency."""
    valid = gold.notna() & predicted.notna()
    gold_positive = gold.loc[valid] == label
    predicted_positive = predicted.loc[valid] == label
    true_positive = int((gold_positive & predicted_positive).sum())
    false_positive = int((~gold_positive & predicted_positive).sum())
    false_negative = int((gold_positive & ~predicted_positive).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def _task_headline_metrics(
    task: dict[str, Any], group: pd.DataFrame, malformed: pd.Series
) -> dict[str, Any]:
    """Score all rows after malformed outputs have been made explicitly wrong."""
    result: dict[str, Any] = {
        "n": int(len(group)),
        "parse_ok": int((~malformed).sum()),
        "parse_err_rate": float(malformed.mean()),
        "mean_latency_s": float(group["latency_s"].mean()),
        "median_latency_s": float(group["latency_s"].median()),
    }
    kind = str(task["label_kind"])
    labels = list(task["labels"])
    if kind == "multi_binary":
        values = []
        for label in labels:
            score = _f1_for_label(
                pd.to_numeric(group[f"gt_{label}"], errors="coerce"),
                pd.to_numeric(group[f"pred_{label}"], errors="coerce"),
                1,
            )
            result[f"f1_{label}"] = score
            values.append(score)
        result["avg_f1"] = float(sum(values) / len(values))
        result["headline_f1"] = result["avg_f1"]
    elif kind == "binary":
        key = str(task["label_key"])
        gold = pd.to_numeric(group[f"gt_{key}"], errors="coerce")
        predicted = pd.to_numeric(group[f"pred_{key}"], errors="coerce")
        score = _f1_for_label(gold, predicted, 1)
        result[f"f1_{key}"] = score
        valid = gold.notna() & predicted.notna()
        result["accuracy"] = (
            float((gold.loc[valid] == predicted.loc[valid]).mean())
            if valid.any()
            else float("nan")
        )
        result["headline_f1"] = score
    elif kind == "categorical":
        key = str(task["label_key"])
        gold = group[f"gt_{key}"]
        predicted = group[f"pred_{key}"]
        values = []
        for label in labels:
            score = _f1_for_label(gold, predicted, label)
            result[f"f1_{label}"] = score
            values.append(score)
        result["avg_f1"] = float(sum(values) / len(values))
        valid = gold.notna() & predicted.notna()
        result["accuracy"] = (
            float((gold.loc[valid] == predicted.loc[valid]).mean())
            if valid.any()
            else float("nan")
        )
        result["headline_f1"] = result["avg_f1"]
    else:
        raise BakeoffError(f"{task['name']}: unknown label kind {kind}")
    return result


def task_metrics_frame(
    predictions: pd.DataFrame,
    tasks: list[dict[str, Any]],
    model_key: str,
) -> pd.DataFrame:
    """Build the auditable per-task metric index consumed by the frontier builder.

    Convention note. This scorer counts a malformed response as WRONG: binary and
    multi-binary predictions are flipped away from gold, and categorical ones get
    a sentinel that cannot match. `code/build_summary.py` uses the opposite
    convention and DROPS unparseable rows before scoring. The two agree only while
    the parse-error rate is zero, which held for every model in the frozen
    bake-off, so published bake-off numbers were unaffected.

    They will not agree for the unconstrained-decoding arm
    (`generation.structured_outputs: none`), whose whole point is to produce parse
    failures. Score that comparison from the raw predictions with
    `code/scoring.py` so both arms use one convention, and report the parse-error
    rate alongside; do not compare a number from this function against a number
    from `output/summary.csv`.

    This function also averages categorical macro F1 over every manifest label,
    including labels with no gold support. That matches how the published v1
    summaries were computed and is kept for continuity with the frozen bake-off
    results; `code/scoring.py` is the corrected implementation.
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        group = predictions.loc[
            (predictions["task"].astype(str) == str(task["name"]))
            & (predictions["model"].astype(str) == model_key)
        ].copy()
        if group.empty:
            raise BakeoffError(f"{model_key}/{task['name']}: no predictions to score")
        malformed = ~(
            group["parse_error"].isna()
            | (group["parse_error"].astype(str).str.strip() == "")
        )
        kind = str(task["label_kind"])
        if kind == "multi_binary":
            for label in task["labels"]:
                gold = pd.to_numeric(
                    group[f"gt_{label}"], errors="raise"
                ).astype(int)
                group.loc[malformed, f"pred_{label}"] = 1 - gold.loc[malformed]
        elif kind == "binary":
            key = str(task["label_key"])
            gold = pd.to_numeric(group[f"gt_{key}"], errors="raise").astype(int)
            group.loc[malformed, f"pred_{key}"] = 1 - gold.loc[malformed]
        elif kind == "categorical":
            key = str(task["label_key"])
            group.loc[malformed, f"pred_{key}"] = "__MALFORMED_INCORRECT__"
        else:
            raise BakeoffError(f"{task['name']}: unknown label kind {kind}")
        metrics = _task_headline_metrics(task, group, malformed)
        rows.append(
            {
                "task": task["name"],
                "model": model_key,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "task"]).reset_index(drop=True)


class BakeoffError(RuntimeError):
    """Raised when the run would violate a frozen benchmark invariant."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_compact_execution(config: dict[str, Any], path: Path) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise BakeoffError(f"{path} execution must be a mapping")
    protocol = execution.get("protocol")
    if protocol not in {"compact_direct", "broad18_targeted_ablation"}:
        raise BakeoffError(
            "compact execution protocol must be compact_direct or broad18_targeted_ablation"
        )
    if "promotion_gate" in config:
        raise BakeoffError("compact configs cannot define a legacy promotion_gate")

    expected_items = int(execution.get("expected_items", -1))
    pilot_items = int(execution.get("pilot_items", -1))
    remainder_items = int(execution.get("remainder_items", -1))
    if expected_items < 1 or pilot_items < 1:
        raise BakeoffError("compact expected_items and pilot_items must be positive")
    if pilot_items + remainder_items != expected_items:
        raise BakeoffError(
            "compact pilot_items and remainder_items must partition expected_items"
        )
    if expected_items != int(config["task_scope"]["expected_items"]):
        raise BakeoffError("compact execution and task_scope item counts disagree")
    if execution.get("pilot_selection") != "first_requests_in_manifest_order":
        raise BakeoffError("compact pilot must use the first requests in manifest order")
    if not bool(execution.get("combine_pilot_with_remainder", False)):
        raise BakeoffError("compact execution must combine pilot and remainder rows")
    if bool(execution.get("rerun_pilot_items", True)):
        raise BakeoffError("compact execution prohibits rerunning pilot items")
    if bool(execution.get("selective_retries", True)):
        raise BakeoffError("compact execution prohibits selective retries")
    if execution.get("partition_selection") != "live_sbatch_test":
        raise BakeoffError("compact partition selection must use a live sbatch test")
    malformed_limit = float(execution.get("max_malformed_rate", -1))
    if malformed_limit != 0.05:
        raise BakeoffError("compact max_malformed_rate must be 0.05")
    if not bool(execution.get("exactly_at_malformed_limit_passes", False)):
        raise BakeoffError("the compact 5% malformed boundary must pass")
    if execution.get("required_gpu_gres") != "6000_blackwell:1":
        raise BakeoffError("compact runs require one typed 6000_blackwell GPU")
    if int(execution.get("expected_gpu_capacity_mib", -1)) != 97887:
        raise BakeoffError("compact runs require the audited 97,887 MiB capacity")
    if not bool(execution.get("reject_oom", False)):
        raise BakeoffError("compact runs must reject OOM failures")
    if not bool(execution.get("reject_over_capacity", False)):
        raise BakeoffError("compact runs must reject over-capacity observations")
    if not bool(execution.get("require_exact_snapshot_path", False)):
        raise BakeoffError("compact runs must require an exact snapshot path")
    if not bool(execution.get("require_offline_inference", False)):
        raise BakeoffError("compact runs must require offline inference")
    output_root = Path(str(execution.get("output_root", "")))
    panel_id = str(execution.get("panel_id", ""))
    if protocol == "broad18_targeted_ablation":
        required_root = Path("output/sidecar/frontier_2026/broad18/targeted_ablation")
    elif panel_id == "frontier_broad_18":
        required_root = Path("output/sidecar/frontier_2026/broad18")
    elif panel_id == "frontier_compact_8":
        required_root = Path("output/sidecar/frontier_2026/compact8")
    else:
        raise BakeoffError(f"unsupported compact panel_id {panel_id!r}")
    try:
        output_root.relative_to(required_root)
    except ValueError as exc:
        raise BakeoffError(
            f"compact output_root must be under {required_root}"
        ) from exc

    models = config["models"]
    if len(models) != int(execution.get("expected_models", -1)):
        raise BakeoffError("compact model count does not match expected_models")
    dates: list[date] = []
    for model_key, model in models.items():
        for field in [
            "artifact_publication_date",
            "artifact_publication_source",
            "quantization",
        ]:
            if not model.get(field):
                raise BakeoffError(f"compact model {model_key!r} is missing {field}")
        try:
            dates.append(date.fromisoformat(str(model["artifact_publication_date"])))
        except ValueError as exc:
            raise BakeoffError(
                f"compact model {model_key!r} has an invalid artifact publication date"
            ) from exc
    if dates != sorted(dates):
        raise BakeoffError("compact models must be ordered by artifact publication date")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise BakeoffError(f"{path} did not parse to a mapping")
    for key in ["run_id", "task_scope", "runtime", "generation", "models"]:
        if key not in config:
            raise BakeoffError(f"{path} is missing required field: {key}")
    if "execution" not in config and "promotion_gate" not in config:
        raise BakeoffError(
            f"{path} must define either compact execution or legacy promotion_gate"
        )
    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise BakeoffError(f"{path} models must be a non-empty mapping")
    aliases = set()
    for model_key, model in models.items():
        for field in ["model_id", "revision", "display_name", "role"]:
            if field not in model:
                raise BakeoffError(f"model {model_key!r} is missing {field}")
        revision = str(model["revision"])
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
            raise BakeoffError(f"model {model_key!r} revision is not a 40-character commit hash")
        llm_kwargs = model.get("llm_kwargs", {})
        if not isinstance(llm_kwargs, Mapping):
            raise BakeoffError(f"model {model_key!r} llm_kwargs must be a mapping")
        unknown_llm_kwargs = set(llm_kwargs) - ALLOWED_MODEL_LLM_KWARGS
        if unknown_llm_kwargs:
            raise BakeoffError(
                f"model {model_key!r} has unsupported llm_kwargs: "
                f"{sorted(unknown_llm_kwargs)}"
            )
        tensor_parallel = int(llm_kwargs.get("tensor_parallel_size", 1))
        refresh_multi_gpu = bool(config.get("refresh_sample")) and config.get("hardware", {}).get("nodes") == 1 and config.get("hardware", {}).get("gpu_count") == tensor_parallel
        if tensor_parallel < 1 or (tensor_parallel != 1 and not refresh_multi_gpu):
            raise BakeoffError(
                f"model {model_key!r} must use tensor_parallel_size=1 in this one-GPU bake-off"
            )
        for field in ["max_num_batched_tokens", "max_num_seqs", "block_size"]:
            if field in llm_kwargs and int(llm_kwargs[field]) < 1:
                raise BakeoffError(f"model {model_key!r} {field} must be positive")
        if "enforce_eager" in llm_kwargs and not isinstance(llm_kwargs["enforce_eager"], bool):
            raise BakeoffError(f"model {model_key!r} enforce_eager must be boolean")
        if "kernel_config" in llm_kwargs:
            kernel_config = llm_kwargs["kernel_config"]
            if (not isinstance(kernel_config, Mapping)
                    or set(kernel_config) != {"enable_flashinfer_autotune", "moe_backend"}
                    or type(kernel_config["enable_flashinfer_autotune"]) is not bool
                    or type(kernel_config["moe_backend"]) is not str
                    or kernel_config["moe_backend"] not in ALLOWED_MODEL_MOE_BACKENDS):
                raise BakeoffError(
                    f"model {model_key!r} kernel_config must pin boolean "
                    "enable_flashinfer_autotune and a supported moe_backend"
                )
        if (
            "moe_backend" in llm_kwargs
            and str(llm_kwargs["moe_backend"]) not in ALLOWED_MODEL_MOE_BACKENDS
        ):
            raise BakeoffError(
                f"model {model_key!r} moe_backend must be one of "
                f"{sorted(ALLOWED_MODEL_MOE_BACKENDS)}"
            )
        if (
            "linear_backend" in llm_kwargs
            and str(llm_kwargs["linear_backend"])
            not in ALLOWED_MODEL_LINEAR_BACKENDS
        ):
            raise BakeoffError(
                f"model {model_key!r} linear_backend must be one of "
                f"{sorted(ALLOWED_MODEL_LINEAR_BACKENDS)}"
            )
        chat_template_kwargs = model.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, Mapping):
            raise BakeoffError(
                f"model {model_key!r} chat_template_kwargs must be a mapping"
            )
        execution_protocol = str(config.get("execution", {}).get("protocol", ""))
        is_targeted_ablation = execution_protocol == "broad18_targeted_ablation"
        if (
            chat_template_kwargs.get("enable_thinking") not in {None, False}
            and not is_targeted_ablation
        ):
            raise BakeoffError(
                f"model {model_key!r} cannot enable thinking in the controlled bake-off"
            )
        conversation_layout = str(model.get("conversation_layout", "system_user"))
        if conversation_layout not in {"system_user", "single_user"}:
            raise BakeoffError(
                f"model {model_key!r} conversation_layout must be system_user or single_user"
            )
        if conversation_layout != "system_user" and not is_targeted_ablation:
            raise BakeoffError(
                f"model {model_key!r} may change conversation layout only in a targeted ablation"
            )
        if chat_template_kwargs.get("reasoning_effort") not in {
            None,
            *ALLOWED_REASONING_EFFORTS,
        }:
            raise BakeoffError(
                f"model {model_key!r} reasoning_effort must be one of "
                f"{sorted(ALLOWED_REASONING_EFFORTS)} when supplied"
            )
        inference_interface = str(
            model.get("inference_interface", "offline_chat")
        )
        if inference_interface not in ALLOWED_INFERENCE_INTERFACES:
            raise BakeoffError(
                f"model {model_key!r} inference_interface must be one of "
                f"{sorted(ALLOWED_INFERENCE_INTERFACES)}"
            )
        if inference_interface == "vllm_responses_harmony":
            if not str(model.get("model_id", "")).startswith("openai/gpt-oss-"):
                raise BakeoffError(
                    "vllm_responses_harmony is restricted to revision-pinned GPT-OSS"
                )
            if chat_template_kwargs.get("reasoning_effort") not in {
                "low",
                "medium",
                "high",
            }:
                raise BakeoffError(
                    "GPT-OSS Responses/Harmony runs require an explicit reasoning_effort"
                )
        if "launch_eligible" in model and not isinstance(model["launch_eligible"], bool):
            raise BakeoffError(f"model {model_key!r} launch_eligible must be boolean")
        alias = str(model_key)
        if alias in aliases:
            raise BakeoffError(f"duplicate model key: {alias}")
        aliases.add(alias)
    generation = config["generation"]
    if float(generation["temperature"]) != 0.0:
        raise BakeoffError("the controlled bake-off requires temperature 0")
    if bool(generation.get("enable_thinking", True)):
        raise BakeoffError("the controlled bake-off requires thinking disabled")
    if generation.get("structured_outputs") not in {"json_schema", "none"}:
        # "none" exists only for the constrained-decoding experiment described in
        # docs/task_source_fidelity_audit.md: the published benchmark gave API
        # models server-side schema enforcement and local models nothing, so the
        # local-versus-API gap confounds model quality with harness parity. The
        # experiment runs each checkpoint twice, holding model, GPU, prompt and
        # items fixed and varying only this setting. It must never be the default
        # for a scoring run, so the config has to say "none" explicitly.
        raise BakeoffError(
            "generation.structured_outputs must be 'json_schema', or 'none' for "
            "the deliberate unconstrained-decoding comparison"
        )
    benchmark_commit = str(config.get("benchmark_commit", ""))
    if len(benchmark_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in benchmark_commit.lower()
    ):
        raise BakeoffError("benchmark_commit must be a full 40-character commit hash")
    if "execution" in config:
        _validate_compact_execution(config, path)
    return config


def assert_model_launch_eligible(model_key: str, model: Mapping[str, Any]) -> None:
    if bool(model.get("launch_eligible", True)):
        return
    note = str(model.get("provenance_note") or "artifact provenance gate failed")
    raise BakeoffError(f"model {model_key!r} is ineligible for launch: {note}")


def next_evidence_stage(
    current_stage: str,
    *,
    gate_passed: bool,
    promotion_triggered: bool | None = None,
) -> str:
    """Apply the fixed pilot -> panel -> conditional full-suite state machine."""
    if current_stage == "fit_parser_pilot":
        return "panel18" if gate_passed else "terminal_ineligible"
    if current_stage == "panel18":
        if not gate_passed:
            return "terminal_ineligible"
        if promotion_triggered is None:
            raise BakeoffError("panel18 transition requires a promotion decision")
        return "full34" if promotion_triggered else "terminal_screened"
    if current_stage == "full34":
        return "terminal_confirmed" if gate_passed else "terminal_failed"
    raise BakeoffError(f"unknown evidence stage: {current_stage}")


def evaluate_fit_parser_pilot(
    metadata: Mapping[str, Any],
    predictions: pd.DataFrame,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_rows: int = 16,
    expected_capacity_mib: int = 97887,
    max_malformed_rate: float = 0.05,
    required_gpu_name_fragment: str | None = None,
) -> dict[str, Any]:
    """Audit a completed Hive pilot without omitting malformed rows."""
    gpu_before = metadata.get("gpu_before_load")
    gpus = gpu_before.get("gpus", []) if isinstance(gpu_before, Mapping) else []
    capacities = [gpu.get("memory_total_mib") for gpu in gpus if isinstance(gpu, Mapping)]
    gpu0 = gpus[0] if len(gpus) == 1 and isinstance(gpus[0], Mapping) else {}
    observed_capacity = capacities[0] if len(capacities) == 1 else None
    peak = metadata.get("observed_peak_gpu_memory_used_mib")
    malformed = (
        ~(
            predictions["parse_error"].isna()
            | (predictions["parse_error"].astype(str).str.strip() == "")
        )
        if "parse_error" in predictions
        else pd.Series([True] * len(predictions), dtype=bool)
    )
    malformed_rate = float(malformed.mean()) if len(predictions) else 1.0
    gpu_name = str(gpu0.get("name", ""))
    error_text = str(metadata.get("error", "")).lower()
    checks = {
        "completed": metadata.get("status") == "completed",
        "exact_model_id": metadata.get("model_id") == expected_model_id,
        "exact_revision": metadata.get("revision") == expected_revision,
        "exact_row_count": len(predictions) == expected_rows,
        "one_typed_gpu": (
            len(gpus) == 1
            and bool(gpu_name)
            and (
                required_gpu_name_fragment is None
                or required_gpu_name_fragment.lower() in gpu_name.lower()
            )
        ),
        "exact_capacity": observed_capacity == expected_capacity_mib,
        "peak_within_capacity": (
            peak is not None
            and observed_capacity is not None
            and int(peak) <= int(observed_capacity)
        ),
        "oom_free": "out of memory" not in error_text and "cuda oom" not in error_text,
        "malformed_rate_within_limit": malformed_rate <= max_malformed_rate,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "observed_capacity_mib": observed_capacity,
        "observed_peak_memory_mib": peak,
        "rows": len(predictions),
        "malformed_rows": int(malformed.sum()),
        "malformed_rate": malformed_rate,
        "next_stage": next_evidence_stage(
            "fit_parser_pilot",
            gate_passed=passed,
        ),
    }


def evaluate_compact_pilot(
    metadata: Mapping[str, Any],
    predictions: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply the fit/parser gate without legacy panel or promotion stages."""
    audit = evaluate_fit_parser_pilot(metadata, predictions, **kwargs)
    audit["protocol"] = str(metadata.get("execution_protocol", "compact_direct"))
    audit["next_stage"] = "remainder" if audit["passed"] else "terminal_ineligible"
    return audit


def compact_execution(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    execution = config.get("execution")
    if isinstance(execution, Mapping) and execution.get("protocol") in {
        "compact_direct",
        "broad18_targeted_ablation",
    }:
        return execution
    return None


def malformed_rate_audit(
    predictions: pd.DataFrame,
    *,
    max_malformed_rate: float,
) -> dict[str, Any]:
    """Apply the inclusive malformed-rate gate without dropping any rows."""
    if "parse_error" not in predictions:
        malformed = pd.Series([True] * len(predictions), dtype=bool)
    else:
        malformed = ~(
            predictions["parse_error"].isna()
            | (predictions["parse_error"].astype(str).str.strip() == "")
        )
    rate = float(malformed.mean()) if len(predictions) else 1.0
    return {
        "passed": rate <= max_malformed_rate,
        "rows": int(len(predictions)),
        "malformed_rows": int(malformed.sum()),
        "malformed_rate": rate,
        "max_malformed_rate": float(max_malformed_rate),
        "boundary_is_inclusive": True,
    }


def assert_compact_prediction_coverage(
    predictions: pd.DataFrame,
    panel: PanelSelection,
    *,
    model_key: str,
) -> None:
    """Require exact manifest-order coverage for a completed compact checkpoint."""
    required = {
        "task",
        "model",
        "item_id",
        "panel_sha256",
        "panel_task_fingerprint",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise BakeoffError(f"compact predictions are missing fields: {sorted(missing)}")
    expected_keys = [
        (request.task_name, request.item_id) for request in benchmark_requests(panel)
    ]
    observed_keys = list(
        zip(
            predictions["task"].astype(str),
            predictions["item_id"].astype(str),
        )
    )
    if len(observed_keys) != len(expected_keys):
        raise BakeoffError(
            f"compact coverage has {len(observed_keys)} rows, expected {len(expected_keys)}"
        )
    if len(observed_keys) != len(set(observed_keys)):
        raise BakeoffError("compact predictions contain duplicate task/item keys")
    if observed_keys != expected_keys:
        raise BakeoffError("compact predictions do not match frozen manifest order and keys")
    if set(predictions["model"].astype(str)) != {model_key}:
        raise BakeoffError("compact predictions contain the wrong model key")
    if set(predictions["panel_sha256"].astype(str)) != {panel.panel_sha256}:
        raise BakeoffError("compact predictions contain the wrong manifest hash")
    for task_name, expected_fingerprint in panel.task_fingerprints.items():
        observed = set(
            predictions.loc[
                predictions["task"].astype(str) == task_name,
                "panel_task_fingerprint",
            ].astype(str)
        )
        if observed != {expected_fingerprint}:
            raise BakeoffError(
                f"{task_name}: compact predictions contain the wrong task fingerprint"
            )


def assert_compact_hardware(
    execution: Mapping[str, Any],
    gpu_snapshot: Mapping[str, Any],
) -> None:
    """Fail before model load unless the allocated GPU is the frozen Hive tier."""
    gpus = gpu_snapshot.get("gpus", [])
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise BakeoffError("compact Hive runs require exactly one allocated GPU")
    gpu = gpus[0]
    if not isinstance(gpu, Mapping):
        raise BakeoffError("compact Hive GPU metadata is malformed")
    required_name = str(execution["required_gpu_name_fragment"])
    observed_name = str(gpu.get("name", ""))
    if required_name.lower() not in observed_name.lower():
        raise BakeoffError(
            f"compact Hive GPU type mismatch: expected {required_name!r}, "
            f"observed {observed_name!r}"
        )
    expected_capacity = int(execution["expected_gpu_capacity_mib"])
    observed_capacity = gpu.get("memory_total_mib")
    if observed_capacity is None or int(observed_capacity) != expected_capacity:
        raise BakeoffError(
            "compact Hive GPU capacity mismatch: "
            f"expected {expected_capacity} MiB, observed {observed_capacity}"
        )


def assert_compact_offline_provenance(
    execution: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Verify that exact weights were resolved before inference went offline."""
    environment = os.environ if environ is None else environ
    if bool(execution.get("require_offline_inference", False)):
        for variable in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
            if str(environment.get(variable, "")) != "1":
                raise BakeoffError(
                    f"compact inference requires {variable}=1 after model download"
                )
    if not bool(execution.get("require_exact_snapshot_path", False)):
        return None
    raw_path = str(environment.get("BAKEOFF_MODEL_SNAPSHOT_PATH", "")).strip()
    if not raw_path:
        raise BakeoffError("compact inference is missing BAKEOFF_MODEL_SNAPSHOT_PATH")
    snapshot_path = Path(raw_path)
    if not snapshot_path.is_dir():
        raise BakeoffError(f"model snapshot path does not exist: {snapshot_path}")
    expected_revision = str(model["revision"])
    if snapshot_path.name.lower() != expected_revision.lower():
        raise BakeoffError(
            "model snapshot provenance mismatch: "
            f"expected revision {expected_revision}, observed path {snapshot_path}"
        )
    return snapshot_path.resolve()


def compact_failure_category(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "snapshot" in text or "revision" in text or "provenance" in text:
        return "provenance"
    if "gpu" in text or "capacity" in text or "memory" in text:
        return "hardware"
    if "malformed" in text or "parse" in text:
        return "reliability"
    return "runtime"


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_head_without_binary(repo: Path) -> str:
    """Read the checkout commit when a pinned container lacks the git program."""
    marker = repo / '.git'
    if marker.is_file():
        pointer = marker.read_text().strip()
        if not pointer.startswith('gitdir: '):
            raise ValueError('Unrecognized Git directory pointer')
        git_dir = (repo / pointer[8:]).resolve()
    else:
        git_dir = marker
    head = (git_dir / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:].strip()
        if not ref.startswith('refs/') or '..' in Path(ref).parts:
            raise ValueError('Invalid Git HEAD reference')
        direct = git_dir / ref
        if direct.exists():
            head = direct.read_text().strip()
        else:
            packed = git_dir / 'packed-refs'
            matches = [line.split(' ', 1)[0] for line in packed.read_text().splitlines()
                       if not line.startswith(('#', '^')) and line.endswith(' ' + ref)]
            if len(matches) != 1:
                raise ValueError('Git HEAD reference is missing or ambiguous')
            head = matches[0]
    if len(head) != 40 or any(char not in '0123456789abcdef' for char in head):
        raise ValueError('Git HEAD is not a SHA-1 commit')
    return head


def assert_benchmark_commit(config: dict[str, Any]) -> str:
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
    except FileNotFoundError:
        try:
            observed = read_git_head_without_binary(REPO)
        except (OSError, ValueError) as exc:
            raise BakeoffError(
                'the benchmark checkout has no readable Git HEAD; the frozen commit cannot be verified'
            ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise BakeoffError(
            "the benchmark checkout has no readable Git HEAD; the frozen commit cannot be verified"
        ) from exc
    expected = str(config["benchmark_commit"])
    if observed != expected:
        raise BakeoffError(
            f"benchmark checkout is {observed}, but the frozen config requires {expected}"
        )
    return observed


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_fingerprint(
    task: dict[str, Any],
    items: list[dict[str, Any]],
    model_alias: str,
    model: dict[str, Any],
    config_hash: str,
    generation: dict[str, Any],
    runtime: dict[str, Any],
    runtime_versions: dict[str, Any] | None = None,
    panel_sha256: str | None = None,
    panel_task_fingerprint: str | None = None,
) -> str:
    """Fingerprint everything that can change a task checkpoint's meaning."""
    payload = {
        "task": task["name"],
        "model_alias": model_alias,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "config_sha256": config_hash,
        "generation": generation,
        "runtime": runtime,
        "runtime_versions": runtime_versions,
        "system_prompt": Path(task["prompt_path"]).read_text(),
        "json_schema": task["json_schema"],
        "items": [
            {
                "item_id": str(item["item_id"]),
                "user_content": str(item["user_content"]),
                "gold": item["gt"],
            }
            for item in items
        ],
    }
    if panel_sha256 is not None:
        payload["panel_sha256"] = panel_sha256
        payload["panel_task_fingerprint"] = panel_task_fingerprint
    return _sha256_json(payload)


def selected_tasks(
    config: dict[str, Any],
    only_task: str | None = None,
    panel: PanelSelection | None = None,
) -> list[dict[str, Any]]:
    if panel is None:
        tasks_dir = REPO / config["task_scope"]["tasks_dir"]
        tasks = load_task_definitions(tasks_dir=tasks_dir)
    else:
        tasks = list(panel.tasks)
    if config.get('refresh_sample'):
        if panel is not None:
            raise BakeoffError('Refresh sample cannot be combined with another panel')
        from refresh_pilot import read, digest, validate_sources, ROOT as REFRESH_ROOT
        selection = config['refresh_sample']
        frozen = read(REFRESH_ROOT/'panel.json')
        if digest(frozen) != selection['panel_sha256']:
            raise BakeoffError('Refresh panel fingerprint changed')
        validate_sources()
        stage = selection['stage']
        if stage not in ['pilot', 'remainder', 'backfill']:
            raise BakeoffError('Refresh stage must be pilot, remainder or backfill')
        requested_keys = None
        if stage == 'backfill':
            key_path = REPO / selection['keys_manifest']
            if file_sha256(key_path) != selection['keys_sha256']:
                raise BakeoffError('Refresh backfill key fingerprint changed')
            key_rows = read(key_path)['keys']
            requested_keys = {(str(t), str(i)) for t, i in key_rows}
            frozen_keys = {(r['task'], str(r['item']['item_id'])) for r in frozen['rows']}
            if not requested_keys or len(requested_keys) != len(key_rows) or not requested_keys <= frozen_keys:
                raise BakeoffError('Refresh backfill keys must be a unique nonempty frozen-panel subset')
        selected = []
        for task in tasks:
            records = [r['item'] for r in frozen['rows'] if r['task'] == task['name']
                       and ((r['task'], str(r['item']['item_id'])) in requested_keys
                            if requested_keys is not None else bool(r['pilot']) == (stage == 'pilot'))]
            if stage != 'backfill' and len(records) != (2 if stage == 'pilot' else 98):
                raise BakeoffError('Refresh task coverage changed')
            if not records:
                continue
            selected.append(dict(task, loader=lambda records=records: records, _refresh_strict_schema=True))
        tasks = selected
    if only_task:
        tasks = [task for task in tasks if task["name"] == only_task]
        if not tasks:
            raise BakeoffError(f"unknown task: {only_task}")
    return tasks


def load_task_items(task: dict[str, Any], limit_items: int | None = None) -> list[dict[str, Any]]:
    items = task["loader"]()
    if limit_items is not None:
        if limit_items < 1:
            raise BakeoffError("--limit-items must be positive")
        items = items[:limit_items]
    ids = [str(item["item_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise BakeoffError(f"{task['name']}: duplicate item_id values")
    missing_gold = [
        item["item_id"]
        for item in items
        if any(pd.isna(value) for value in item["gt"].values())
    ]
    if missing_gold:
        raise BakeoffError(
            f"{task['name']}: {len(missing_gold)} items have missing gold labels; "
            f"first={missing_gold[:5]}"
        )
    return items


def validate_scope(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    limit_items: int | None = None,
    only_task: str | None = None,
    expected_scope: tuple[int, int] | None = None,
) -> dict[str, Any]:
    counts = {task["name"]: len(load_task_items(task, limit_items)) for task in tasks}
    total = sum(counts.values())
    if only_task is None and limit_items is None:
        if expected_scope is None:
            expected_tasks = int(config["task_scope"]["expected_tasks"])
            expected_items = int(config["task_scope"]["expected_items"])
        else:
            expected_tasks, expected_items = expected_scope
        if len(tasks) != expected_tasks or total != expected_items:
            raise BakeoffError(
                f"frozen scope mismatch: observed {len(tasks)} tasks/{total} items, "
                f"expected {expected_tasks}/{expected_items}"
            )
    return {"task_counts": counts, "total_tasks": len(tasks), "total_items": total}


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_path(path)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def require_sidecar_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    sidecar_root = (REPO / "output" / "sidecar").resolve()
    try:
        relative = resolved.relative_to(sidecar_root)
    except ValueError as exc:
        raise BakeoffError(
            f"output directory must be under {sidecar_root}; refusing {resolved}"
        ) from exc
    if not relative.parts:
        raise BakeoffError("output directory must be a run-specific child of output/sidecar")
    return resolved


def task_done_path(task_csv_path: Path) -> Path:
    return task_csv_path.with_suffix(".done.json")


def refuse_uncommitted_checkpoint(path: Path, task_name: str) -> None:
    """Preserve generated rows when their completion marker is missing or invalid."""
    if path.exists() or task_done_path(path).exists():
        raise BakeoffError(
            f"{task_name}: existing task output lacks a valid matching completion "
            "marker; refusing to overwrite possible model responses"
        )


def task_checkpoint_complete(
    path: Path,
    task_name: str,
    model_alias: str,
    expected_item_ids: list[str],
    expected_fingerprint: str | None = None,
    expected_panel_sha256: str | None = None,
    expected_panel_task_fingerprint: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    required = {"task", "model", "item_id", "parse_error"}
    if not required.issubset(frame.columns):
        return False
    if len(frame) != len(expected_item_ids):
        return False
    if frame["item_id"].astype(str).duplicated().any():
        return False
    if set(frame["item_id"].astype(str)) != set(expected_item_ids):
        return False
    if set(frame["task"].astype(str)) != {task_name}:
        return False
    if set(frame["model"].astype(str)) != {model_alias}:
        return False
    if expected_panel_sha256 is not None:
        if "panel_sha256" not in frame.columns:
            return False
        if set(frame["panel_sha256"].astype(str)) != {expected_panel_sha256}:
            return False
    if expected_panel_task_fingerprint is not None:
        if "panel_task_fingerprint" not in frame.columns:
            return False
        if set(frame["panel_task_fingerprint"].astype(str)) != {
            expected_panel_task_fingerprint
        }:
            return False
    if expected_fingerprint is not None:
        done_path = task_done_path(path)
        if not done_path.exists():
            return False
        try:
            done = json.loads(done_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if done.get("task_fingerprint") != expected_fingerprint:
            return False
        if (
            expected_panel_sha256 is not None
            and done.get("panel_sha256") != expected_panel_sha256
        ):
            return False
        if (
            expected_panel_task_fingerprint is not None
            and done.get("panel_task_fingerprint")
            != expected_panel_task_fingerprint
        ):
            return False
        if done.get("task") != task_name or done.get("model_key") != model_alias:
            return False
        if int(done.get("rows", -1)) != len(expected_item_ids):
            return False
        checkpoint_hash = done.get("checkpoint_sha256")
        if not checkpoint_hash or checkpoint_hash != file_sha256(path):
            return False
    return True


def build_conversations(
    system_prompt: str,
    items: list[dict[str, Any]],
    conversation_layout: str = "system_user",
) -> list[list[dict[str, str]]]:
    if conversation_layout == "single_user":
        return [
            [
                {
                    "role": "user",
                    "content": f"{system_prompt.rstrip()}\n\n{str(item['user_content'])}",
                }
            ]
            for item in items
        ]
    if conversation_layout != "system_user":
        raise BakeoffError(f"unsupported conversation layout: {conversation_layout}")
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(item["user_content"])},
        ]
        for item in items
    ]


def chat_token_count(
    tokenizer: Any,
    conversation: list[dict[str, str]],
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> int:
    template_kwargs = (
        {"enable_thinking": False}
        if chat_template_kwargs is None
        else dict(chat_template_kwargs)
    )
    token_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise BakeoffError("chat template returned a mapping without input_ids")
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def preflight_prompt_lengths(
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    max_model_len: int,
    max_tokens: int,
    margin_tokens: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    lengths = [
        chat_token_count(tokenizer, conversation, chat_template_kwargs)
        for conversation in conversations
    ]
    budget = max_model_len - max_tokens - margin_tokens
    too_long = [(idx, length) for idx, length in enumerate(lengths) if length > budget]
    if too_long:
        raise BakeoffError(
            f"{len(too_long)} prompts exceed the {budget}-token input budget; "
            f"first={too_long[:5]}. The runner does not silently truncate benchmark inputs."
        )
    return lengths


def structured_output_kwargs(
    schema: dict[str, Any], mode: str = "json_schema"
) -> tuple[dict[str, Any], str]:
    """Sampling kwargs that constrain generation to the task's JSON schema.

    `mode="none"` returns no kwargs, so the model generates freely and the run
    depends on the prompt's own JSON instruction plus the text parser, exactly as
    the published Ollama runs did. That is the comparison arm of the
    constrained-decoding experiment; it is never a default.
    """
    if mode == "none":
        return {}, "none_unconstrained"
    if mode != "json_schema":
        raise BakeoffError(f"unsupported structured-output mode: {mode!r}")
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}, "StructuredOutputsParams"
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}, "GuidedDecodingParams"


def _expand_gpu_indices(raw: str) -> set[int]:
    indices: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token and all(part.isdigit() for part in token.split("-", 1)):
            start, end = (int(part) for part in token.split("-", 1))
            indices.update(range(start, end + 1))
        elif token.isdigit():
            indices.add(int(token))
    return indices


def query_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return {"gpus": [], "total_used_mib": None}
    allocation_raw = (
        os.environ.get("SLURM_STEP_GPUS")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or ""
    )
    allocated_indices = _expand_gpu_indices(allocation_raw)
    allocated_uuids = {
        token.strip() for token in allocation_raw.split(",") if token.strip().startswith("GPU-")
    }
    all_gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, uuid, name, total, used = parts
        all_gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    if allocated_indices:
        selected = [gpu for gpu in all_gpus if gpu["index"] in allocated_indices]
    elif allocated_uuids:
        selected = [gpu for gpu in all_gpus if gpu["uuid"] in allocated_uuids]
    else:
        selected = all_gpus
    filter_miss = bool(allocation_raw and not selected and all_gpus)
    if filter_miss:
        selected = all_gpus
    return {
        "gpus": selected,
        "allocation_env": allocation_raw or None,
        "allocation_filter_miss": filter_miss,
        "total_used_mib": (
            sum(gpu["memory_used_mib"] for gpu in selected) if selected else None
        ),
    }


class GpuMemoryMonitor:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self.max_total_used_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = query_gpu().get("total_used_mib")
            if used is not None:
                self.max_total_used_mib = max(self.max_total_used_mib or 0, int(used))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_versions() -> dict[str, Any]:
    packages = [
        "vllm",
        "torch",
        "transformers",
        "pandas",
        "PyYAML",
        "flashinfer-python",
        "xgrammar",
        "compressed-tensors",
        "cuda-toolkit",
        "nvidia-cuda-cccl",
        "nvidia-cuda-crt",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-nvvm",
    ]
    details: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {name: package_version(name) for name in packages},
        "nvcc": _command_output(["nvcc", "--version"]),
        "nvidia_driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
    }
    lock_path = os.environ.get("BAKEOFF_RUNTIME_LOCK")
    details["runtime_lock"] = (
        {"path": lock_path, "sha256": file_sha256(Path(lock_path))}
        if lock_path and Path(lock_path).is_file()
        else None
    )
    try:
        import torch

        details["torch_cuda"] = torch.version.cuda
    except ImportError:
        details["torch_cuda"] = None
    return details


def assert_runtime_versions(runtime: dict[str, Any], details: dict[str, Any]) -> None:
    expected_vllm = str(runtime["vllm_version"])
    actual_vllm = details["packages"].get("vllm")
    if actual_vllm != expected_vllm:
        raise BakeoffError(
            f"vLLM version mismatch: expected {expected_vllm}, observed {actual_vllm}"
        )
    expected_python = str(runtime["python_version"])
    actual_python = ".".join(details["python"].split(".")[:2])
    if actual_python != expected_python:
        raise BakeoffError(
            f"Python version mismatch: expected {expected_python}, observed {details['python']}"
        )
    expected_toolkit = runtime.get("cuda_toolkit_meta_version")
    if expected_toolkit is not None:
        actual_toolkit = details["packages"].get("cuda-toolkit")
        if actual_toolkit != str(expected_toolkit):
            raise BakeoffError(
                "CUDA toolkit meta-package mismatch: "
                f"expected {expected_toolkit}, observed {actual_toolkit}"
            )
    expected_components = runtime.get("cuda_component_versions", {})
    if not isinstance(expected_components, Mapping):
        raise BakeoffError("runtime cuda_component_versions must be a mapping")
    for package, expected_version in expected_components.items():
        actual_version = details["packages"].get(str(package))
        if actual_version != str(expected_version):
            raise BakeoffError(
                f"CUDA component mismatch for {package}: "
                f"expected {expected_version}, observed {actual_version}"
            )
    expected_lock_sha256 = runtime.get("runtime_lock_sha256")
    if expected_lock_sha256 is not None:
        runtime_lock = details.get("runtime_lock")
        actual_lock_sha256 = (
            runtime_lock.get("sha256") if isinstance(runtime_lock, Mapping) else None
        )
        if actual_lock_sha256 != str(expected_lock_sha256):
            raise BakeoffError(
                "runtime lock SHA-256 mismatch: "
                f"expected {expected_lock_sha256}, observed {actual_lock_sha256}"
            )


def assert_refresh_runtime_requirements(runtime: dict[str, Any]) -> dict[str, Any]:
    """Pin container and offload controls before any model weights are loaded."""
    required = runtime.get("required_environment", {})
    if not isinstance(required, Mapping) or set(required) - {"VLLM_PLE_CPU_OFFLOAD", "VLLM_FLOAT32_MATMUL_PRECISION"}:
        raise BakeoffError("Unsupported refresh required_environment setting")
    observed = {key: os.environ.get(key) for key in required}
    if observed != dict(required):
        raise BakeoffError("Refresh required environment differs from pinned configuration")
    image_sha256 = runtime.get("container_image_sha256")
    image_path = os.environ.get("BAKEOFF_CONTAINER_IMAGE_PATH")
    if image_sha256:
        if not image_path or not Path(image_path).is_file():
            raise BakeoffError("Pinned refresh container image is unavailable")
        if file_sha256(Path(image_path)) != image_sha256:
            raise BakeoffError("Refresh container image SHA-256 mismatch")
    return {"required_environment": observed,
            "container_image_path": image_path if image_sha256 else None,
            "container_image_sha256": image_sha256 if image_sha256 else None}


def create_llm(model: dict[str, Any], runtime: dict[str, Any], generation: dict[str, Any]):
    from vllm import LLM

    kwargs = {
        "model": model["model_id"],
        "revision": model["revision"],
        "tokenizer_revision": model["revision"],
        "max_model_len": int(runtime["max_model_len"]),
        "gpu_memory_utilization": float(runtime["gpu_memory_utilization"]),
        "seed": int(generation["seed"]),
        "trust_remote_code": False,
    }
    if bool(model.get("language_model_only", False)):
        kwargs["language_model_only"] = True
    if model.get("reviewed_remote_config"):
        # This one pinned configuration module was reviewed before submission.
        # It imports only typing and Transformers configuration classes.
        expected = "836c3e4aff06f88bd2891b18292424ed7a90158b508d9235d490a33d9e3cba32"
        if (model["model_id"] != "nvidia/MiniMax-M3-NVFP4"
                or model["revision"] != "901464083161bf8612a29ff7ad29914cd4ab4a85"
                or model["reviewed_remote_config"] != expected):
            raise BakeoffError("Remote configuration has not been reviewed")
        snapshot = assert_compact_offline_provenance(
            {"require_exact_snapshot_path": True, "require_offline_inference": True}, model)
        if file_sha256(snapshot / "configuration_minimax_m3_vl.py") != expected:
            raise BakeoffError("Reviewed remote configuration checksum mismatch")
        kwargs.update(trust_remote_code=True, code_revision=model["revision"])
    kwargs.update(dict(model.get("llm_kwargs", {})))
    return LLM(**kwargs), kwargs


def responses_request_payload(
    *,
    model_id: str,
    conversation: list[dict[str, str]],
    schema: Mapping[str, Any],
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Build one vLLM Responses request with a Harmony-aware final JSON schema."""
    return {
        "model": model_id,
        "input": conversation,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "classification",
                "schema": dict(schema),
                "strict": True,
            }
        },
    }


def responses_output_text(response: Mapping[str, Any]) -> str:
    """Extract only user-visible final text, never Harmony reasoning content."""
    texts: list[str] = []
    for output in response.get("output", []):
        if not isinstance(output, Mapping) or output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                texts.append(str(content.get("text", "")))
    return "\n".join(texts).strip()


def harmony_generation_error_to_malformed(detail: str) -> dict[str, Any] | None:
    """Convert a model-generated invalid Harmony header into one malformed row.

    Other HTTP failures remain fatal infrastructure errors.  This narrow case is
    the Responses server's representation of an invalid generated response, so
    retaining it as wrong follows the benchmark's malformed-output rule.
    """
    parse_error = None
    if "unexpected tokens remaining in message header" in detail:
        parse_error = "harmony_invalid_message_header"
    else:
        try:
            error = json.loads(detail).get("error", {})
        except (ValueError, AttributeError):
            error = {}
        if (isinstance(error, Mapping)
                and error.get("type") == "BadRequestError"
                and str(error.get("message", "")).startswith("Unknown channel: ")):
            parse_error = "harmony_unknown_generated_channel"
    if parse_error is None:
        return None
    return {
        "output": [],
        "status": "failed",
        "usage": {},
        "_benchmark_parse_error": parse_error,
        "_benchmark_token_usage_status": "unavailable",
    }


class VllmResponsesBackend:
    """A job-local vLLM server used only for GPT-OSS Harmony responses."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[Any],
        base_url: str,
        tokenizer: Any,
        max_workers: int,
    ) -> None:
        self.process = process
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.max_workers = max_workers

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/responses",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            malformed = harmony_generation_error_to_malformed(detail)
            if malformed is not None:
                return malformed
            raise BakeoffError(
                f"vLLM Responses request failed with HTTP {exc.code}: {detail}"
            ) from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise BakeoffError("vLLM Responses endpoint returned a non-object payload")
        return parsed


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_vllm_responses_backend(
    model: dict[str, Any],
    runtime: dict[str, Any],
    snapshot_path: Path,
) -> tuple[VllmResponsesBackend, dict[str, Any]]:
    """Start the documented online renderer needed by GPT-OSS Harmony output."""
    from transformers import AutoTokenizer

    port = _available_local_port()
    max_workers = int(model.get("llm_kwargs", {}).get("max_num_seqs", 16))
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(snapshot_path),
        "--served-model-name",
        str(model["model_id"]),
        "--max-model-len",
        str(int(runtime["max_model_len"])),
        "--gpu-memory-utilization",
        str(float(runtime["gpu_memory_utilization"])),
        "--max-num-seqs",
        str(max_workers),
        "--generation-config",
        "vllm",
        "--disable-log-stats",
    ]
    process = subprocess.Popen(command)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BakeoffError(
                f"vLLM Responses server exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    else:
        process.terminate()
        raise BakeoffError("vLLM Responses server did not become healthy within 900 seconds")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), local_files_only=True, trust_remote_code=False
    )
    backend = VllmResponsesBackend(
        process=process,
        base_url=base_url,
        tokenizer=tokenizer,
        max_workers=max_workers,
    )
    metadata = {
        "model": str(snapshot_path),
        "served_model_name": str(model["model_id"]),
        "max_model_len": int(runtime["max_model_len"]),
        "gpu_memory_utilization": float(runtime["gpu_memory_utilization"]),
        "max_num_seqs": max_workers,
        "inference_interface": "vllm_responses_harmony",
    }
    return backend, metadata


def resolved_chat_template_kwargs(model: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_thinking": False}
    kwargs.update(dict(model.get("chat_template_kwargs", {})))
    return kwargs


def parse_refresh_content(text: str, task: dict[str, Any]):
    predictions, error = parse_content(text, task)
    if not task.get("_refresh_strict_schema"):
        return predictions, error
    try:
        obj = json.loads(text)
        keys = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
        if not isinstance(obj, dict) or set(obj) != set(keys):
            raise ValueError("Missing or unexpected output fields")
        if task["label_kind"] == "categorical":
            if obj[task["label_key"]] not in task["labels"]:
                raise ValueError("Unknown categorical label")
        elif any(type(value) is not int or value not in (0, 1) for value in obj.values()):
            raise ValueError("Binary fields must be integers 0 or 1")
    except (ValueError, TypeError) as exc:
        error = "schema_invalid: " + str(exc)[:160]
    return predictions, error


def prediction_rows(
    task: dict[str, Any],
    items: list[dict[str, Any]],
    outputs: list[Any],
    prompt_lengths: list[int],
    model_alias: str,
    model: dict[str, Any],
    effective_latency_s: float,
    structured_backend: str,
    checkpoint_metadata: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if len(outputs) != len(items):
        raise BakeoffError(
            f"{task['name']}: vLLM returned {len(outputs)} outputs for {len(items)} items"
        )
    rows = []
    for item, request_output, prompt_tokens in zip(items, outputs, prompt_lengths):
        if not request_output.outputs:
            text = ""
            output_tokens = 0
            finish_reason = "missing_output"
        else:
            completion = request_output.outputs[0]
            text = completion.text.strip()
            output_tokens = len(completion.token_ids or [])
            finish_reason = getattr(completion, "finish_reason", None)
        predictions, parse_error = parse_refresh_content(text, task)
        row = {
            "task": task["name"],
            "model": model_alias,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "item_id": str(item["item_id"]),
            "latency_s": effective_latency_s,
            "eval_count": output_tokens,
            "prompt_tokens": prompt_tokens,
            "token_usage_status": "reported",
            "parse_error": parse_error,
            "finish_reason": finish_reason,
            "structured_output_backend": structured_backend,
            "raw_content_preview": text[:200],
        }
        if task.get("_refresh_strict_schema"):
            row["raw_content"] = text
        if checkpoint_metadata:
            row.update(checkpoint_metadata)
        for key, value in predictions.items():
            row[f"pred_{key}"] = value
        for key, value in item["gt"].items():
            row[f"gt_{key}"] = value
        rows.append(row)
    return rows


def generate_responses_task_batch(
    *,
    backend: VllmResponsesBackend,
    tokenizer: Any,
    task: dict[str, Any],
    items: list[dict[str, Any]],
    model_alias: str,
    model: dict[str, Any],
    runtime: Mapping[str, Any],
    generation: Mapping[str, Any],
    chat_template_kwargs: Mapping[str, Any],
    checkpoint_metadata: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Send separate logical requests through GPT-OSS's Harmony renderer."""
    system_prompt = Path(task["prompt_path"]).read_text()
    conversations = build_conversations(
        system_prompt,
        items,
        conversation_layout=str(model.get("conversation_layout", "system_user")),
    )
    prompt_lengths = preflight_prompt_lengths(
        tokenizer,
        conversations,
        max_model_len=int(runtime["max_model_len"]),
        max_tokens=int(generation["max_tokens"]),
        margin_tokens=int(generation["prompt_margin_tokens"]),
        chat_template_kwargs=chat_template_kwargs,
    )
    reasoning_effort = str(chat_template_kwargs["reasoning_effort"])
    payloads = [
        responses_request_payload(
            model_id=str(model["model_id"]),
            conversation=conversation,
            schema=task["json_schema"],
            max_output_tokens=int(generation["max_tokens"]),
            reasoning_effort=reasoning_effort,
        )
        for conversation in conversations
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(backend.max_workers, len(payloads))
    ) as pool:
        responses = list(pool.map(backend.post, payloads))
    generation_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for item, prompt_tokens, response in zip(items, prompt_lengths, responses):
        text = responses_output_text(response)
        predictions, parse_error = parse_refresh_content(text, task)
        parse_error = response.get("_benchmark_parse_error") or parse_error
        usage = response.get("usage", {})
        output_tokens = int(usage.get("output_tokens") or 0)
        status = str(response.get("status") or "unknown")
        if not text and parse_error is None:
            parse_error = "missing_final_output"
        row = {
            "task": task["name"],
            "model": model_alias,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "item_id": str(item["item_id"]),
            "latency_s": generation_seconds / len(items),
            "eval_count": output_tokens,
            "prompt_tokens": prompt_tokens,
            "token_usage_status": response.get("_benchmark_token_usage_status", "reported"),
            "parse_error": parse_error,
            "finish_reason": status,
            "structured_output_backend": "vllm_responses_harmony_json_schema",
            "raw_content_preview": text[:200],
        }
        if task.get("_refresh_strict_schema"):
            row["raw_content"] = text
        if checkpoint_metadata:
            row.update(checkpoint_metadata)
        for key, value in predictions.items():
            row[f"pred_{key}"] = value
        for key, value in item["gt"].items():
            row[f"gt_{key}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, {
        "generation_seconds": generation_seconds,
        "items_per_second": len(items) / generation_seconds,
        "max_prompt_tokens": max(prompt_lengths),
        "mean_output_tokens": float(frame["eval_count"].mean()),
        "parse_ok": int(frame["parse_error"].isna().sum()),
    }


def generate_task_batch(
    *,
    llm: Any,
    tokenizer: Any,
    task: dict[str, Any],
    items: list[dict[str, Any]],
    model_alias: str,
    model: dict[str, Any],
    runtime: Mapping[str, Any],
    generation: Mapping[str, Any],
    chat_template_kwargs: Mapping[str, Any],
    checkpoint_metadata: dict[str, str] | None = None,
    use_tqdm: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one non-empty atomic batch and return rows plus timing evidence."""
    if not items:
        raise BakeoffError(f"{task['name']}: cannot generate an empty item batch")
    if isinstance(llm, VllmResponsesBackend):
        return generate_responses_task_batch(
            backend=llm,
            tokenizer=tokenizer,
            task=task,
            items=items,
            model_alias=model_alias,
            model=model,
            runtime=runtime,
            generation=generation,
            chat_template_kwargs=chat_template_kwargs,
            checkpoint_metadata=checkpoint_metadata,
        )
    system_prompt = Path(task["prompt_path"]).read_text()
    conversations = build_conversations(
        system_prompt,
        items,
        conversation_layout=str(model.get("conversation_layout", "system_user")),
    )
    prompt_lengths = preflight_prompt_lengths(
        tokenizer,
        conversations,
        max_model_len=int(runtime["max_model_len"]),
        max_tokens=int(generation["max_tokens"]),
        margin_tokens=int(generation["prompt_margin_tokens"]),
        chat_template_kwargs=chat_template_kwargs,
    )
    structured_kwargs, structured_backend = structured_output_kwargs(
        task["json_schema"], str(generation.get("structured_outputs", "json_schema"))
    )
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_tokens"]),
        seed=int(generation["seed"]),
        **structured_kwargs,
    )
    generation_started = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        use_tqdm=use_tqdm,
        chat_template_kwargs=dict(chat_template_kwargs),
    )
    generation_seconds = time.perf_counter() - generation_started
    frame = pd.DataFrame(
        prediction_rows(
            task=task,
            items=items,
            outputs=outputs,
            prompt_lengths=prompt_lengths,
            model_alias=model_alias,
            model=model,
            effective_latency_s=generation_seconds / len(items),
            structured_backend=structured_backend,
            checkpoint_metadata=checkpoint_metadata,
        )
    )
    return frame, {
        "generation_seconds": generation_seconds,
        "items_per_second": len(items) / generation_seconds,
        "max_prompt_tokens": max(prompt_lengths),
        "mean_output_tokens": float(frame["eval_count"].mean()),
        "parse_ok": int(frame["parse_error"].isna().sum()),
    }


def observed_peak_gpu_memory_mib(monitor: GpuMemoryMonitor) -> int | None:
    current = query_gpu().get("total_used_mib")
    values = [
        int(value)
        for value in [monitor.max_total_used_mib, current]
        if value is not None
    ]
    return max(values) if values else None


def merge_task_checkpoints(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    model_alias: str,
    model: dict[str, Any],
    config_hash: str,
    generation: dict[str, Any],
    runtime: dict[str, Any],
    runtime_version_details: dict[str, Any],
    limit_items: int | None,
    panel: PanelSelection | None = None,
) -> pd.DataFrame:
    frames = []
    for task in tasks:
        items = load_task_items(task, limit_items)
        expected_ids = [str(item["item_id"]) for item in items]
        panel_task_fingerprint = (
            panel.task_fingerprints[task["name"]] if panel else None
        )
        fingerprint = task_fingerprint(
            task,
            items,
            model_alias=model_alias,
            model=model,
            config_hash=config_hash,
            generation=generation,
            runtime=runtime,
            runtime_versions=runtime_version_details,
            panel_sha256=panel.panel_sha256 if panel else None,
            panel_task_fingerprint=panel_task_fingerprint,
        )
        path = output_dir / "tasks" / f"{task['name']}.csv"
        if not task_checkpoint_complete(
            path,
            task["name"],
            model_alias,
            expected_ids,
            expected_fingerprint=fingerprint,
            expected_panel_sha256=panel.panel_sha256 if panel else None,
            expected_panel_task_fingerprint=panel_task_fingerprint,
        ):
            raise BakeoffError(f"cannot merge incomplete checkpoint: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if merged.duplicated(["task", "model", "item_id"]).any():
        raise BakeoffError("merged predictions contain duplicate task/model/item_id keys")
    write_csv_atomic(output_dir / "predictions.csv", merged)
    return merged


def load_task_result(
    task_csv_path: Path,
    expected_fingerprint: str,
    expected_panel_sha256: str | None = None,
    expected_panel_task_fingerprint: str | None = None,
) -> dict[str, Any]:
    done_path = task_done_path(task_csv_path)
    done = json.loads(done_path.read_text())
    if done.get("task_fingerprint") != expected_fingerprint:
        raise BakeoffError(f"stale task manifest: {done_path}")
    if (
        expected_panel_sha256 is not None
        and done.get("panel_sha256") != expected_panel_sha256
    ):
        raise BakeoffError(f"stale panel manifest: {done_path}")
    if (
        expected_panel_task_fingerprint is not None
        and done.get("panel_task_fingerprint")
        != expected_panel_task_fingerprint
    ):
        raise BakeoffError(f"stale panel task fingerprint: {done_path}")
    return done


def resume_identity_matches(prior_metadata: dict[str, Any], plan: dict[str, Any]) -> bool:
    fields = [
        "config_sha256",
        "model_key",
        "revision",
        "only_task",
        "limit_items",
        "task_counts",
        "panel_sha256",
        "execution_protocol",
        "pilot_items",
        "remainder_items",
    ]
    return all(prior_metadata.get(field) == plan.get(field) for field in fields)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = load_config(config_path)
    execution = compact_execution(config)
    panel = None
    panel_manifest_path = getattr(args, "panel_manifest", None)
    explicit_panel_override = panel_manifest_path is not None
    if panel_manifest_path is None and execution is not None:
        configured_manifest = Path(str(execution["panel_manifest"]))
        panel_manifest_path = (
            configured_manifest
            if configured_manifest.is_absolute()
            else REPO / configured_manifest
        )
    if panel_manifest_path is not None:
        tasks_dir = REPO / config["task_scope"]["tasks_dir"]
        panel = load_panel_manifest(panel_manifest_path, tasks_dir=tasks_dir)
        assert_panel_commit(panel, str(config["benchmark_commit"]))
        assert_panel_checkout(panel, REPO)
        if execution is not None and explicit_panel_override:
            execution = {
                **execution,
                "panel_manifest": str(panel.manifest_path),
                "panel_id": panel.panel_id,
                "expected_items": panel.expected_items,
                "pilot_items": panel.pilot_items,
                "remainder_items": panel.remainder_items,
            }
    else:
        assert_benchmark_commit(config)
    frozen_config_hash = config_sha256(config_path)
    if args.model_key not in config["models"]:
        raise BakeoffError(
            f"unknown model key {args.model_key!r}; choices={sorted(config['models'])}"
        )
    model = config["models"][args.model_key]
    assert_model_launch_eligible(args.model_key, model)
    if execution is not None:
        if args.only_task is not None or args.limit_items is not None:
            raise BakeoffError(
                "compact direct runs do not allow --only-task or --limit-items"
            )
        if panel is None:
            raise BakeoffError("compact direct runs require the frozen compact manifest")
        if panel.panel_id != str(execution["panel_id"]):
            raise BakeoffError(
                f"compact panel mismatch: expected {execution['panel_id']}, "
                f"observed {panel.panel_id}"
            )
        if panel.expected_items != int(execution["expected_items"]):
            raise BakeoffError("compact config and manifest item counts disagree")
        if panel.pilot_items != int(execution["pilot_items"]):
            raise BakeoffError("compact config and manifest pilot counts disagree")
    tasks = selected_tasks(config, args.only_task, panel=panel)
    expected_scope = (
        (panel.expected_tasks, panel.expected_items) if panel is not None else None
    )
    scope = validate_scope(
        config,
        tasks,
        args.limit_items,
        args.only_task,
        expected_scope=expected_scope,
    )

    panel_plan = (
        {
            "panel_id": panel.panel_id,
            "panel_manifest": str(panel.manifest_path),
            "panel_sha256": panel.panel_sha256,
            "panel_task_fingerprints": panel.task_fingerprints,
        }
        if panel is not None
        else {}
    )

    plan = {
        "run_id": config["run_id"],
        "benchmark_commit": config["benchmark_commit"],
        "model_key": args.model_key,
        "model_id": model["model_id"],
        "revision": model["revision"],
        "config_sha256": frozen_config_hash,
        "only_task": args.only_task,
        "limit_items": args.limit_items,
        "execution_protocol": execution.get("protocol") if execution else None,
        "pilot_items": int(execution["pilot_items"]) if execution else None,
        "remainder_items": int(execution["remainder_items"]) if execution else None,
        "artifact_publication_date": (
            str(model["artifact_publication_date"])
            if model.get("artifact_publication_date") is not None
            else None
        ),
        "artifact_publication_source": model.get("artifact_publication_source"),
        "quantization": model.get("quantization"),
        **panel_plan,
        **scope,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return

    if args.output_dir is None:
        if execution is not None:
            args.output_dir = REPO / str(execution["output_root"]) / args.model_key
        else:
            run_root = (
                "frontier_2026" if panel_manifest_path else "hive_model_bakeoff_20260804"
            )
            args.output_dir = REPO / "output" / "sidecar" / run_root / args.model_key
    output_dir = require_sidecar_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    generation = config["generation"]
    metadata_path = output_dir / "run_metadata.json"
    prior_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        prior_metadata = json.loads(metadata_path.read_text())
        same_run = resume_identity_matches(prior_metadata, plan)
        if not same_run and not args.force:
            raise BakeoffError(
                f"{metadata_path} belongs to a different config/model revision; "
                "use a new output directory"
            )
        if not same_run:
            prior_metadata = {}

    version_details = runtime_versions()
    gpu_before_load = query_gpu()
    monitor = GpuMemoryMonitor()
    monitor.start()
    attempts = [] if args.force else list(prior_metadata.get("attempts", []))
    task_results = {} if args.force else dict(prior_metadata.get("task_results", {}))
    run_metadata = {
        **plan,
        "started_at": prior_metadata.get("started_at", now_utc()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu_before_load": gpu_before_load,
        "runtime_versions": version_details,
        "generation": generation,
        "runtime": runtime,
        "attempts": attempts,
        "task_results": task_results,
        "status": "running",
    }
    write_json_atomic(metadata_path, run_metadata)
    responses_backend: VllmResponsesBackend | None = None

    try:
        assert_runtime_versions(runtime, version_details)
        if config.get("refresh_sample"):
            run_metadata["refresh_runtime_requirements"] = assert_refresh_runtime_requirements(runtime)
            write_json_atomic(metadata_path, run_metadata)
            expected_gpus = int(model.get("llm_kwargs", {}).get("tensor_parallel_size", 1))
            if len(gpu_before_load.get("gpus", [])) != expected_gpus:
                raise BakeoffError("Refresh allocated GPU count differs from tensor parallel size")
            if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1:
                raise BakeoffError("Refresh inference requires a single-node allocation")
        if execution is not None:
            assert_compact_hardware(execution, gpu_before_load)
            snapshot_path = assert_compact_offline_provenance(execution, model)
            run_metadata["model_snapshot_path"] = (
                str(snapshot_path) if snapshot_path is not None else None
            )
            run_metadata["provenance_status"] = "qualified"
            write_json_atomic(metadata_path, run_metadata)
        elif config.get("refresh_sample"):
            snapshot_path = assert_compact_offline_provenance(
                {"require_exact_snapshot_path": True, "require_offline_inference": True}, model
            )
            run_metadata["model_snapshot_path"] = str(snapshot_path)
            run_metadata["provenance_status"] = "qualified"
            write_json_atomic(metadata_path, run_metadata)
        load_started = time.perf_counter()
        if str(model.get("inference_interface", "offline_chat")) == (
            "vllm_responses_harmony"
        ):
            if (execution is None and not config.get("refresh_sample")) or snapshot_path is None:
                raise BakeoffError(
                    "Responses/Harmony inference requires compact offline provenance"
                )
            responses_backend, llm_kwargs = start_vllm_responses_backend(
                model, runtime, snapshot_path
            )
            llm = responses_backend
            tokenizer = responses_backend.tokenizer
        else:
            llm, llm_kwargs = create_llm(model, runtime, generation)
            tokenizer = llm.get_tokenizer()
        startup_seconds = time.perf_counter() - load_started
        chat_template_kwargs = resolved_chat_template_kwargs(model)
        run_metadata["llm_kwargs"] = llm_kwargs
        run_metadata["chat_template_kwargs"] = chat_template_kwargs
        run_metadata["gpu_after_load"] = query_gpu()
        run_metadata["attempts"].append(
            {
                "started_at": now_utc(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "startup_seconds": startup_seconds,
                "gpu_after_load": run_metadata["gpu_after_load"],
            }
        )
        attempt_startups = [
            float(attempt["startup_seconds"])
            for attempt in run_metadata["attempts"]
            if attempt.get("startup_seconds") is not None
        ]
        run_metadata["startup_seconds"] = median(attempt_startups)
        run_metadata["startup_seconds_total_attempts"] = sum(attempt_startups)
        write_json_atomic(metadata_path, run_metadata)

        pilot_task_name: str | None = None
        pilot_frame: pd.DataFrame | None = None
        pilot_batch_result: dict[str, Any] | None = None
        if execution is not None:
            pilot_count = int(execution["pilot_items"])
            pilot_task = tasks[0]
            pilot_task_name = str(pilot_task["name"])
            pilot_items = load_task_items(pilot_task)[:pilot_count]
            if len(pilot_items) != pilot_count:
                raise BakeoffError(
                    f"compact pilot resolved {len(pilot_items)} rows, expected {pilot_count}"
                )
            pilot_ids = [str(item["item_id"]) for item in pilot_items]
            pilot_panel_metadata = panel.row_metadata(pilot_task_name)
            pilot_task_fingerprint = panel.task_fingerprints[pilot_task_name]
            pilot_fingerprint = task_fingerprint(
                pilot_task,
                pilot_items,
                model_alias=args.model_key,
                model=model,
                config_hash=frozen_config_hash,
                generation=generation,
                runtime=runtime,
                runtime_versions=version_details,
                panel_sha256=panel.panel_sha256,
                panel_task_fingerprint=pilot_task_fingerprint,
            )
            pilot_path = output_dir / "pilot" / "predictions.csv"
            pilot_audit_path = output_dir / "pilot" / "audit.json"
            pilot_checkpoint_ok = (
                not args.force
                and task_checkpoint_complete(
                    pilot_path,
                    pilot_task_name,
                    args.model_key,
                    pilot_ids,
                    expected_fingerprint=pilot_fingerprint,
                    expected_panel_sha256=panel.panel_sha256,
                    expected_panel_task_fingerprint=pilot_task_fingerprint,
                )
                and pilot_audit_path.exists()
            )
            if pilot_checkpoint_ok:
                pilot_frame = pd.read_csv(pilot_path, low_memory=False)
                pilot_batch_result = load_task_result(
                    pilot_path,
                    pilot_fingerprint,
                    expected_panel_sha256=panel.panel_sha256,
                    expected_panel_task_fingerprint=pilot_task_fingerprint,
                )
                pilot_audit = json.loads(pilot_audit_path.read_text())
                if not bool(pilot_audit.get("passed", False)):
                    raise BakeoffError("stored compact pilot did not pass its frozen gate")
                print(
                    f"[pilot] {pilot_task_name}: exact 16-row checkpoint, reusing",
                    flush=True,
                )
            else:
                print(
                    f"[pilot] {pilot_task_name}: generating first {pilot_count} requests",
                    flush=True,
                )
                pilot_frame, generated = generate_task_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    task=pilot_task,
                    items=pilot_items,
                    model_alias=args.model_key,
                    model=model,
                    runtime=runtime,
                    generation=generation,
                    chat_template_kwargs=chat_template_kwargs,
                    checkpoint_metadata=pilot_panel_metadata,
                )
                write_csv_atomic(pilot_path, pilot_frame)
                pilot_batch_result = {
                    "task": pilot_task_name,
                    "model_key": args.model_key,
                    "model_id": model["model_id"],
                    "model_revision": model["revision"],
                    "config_sha256": frozen_config_hash,
                    "task_fingerprint": pilot_fingerprint,
                    **pilot_panel_metadata,
                    "completed_at": now_utc(),
                    "rows": len(pilot_items),
                    "items": len(pilot_items),
                    **generated,
                    "gpu_after_pilot": query_gpu(),
                    "checkpoint": str(pilot_path),
                    "checkpoint_sha256": file_sha256(pilot_path),
                }
                write_json_atomic(task_done_path(pilot_path), pilot_batch_result)
                pilot_peak = observed_peak_gpu_memory_mib(monitor)
                pilot_metadata = {
                    **run_metadata,
                    "status": "completed",
                    "observed_peak_gpu_memory_used_mib": pilot_peak,
                }
                pilot_audit = evaluate_compact_pilot(
                    pilot_metadata,
                    pilot_frame,
                    expected_model_id=str(model["model_id"]),
                    expected_revision=str(model["revision"]),
                    expected_rows=pilot_count,
                    expected_capacity_mib=int(execution["expected_gpu_capacity_mib"]),
                    max_malformed_rate=float(execution["max_malformed_rate"]),
                    required_gpu_name_fragment=str(
                        execution["required_gpu_name_fragment"]
                    ),
                )
                pilot_audit.update(
                    {
                        "pilot_predictions": str(pilot_path),
                        "pilot_predictions_sha256": file_sha256(pilot_path),
                        "pilot_checkpoint_sha256": file_sha256(
                            task_done_path(pilot_path)
                        ),
                    }
                )
                write_json_atomic(pilot_audit_path, pilot_audit)
                if not pilot_audit["passed"]:
                    raise BakeoffError(
                        "compact pilot failed fit, provenance, capacity, OOM, or "
                        "malformed-output gates"
                    )
            run_metadata["pilot"] = pilot_audit
            run_metadata["evidence_stage"] = "remainder"
            write_json_atomic(metadata_path, run_metadata)

        for task_index, task in enumerate(tasks, start=1):
            items = load_task_items(task, args.limit_items)
            expected_ids = [str(item["item_id"]) for item in items]
            task_path = output_dir / "tasks" / f"{task['name']}.csv"
            panel_row_metadata = panel.row_metadata(task["name"]) if panel else None
            panel_task_fingerprint = (
                panel.task_fingerprints[task["name"]] if panel else None
            )
            fingerprint = task_fingerprint(
                task,
                items,
                model_alias=args.model_key,
                model=model,
                config_hash=frozen_config_hash,
                generation=generation,
                runtime=runtime,
                runtime_versions=version_details,
                panel_sha256=panel.panel_sha256 if panel else None,
                panel_task_fingerprint=panel_task_fingerprint,
            )
            if not args.force and task_checkpoint_complete(
                task_path,
                task["name"],
                args.model_key,
                expected_ids,
                expected_fingerprint=fingerprint,
                expected_panel_sha256=panel.panel_sha256 if panel else None,
                expected_panel_task_fingerprint=panel_task_fingerprint,
            ):
                run_metadata["task_results"][task["name"]] = load_task_result(
                    task_path,
                    fingerprint,
                    expected_panel_sha256=panel.panel_sha256 if panel else None,
                    expected_panel_task_fingerprint=panel_task_fingerprint,
                )
                write_json_atomic(metadata_path, run_metadata)
                print(
                    f"[{task_index}/{len(tasks)}] {task['name']}: complete checkpoint, skipping",
                    flush=True,
                )
                continue

            if not args.force:
                refuse_uncommitted_checkpoint(task_path, task["name"])

            batch_items = items
            pilot_rows_reused = 0
            if execution is not None and task["name"] == pilot_task_name:
                batch_items = items[int(execution["pilot_items"]) :]
                pilot_rows_reused = int(execution["pilot_items"])
            print(
                f"[{task_index}/{len(tasks)}] {task['name']}: "
                f"generating {len(batch_items)} remaining items",
                flush=True,
            )
            generated_frame, generated = generate_task_batch(
                llm=llm,
                tokenizer=tokenizer,
                task=task,
                items=batch_items,
                model_alias=args.model_key,
                model=model,
                runtime=runtime,
                generation=generation,
                chat_template_kwargs=chat_template_kwargs,
                checkpoint_metadata=panel_row_metadata,
            )
            if pilot_rows_reused:
                if pilot_frame is None or pilot_batch_result is None:
                    raise BakeoffError("compact remainder is missing its pilot evidence")
                frame = pd.concat(
                    [pilot_frame, generated_frame], ignore_index=True, sort=False
                )
                generation_seconds = float(
                    pilot_batch_result["generation_seconds"]
                ) + float(generated["generation_seconds"])
                max_prompt_tokens = max(
                    int(pilot_batch_result["max_prompt_tokens"]),
                    int(generated["max_prompt_tokens"]),
                )
            else:
                frame = generated_frame
                generation_seconds = float(generated["generation_seconds"])
                max_prompt_tokens = int(generated["max_prompt_tokens"])
            if len(frame) != len(items) or frame["item_id"].astype(str).duplicated().any():
                raise BakeoffError(
                    f"{task['name']}: pilot plus remainder did not form exact coverage"
                )
            write_csv_atomic(task_path, frame)
            parse_ok = int(frame["parse_error"].isna().sum())
            task_result = {
                "task": task["name"],
                "model_key": args.model_key,
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "config_sha256": frozen_config_hash,
                "task_fingerprint": fingerprint,
                **(panel_row_metadata or {}),
                "completed_at": now_utc(),
                "rows": len(items),
                "items": len(items),
                "parse_ok": parse_ok,
                "generation_seconds": generation_seconds,
                "items_per_second": len(items) / generation_seconds,
                "max_prompt_tokens": max_prompt_tokens,
                "mean_output_tokens": float(frame["eval_count"].mean()),
                "pilot_rows_reused": pilot_rows_reused,
                "gpu_after_task": query_gpu(),
                "checkpoint": str(task_path),
                "checkpoint_sha256": file_sha256(task_path),
            }
            write_json_atomic(task_done_path(task_path), task_result)
            run_metadata["task_results"][task["name"]] = task_result
            write_json_atomic(metadata_path, run_metadata)
            print(
                f"  wrote {task_path.name}: {parse_ok}/{len(items)} parsed; "
                f"{task_result['items_per_second']:.2f} items/s",
                flush=True,
            )

        merged = merge_task_checkpoints(
            output_dir,
            tasks,
            args.model_key,
            model,
            frozen_config_hash,
            generation,
            runtime,
            version_details,
            args.limit_items,
            panel=panel,
        )
        task_metrics_path = output_dir / "task_metrics.csv"
        write_csv_atomic(
            task_metrics_path,
            task_metrics_frame(merged, tasks, args.model_key),
        )
        complete_task_results = {}
        for task in tasks:
            items = load_task_items(task, args.limit_items)
            panel_task_fingerprint = (
                panel.task_fingerprints[task["name"]] if panel else None
            )
            fingerprint = task_fingerprint(
                task,
                items,
                model_alias=args.model_key,
                model=model,
                config_hash=frozen_config_hash,
                generation=generation,
                runtime=runtime,
                runtime_versions=version_details,
                panel_sha256=panel.panel_sha256 if panel else None,
                panel_task_fingerprint=panel_task_fingerprint,
            )
            task_path = output_dir / "tasks" / f"{task['name']}.csv"
            complete_task_results[task["name"]] = load_task_result(
                task_path,
                fingerprint,
                expected_panel_sha256=panel.panel_sha256 if panel else None,
                expected_panel_task_fingerprint=panel_task_fingerprint,
            )
        observed_peak = observed_peak_gpu_memory_mib(monitor)
        if responses_backend is not None:
            responses_backend.stop()
        monitor.stop()
        run_metadata["task_results"] = complete_task_results
        run_metadata["completed_at"] = now_utc()
        run_metadata["rows_written"] = len(merged)
        run_metadata["parse_ok"] = int(merged["parse_error"].isna().sum())
        run_metadata["predictions_sha256"] = file_sha256(output_dir / "predictions.csv")
        run_metadata["task_metrics_sha256"] = file_sha256(task_metrics_path)
        run_metadata["observed_peak_gpu_memory_used_mib"] = observed_peak
        baseline_used = gpu_before_load.get("total_used_mib")
        run_metadata["observed_peak_gpu_memory_delta_mib"] = (
            observed_peak - baseline_used
            if observed_peak is not None and baseline_used is not None
            else None
        )
        if execution is not None:
            assert_compact_prediction_coverage(
                merged,
                panel,
                model_key=args.model_key,
            )
            reliability = malformed_rate_audit(
                merged,
                max_malformed_rate=float(execution["max_malformed_rate"]),
            )
            identity_ok = (
                set(merged["model_id"].astype(str)) == {str(model["model_id"])}
                and set(merged["model_revision"].astype(str))
                == {str(model["revision"])}
            )
            exact_coverage = True
            capacity_ok = (
                observed_peak is not None
                and int(observed_peak) <= int(execution["expected_gpu_capacity_mib"])
            )
            run_metadata["qualification"] = {
                "passed": bool(
                    reliability["passed"]
                    and identity_ok
                    and exact_coverage
                    and capacity_ok
                ),
                "reliability": reliability,
                "exact_model_identity": identity_ok,
                "exact_coverage": exact_coverage,
                "within_gpu_capacity": capacity_ok,
                "oom_free": True,
                "pilot_rows": int(execution["pilot_items"]),
                "remainder_rows": int(execution["remainder_items"]),
            }
            if not run_metadata["qualification"]["passed"]:
                write_json_atomic(metadata_path, run_metadata)
                raise BakeoffError(
                    "compact full benchmark failed coverage, identity, capacity, or "
                    "malformed-output qualification"
                )
            run_metadata["evidence_stage"] = "terminal_qualified"
        run_metadata["status"] = "completed"
        run_metadata.pop("error", None)
        write_json_atomic(metadata_path, run_metadata)
        print(
            f"DONE: {len(merged)} rows; {run_metadata['parse_ok']} parsed; "
            f"output={output_dir / 'predictions.csv'}",
            flush=True,
        )
    except Exception as exc:
        if responses_backend is not None:
            responses_backend.stop()
        monitor.stop()
        run_metadata["failed_at"] = now_utc()
        run_metadata["status"] = "failed"
        run_metadata["error"] = repr(exc)
        if execution is not None:
            run_metadata["qualification_status"] = "ineligible"
            run_metadata["failure_category"] = compact_failure_category(exc)
            run_metadata["oom_free"] = compact_failure_category(exc) != "oom"
        run_metadata["observed_peak_gpu_memory_used_mib"] = monitor.max_total_used_mib
        baseline_used = gpu_before_load.get("total_used_mib")
        run_metadata["observed_peak_gpu_memory_delta_mib"] = (
            monitor.max_total_used_mib - baseline_used
            if monitor.max_total_used_mib is not None and baseline_used is not None
            else None
        )
        write_json_atomic(metadata_path, run_metadata)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        help="Frozen multi-task panel manifest; validates exact task inputs before inference.",
    )
    parser.add_argument("--only-task")
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
