#!/usr/bin/env python3
"""Audit multi-model Broad18 patterns without running new model inference.

The analysis is explicitly post-outcome exploratory. It reuses the frozen
Broad18 panel, its immutable checkpoint registry, and the benchmark's existing
fail-closed evidence validators. Hard-label behavior is described as
selectivity or label propensity, not probability calibration.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "polsci-open-bench-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from build_architecture_adjusted_plots import (  # noqa: E402
    LINEAGES,
    architecture_data,
)
from analyze_heterogeneity_drivers import load_spec as load_heterogeneity_spec  # noqa: E402
from build_broad_robustness import TASK_TO_SOURCE  # noqa: E402
from build_compact_frontier import (  # noqa: E402
    _score_task_with_malformed_predictions,
    _validate_compact_rows,
    _validate_hive_evidence,
    bootstrap_indices,
    file_sha256,
    load_compact_registry,
    resolve_repo_path,
)
from panel_manifest import PanelSelection, load_panel_manifest  # noqa: E402


DEFAULT_ROOT = REPO / "output" / "sidecar" / "frontier_2026"
DEFAULT_MANIFEST = REPO / "experiments" / "frontier_broad_18.yaml"
DEFAULT_REGISTRY = REPO / "experiments" / "frontier_broad_checkpoints_2026.yaml"
DEFAULT_OUTPUT = DEFAULT_ROOT / "multi_model_patterns"
DEFAULT_TASK_METADATA = REPO / "output" / "task_length_audit.csv"
DEFAULT_COMPUTE_STAIRCASE = (
    DEFAULT_ROOT / "twitter_claims" / "05_compute_class_staircase.csv"
)
DEFAULT_HETEROGENEITY_SPEC = REPO / "experiments" / "heterogeneity_drivers_20260824.yaml"
EXTERNAL_EVENT_SUMMARY_SHA256 = (
    "6151f95fe0d52209b80cd9bc40ccaed0e8b8e320e2cac8121a74d9b4d3102459"
)
EXPLORATORY_STATUS = "post_outcome_exploratory"

BLUE = "#2f6f9f"
DARK = "#1d4f77"
RED = "#a4443a"
GRAY = "#777777"
LIGHT_GRAY = "#d8d8d8"

GEMMA_LINEAGE = (
    "gemma3_27b_it_fp8_dynamic_hive",
    "gemma4_31b_it_qat_w4a16_hive",
)
ANALYSIS_LINEAGES = {
    **LINEAGES,
    "Gemma dense 27–31B": GEMMA_LINEAGE,
}
REPLICATED_DENSE_LINEAGES = {
    name: ids for name, ids in LINEAGES.items()
}

SHORT_TASK_NAMES = {
    "burnham_covid_threat_minimization": "COVID\nthreat",
    "burnham_polnli_entailment": "PolNLI\nclaim",
    "burnham_polnli_event_entailment": "PolNLI\nevent",
    "burnham_trump_stance": "Trump\nstance",
    "cap_crs_policy_topic": "CRS\npolicy",
    "cap_party_platform_policy_topic": "Platform\npolicy",
    "chae_semeval_stance": "SemEval\nstance",
    "dicocco_manifesto_populism": "Manifesto\npopulism",
    "douglass_icbe_sentence_event_type": "ICBe\nevent",
    "erlich_ati_topics": "ATI\ntopics",
    "gilardi_relevance": "Tweet\nrelevance",
    "gilardi_stance": "Section 230\nstance",
    "halterman_keith_bfrs": "BFRS\nevent",
    "haunss_papea_fgz_forms": "PAPEA\nprotest",
    "muller_fujimura_campaign_policy_area": "Campaign\npolicy",
    "rheault_line_of_fire_incivility": "Line of Fire\nincivility",
    "theocharis_dynamics_incivility": "Dynamics\nincivility",
    "twitcivility_impoliteness": "TwitCivility\nimpoliteness",
}


class MultiModelPatternError(ValueError):
    """Raised when frozen evidence or an analysis invariant fails."""


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _parse_valid(frame: pd.DataFrame) -> np.ndarray:
    values = frame["parse_error"]
    return (
        values.isna() | values.astype(str).str.strip().eq("")
    ).to_numpy(dtype=bool)


def binary_metrics(
    gold: Iterable[Any],
    prediction: Iterable[Any],
    valid: Iterable[bool] | None = None,
) -> dict[str, float]:
    """Score hard binary labels, counting invalid outputs as gold-class misses."""
    gold_array = pd.to_numeric(pd.Series(list(gold)), errors="raise").to_numpy(dtype=int)
    prediction_array = pd.to_numeric(
        pd.Series(list(prediction)), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isin(gold_array, [0, 1]).all():
        raise MultiModelPatternError("binary gold labels must be 0/1")
    observed_valid = np.isin(prediction_array, [0, 1])
    if valid is not None:
        supplied = np.asarray(list(valid), dtype=bool)
        if len(supplied) != len(gold_array):
            raise MultiModelPatternError("binary validity mask has the wrong length")
        observed_valid &= supplied
    scored_prediction = np.where(observed_valid, prediction_array, 1 - gold_array).astype(int)
    gold_positive = gold_array == 1
    predicted_positive = scored_prediction == 1
    tp = int((gold_positive & predicted_positive).sum())
    fp = int((~gold_positive & predicted_positive).sum())
    fn = int((gold_positive & ~predicted_positive).sum())
    tn = int((~gold_positive & ~predicted_positive).sum())
    positive_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_ratio(2 * tn, 2 * tn + fp + fn)
    valid_count = int(observed_valid.sum())
    predicted_positive_rate = (
        float((prediction_array[observed_valid] == 1).mean()) if valid_count else np.nan
    )
    matched_gold_positive_rate = (
        float(gold_positive[observed_valid].mean()) if valid_count else np.nan
    )
    return {
        "predicted_positive_rate": predicted_positive_rate,
        "gold_positive_rate": matched_gold_positive_rate,
        "positive_rate_bias": predicted_positive_rate - matched_gold_positive_rate,
        "precision": _safe_ratio(tp, tp + fp),
        "positive_recall": _safe_ratio(tp, tp + fn),
        "negative_recall": _safe_ratio(tn, tn + fp),
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "symmetric_macro_f1": (positive_f1 + negative_f1) / 2,
        "accuracy": _safe_ratio(tp + tn, len(gold_array)),
        "valid_prediction_rate": _safe_ratio(valid_count, len(gold_array)),
    }


def categorical_metrics(
    gold: Iterable[Any],
    prediction: Iterable[Any],
    labels: Iterable[Any],
    valid: Iterable[bool] | None = None,
) -> dict[str, float]:
    """Score configured categorical labels; invalid outputs are always wrong."""
    configured = [str(label) for label in labels]
    if not configured or len(set(configured)) != len(configured):
        raise MultiModelPatternError("categorical labels must be unique and nonempty")
    gold_array = pd.Series(list(gold)).astype(str).to_numpy()
    prediction_array = pd.Series(list(prediction)).astype(str).to_numpy()
    if not np.isin(gold_array, configured).all():
        raise MultiModelPatternError("categorical gold labels are outside the schema")
    observed_valid = np.isin(prediction_array, configured)
    if valid is not None:
        supplied = np.asarray(list(valid), dtype=bool)
        if len(supplied) != len(gold_array):
            raise MultiModelPatternError("categorical validity mask has the wrong length")
        observed_valid &= supplied
    scored_prediction = prediction_array.astype(object)
    scored_prediction[~observed_valid] = "__MALFORMED_INCORRECT__"
    recalls, f1s = [], []
    supports = []
    for label in configured:
        gold_label = gold_array == label
        predicted_label = scored_prediction == label
        tp = int((gold_label & predicted_label).sum())
        fp = int((~gold_label & predicted_label).sum())
        fn = int((gold_label & ~predicted_label).sum())
        recalls.append(_safe_ratio(tp, tp + fn))
        f1s.append(_safe_ratio(2 * tp, 2 * tp + fp + fn))
        supports.append(int(gold_label.sum()))
    present_supports = [support for support in supports if support > 0]
    if not present_supports:
        raise MultiModelPatternError("categorical task has no observed gold labels")
    minimum_support = min(present_supports)
    rare_indices = [
        index for index, support in enumerate(supports) if support == minimum_support
    ]
    valid_predictions = prediction_array[observed_valid]
    counts = np.array(
        [(valid_predictions == label).sum() for label in configured], dtype=float
    )
    probabilities = counts[counts > 0] / counts.sum() if counts.sum() else np.array([])
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0
    normalized_entropy = entropy / np.log(len(configured)) if len(configured) > 1 else 0.0
    return {
        "accuracy": float((scored_prediction == gold_array).mean()),
        "symmetric_macro_f1": float(np.mean(f1s)),
        "label_coverage": _safe_ratio(int((counts > 0).sum()), len(configured)),
        "prediction_entropy": normalized_entropy,
        "rare_class_recall": float(np.mean([recalls[index] for index in rare_indices])),
        "valid_prediction_rate": float(observed_valid.mean()),
    }


def multi_binary_metrics(
    frame: pd.DataFrame,
    task: Mapping[str, Any],
    malformed: np.ndarray,
) -> dict[str, float]:
    """Average binary-decision metrics while treating row-level failures as wrong."""
    per_label = []
    valid_rows = ~np.asarray(malformed, dtype=bool)
    for label in task["labels"]:
        per_label.append(
            binary_metrics(
                frame[f"gt_{label}"], frame[f"pred_{label}"], valid=valid_rows
            )
        )
    keys = per_label[0].keys()
    return {key: float(np.mean([row[key] for row in per_label])) for key in keys}


def _f1_draws_for_label(
    gold: np.ndarray, prediction: np.ndarray, label: Any, indices: np.ndarray
) -> np.ndarray:
    sampled_gold = gold[indices]
    sampled_prediction = prediction[indices]
    tp = ((sampled_gold == label) & (sampled_prediction == label)).sum(axis=1)
    fp = ((sampled_gold != label) & (sampled_prediction == label)).sum(axis=1)
    fn = ((sampled_gold == label) & (sampled_prediction != label)).sum(axis=1)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros(len(indices), dtype=float),
        where=denominator != 0,
    )


def task_score_draws(
    scored: pd.DataFrame,
    task: Mapping[str, Any],
    indices: np.ndarray,
    *,
    symmetric: bool,
) -> np.ndarray:
    """Apply the shared item bootstrap to headline or symmetric task F1."""
    kind = str(task["label_kind"])
    if kind == "categorical":
        key = str(task["label_key"])
        gold = scored[f"gt_{key}"].astype(str).to_numpy()
        prediction = scored[f"pred_{key}"].astype(str).to_numpy()
        return np.mean(
            np.stack(
                [
                    _f1_draws_for_label(gold, prediction, str(label), indices)
                    for label in task["labels"]
                ],
                axis=1,
            ),
            axis=1,
        )
    keys = task["labels"] if kind == "multi_binary" else [task["label_key"]]
    label_values = [0, 1] if symmetric else [1]
    components = []
    for key in keys:
        gold = pd.to_numeric(scored[f"gt_{key}"], errors="raise").to_numpy(dtype=int)
        prediction = pd.to_numeric(
            scored[f"pred_{key}"], errors="raise"
        ).to_numpy(dtype=int)
        for label in label_values:
            components.append(_f1_draws_for_label(gold, prediction, label, indices))
    return np.mean(np.stack(components, axis=1), axis=1)


def bootstrap_weight_matrices(
    shared_indices: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert shared item indices to compact count matrices for fast rescoring."""
    result: dict[str, np.ndarray] = {}
    for task, indices in shared_indices.items():
        if indices.ndim != 2 or indices.shape[1] < 1:
            raise MultiModelPatternError(f"invalid bootstrap index matrix: {task}")
        draws, item_count = indices.shape
        if indices.min() < 0 or indices.max() >= item_count:
            raise MultiModelPatternError(f"bootstrap indices are out of range: {task}")
        weights = np.zeros((draws, item_count), dtype=np.float32)
        rows = np.repeat(np.arange(draws, dtype=np.int64), item_count)
        np.add.at(weights, (rows, indices.reshape(-1)), 1)
        if not np.allclose(weights.sum(axis=1), item_count):
            raise MultiModelPatternError(f"bootstrap weights do not preserve draws: {task}")
        result[task] = weights
    return result


def task_score_draw_pair(
    scored: pd.DataFrame,
    task: Mapping[str, Any],
    weights: np.ndarray,
) -> np.ndarray:
    """Return headline and symmetric F1 draws using one matrix multiplication."""
    if weights.ndim != 2 or weights.shape[1] != len(scored):
        raise MultiModelPatternError("bootstrap weight matrix has the wrong shape")
    kind = str(task["label_kind"])
    components: list[tuple[str, np.ndarray, np.ndarray]] = []
    if kind == "categorical":
        key = str(task["label_key"])
        gold = scored[f"gt_{key}"].astype(str).to_numpy()
        prediction = scored[f"pred_{key}"].astype(str).to_numpy()
        for label in task["labels"]:
            components.append(("headline", gold == str(label), prediction == str(label)))
    else:
        keys = task["labels"] if kind == "multi_binary" else [task["label_key"]]
        for key in keys:
            gold = pd.to_numeric(scored[f"gt_{key}"], errors="raise").to_numpy(dtype=int)
            prediction = pd.to_numeric(
                scored[f"pred_{key}"], errors="raise"
            ).to_numpy(dtype=int)
            for label in (0, 1):
                components.append(
                    ("headline" if label == 1 else "negative", gold == label, prediction == label)
                )
    packed = []
    for _, gold_label, prediction_label in components:
        packed.extend([gold_label & prediction_label, gold_label, prediction_label])
    sufficient = weights @ np.column_stack(packed).astype(np.float32)
    sufficient = sufficient.astype(float, copy=False)
    f1_columns = []
    for index in range(len(components)):
        tp = sufficient[:, 3 * index]
        gold_count = sufficient[:, 3 * index + 1]
        prediction_count = sufficient[:, 3 * index + 2]
        denominator = gold_count + prediction_count
        f1_columns.append(
            np.divide(
                2 * tp,
                denominator,
                out=np.zeros(len(weights), dtype=float),
                where=denominator != 0,
            )
        )
    f1 = np.stack(f1_columns, axis=1)
    headline_columns = [
        index for index, (role, _, _) in enumerate(components) if role == "headline"
    ]
    headline = f1[:, headline_columns].mean(axis=1)
    symmetric = headline if kind == "categorical" else f1.mean(axis=1)
    return np.stack([headline, symmetric], axis=1)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _leave_one_out_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = [
        _linear_slope(np.delete(x, index), np.delete(y, index))
        for index in range(len(x))
    ]
    return float(np.nanmin(values)), float(np.nanmax(values))


def _rank(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if (
        len(x_array) < 2
        or len(x_array) != len(y_array)
        or not np.isfinite(x_array).all()
        or not np.isfinite(y_array).all()
        or np.allclose(x_array, x_array[0])
        or np.allclose(y_array, y_array[0])
    ):
        return np.nan
    return float(spearmanr(x_array, y_array).statistic)


def _permutation_rank(
    x: np.ndarray, y: np.ndarray, *, iterations: int, seed: int
) -> tuple[float, float]:
    observed = _rank(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([_rank(x, rng.permutation(y)) for _ in range(iterations)])
    p_value = (1 + int((null >= observed).sum())) / (iterations + 1)
    return observed, float(p_value)


def _leave_one_out_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = [
        _rank(np.delete(x, index), np.delete(y, index)) for index in range(len(x))
    ]
    return float(np.nanmin(values)), float(np.nanmax(values))


def _partial_rank(x: Iterable[float], y: Iterable[float], control: Iterable[float]) -> float:
    """Partial Spearman association obtained by residualizing average ranks."""
    x_rank = pd.Series(list(x), dtype=float).rank(method="average").to_numpy()
    y_rank = pd.Series(list(y), dtype=float).rank(method="average").to_numpy()
    z_rank = pd.Series(list(control), dtype=float).rank(method="average").to_numpy()
    if len(x_rank) < 3 or np.allclose(z_rank, z_rank[0]):
        return _rank(x_rank, y_rank)
    design = np.column_stack([np.ones(len(z_rank)), z_rank])
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    if np.allclose(x_residual, 0) or np.allclose(y_residual, 0):
        return np.nan
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def _posttraining_developer(checkpoint_id: str) -> str:
    if "nemotron" in checkpoint_id:
        return "NVIDIA"
    if checkpoint_id.startswith("deepseek"):
        return "DeepSeek"
    if checkpoint_id.startswith("qwen"):
        return "Qwen"
    if checkpoint_id.startswith("llama"):
        return "Meta"
    if checkpoint_id.startswith("gemma"):
        return "Google"
    if checkpoint_id.startswith("mistral"):
        return "Mistral"
    if checkpoint_id.startswith("gpt_oss"):
        return "OpenAI"
    if checkpoint_id.startswith("glm"):
        return "Z.ai"
    raise MultiModelPatternError(f"unknown post-training developer: {checkpoint_id}")


def _task_metadata(benchmark: PanelSelection, path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {"task", "effective_label_count"}
    if not required.issubset(metadata) or metadata["task"].duplicated().any():
        raise MultiModelPatternError("task metadata is incomplete or duplicated")
    specs = pd.DataFrame(
        [
            {
                "task": spec.name,
                "annotation_type": spec.annotation_type,
                "complexity": spec.complexity,
            }
            for spec in benchmark.task_specs
        ]
    )
    result = specs.merge(
        metadata[["task", "effective_label_count"]], on="task", validate="one_to_one"
    )
    result["source_family"] = result["task"].map(TASK_TO_SOURCE)
    if result["source_family"].isna().any() or result["source_family"].nunique() != 13:
        raise MultiModelPatternError("Broad18 source-family mapping drifted")
    result["log_effective_label_count"] = np.log(result["effective_label_count"])
    return result


def audit_and_score(
    benchmark: PanelSelection,
    registry: Mapping[str, Any],
    checkpoints: pd.DataFrame,
    existing_task_scores: pd.DataFrame,
    shared_weights: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Validate all evidence and return one metric row per model and task."""
    eligible = architecture_data(checkpoints)
    registry_by_id = {
        str(checkpoint["checkpoint_id"]): checkpoint
        for checkpoint in registry["checkpoints"]
    }
    if set(eligible["checkpoint_id"]) - set(registry_by_id):
        raise MultiModelPatternError("eligible checkpoints are missing from the registry")
    expected = existing_task_scores.loc[
        existing_task_scores["checkpoint_id"].isin(eligible["checkpoint_id"])
    ].copy()
    if len(expected) != 21 * 18 or expected.duplicated(["checkpoint_id", "task"]).any():
        raise MultiModelPatternError("existing model-by-task evidence is not 21 by 18")
    expected_scores = expected.set_index(["checkpoint_id", "task"])["headline_f1"]
    checkpoint_evidence = checkpoints.set_index("checkpoint_id")
    tasks = {str(task["name"]): task for task in benchmark.tasks}
    rows: list[dict[str, Any]] = []
    draws: dict[str, dict[str, np.ndarray]] = {}
    for model in eligible.itertuples(index=False):
        checkpoint = registry_by_id[model.checkpoint_id]
        if checkpoint.get("result_status") != "completed":
            raise MultiModelPatternError(f"eligible checkpoint is not completed: {model.checkpoint_id}")
        result = checkpoint["result"]
        predictions_path = resolve_repo_path(result["predictions_path"])
        raw = pd.read_csv(predictions_path, dtype={"item_id": str}, low_memory=False)
        selected = _validate_compact_rows(raw, benchmark, checkpoint)
        _, metadata_sha = _validate_hive_evidence(
            checkpoint, result, predictions_path, benchmark
        )
        evidence_row = checkpoint_evidence.loc[model.checkpoint_id]
        if file_sha256(predictions_path) != str(evidence_row["predictions_sha256"]):
            raise MultiModelPatternError(f"prediction hash drift: {model.checkpoint_id}")
        if metadata_sha != str(evidence_row["metadata_sha256"]):
            raise MultiModelPatternError(f"metadata hash drift: {model.checkpoint_id}")
        draws[model.checkpoint_id] = {}
        for task_name in benchmark.task_names:
            task = tasks[task_name]
            group = selected.loc[selected["task"].astype(str).eq(task_name)].copy()
            headline, scored, malformed = _score_task_with_malformed_predictions(group, task)
            expected_score = float(expected_scores.loc[(model.checkpoint_id, task_name)])
            if not np.isclose(float(headline["headline_f1"]), expected_score, atol=1e-12):
                raise MultiModelPatternError(
                    f"headline score mismatch: {model.checkpoint_id}/{task_name}"
                )
            common = {
                "checkpoint_id": model.checkpoint_id,
                "display_name": model.display_name,
                "plot_date": pd.Timestamp(model.plot_date).date().isoformat(),
                "posttraining_developer": _posttraining_developer(model.checkpoint_id),
                "task": task_name,
                "label_kind": str(task["label_kind"]),
                "n": len(group),
                "malformed_count": int(np.asarray(malformed).sum()),
                "malformed_rate": float(np.asarray(malformed).mean()),
                "headline_f1": expected_score,
            }
            kind = str(task["label_kind"])
            parse_valid = _parse_valid(group)
            if kind == "binary":
                key = str(task["label_key"])
                metrics = binary_metrics(
                    group[f"gt_{key}"], group[f"pred_{key}"], valid=parse_valid
                )
            elif kind == "categorical":
                key = str(task["label_key"])
                metrics = categorical_metrics(
                    group[f"gt_{key}"],
                    group[f"pred_{key}"],
                    task["labels"],
                    valid=parse_valid,
                )
            elif kind == "multi_binary":
                metrics = multi_binary_metrics(group, task, np.asarray(malformed))
            else:
                raise MultiModelPatternError(f"unknown label kind: {kind}")
            common.update(metrics)
            if kind == "categorical" and not np.isclose(
                common["symmetric_macro_f1"], expected_score, atol=1e-12
            ):
                raise MultiModelPatternError("categorical macro-F1 reproduction failed")
            rows.append(common)
            score_draws = task_score_draw_pair(scored, task, shared_weights[task_name])
            if (
                score_draws.shape != (len(shared_weights[task_name]), 2)
                or not np.isfinite(score_draws).all()
            ):
                raise MultiModelPatternError(
                    f"bootstrap draws are incomplete: {model.checkpoint_id}/{task_name}"
                )
            draws[model.checkpoint_id][task_name] = score_draws
    metrics = pd.DataFrame(rows)
    if len(metrics) != 378 or metrics.duplicated(["checkpoint_id", "task"]).any():
        raise MultiModelPatternError("audited metric panel is not exactly 21 by 18")
    return metrics, draws


def build_model_summary(
    metrics: pd.DataFrame, checkpoints: pd.DataFrame
) -> pd.DataFrame:
    mapped = architecture_data(checkpoints).copy()
    mapped["posttraining_developer"] = mapped["checkpoint_id"].map(_posttraining_developer)
    rows = []
    for checkpoint_id, group in metrics.groupby("checkpoint_id", sort=False):
        binary = group.loc[group["label_kind"].eq("binary")]
        nonbinary = group.loc[~group["label_kind"].eq("binary")]
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "overall_headline_f1": group["headline_f1"].mean(),
                "overall_symmetric_f1": group["symmetric_macro_f1"].mean(),
                "binary_headline_f1": binary["headline_f1"].mean(),
                "binary_symmetric_f1": binary["symmetric_macro_f1"].mean(),
                "nonbinary_headline_f1": nonbinary["headline_f1"].mean(),
                "nonbinary_symmetric_f1": nonbinary["symmetric_macro_f1"].mean(),
                "binary_predicted_positive_rate": binary["predicted_positive_rate"].mean(),
                "binary_gold_positive_rate": binary["gold_positive_rate"].mean(),
                "binary_positive_rate_bias": binary["positive_rate_bias"].mean(),
                "binary_precision": binary["precision"].mean(),
                "binary_positive_recall": binary["positive_recall"].mean(),
                "binary_negative_recall": binary["negative_recall"].mean(),
            }
        )
    summary = mapped.merge(pd.DataFrame(rows), on="checkpoint_id", validate="one_to_one")
    if not np.allclose(summary["mean_f1"], summary["overall_headline_f1"], atol=1e-12):
        raise MultiModelPatternError("model means do not reproduce checkpoint summary")
    return summary.sort_values("plot_date").reset_index(drop=True)


def attach_validated_compute_staircase(
    summary: pd.DataFrame, staircase_path: Path
) -> pd.DataFrame:
    """Reuse and audit the existing active-parameter-class frontier staircase."""
    if not staircase_path.exists():
        raise MultiModelPatternError(
            f"existing compute-class staircase is missing: {staircase_path}"
        )
    staircase = pd.read_csv(staircase_path)
    required = {
        "checkpoint_id",
        "plot_date",
        "mean_f1",
        "active_parameters_b",
        "compute_class",
        "class_frontier_f1",
        "class_frontier_setter",
    }
    if (
        not required.issubset(staircase.columns)
        or staircase["checkpoint_id"].duplicated().any()
        or set(staircase["checkpoint_id"]) != set(summary["checkpoint_id"])
    ):
        raise MultiModelPatternError("existing compute-class staircase coverage drifted")
    observed = staircase.set_index("checkpoint_id").loc[summary["checkpoint_id"]].copy()
    observed_dates = pd.to_datetime(observed["plot_date"]).dt.date.astype(str).to_numpy()
    summary_dates = pd.to_datetime(summary["plot_date"]).dt.date.astype(str).to_numpy()
    if (
        not np.array_equal(observed_dates, summary_dates)
        or not np.allclose(observed["mean_f1"], summary["mean_f1"], atol=1e-12)
        or not np.allclose(
            observed["active_parameters_b"], summary["active_parameters_b"], atol=1e-12
        )
        or not np.array_equal(
            observed["compute_class"].astype(str), summary["compute_class"].astype(str)
        )
    ):
        raise MultiModelPatternError("existing compute-class staircase inputs drifted")
    recomputed_parts = []
    for compute_class, group in summary.groupby("compute_class", sort=False):
        part = group.sort_values("plot_date")[["checkpoint_id", "mean_f1"]].copy()
        part["expected_frontier"] = part["mean_f1"].cummax()
        part["expected_setter"] = part["mean_f1"].gt(
            part["expected_frontier"].shift(fill_value=-np.inf)
        )
        recomputed_parts.append(part)
    recomputed = pd.concat(recomputed_parts).set_index("checkpoint_id").loc[
        summary["checkpoint_id"]
    ]
    observed_setter = observed["class_frontier_setter"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if (
        observed_setter.isna().any()
        or not np.allclose(
            observed["class_frontier_f1"], recomputed["expected_frontier"], atol=1e-12
        )
        or not np.array_equal(
            observed_setter.to_numpy(dtype=bool),
            recomputed["expected_setter"].to_numpy(dtype=bool),
        )
    ):
        raise MultiModelPatternError("existing compute-class staircase calculation drifted")
    result = summary.copy()
    result["class_frontier_f1"] = observed["class_frontier_f1"].to_numpy(dtype=float)
    result["class_frontier_setter"] = observed_setter.to_numpy(dtype=bool)
    try:
        staircase_source = str(staircase_path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        staircase_source = str(staircase_path.resolve())
    result["compute_staircase_source"] = staircase_source
    return result


def build_task_trends(metrics: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    work = metrics.merge(metadata, on="task", validate="many_to_one")
    work["years"] = (
        pd.to_datetime(work["plot_date"]) - pd.to_datetime(work["plot_date"]).min()
    ).dt.days / 365.25
    metric_names = [
        "headline_f1",
        "symmetric_macro_f1",
        "predicted_positive_rate",
        "positive_rate_bias",
        "precision",
        "positive_recall",
        "negative_recall",
        "label_coverage",
        "prediction_entropy",
        "rare_class_recall",
        "accuracy",
    ]
    rows = []
    scopes: dict[str, tuple[str, ...] | None] = {
        "all_21": None,
        **REPLICATED_DENSE_LINEAGES,
    }
    for scope, checkpoint_ids in scopes.items():
        scoped = (
            work
            if checkpoint_ids is None
            else work.loc[work["checkpoint_id"].isin(checkpoint_ids)]
        )
        for task_name, group in scoped.groupby("task", sort=True):
            base = group.iloc[0]
            for metric in metric_names:
                valid = group[["years", metric]].dropna()
                if len(valid) < 3:
                    continue
                x = valid["years"].to_numpy(dtype=float)
                y = valid[metric].to_numpy(dtype=float)
                loo_low, loo_high = _leave_one_out_slope(x, y)
                rows.append(
                    {
                        "scope": scope,
                        "task": task_name,
                        "label_kind": base["label_kind"],
                        "annotation_type": base["annotation_type"],
                        "complexity": base["complexity"],
                        "source_family": base["source_family"],
                        "effective_label_count": base["effective_label_count"],
                        "log_effective_label_count": base["log_effective_label_count"],
                        "metric": metric,
                        "models": len(valid),
                        "slope_per_year": _linear_slope(x, y),
                        "spearman_rho": _rank(x, y),
                        "loo_min_slope": loo_low,
                        "loo_max_slope": loo_high,
                    }
                )
    return pd.DataFrame(rows)


def build_lineage_changes(
    summary: pd.DataFrame,
    draws: Mapping[str, Mapping[str, np.ndarray]],
    task_names: Iterable[str],
) -> pd.DataFrame:
    task_names = list(task_names)
    rows = []
    for lineage, ids in ANALYSIS_LINEAGES.items():
        part = summary.loc[summary["checkpoint_id"].isin(ids)].sort_values("plot_date")
        if tuple(part["checkpoint_id"]) != tuple(ids):
            raise MultiModelPatternError(f"lineage chronology or coverage drifted: {lineage}")
        first, latest = part.iloc[0], part.iloc[-1]
        first_draws = np.mean(
            np.stack([draws[first.checkpoint_id][task][:, 0] for task in task_names], axis=1),
            axis=1,
        )
        latest_draws = np.mean(
            np.stack([draws[latest.checkpoint_id][task][:, 0] for task in task_names], axis=1),
            axis=1,
        )
        difference = latest_draws - first_draws
        rows.append(
            {
                "lineage": lineage,
                "models": len(part),
                "first_checkpoint_id": first.checkpoint_id,
                "latest_checkpoint_id": latest.checkpoint_id,
                "first_date": pd.Timestamp(first.plot_date).date().isoformat(),
                "latest_date": pd.Timestamp(latest.plot_date).date().isoformat(),
                "headline_delta_points": 100
                * (latest.overall_headline_f1 - first.overall_headline_f1),
                "paired_item_ci_low_points": 100 * np.quantile(difference, 0.025),
                "paired_item_ci_high_points": 100 * np.quantile(difference, 0.975),
                "binary_positive_rate_bias_delta_points": 100
                * (latest.binary_positive_rate_bias - first.binary_positive_rate_bias),
                "binary_precision_delta_points": 100
                * (latest.binary_precision - first.binary_precision),
                "binary_recall_delta_points": 100
                * (latest.binary_positive_recall - first.binary_positive_recall),
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_profiles(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    draws: Mapping[str, Mapping[str, np.ndarray]],
    task_names: Iterable[str],
) -> pd.DataFrame:
    task_names = list(task_names)
    scores = metrics.pivot(index="checkpoint_id", columns="task", values="headline_f1").loc[
        summary["checkpoint_id"], task_names
    ]
    model_means = scores.mean(axis=1)
    developers = summary.set_index("checkpoint_id")["posttraining_developer"]
    rows = []
    for left, right in combinations(scores.index, 2):
        task_differences = scores.loc[left] - scores.loc[right]
        mean_gap = abs(float(model_means[left] - model_means[right]))
        row = {
            "left_checkpoint_id": left,
            "right_checkpoint_id": right,
            "cross_developer": developers[left] != developers[right],
            "overall_gap_points": 100 * mean_gap,
            "mean_absolute_task_gap_points": 100 * float(task_differences.abs().mean()),
            "task_profile_spearman_rho": _rank(scores.loc[left], scores.loc[right]),
            "near_tied_half_point": mean_gap <= 0.005,
            "task_gap_ci_low_points": np.nan,
            "task_gap_ci_high_points": np.nan,
        }
        if row["near_tied_half_point"]:
            per_task = []
            for task in task_names:
                per_task.append(
                    np.abs(draws[left][task][:, 0] - draws[right][task][:, 0])
                )
            distribution = np.mean(np.stack(per_task, axis=1), axis=1)
            row["task_gap_ci_low_points"] = 100 * np.quantile(distribution, 0.025)
            row["task_gap_ci_high_points"] = 100 * np.quantile(distribution, 0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def _trend_scope_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = [
        "binary_positive_rate_bias",
        "binary_precision",
        "binary_positive_recall",
        "binary_negative_recall",
        "overall_headline_f1",
    ]
    scopes: dict[str, tuple[str, ...] | None] = {"all_21": None, **REPLICATED_DENSE_LINEAGES}
    rows = []
    start = pd.to_datetime(summary["plot_date"]).min()
    for scope, ids in scopes.items():
        part = summary if ids is None else summary.loc[summary["checkpoint_id"].isin(ids)]
        part = part.sort_values("plot_date")
        x = ((pd.to_datetime(part["plot_date"]) - start).dt.days / 365.25).to_numpy()
        for metric in metrics:
            y = part[metric].to_numpy(dtype=float)
            loo_low, loo_high = _leave_one_out_slope(x, y) if len(part) > 2 else (np.nan, np.nan)
            rows.append(
                {
                    "section": "release_date_trend",
                    "result": metric,
                    "scope": scope,
                    "estimate": _linear_slope(x, y),
                    "units": "proportion_per_year",
                    "n": len(part),
                    "interval_low": loo_low,
                    "interval_high": loo_high,
                    "uncertainty_type": "leave_one_model_out_range",
                    "p_value": np.nan,
                    "p_value_type": "none",
                    "evidentiary_status": EXPLORATORY_STATUS,
                    "interpretation": "Hard-label behavior; date, family, and model design remain confounded.",
                }
            )
    return rows


def _decision_space_rows(
    trends: pd.DataFrame, *, locked_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    hypothesis = next(
        (row for row in locked_spec["hypotheses"] if row.get("id") == "H1_decision_space"),
        None,
    )
    if (
        not isinstance(hypothesis, Mapping)
        or hypothesis.get("status") != "testable"
        or hypothesis.get("moderator") != "log_effective_label_count"
        or hypothesis.get("direction") != "positive"
    ):
        raise MultiModelPatternError("locked decision-space hypothesis drifted")
    locked_scope_names = {
        "Qwen dense 27–32B": "qwen_dense_27_32b",
        "Qwen dense 72B": "qwen_dense_72b",
        "Llama dense 70B": "llama_dense_70b",
    }
    for display_name, locked_name in locked_scope_names.items():
        if tuple(locked_spec["lineages"].get(locked_name, [])) != tuple(
            REPLICATED_DENSE_LINEAGES[display_name]
        ):
            raise MultiModelPatternError(f"locked lineage definition drifted: {display_name}")
    rows = []
    lineage_symmetric_rhos: list[float] = []
    for score_metric in ("headline_f1", "symmetric_macro_f1"):
        for scope_index, scope in enumerate(("all_21", *REPLICATED_DENSE_LINEAGES)):
            task = trends.loc[
                trends["scope"].eq(scope) & trends["metric"].eq(score_metric)
            ].copy()
            source = task.groupby("source_family", as_index=False).agg(
                log_effective_label_count=("log_effective_label_count", "mean"),
                slope_per_year=("slope_per_year", "mean"),
            )
            if score_metric == "headline_f1":
                rho, p_value = _permutation_rank(
                    source["log_effective_label_count"].to_numpy(),
                    source["slope_per_year"].to_numpy(),
                    iterations=int(locked_spec["permutation_draws"]),
                    seed=int(locked_spec["seed"]) + scope_index,
                )
                p_value_type = "locked_h1_one_sided_positive_permutation_post_outcome"
            else:
                rho = _rank(
                    source["log_effective_label_count"], source["slope_per_year"]
                )
                p_value = np.nan
                p_value_type = "none_symmetric_rescoring_sensitivity"
            loo_low, loo_high = _leave_one_out_rank(
                source["log_effective_label_count"].to_numpy(),
                source["slope_per_year"].to_numpy(),
            )
            if score_metric == "symmetric_macro_f1" and scope != "all_21":
                lineage_symmetric_rhos.append(rho)
            status = (
                "suggestive_exploratory"
                if rho > 0 and loo_low > 0
                else "mixed_exploratory"
            )
            rows.append(
                {
                    "section": "decision_space",
                    "result": f"effective_labels_vs_{score_metric}_trend",
                    "scope": (
                        "all_21:13_source_families"
                        if scope == "all_21"
                        else f"{scope}:13_source_families"
                    ),
                    "estimate": rho,
                    "units": "spearman_rho",
                    "n": len(source),
                    "interval_low": loo_low,
                    "interval_high": loo_high,
                    "uncertainty_type": "leave_one_source_out_range",
                    "p_value": p_value,
                    "p_value_type": p_value_type,
                    "evidentiary_status": status,
                    "interpretation": "Exploratory moderator; task format, scorer, and ontology remain confounded.",
                }
            )
    positive_lineages = int(np.sum(np.asarray(lineage_symmetric_rhos) > 0))
    rows.append(
        {
            "section": "decision_space",
            "result": "replicated_dense_lineages_positive_after_symmetric_rescoring",
            "scope": "three_replicated_dense_lineages",
            "estimate": float(positive_lineages),
            "units": "lineages",
            "n": len(lineage_symmetric_rhos),
            "interval_low": np.nan,
            "interval_high": np.nan,
            "uncertainty_type": "none",
            "p_value": np.nan,
            "p_value_type": "none",
            "evidentiary_status": (
                "suggestive_exploratory" if positive_lineages >= 2 else "not_supported"
            ),
            "interpretation": "A sign recurrence check only; the lineage-specific correlations are weak and post-outcome.",
        }
    )
    categorical = trends.loc[
        trends["scope"].eq("all_21")
        & trends["label_kind"].eq("categorical")
        & trends["metric"].isin(["headline_f1", "label_coverage", "rare_class_recall"])
    ].pivot(index="task", columns="metric", values="slope_per_year")
    categorical = categorical.join(
        trends.loc[
            trends["scope"].eq("all_21") & trends["label_kind"].eq("categorical")
        ]
        .drop_duplicates("task")
        .set_index("task")[["log_effective_label_count", "source_family"]]
    ).dropna()
    for result, left, right in (
        ("effective_labels_vs_label_coverage_trend", "log_effective_label_count", "label_coverage"),
        ("effective_labels_vs_rare_recall_trend", "log_effective_label_count", "rare_class_recall"),
        ("headline_vs_label_coverage_trend", "headline_f1", "label_coverage"),
        ("headline_vs_rare_recall_trend", "headline_f1", "rare_class_recall"),
    ):
        source = categorical.groupby("source_family", as_index=False)[[left, right]].mean()
        rho = _rank(source[left], source[right])
        loo_low, loo_high = _leave_one_out_rank(
            source[left].to_numpy(), source[right].to_numpy()
        )
        rows.append(
            {
                "section": "categorical_mechanism",
                "result": result,
                "scope": "categorical_source_families",
                "estimate": rho,
                "units": "spearman_rho",
                "n": len(source),
                "interval_low": loo_low,
                "interval_high": loo_high,
                "uncertainty_type": "leave_one_source_out_range",
                "p_value": np.nan,
                "p_value_type": "none",
                "evidentiary_status": "exploratory_mechanism_check",
                "interpretation": "Describes hard-label coverage or rare-class recall; it does not identify training causes.",
            }
        )
    categorical_source = categorical.groupby("source_family", as_index=False)[
        [
            "log_effective_label_count",
            "headline_f1",
            "label_coverage",
            "rare_class_recall",
        ]
    ].mean()
    for control in ("label_coverage", "rare_class_recall"):
        rows.append(
            {
                "section": "categorical_mechanism",
                "result": f"effective_labels_vs_f1_trend_partial_{control}",
                "scope": "eight_categorical_source_families",
                "estimate": _partial_rank(
                    categorical_source["log_effective_label_count"],
                    categorical_source["headline_f1"],
                    categorical_source[control],
                ),
                "units": "partial_rank_correlation",
                "n": len(categorical_source),
                "interval_low": np.nan,
                "interval_high": np.nan,
                "uncertainty_type": "none",
                "p_value": np.nan,
                "p_value_type": "none",
                "evidentiary_status": "exploratory_mechanism_check",
                "interpretation": "Residualizes average ranks on the named hard-label mechanism; eight source families are too few for a causal explanation.",
            }
        )
    return rows


def build_pattern_summary(
    benchmark: PanelSelection,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    lineages: pd.DataFrame,
    pairs: pd.DataFrame,
    trends: pd.DataFrame,
    draws: Mapping[str, Mapping[str, np.ndarray]],
    *,
    iterations: int,
    seed: int,
    external_event_path: Path,
    locked_heterogeneity_spec: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        result: str,
        scope: str,
        estimate: float,
        units: str,
        n: int,
        status: str,
        interpretation: str,
        interval_low: float = np.nan,
        interval_high: float = np.nan,
        uncertainty_type: str = "none",
        p_value: float = np.nan,
        p_value_type: str = "none",
    ) -> None:
        rows.append(
            {
                "section": section,
                "result": result,
                "scope": scope,
                "estimate": estimate,
                "units": units,
                "n": n,
                "interval_low": interval_low,
                "interval_high": interval_high,
                "uncertainty_type": uncertainty_type,
                "p_value": p_value,
                "p_value_type": p_value_type,
                "evidentiary_status": status,
                "interpretation": interpretation,
            }
        )

    medium = summary.loc[summary["compute_class"].eq("Medium (10–40B active)")]
    large = summary.loc[summary["compute_class"].eq("Large (>40B active)")]
    medium_frontier = float(medium["class_frontier_f1"].iloc[-1])
    large_frontier = float(large["class_frontier_f1"].iloc[-1])
    catching = medium.loc[medium["class_frontier_f1"].ge(large_frontier)].iloc[0]
    large_setter = large.loc[large["mean_f1"].idxmax()]
    selected_frontier_difference = np.mean(
        np.stack(
            [
                draws[catching.checkpoint_id][task][:, 0]
                - draws[large_setter.checkpoint_id][task][:, 0]
                for task in benchmark.task_names
            ],
            axis=1,
        ),
        axis=1,
    )
    add(
        "frontier_compression",
        "medium_minus_large_tested_frontier",
        "21_checkpoint_roster",
        100 * (medium_frontier - large_frontier),
        "f1_points",
        21,
        "supported_descriptive",
        f"In point estimates, the existing compute-class staircase shows that {catching.display_name} first edged past the earlier best tested large-model average on {pd.Timestamp(catching.plot_date).date()}; the frontier setters are outcome-selected.",
        100 * np.quantile(selected_frontier_difference, .025),
        100 * np.quantile(selected_frontier_difference, .975),
        "paired_item_bootstrap_95pct_fixed_selected_pair_conditional_on_18_tasks",
    )
    date_numeric = pd.to_datetime(summary["plot_date"]).map(pd.Timestamp.toordinal).to_numpy()
    add(
        "frontier_compression",
        "release_date_vs_average_f1",
        "21_checkpoint_roster",
        _rank(date_numeric, summary["mean_f1"]),
        "spearman_rho",
        21,
        "not_general_temporal_improvement",
        "Release date is not a monotone predictor of average F1 in this non-exhaustive roster.",
    )
    for row in lineages.itertuples(index=False):
        add(
            "lineage_change",
            row.lineage,
            "first_to_latest",
            row.headline_delta_points,
            "f1_points",
            row.models,
            "supported_descriptive",
            "First-to-latest change within an approximately matched developer/model sequence.",
            row.paired_item_ci_low_points,
            row.paired_item_ci_high_points,
            "paired_item_bootstrap_95pct_conditional_on_18_tasks",
        )

    profiles = metrics.pivot(index="checkpoint_id", columns="task", values="headline_f1")
    grand = float(profiles.to_numpy().mean())
    task_component = profiles.mean(axis=0) - grand
    model_component = profiles.mean(axis=1) - grand
    interaction = profiles.subtract(profiles.mean(axis=0), axis=1).subtract(
        profiles.mean(axis=1), axis=0
    ) + grand
    total_ss = float(((profiles - grand) ** 2).to_numpy().sum())
    interaction_share = float((interaction**2).to_numpy().sum() / total_ss)
    cross = pairs.loc[pairs["cross_developer"]]
    near = pairs.loc[pairs["near_tied_half_point"]]
    developers = summary.set_index("checkpoint_id")["posttraining_developer"]
    developer_profiles = profiles.join(developers).groupby("posttraining_developer").mean()
    developer_pair_rhos = [
        _rank(developer_profiles.loc[left], developer_profiles.loc[right])
        for left, right in combinations(developer_profiles.index, 2)
    ]
    add(
        "task_profiles",
        "median_cross_developer_profile_correlation",
        "cross_developer_model_pairs",
        float(cross["task_profile_spearman_rho"].median()),
        "spearman_rho",
        len(cross),
        "supported_descriptive",
        "Checkpoint pairs from different developers have similar raw task-score rankings; the pairs are dependent.",
    )
    add(
        "task_profiles",
        "median_developer_equal_profile_correlation",
        "eight_developer_mean_profiles",
        float(np.median(developer_pair_rhos)),
        "spearman_rho",
        len(developer_profiles),
        "supported_descriptive",
        "The ranking result survives equal weighting of the eight post-training developers.",
    )
    add(
        "task_profiles",
        "model_by_task_interaction_variance_share",
        "21_by_18_score_panel",
        interaction_share,
        "share_of_total_sum_squares",
        len(metrics),
        "supported_descriptive",
        "Observed task-centered deviations remain after removing task and model mean levels; this residual share includes item-sampling noise.",
    )
    add(
        "task_profiles",
        "near_tied_pair_mean_absolute_task_gap",
        "overall_gap_at_most_half_f1_point",
        float(near["mean_absolute_task_gap_points"].mean()),
        "f1_points",
        len(near),
        "appendix_descriptive",
        "Near-equal aggregate scores conceal larger task-level differences; pairs are outcome-selected and the estimate is descriptive.",
    )
    winners = metrics.loc[
        metrics.groupby("task")["headline_f1"].transform("max").eq(metrics["headline_f1"])
    ]
    add(
        "task_profiles",
        "unique_task_winners",
        "18_tasks",
        float(winners["checkpoint_id"].nunique()),
        "models",
        18,
        "appendix_descriptive",
        "Ties count every model sharing a task's highest point estimate.",
    )

    trend_scope_rows = _trend_scope_rows(summary)
    rows.extend(trend_scope_rows)
    binary_trends = trends.loc[
        trends["scope"].eq("all_21")
        & trends["label_kind"].eq("binary")
        & trends["metric"].isin(
            ["predicted_positive_rate", "precision", "positive_recall", "negative_recall"]
        )
    ]
    for metric, expected_direction in (
        ("predicted_positive_rate", "negative"),
        ("precision", "positive"),
        ("positive_recall", "negative"),
        ("negative_recall", "positive"),
    ):
        task = binary_trends.loc[binary_trends["metric"].eq(metric)]
        source = task.groupby("source_family", as_index=False)["slope_per_year"].mean()
        if expected_direction == "negative":
            successes = int((source["slope_per_year"] < 0).sum())
        else:
            successes = int((source["slope_per_year"] > 0).sum())
        p_value = binomtest(successes, len(source), 0.5, alternative="greater").pvalue
        add(
            "binary_selectivity",
            f"source_families_in_expected_direction_{metric}",
            "binary_source_families",
            float(successes),
            "source_families",
            len(source),
            "supported_exploratory" if successes == len(source) else "mixed",
            "Direction is evaluated across source-family-average task slopes; the post-outcome sign test is nominal and the metrics are mechanically related.",
            p_value=p_value,
            p_value_type="post_outcome_nominal_exact_sign_test",
        )
    dense_direction_metrics = (
        ("binary_positive_rate_bias", "negative"),
        ("binary_precision", "positive"),
        ("binary_positive_recall", "negative"),
        ("binary_negative_recall", "positive"),
    )
    for metric, expected_direction in dense_direction_metrics:
        values = [
            float(row["estimate"])
            for row in trend_scope_rows
            if row["result"] == metric and row["scope"] != "all_21"
        ]
        successes = int(
            np.sum(np.asarray(values) < 0)
            if expected_direction == "negative"
            else np.sum(np.asarray(values) > 0)
        )
        add(
            "binary_selectivity",
            f"replicated_dense_lineages_in_expected_direction_{metric}",
            "three_replicated_dense_lineages",
            float(successes),
            "lineages",
            len(values),
            "supported_exploratory" if successes >= 2 else "mixed",
            "Point-sign recurrence across the three dense lineages with at least three checkpoints; see release-date rows for leave-one-model-out ranges.",
        )
    score_trends = trends.loc[
        trends["metric"].isin(["headline_f1", "symmetric_macro_f1"])
    ]
    for scope in ("all_21", *REPLICATED_DENSE_LINEAGES):
        for metric in ("headline_f1", "symmetric_macro_f1"):
            part = score_trends.loc[
                score_trends["scope"].eq(scope) & score_trends["metric"].eq(metric)
            ]
            binary = part.loc[part["label_kind"].eq("binary"), "slope_per_year"]
            nonbinary = part.loc[~part["label_kind"].eq("binary"), "slope_per_year"]
            observed = float(binary.mean() - nonbinary.mean())
            loo = []
            for index in part.index:
                reduced = part.drop(index)
                loo.append(
                    reduced.loc[
                        reduced["label_kind"].eq("binary"), "slope_per_year"
                    ].mean()
                    - reduced.loc[
                        ~reduced["label_kind"].eq("binary"), "slope_per_year"
                    ].mean()
                )
            add(
                "scoring_sensitivity",
                f"binary_minus_nonbinary_{metric}_trend",
                f"{scope}:18_tasks",
                observed * 100,
                "f1_points_per_year",
                len(part),
                "supporting_exploratory" if max(loo) < 0 else "mixed_exploratory",
                "A less negative symmetric-F1 contrast indicates that positive-class scoring amplifies, but does not necessarily explain, the binary-task lag.",
                min(loo) * 100,
                max(loo) * 100,
                "leave_one_task_out_range",
            )

    rows.extend(
        _decision_space_rows(trends, locked_spec=locked_heterogeneity_spec)
    )
    if (
        not external_event_path.exists()
        or file_sha256(external_event_path) != EXTERNAL_EVENT_SUMMARY_SHA256
    ):
        raise MultiModelPatternError("frozen external event summary identity failed")
    external = pd.read_csv(external_event_path)
    if (
        not {"aggregation", "delta_f1_points"}.issubset(external.columns)
        or external["aggregation"].duplicated().any()
        or set(external["aggregation"]) != {"equal_task", "equal_source_family"}
        or not np.isfinite(pd.to_numeric(external["delta_f1_points"], errors="coerce")).all()
    ):
        raise MultiModelPatternError("frozen external event summary schema drifted")
    for aggregation, scope in (
        ("equal_task", "four_out_of_database_event_tasks"),
        ("equal_source_family", "four_external_source_families"),
    ):
        estimate = float(
            external.loc[
                external["aggregation"].eq(aggregation), "delta_f1_points"
            ].iloc[0]
        )
        reversed_pattern = estimate < 0
        add(
            "external_validity",
            f"external_event_{aggregation}",
            scope,
            estimate,
            "f1_points",
            4,
            "failed_generalization" if reversed_pattern else "external_pattern_not_reversed",
            (
                "The repository event pattern reversed externally; do not generalize task-category progress."
                if reversed_pattern
                else "The external evidence no longer reverses the repository event pattern."
            ),
        )
    add(
        "audit",
        "frozen_task_item_keys_per_checkpoint",
        benchmark.panel_id,
        float(benchmark.expected_items),
        "task_item_keys",
        int(summary["checkpoint_id"].nunique()),
        "passed",
        "Every checkpoint passed the manifest task-item-key, gold-label, task-fingerprint, and coverage validators.",
    )
    add(
        "audit",
        "immutable_model_identity_and_revision_evidence",
        "eligible_checkpoint_roster",
        float(summary["checkpoint_id"].nunique()),
        "checkpoints",
        int(summary["checkpoint_id"].nunique()),
        "passed",
        "Registry identities, immutable revisions, metadata evidence, and prediction hashes were revalidated.",
    )
    add(
        "audit",
        "reproduced_existing_headline_f1_values",
        "21_checkpoints_by_18_tasks",
        float(len(metrics)),
        "model_task_scores",
        len(metrics),
        "passed",
        "Every existing model-task headline F1 was reproduced within 1e-12.",
    )
    add(
        "audit",
        "malformed_outputs_scored_as_gold_class_misses",
        "all_prediction_rows",
        float(metrics["malformed_count"].sum()),
        "outputs",
        int(metrics["n"].sum()),
        "passed",
        "Malformed outputs remain invalid for propensity summaries and are forced to the wrong class for scoring.",
    )
    add(
        "audit",
        "reused_compute_class_staircase",
        "active_parameter_classes",
        float(summary["class_frontier_setter"].sum()),
        "frontier_updates",
        len(summary),
        "passed",
        f"Validated and reused {summary['compute_staircase_source'].iloc[0]} rather than generating a duplicate staircase.",
    )
    add(
        "audit",
        "reused_locked_decision_space_specification",
        str(locked_heterogeneity_spec["analysis_id"]),
        float(locked_heterogeneity_spec["permutation_draws"]),
        "permutation_draws",
        13,
        "passed",
        f"Headline decision-space p-values reuse the locked post-outcome exploratory seed {locked_heterogeneity_spec['seed']} and its hash-validated inputs; symmetric rescoring is reported as a sensitivity without a new p-value.",
    )
    add(
        "audit",
        "shared_paired_item_bootstrap",
        "21_checkpoints_by_18_tasks",
        float(iterations),
        "draws",
        len(metrics),
        "passed",
        f"All checkpoints use the same within-task item draws with seed {seed}.",
    )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_binary_selectivity(summary: pd.DataFrame, output: Path) -> None:
    data = summary.sort_values("plot_date").copy()
    panels = [
        ("binary_positive_rate_bias", "Positive-rate bias", 100, "pp vs. gold"),
        ("binary_precision", "Precision", 100, "%"),
        ("binary_positive_recall", "Recall", 100, "%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.8), sharex=True)
    x = pd.to_datetime(data["plot_date"])
    years = (x - x.min()).dt.days.to_numpy() / 365.25
    for ax, (column, title, scale, unit) in zip(axes, panels):
        values = scale * data[column].to_numpy(dtype=float)
        dense = data["architecture"].eq("dense").to_numpy()
        ax.scatter(x[dense], values[dense], s=27, color=DARK, label="Dense", zorder=3)
        ax.scatter(
            x[~dense], values[~dense], s=38, marker="^", facecolors="white",
            edgecolors=RED, linewidths=1.3, label="MoE", zorder=3,
        )
        fit = np.polyfit(years, values, 1)
        xline = pd.date_range(x.min(), x.max(), periods=100)
        yline = fit[0] * ((xline - x.min()).days / 365.25) + fit[1]
        ax.plot(xline, yline, color=GRAY, lw=1.4, zorder=2)
        ax.axhline(0, color=LIGHT_GRAY, lw=.8, zorder=1) if column.endswith("bias") else None
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0, labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for spine in ax.spines.values():
            spine.set_color("#b3b3b3")
            spine.set_linewidth(.8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        "Later models make fewer positive calls on binary tasks",
        x=.08, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        .08, .89,
        "Equal-task means across eight tasks; hard-label selectivity, not probability calibration",
        fontsize=8, color="0.4",
    )
    fig.tight_layout(rect=(.02, .03, .99, .84))
    _save_figure(fig, output)


def plot_task_profiles(metrics: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    task_order = metrics.drop_duplicates("task")["task"].tolist()
    model_order = summary.sort_values("plot_date")["checkpoint_id"].tolist()
    matrix = metrics.pivot(index="checkpoint_id", columns="task", values="headline_f1").loc[
        model_order, task_order
    ]
    centered = 100 * matrix.subtract(matrix.mean(axis=0), axis=1)
    labels = (
        summary.set_index("checkpoint_id").loc[model_order, "display_name"]
        .str.replace(" Instruct", "", regex=False)
        .str.replace(" Dynamic", "", regex=False)
        .tolist()
    )
    limit = max(8.0, float(np.quantile(np.abs(centered.to_numpy()), .95)))
    fig = plt.figure(figsize=(13.2, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.2, 7.8), wspace=.04)
    mean_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=mean_ax)
    positions = np.arange(len(model_order))
    means = 100 * matrix.mean(axis=1).to_numpy()
    mean_ax.plot(means, positions, color=GRAY, lw=1.0, zorder=1)
    mean_ax.scatter(means, positions, color=DARK, s=18, zorder=2)
    mean_ax.set_yticks(positions, labels, fontsize=6.5)
    mean_ax.set_ylim(len(model_order) - .5, -.5)
    padding = max(1.0, .08 * (means.max() - means.min()))
    mean_ax.set_xlim(means.min() - padding, means.max() + padding)
    mean_ax.set_xlabel("Mean F1 (%)", fontsize=8)
    mean_ax.set_title("18-task average", fontsize=9, fontweight="bold")
    mean_ax.tick_params(length=0, labelsize=6.5)
    image = ax.imshow(centered.to_numpy(), aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(
        range(len(task_order)),
        [SHORT_TASK_NAMES[task].replace("\n", " ") for task in task_order],
        fontsize=6,
        rotation=43,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(length=0)
    ordered_summary = summary.sort_values("plot_date").reset_index(drop=True)
    for index in range(len(model_order) - 1):
        if pd.Timestamp(ordered_summary.iloc[index].plot_date).year != pd.Timestamp(
            ordered_summary.iloc[index + 1].plot_date
        ).year:
            ax.axhline(index + .5, color="white", lw=1.4)
            mean_ax.axhline(index + .5, color=LIGHT_GRAY, lw=.8)
    colorbar = fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    colorbar.set_label("F1 points above or below each task's model average", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=0)
    fig.suptitle(
        "Similar averages conceal different task profiles",
        x=.07, ha="left", fontsize=15, fontweight="bold", y=.985,
    )
    fig.text(
        .07, .945,
        "Each column is centered on the 21-model mean for that task; models are ordered by release date",
        fontsize=8, color="0.4",
    )
    for current_ax in (ax, mean_ax):
        for spine in current_ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=.18, right=.97, bottom=.22, top=.89)
    _save_figure(fig, output)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_analysis(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    task_metadata_path: Path = DEFAULT_TASK_METADATA,
    compute_staircase_path: Path | None = None,
    heterogeneity_spec_path: Path = DEFAULT_HETEROGENEITY_SPEC,
    iterations: int | None = None,
    seed: int | None = None,
    make_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    benchmark = load_panel_manifest(manifest_path)
    registry = load_compact_registry(registry_path, benchmark)
    iterations = int(iterations or benchmark.bootstrap["iterations"])
    seed = int(seed if seed is not None else benchmark.bootstrap["seed"])
    if iterations < 100:
        raise MultiModelPatternError("bootstrap iterations must be at least 100")
    shared_indices = bootstrap_indices(benchmark, iterations=iterations, seed=seed)
    shared_weights = bootstrap_weight_matrices(shared_indices)
    del shared_indices
    checkpoints = pd.read_csv(root / "broad18" / "checkpoint_summary.csv")
    existing_task_scores = pd.read_csv(root / "broad18" / "model_by_task.csv")
    locked_heterogeneity_spec = load_heterogeneity_spec(heterogeneity_spec_path)
    locked_metadata_path = (
        REPO / locked_heterogeneity_spec["inputs"]["task_metadata"]["path"]
    ).resolve()
    locked_scores_path = (
        REPO / locked_heterogeneity_spec["inputs"]["longitudinal_scores"]["path"]
    ).resolve()
    if (
        task_metadata_path.resolve() != locked_metadata_path
        or (root / "broad18" / "model_by_task.csv").resolve() != locked_scores_path
    ):
        raise MultiModelPatternError(
            "decision-space inputs do not match the locked exploratory specification"
        )
    metadata = _task_metadata(benchmark, task_metadata_path)
    raw_metrics, draws = audit_and_score(
        benchmark, registry, checkpoints, existing_task_scores, shared_weights
    )
    summary = build_model_summary(raw_metrics, checkpoints)
    summary = attach_validated_compute_staircase(
        summary,
        compute_staircase_path
        if compute_staircase_path is not None
        else root / "twitter_claims" / "05_compute_class_staircase.csv",
    )
    trends = build_task_trends(raw_metrics, metadata)
    metrics = raw_metrics.merge(metadata, on="task", validate="many_to_one")
    lineages = build_lineage_changes(summary, draws, benchmark.task_names)
    pairs = build_pairwise_profiles(metrics, summary, draws, benchmark.task_names)
    patterns = build_pattern_summary(
        benchmark,
        metrics,
        summary,
        lineages,
        pairs,
        trends,
        draws,
        iterations=iterations,
        seed=seed,
        external_event_path=root / "event_progress" / "external_confirmation_summary.csv",
        locked_heterogeneity_spec=locked_heterogeneity_spec,
    )
    outputs = {
        "model_task_metrics": metrics.sort_values(["plot_date", "checkpoint_id", "task"]),
        "model_summary": summary,
        "lineage_changes": lineages,
        "pairwise_profiles": pairs,
        "task_trends": trends,
        "pattern_summary": patterns,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output_dir / f"{name}.csv")
    if make_figures:
        plot_binary_selectivity(summary, output_dir / "01_binary_selectivity")
        plot_task_profiles(metrics, summary, output_dir / "02_task_profiles")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-metadata", type=Path, default=DEFAULT_TASK_METADATA)
    parser.add_argument("--compute-staircase", type=Path, default=DEFAULT_COMPUTE_STAIRCASE)
    parser.add_argument(
        "--heterogeneity-spec", type=Path, default=DEFAULT_HETEROGENEITY_SPEC
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    build_analysis(
        manifest_path=args.manifest,
        registry_path=args.registry,
        root=args.root,
        output_dir=args.output_dir,
        task_metadata_path=args.task_metadata,
        compute_staircase_path=args.compute_staircase,
        heterogeneity_spec_path=args.heterogeneity_spec,
        iterations=args.iterations,
        seed=args.seed,
        make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
