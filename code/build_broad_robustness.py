#!/usr/bin/env python3
"""Stress-test the Broad18 open-model comparison using existing evidence only.

This script does not run inference.  It reads the frozen Broad18 manifest,
checkpoint registry, task metadata, run metadata, and the three published
Broad18 CSVs.  It compares Qwen3.6 27B with two 2024 Llama 70B baselines under
four transparent weighting rules and reports heterogeneity across the frozen
task strata and analysis-defined source families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = REPO / "experiments" / "frontier_broad_18.yaml"
DEFAULT_REGISTRY = REPO / "experiments" / "frontier_broad_checkpoints_2026.yaml"
DEFAULT_BROAD_DIR = REPO / "output" / "sidecar" / "frontier_2026" / "broad18"

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260820
CANDIDATE_ID = "qwen3_6_27b_fp8_hive"
BASELINE_IDS = (
    "llama3_1_70b_instruct_fp8_dynamic_hive",
    "llama3_3_70b_instruct_fp8_dynamic_hive",
)
PRIMARY_BASELINE_ID = BASELINE_IDS[0]


class BroadRobustnessError(ValueError):
    """Raised when the frozen evidence needed for the stress test has drifted."""


# These are analysis-defined source/publication families, not verified
# independent draws and not annotation families. Stable machine IDs are
# intentionally separate from display labels. Bootstrap order is the lexical
# order of those IDs, so manifest rows and label wording cannot change the
# deterministic draws.
SOURCE_CLUSTERS: dict[str, dict[str, Any]] = {
    "burnham_2025": {
        "label": "Burnham (2025)",
        "year": 2025,
        "tasks": (
            "burnham_covid_threat_minimization",
            "burnham_polnli_entailment",
            "burnham_polnli_event_entailment",
            "burnham_trump_stance",
        ),
    },
    "policy_agendas_2025": {
        "label": "Policy Agendas Project (2025)",
        "year": 2025,
        "tasks": (
            "cap_crs_policy_topic",
            "cap_party_platform_policy_topic",
        ),
    },
    "chae_davidson_2026": {
        "label": "Chae and Davidson (2026)",
        "year": 2026,
        "tasks": ("chae_semeval_stance",),
    },
    "dicocco_monechi_2022": {
        "label": "Di Cocco and Monechi (2022)",
        "year": 2022,
        "tasks": ("dicocco_manifesto_populism",),
    },
    "douglass_2024": {
        "label": "Douglass et al. (2024)",
        "year": 2024,
        "tasks": ("douglass_icbe_sentence_event_type",),
    },
    "erlich_2022": {
        "label": "Erlich et al. (2022)",
        "year": 2022,
        "tasks": ("erlich_ati_topics",),
    },
    "gilardi_2023": {
        "label": "Gilardi et al. (2023)",
        "year": 2023,
        "tasks": ("gilardi_relevance", "gilardi_stance"),
    },
    "halterman_keith_2026": {
        "label": "Halterman and Keith (2026)",
        "year": 2026,
        "tasks": ("halterman_keith_bfrs",),
    },
    "haunss_2025": {
        "label": "Haunss et al. (2025)",
        "year": 2025,
        "tasks": ("haunss_papea_fgz_forms",),
    },
    "muller_fujimura_2025": {
        "label": "Müller and Fujimura (2025)",
        "year": 2025,
        "tasks": ("muller_fujimura_campaign_policy_area",),
    },
    "rheault_2019": {
        "label": "Rheault et al. (2019)",
        "year": 2019,
        "tasks": ("rheault_line_of_fire_incivility",),
    },
    "theocharis_2020": {
        "label": "Theocharis et al. (2020)",
        "year": 2020,
        "tasks": ("theocharis_dynamics_incivility",),
    },
    "pendzel_2023": {
        "label": "Pendzel et al. (2023)",
        "year": 2023,
        "tasks": ("twitcivility_impoliteness",),
    },
}

AGGREGATIONS = (
    ("equal_task", "Equal task", None),
    ("equal_annotation_family", "Equal annotation family", "annotation_family"),
    ("equal_source_family", "Equal source family", "source_cluster_id"),
    ("equal_complexity", "Equal complexity stratum", "complexity"),
)

DISPLAY_NAMES = {
    CANDIDATE_ID: "Qwen3.6 27B FP8",
    BASELINE_IDS[0]: "Llama 3.1 70B",
    BASELINE_IDS[1]: "Llama 3.3 70B",
}


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _source_task_map() -> dict[str, str]:
    task_to_source: dict[str, str] = {}
    for source_id, source in SOURCE_CLUSTERS.items():
        for task in source["tasks"]:
            if task in task_to_source:
                raise BroadRobustnessError(f"task {task} appears in two source clusters")
            task_to_source[str(task)] = source_id
    return task_to_source


TASK_TO_SOURCE = _source_task_map()


def source_cluster_order(task_names: Sequence[str]) -> tuple[str, ...]:
    """Return lexically ordered stable source IDs after exact task validation."""
    observed: set[str] = set()
    for task_name in task_names:
        try:
            source_id = TASK_TO_SOURCE[str(task_name)]
        except KeyError as exc:
            raise BroadRobustnessError(
                f"task {task_name} has no explicit source cluster"
            ) from exc
        observed.add(source_id)
    if len(observed) != 13:
        raise BroadRobustnessError(
            f"Broad18 must contain exactly 13 source clusters, found {len(observed)}"
        )
    return tuple(sorted(observed))


def source_cluster_bootstrap(
    source_deltas: Mapping[str, float],
    source_order: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample the 13 source-family means with replacement."""
    if iterations <= 0:
        raise BroadRobustnessError("bootstrap iterations must be positive")
    if tuple(source_deltas) != tuple(source_order):
        raise BroadRobustnessError(
            "source deltas must use lexically ordered stable source IDs"
        )
    values = np.asarray([source_deltas[source] for source in source_order], dtype=float)
    if len(values) != 13 or not np.isfinite(values).all():
        raise BroadRobustnessError("source bootstrap requires 13 finite cluster means")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(iterations, len(values)))
    return values[sampled].mean(axis=1)


def _load_manifest(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise BroadRobustnessError("Broad18 manifest must be a mapping")
    tasks = raw.get("tasks")
    if (
        raw.get("panel_id") != "frontier_broad_18"
        or int(raw.get("expected_tasks", -1)) != 18
        or int(raw.get("expected_items", -1)) != 3600
        or not isinstance(tasks, list)
        or len(tasks) != 18
    ):
        raise BroadRobustnessError("expected the frozen 18-task/3,600-item Broad18 manifest")

    rows: list[dict[str, Any]] = []
    task_names = [str(task["name"]) for task in tasks]
    source_order = source_cluster_order(task_names)
    if set(task_names) != set(TASK_TO_SOURCE):
        missing = sorted(set(task_names) - set(TASK_TO_SOURCE))
        extra = sorted(set(TASK_TO_SOURCE) - set(task_names))
        raise BroadRobustnessError(
            f"source-cluster task coverage drifted; missing={missing}, extra={extra}"
        )
    for order, task in enumerate(tasks):
        task_name = str(task["name"])
        if int(task.get("expected_items", -1)) != 200:
            raise BroadRobustnessError(f"{task_name}: expected 200 Broad18 items")
        source_id = TASK_TO_SOURCE[task_name]
        source = SOURCE_CLUSTERS[source_id]
        task_path = REPO / "tasks" / f"{task_name}.yaml"
        task_definition = yaml.safe_load(task_path.read_text())
        citation = str(task_definition.get("source", ""))
        if str(source["year"]) not in citation:
            raise BroadRobustnessError(
                f"{task_name}: source publication year disagrees with task metadata"
            )
        year = int(source["year"])
        rows.append(
            {
                "task": task_name,
                "task_order": order,
                "annotation_family": str(task["annotation_type"]),
                "complexity": str(task["complexity"]),
                "source_cluster_id": source_id,
                "source_cluster_label": str(source["label"]),
                "source_cluster_order": source_order.index(source_id),
                "source_publication_year": year,
                "source_citation": citation,
                "publication_period": "2025_2026" if year >= 2025 else "through_2024",
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata["annotation_family"].nunique() != 5:
        raise BroadRobustnessError("Broad18 must contain five annotation families")
    if metadata["complexity"].nunique() != 3:
        raise BroadRobustnessError("Broad18 must contain three complexity strata")
    period_counts = metadata["publication_period"].value_counts().to_dict()
    if period_counts != {"2025_2026": 10, "through_2024": 8}:
        raise BroadRobustnessError(f"source-publication timing drifted: {period_counts}")
    return raw, metadata


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    checkpoints = raw.get("checkpoints") if isinstance(raw, dict) else None
    if not isinstance(checkpoints, list):
        raise BroadRobustnessError("Broad18 checkpoint registry is malformed")
    result = {str(row["checkpoint_id"]): row for row in checkpoints}
    expected = {CANDIDATE_ID, *BASELINE_IDS}
    if not expected.issubset(result):
        raise BroadRobustnessError("target checkpoint missing from Broad18 registry")
    for checkpoint_id in expected:
        row = result[checkpoint_id]
        if row.get("result_status") != "completed" or not bool(row.get("immutable")):
            raise BroadRobustnessError(f"{checkpoint_id}: completed immutable evidence required")
    return result


def _load_target_scores(
    path: Path, task_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path)
    required = {
        "checkpoint_id",
        "display_name",
        "task",
        "n",
        "headline_f1",
        "malformed_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BroadRobustnessError(f"model_by_task.csv missing columns {missing}")
    checkpoint_ids = (CANDIDATE_ID, *BASELINE_IDS)
    selected = frame.loc[frame["checkpoint_id"].isin(checkpoint_ids)].copy()
    names: dict[str, str] = {}
    expected_tasks = set(task_metadata["task"])
    for checkpoint_id in checkpoint_ids:
        group = selected.loc[selected["checkpoint_id"] == checkpoint_id]
        if len(group) != 18 or set(group["task"].astype(str)) != expected_tasks:
            raise BroadRobustnessError(f"{checkpoint_id}: incomplete Broad18 task scores")
        if group["task"].duplicated().any() or set(group["n"].astype(int)) != {200}:
            raise BroadRobustnessError(f"{checkpoint_id}: task score coverage drifted")
        if not np.isfinite(group["headline_f1"].astype(float)).all():
            raise BroadRobustnessError(f"{checkpoint_id}: non-finite task score")
        names[checkpoint_id] = str(group["display_name"].iloc[0])
    wide = selected.pivot(index="task", columns="checkpoint_id", values="headline_f1")
    wide = (
        task_metadata.set_index("task")
        .join(wide, how="left", validate="one_to_one")
        .reset_index()
        .sort_values("task_order")
        .reset_index(drop=True)
    )
    if wide[list(checkpoint_ids)].isna().any().any():
        raise BroadRobustnessError("target task-score matrix is incomplete")
    return wide, names


def _validate_summary_evidence(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    target_ids = {CANDIDATE_ID, *BASELINE_IDS}
    selected = summary.loc[summary["checkpoint_id"].isin(target_ids)].copy()
    if len(selected) != 3 or set(selected["checkpoint_id"]) != target_ids:
        raise BroadRobustnessError("checkpoint_summary.csv lacks the three target checkpoints")
    for _, row in selected.iterrows():
        checkpoint_id = str(row["checkpoint_id"])
        if (
            not bool(row["frontier_eligible"])
            or not bool(row["identity_verified"])
            or int(row["observed_tasks"]) != 18
            or int(row["observed_items"]) != 3600
            or row["coverage_audit"] != "passed_exact_18_task_3600_item_keys_and_gold"
            or row["provenance_audit"]
            != "passed_prompt_schema_input_model_and_hash_audit"
        ):
            raise BroadRobustnessError(f"{checkpoint_id}: Broad18 evidence audit failed")
    return selected.set_index("checkpoint_id", verify_integrity=True)


def _existing_primary_uncertainty(path: Path) -> dict[str, float]:
    sensitivity = pd.read_csv(path)
    match = sensitivity.loc[
        (sensitivity["checkpoint_id"] == CANDIDATE_ID)
        & (sensitivity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
    ]
    if len(match) != 1:
        raise BroadRobustnessError("primary Broad18 sensitivity row is missing")
    row = match.iloc[0]
    if int(row["task_count"]) != 18:
        raise BroadRobustnessError("primary sensitivity row must cover 18 tasks")
    fields = (
        "paired_item_ci_low",
        "paired_item_ci_high",
        "paired_item_probability_positive",
        "task_bootstrap_ci_low",
        "task_bootstrap_ci_high",
        "task_bootstrap_probability_positive",
        "leave_one_task_out_min_delta",
        "leave_one_task_out_max_delta",
    )
    return {field: float(row[field]) for field in fields} | {
        "tasks_better": int(row["tasks_better"]),
        "tasks_tied": int(row["tasks_tied"]),
        "tasks_worse": int(row["tasks_worse"]),
    }


def _weighted_mean(frame: pd.DataFrame, value: str, group: str | None) -> float:
    if group is None:
        return float(frame[value].mean())
    return float(frame.groupby(group, sort=False)[value].mean().mean())


def aggregation_summary(
    scores: pd.DataFrame,
    names: Mapping[str, str],
    primary_uncertainty: Mapping[str, float],
    source_order: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_id in BASELINE_IDS:
        work = scores.copy()
        work["delta_f1"] = work[CANDIDATE_ID] - work[baseline_id]
        source_means = (
            work.groupby("source_cluster_id", sort=False)["delta_f1"]
            .mean()
            .reindex(source_order)
        )
        if source_means.isna().any():
            raise BroadRobustnessError("source-family delta matrix is incomplete")
        ordered_source_means = {
            source_id: float(source_means.loc[source_id]) for source_id in source_order
        }
        source_draws = source_cluster_bootstrap(
            ordered_source_means,
            source_order,
            iterations=iterations,
            seed=seed,
        )
        for aggregation_id, aggregation_label, group in AGGREGATIONS:
            row: dict[str, Any] = {
                "record_type": "aggregation",
                "candidate_checkpoint_id": CANDIDATE_ID,
                "candidate_display_name": names[CANDIDATE_ID],
                "baseline_checkpoint_id": baseline_id,
                "baseline_display_name": names[baseline_id],
                "aggregation_id": aggregation_id,
                "aggregation_label": aggregation_label,
                "group_count": 18 if group is None else int(work[group].nunique()),
                "task_count": 18,
                "baseline_mean_f1": _weighted_mean(work, baseline_id, group),
                "candidate_mean_f1": _weighted_mean(work, CANDIDATE_ID, group),
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
                "uncertainty_note": "",
            }
            row["delta_f1"] = row["candidate_mean_f1"] - row["baseline_mean_f1"]
            row["delta_f1_points"] = 100.0 * row["delta_f1"]
            if aggregation_id == "equal_source_family":
                row.update(
                    {
                        "source_cluster_ci_low": float(np.quantile(source_draws, 0.025)),
                        "source_cluster_ci_high": float(np.quantile(source_draws, 0.975)),
                        "source_cluster_probability_positive": float(
                            np.mean(source_draws > 0)
                        ),
                        "uncertainty_note": (
                            "95% percentile interval from resampling 13 source-family "
                            "means; cluster order is lexical order of stable source IDs"
                        ),
                    }
                )
            if baseline_id == PRIMARY_BASELINE_ID and aggregation_id == "equal_task":
                row.update(primary_uncertainty)
                row["uncertainty_note"] = (
                    "Existing paired within-task item and task-bootstrap intervals; "
                    "both condition on the frozen task set/design"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _metadata_for_checkpoint(
    registry_row: Mapping[str, Any], checkpoint_id: str
) -> dict[str, Any]:
    result = registry_row.get("result")
    if not isinstance(result, Mapping):
        raise BroadRobustnessError(f"{checkpoint_id}: result metadata missing")
    metadata_path = _resolve_repo_path(str(result.get("metadata_path", "")))
    if not metadata_path.is_file():
        raise BroadRobustnessError(f"{checkpoint_id}: run metadata not found")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("model_id") != registry_row.get("model_id")
        or metadata.get("revision") != registry_row.get("identity")
        or len(str(registry_row.get("identity", ""))) != 40
    ):
        raise BroadRobustnessError(f"{checkpoint_id}: immutable model identity mismatch")
    return metadata


def protocol_audit_rows(
    registry: Mapping[str, Mapping[str, Any]],
    evidence_summary: pd.DataFrame,
    names: Mapping[str, str],
) -> pd.DataFrame:
    candidate_registry = registry[CANDIDATE_ID]
    candidate_metadata = _metadata_for_checkpoint(candidate_registry, CANDIDATE_ID)
    candidate_result = candidate_registry["result"]
    if (
        candidate_result.get("source_protocol") != "legacy_full34"
        or int(candidate_metadata.get("rows_written", -1)) != 16_425
        or len(candidate_metadata.get("task_counts", {})) != 34
    ):
        raise BroadRobustnessError("Qwen3.6 must retain audited reused full34 provenance")

    rows: list[dict[str, Any]] = []
    for baseline_id in BASELINE_IDS:
        baseline_registry = registry[baseline_id]
        baseline_metadata = _metadata_for_checkpoint(baseline_registry, baseline_id)
        baseline_result = baseline_registry["result"]
        if (
            baseline_result.get("source_protocol") != "broad18"
            or int(baseline_metadata.get("rows_written", -1)) != 3600
            or len(baseline_metadata.get("task_counts", {})) != 18
        ):
            raise BroadRobustnessError(f"{baseline_id}: direct Broad18 provenance drifted")

        base_generation = baseline_metadata.get("generation", {})
        cand_generation = candidate_metadata.get("generation", {})
        base_runtime = baseline_metadata.get("runtime", {})
        cand_runtime = candidate_metadata.get("runtime", {})

        def add(
            field: str,
            baseline_value: Any,
            candidate_value: Any,
            status: str,
            note: str,
        ) -> None:
            rows.append(
                {
                    "record_type": "protocol_audit",
                    "candidate_checkpoint_id": CANDIDATE_ID,
                    "candidate_display_name": names[CANDIDATE_ID],
                    "baseline_checkpoint_id": baseline_id,
                    "baseline_display_name": names[baseline_id],
                    "audit_field": field,
                    "baseline_value": str(baseline_value),
                    "candidate_value": str(candidate_value),
                    "audit_status": status,
                    "audit_note": note,
                }
            )

        provenance = "passed_prompt_schema_input_model_and_hash_audit"
        coverage = "passed_exact_18_task_3600_item_keys_and_gold"
        for field, note in (
            (
                "prompt_contract",
                "Broad18 selected-task prompts passed the builder provenance audit; "
                "legacy metadata does not contain a Broad18 panel hash.",
            ),
            (
                "output_schema_contract",
                "Broad18 selected-task schemas passed the builder provenance audit; "
                "raw run-level panel hashes are not asserted equal.",
            ),
        ):
            add(field, provenance, provenance, "matched_by_broad_audit", note)
        add(
            "selected_item_and_gold_coverage",
            coverage,
            coverage,
            "matched_by_broad_audit",
            "Both reported scores use the same 18 tasks and 3,600 selected item/gold keys.",
        )

        matched_settings = (
            ("generation.temperature", base_generation, cand_generation, "temperature"),
            ("generation.max_tokens", base_generation, cand_generation, "max_tokens"),
            (
                "generation.structured_outputs",
                base_generation,
                cand_generation,
                "structured_outputs",
            ),
            (
                "generation.enable_thinking",
                base_generation,
                cand_generation,
                "enable_thinking",
            ),
            ("runtime.vllm_version", base_runtime, cand_runtime, "vllm_version"),
            (
                "runtime.cuda_toolkit_meta_version",
                base_runtime,
                cand_runtime,
                "cuda_toolkit_meta_version",
            ),
            (
                "runtime.runtime_lock_sha256",
                base_runtime,
                cand_runtime,
                "runtime_lock_sha256",
            ),
        )
        for field, base_parent, cand_parent, key in matched_settings:
            base_value = base_parent.get(key)
            cand_value = cand_parent.get(key)
            if base_value != cand_value:
                raise BroadRobustnessError(f"protocol setting {field} unexpectedly differs")
            add(
                field,
                base_value,
                cand_value,
                "matched_exactly",
                "Exact values match in the retained run metadata.",
            )

        add(
            "immutable_model_identity",
            baseline_registry["identity"],
            candidate_registry["identity"],
            "verified_independently",
            "Different models necessarily have different revision hashes; each identity was verified.",
        )
        add(
            "evidence_protocol",
            baseline_result["source_protocol"],
            candidate_result["source_protocol"],
            "differing_caveat",
            "Baseline evidence was run directly on Broad18; candidate evidence reuses audited rows.",
        )
        add(
            "source_evidence_scope",
            "18 tasks / 3,600 rows (direct)",
            "34 tasks / 16,425 rows (reused source run)",
            "differing_caveat",
            "Only the candidate's exact 18-task/3,600-item Broad18 selection enters reported scores.",
        )
        base_seed = base_generation.get("seed")
        cand_seed = cand_generation.get("seed")
        if base_seed == cand_seed:
            raise BroadRobustnessError("expected the retained generation-seed mismatch")
        add(
            "generation.seed",
            base_seed,
            cand_seed,
            "differing_caveat",
            "Temperature is zero, but exact protocol parity is not claimed because seeds differ.",
        )

        # Tie the text audit above to the authoritative published checkpoint rows.
        for checkpoint_id in (baseline_id, CANDIDATE_ID):
            if evidence_summary.loc[checkpoint_id, "provenance_audit"] != provenance:
                raise BroadRobustnessError(f"{checkpoint_id}: provenance audit drifted")
            if evidence_summary.loc[checkpoint_id, "coverage_audit"] != coverage:
                raise BroadRobustnessError(f"{checkpoint_id}: coverage audit drifted")
    return pd.DataFrame(rows)


def heterogeneity_frame(
    scores: pd.DataFrame, names: Mapping[str, str], source_order: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    annotation_order = list(dict.fromkeys(scores["annotation_family"].astype(str)))
    complexity_order = list(dict.fromkeys(scores["complexity"].astype(str)))
    dimensions: tuple[tuple[str, Sequence[str], Mapping[str, str]], ...] = (
        (
            "task",
            scores["task"].astype(str).tolist(),
            {task: task for task in scores["task"].astype(str)},
        ),
        (
            "annotation_family",
            annotation_order,
            {value: value.title() for value in annotation_order},
        ),
        (
            "complexity",
            complexity_order,
            {value: value.title() for value in complexity_order},
        ),
        (
            "source_family",
            source_order,
            {source: str(SOURCE_CLUSTERS[source]["label"]) for source in source_order},
        ),
        (
            "publication_period",
            ("2025_2026", "through_2024"),
            {
                "2025_2026": "Sources published 2025–2026",
                "through_2024": "Sources published through 2024",
            },
        ),
    )
    dimension_column = {
        "task": "task",
        "annotation_family": "annotation_family",
        "complexity": "complexity",
        "source_family": "source_cluster_id",
        "publication_period": "publication_period",
    }
    for baseline_id in BASELINE_IDS:
        work = scores.copy()
        work["delta_f1"] = work[CANDIDATE_ID] - work[baseline_id]
        for dimension, levels, labels in dimensions:
            column = dimension_column[dimension]
            for level_order, level_id in enumerate(levels):
                group = work.loc[work[column].astype(str) == str(level_id)].copy()
                if group.empty:
                    raise BroadRobustnessError(f"empty heterogeneity cell {dimension}/{level_id}")
                deltas = group["delta_f1"].to_numpy(dtype=float)
                rows.append(
                    {
                        "candidate_checkpoint_id": CANDIDATE_ID,
                        "candidate_display_name": names[CANDIDATE_ID],
                        "baseline_checkpoint_id": baseline_id,
                        "baseline_display_name": names[baseline_id],
                        "dimension": dimension,
                        "level_id": str(level_id),
                        "level_label": str(labels[level_id]),
                        "level_order": level_order,
                        "task_count": len(group),
                        "source_family_count": int(group["source_cluster_id"].nunique()),
                        "source_year_min": int(group["source_publication_year"].min()),
                        "source_year_max": int(group["source_publication_year"].max()),
                        "task_names": "|".join(group["task"].astype(str)),
                        "baseline_mean_f1": float(group[baseline_id].mean()),
                        "candidate_mean_f1": float(group[CANDIDATE_ID].mean()),
                        "delta_f1": float(deltas.mean()),
                        "delta_f1_points": float(100.0 * deltas.mean()),
                        "tasks_better": int((deltas > 0).sum()),
                        "tasks_tied": int((deltas == 0).sum()),
                        "tasks_worse": int((deltas < 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_diagnostics(
    summary: pd.DataFrame, heterogeneity: pd.DataFrame, output_dir: Path
) -> None:
    aggregations = summary.loc[summary["record_type"] == "aggregation"].copy()
    primary = heterogeneity.loc[
        (heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
        & heterogeneity["dimension"].isin(["complexity", "annotation_family"])
    ].copy()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 5.8), sharex=True)
    baseline_colors = {
        BASELINE_IDS[0]: "#0072B2",
        BASELINE_IDS[1]: "#D55E00",
    }
    aggregation_order = [item[0] for item in AGGREGATIONS]
    aggregation_labels = [item[1] for item in AGGREGATIONS]
    y = np.arange(len(aggregation_order), dtype=float)
    offsets = {BASELINE_IDS[0]: -0.12, BASELINE_IDS[1]: 0.12}
    for baseline_id in BASELINE_IDS:
        group = (
            aggregations.loc[aggregations["baseline_checkpoint_id"] == baseline_id]
            .set_index("aggregation_id")
            .loc[aggregation_order]
        )
        values = group["delta_f1_points"].to_numpy(dtype=float)
        left.scatter(
            values,
            y + offsets[baseline_id],
            s=31,
            color=baseline_colors[baseline_id],
            label=DISPLAY_NAMES[baseline_id],
            zorder=3,
        )
        source_row = group.loc["equal_source_family"]
        source_y = y[aggregation_order.index("equal_source_family")] + offsets[baseline_id]
        low = 100.0 * float(source_row["source_cluster_ci_low"])
        high = 100.0 * float(source_row["source_cluster_ci_high"])
        point = float(source_row["delta_f1_points"])
        left.hlines(
            source_y,
            low,
            high,
            color=baseline_colors[baseline_id],
            linewidth=1.15,
            zorder=2,
        )
        left.vlines(
            [low, high],
            source_y - 0.045,
            source_y + 0.045,
            color=baseline_colors[baseline_id],
            linewidth=0.9,
            zorder=2,
        )
        for point_value, point_y in zip(values, y + offsets[baseline_id]):
            left.annotate(
                f"{point_value:+.1f}",
                (point_value, point_y),
                xytext=(4 if point_value >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if point_value >= 0 else "right",
                va="center",
                color=baseline_colors[baseline_id],
                fontsize=7.5,
            )
    left.set_yticks(y, aggregation_labels)
    left.set_ylim(len(y) - 0.55, -0.55)
    left.set_title("Aggregation sensitivity", loc="left")
    left.legend(frameon=False, loc="lower left")

    complexity_levels = ("low", "medium", "high")
    annotation_levels = ("claims", "events", "issues", "position", "relevance")
    right_specs = [
        ("complexity", level, f"Complexity · {level.title()}")
        for level in complexity_levels
    ] + [
        ("annotation_family", level, f"Annotation · {level.title()}")
        for level in annotation_levels
    ]
    right_y = np.asarray([0, 1, 2, 4, 5, 6, 7, 8], dtype=float)
    right_values: list[float] = []
    right_labels: list[str] = []
    right_colors: list[str] = []
    for dimension, level, label in right_specs:
        match = primary.loc[
            (primary["dimension"] == dimension) & (primary["level_id"] == level)
        ]
        if len(match) != 1:
            raise BroadRobustnessError(f"missing plotted heterogeneity row {dimension}/{level}")
        right_values.append(float(match.iloc[0]["delta_f1_points"]))
        right_labels.append(label)
        right_colors.append("#0072B2" if dimension == "complexity" else "#666666")
    right.scatter(right_values, right_y, s=31, color=right_colors, zorder=3)
    for point_value, point_y, color in zip(right_values, right_y, right_colors):
        right.annotate(
            f"{point_value:+.1f}",
            (point_value, point_y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=7.5,
        )
    right.set_yticks(right_y, right_labels)
    right.set_ylim(8.55, -0.55)
    right.set_title("Where Qwen3.6 differs from Llama 3.1", loc="left")

    all_x = list(aggregations["delta_f1_points"].astype(float)) + right_values
    source_lows = 100.0 * aggregations["source_cluster_ci_low"].dropna().to_numpy(float)
    source_highs = 100.0 * aggregations["source_cluster_ci_high"].dropna().to_numpy(float)
    all_x.extend(source_lows.tolist())
    all_x.extend(source_highs.tolist())
    xmin = min(-10.0, float(np.floor(min(all_x) - 0.75)))
    xmax = max(7.0, float(np.ceil(max(all_x) + 0.75)))
    for axis in (left, right):
        axis.axvline(0, color="#999999", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        axis.set_xlim(xmin, xmax)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#888888")
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", colors="#555555")
        axis.set_xlabel("Qwen3.6 minus baseline (F1 points)")

    fig.suptitle(
        "Small average gains conceal offsetting task-level changes",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=13,
    )
    fig.text(
        0.065,
        0.018,
        "Whiskers: 95% source-family bootstrap interval (equal-source estimate only). "
        "All other marks are point estimates.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.05, 0.055, 0.995, 0.95), w_pad=2.4)
    for suffix, kwargs in (
        ("png", {"dpi": 220}),
        ("pdf", {}),
    ):
        fig.savefig(
            output_dir / f"robustness_diagnostics.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def build_outputs(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    broad_dir: Path = DEFAULT_BROAD_DIR,
    output_dir: Path = DEFAULT_BROAD_DIR,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build robustness tables and diagnostic figures from retained evidence."""
    _, task_metadata = _load_manifest(manifest_path)
    registry = _load_registry(registry_path)
    scores, names = _load_target_scores(broad_dir / "model_by_task.csv", task_metadata)
    evidence_summary = _validate_summary_evidence(broad_dir / "checkpoint_summary.csv")
    primary_uncertainty = _existing_primary_uncertainty(
        broad_dir / "task_sensitivity.csv"
    )
    source_order = source_cluster_order(scores["task"].astype(str).tolist())

    estimates = aggregation_summary(
        scores,
        names,
        primary_uncertainty,
        source_order,
        iterations=iterations,
        seed=seed,
    )
    audit = protocol_audit_rows(registry, evidence_summary, names)
    robustness = pd.concat([estimates, audit], ignore_index=True, sort=False)
    heterogeneity = heterogeneity_frame(scores, names, source_order)

    primary_tasks = heterogeneity.loc[
        (heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
        & (heterogeneity["dimension"] == "task")
    ]
    primary_families = heterogeneity.loc[
        (heterogeneity["baseline_checkpoint_id"] == PRIMARY_BASELINE_ID)
        & (heterogeneity["dimension"] == "annotation_family")
    ]
    if (
        int(primary_tasks["tasks_better"].sum()) != 11
        or int(primary_tasks["tasks_worse"].sum()) != 7
        or int((primary_families["delta_f1"] > 0).sum()) != 4
    ):
        raise BroadRobustnessError("frozen task/family direction counts drifted")

    output_dir.mkdir(parents=True, exist_ok=True)
    robustness.to_csv(output_dir / "robustness_summary.csv", index=False)
    heterogeneity.to_csv(output_dir / "source_heterogeneity.csv", index=False)
    _plot_diagnostics(robustness, heterogeneity, output_dir)
    return robustness, heterogeneity


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--broad-dir", type=Path, default=DEFAULT_BROAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BROAD_DIR)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    robustness, heterogeneity = build_outputs(
        manifest_path=args.manifest,
        registry_path=args.registry,
        broad_dir=args.broad_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(
        f"wrote {len(robustness)} robustness rows and "
        f"{len(heterogeneity)} heterogeneity rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
