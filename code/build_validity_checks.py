#!/usr/bin/env python3
"""Build parsimonious checks for task choice, leakage, and model-size fairness."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "output" / "sidecar" / "frontier_2026"

QWEN_FAMILY = (
    "qwen1_5_32b_chat_hive",
    "qwen2_5_32b_instruct_bf16_hive",
    "qwen3_32b_bf16_hive",
    "qwen3_6_27b_fp8_hive",
)
LLAMA_FAMILY = (
    "llama3_70b_instruct_fp8_hive",
    "llama3_1_70b_instruct_fp8_dynamic_hive",
    "llama3_3_70b_instruct_fp8_dynamic_hive",
)
PARAMETERS = {
    "qwen1_5_32b_chat_hive": 32,
    "qwen2_5_32b_instruct_bf16_hive": 32,
    "qwen3_32b_bf16_hive": 32,
    "qwen3_6_27b_fp8_hive": 27,
    "llama3_70b_instruct_fp8_hive": 70,
    "llama3_1_70b_instruct_fp8_dynamic_hive": 70,
    "llama3_3_70b_instruct_fp8_dynamic_hive": 70,
}


def _external_delta(root: Path) -> tuple[float, int]:
    rows = []
    for model, label in (
        ("llama3_1_70b_external_votes", "older"),
        ("qwen3_6_27b_external_votes", "newer"),
    ):
        path = root / "human_votes" / "external_runs" / model / "task_metrics.csv"
        frame = pd.read_csv(path)[["task", "headline_f1"]].assign(model=label)
        rows.append(frame)
    for model, label in (
        ("llama3_1_70b_human_votes", "older"),
        ("qwen3_6_27b_human_votes", "newer"),
    ):
        path = root / "human_votes" / "alia" / model / "task_metrics.csv"
        frame = pd.read_csv(path)[["task", "headline_f1"]].assign(model=label)
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    pair = data.pivot(index="task", columns="model", values="headline_f1")
    if len(pair) != 4 or pair.isna().any().any():
        raise ValueError("external comparison requires four paired tasks")
    return float((pair.newer - pair.older).mean()), len(pair)


def build_checks(root: Path = DEFAULT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = pd.read_csv(root / "broad18" / "holdout_task_deltas.csv")
    if holdout.groupby("set").size().to_dict() != {"Broad18": 18, "Held-out 16": 16}:
        raise ValueError("expected the frozen 18-task set and prespecified 16-task holdout")
    task_rows = []
    for name in ("Broad18", "Held-out 16"):
        values = holdout.loc[holdout["set"].eq(name), "delta_f1"]
        task_rows.append({"evidence_set": name, "task_count": len(values),
                          "delta_f1": float(values.mean()),
                          "scope": "repository tasks"})
    external_delta, external_n = _external_delta(root)
    task_rows.append({"evidence_set": "External 4", "task_count": external_n,
                      "delta_f1": external_delta, "scope": "out-of-database tasks"})
    task_checks = pd.DataFrame(task_rows)
    task_checks["delta_f1_points"] = 100 * task_checks["delta_f1"]

    heterogeneity = pd.read_csv(root / "broad18" / "source_heterogeneity.csv")
    timing = heterogeneity.loc[
        (heterogeneity["baseline_checkpoint_id"] == "llama3_1_70b_instruct_fp8_dynamic_hive")
        & (heterogeneity["dimension"] == "publication_period")
    ].copy()
    if set(timing["task_count"]) != {8, 10} or len(timing) != 2:
        raise ValueError("source timing check requires the frozen 8/10 task split")
    leakage = timing[["level_id", "level_label", "task_count", "delta_f1", "delta_f1_points"]].copy()
    leakage["interpretation"] = (
        "Exact benchmark-source exposure is not supported as the plateau explanation; "
        "underlying-data exposure remains unresolved."
    )

    checkpoints = pd.read_csv(root / "broad18" / "checkpoint_summary.csv")
    eligible = checkpoints.loc[
        checkpoints["checkpoint_id"].isin(PARAMETERS)
        & checkpoints["result_status"].eq("completed")
        & checkpoints["frontier_eligible"].eq(True)
    ].copy()
    expected = set(PARAMETERS)
    if set(eligible["checkpoint_id"]) != expected:
        missing = sorted(expected - set(eligible["checkpoint_id"]))
        raise ValueError(f"matched-family roster is incomplete: {missing}")
    eligible["parameter_count_b"] = eligible["checkpoint_id"].map(PARAMETERS)
    eligible["family_series"] = eligible["checkpoint_id"].map(
        {key: "Qwen dense 27–32B" for key in QWEN_FAMILY}
        | {key: "Llama dense 70B" for key in LLAMA_FAMILY}
    )
    eligible["plot_date"] = pd.to_datetime(eligible["plot_date"])
    size_checks = eligible[["family_series", "checkpoint_id", "display_name", "plot_date",
                            "parameter_count_b", "mean_f1", "ci_low", "ci_high"]].sort_values(
        ["family_series", "plot_date"]
    )
    return task_checks, leakage, size_checks


def render(task_checks: pd.DataFrame, size_checks: pd.DataFrame, output_base: Path) -> None:
    blue, gray, red = "#1676A3", "#666666", "#B14A42"
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [0.9, 1.4]})
    y = list(range(len(task_checks)))[::-1]
    left.axvline(0, color="0.75", lw=.8)
    colors = [blue if value >= 0 else red for value in task_checks.delta_f1_points]
    left.scatter(task_checks.delta_f1_points, y, c=colors, s=65, zorder=3)
    left.set_yticks(y, [f"{row.evidence_set}\n{row.task_count} tasks" for row in task_checks.itertuples()])
    left.set_xlabel("Qwen3.6 minus Llama 3.1 (F1 points)")
    left.set_title("The plateau survives other task sets", loc="left", fontsize=13, fontweight="bold")
    for ypos, row in zip(y, task_checks.itertuples()):
        left.text(row.delta_f1_points + .25, ypos, f"{row.delta_f1_points:+.1f}",
                  ha="left", va="center")

    for family, color, marker in (("Qwen dense 27–32B", blue, "o"), ("Llama dense 70B", gray, "s")):
        part = size_checks.loc[size_checks.family_series.eq(family)].sort_values("plot_date")
        right.plot(part.plot_date, 100 * part.mean_f1, color=color, marker=marker, lw=1.6, ms=6, label=family)
        for row in part.itertuples():
            short = str(row.display_name).replace(" Instruct", "").replace(" FP8 Dynamic", "").replace(" BF16", "")
            offset = {
                "llama3_70b_instruct_fp8_hive": (4, 8),
                "llama3_1_70b_instruct_fp8_dynamic_hive": (4, -15),
            }.get(row.checkpoint_id, (4, 6))
            right.annotate(short, (row.plot_date, 100 * row.mean_f1), xytext=offset, textcoords="offset points", fontsize=8, color=color)
    right.set_ylabel("Mean F1 across 18 tasks")
    right.set_xlabel("Artifact publication date")
    right.set_title("Holding model family and size roughly fixed", loc="left", fontsize=13, fontweight="bold")
    right.legend(frameon=False, loc="lower right")
    for ax in (left, right):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
    fig.suptitle("Two direct checks on the apparent open-model plateau", x=.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(.06, .01, "Task-set checks compare the same two immutable models. Size checks include every tested dense checkpoint in the two named within-family series.", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=(0, .04, 1, .94))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "validity_checks")
    args = parser.parse_args()
    task, leakage, size = build_checks(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task.to_csv(args.output_dir / "task_set_sensitivity.csv", index=False)
    leakage.to_csv(args.output_dir / "leakage_timing_check.csv", index=False)
    size.to_csv(args.output_dir / "matched_family_roster.csv", index=False)
    render(task, size, args.output_dir / "validity_checks")


if __name__ == "__main__":
    main()
