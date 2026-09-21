#!/usr/bin/env python3
"""Plot average trends and task-level shifts for Broad18 and its holdout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from task_registry import load_task_definitions  # noqa: E402

ROOT = REPO / "output" / "sidecar" / "frontier_2026" / "broad18"
MANIFEST = REPO / "experiments" / "frontier_broad_18.yaml"
LLAMA_ID = "llama3_1_70b_instruct_fp8_dynamic_hive"
QWEN_ID = "qwen3_6_27b_fp8_hive"
HOLDOUT_LLAMA = (
    ROOT / "holdout16" / "llama3_1_70b_instruct_fp8_dynamic_full34" / "task_metrics.csv"
)
ORIGINAL_METRICS = (
    REPO / "output" / "sidecar" / "hive_model_bakeoff_20260804" / "comparison" / "task_metrics.csv"
)
EXTENSION_METRICS = (
    REPO
    / "output"
    / "sidecar"
    / "hive_model_bakeoff_extension_cuda130_native_cutlass_20260805"
    / "comparison"
    / "task_metrics.csv"
)

DATES = {
    "llama3_1_70b_instruct_fp8_dynamic_full34": "2024-08-23",
    "qwen3_30b_a3b_instruct_2507_fp8": "2025-07-21",
    "glm4_7_flash": "2026-01-19",
    "qwen3_6_35b_a3b_fp8": "2026-02-25",
    "mistral_small_4_119b_nvfp4": "2026-03-16",
    "qwen3_6_27b_fp8": "2026-04-21",
    "gemma4_31b_it_qat_w4a16": "2026-06-04",
}

FAMILY_COLORS = {
    "Claims": "#7A7A7A",
    "Events": "#0072B2",
    "Policy topics": "#009E73",
    "Position / stance": "#CC79A7",
    "Relevance / incivility": "#D55E00",
    "Causal relation": "#E69F00",
    "Discourse function": "#56B4E9",
}


def broad_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ROOT / "checkpoint_summary.csv")
    summary = summary.loc[(summary["series"] == "open_hive") & summary["mean_f1"].notna()].copy()
    summary["plot_date"] = pd.to_datetime(summary["plot_date"])
    tasks = pd.read_csv(ROOT / "model_by_task.csv")
    return summary, tasks


def holdout_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    broad_names = {row["name"] for row in yaml.safe_load(MANIFEST.read_text())["tasks"]}
    definitions = {task["name"]: task for task in load_task_definitions()}
    holdout = sorted(set(definitions) - broad_names)
    if len(holdout) != 16:
        raise ValueError(f"expected 16 holdout tasks, found {len(holdout)}")

    original = pd.read_csv(ORIGINAL_METRICS)
    original = original.loc[original["model"] != "qwen3_6_27b_fp8"]
    extension = pd.read_csv(EXTENSION_METRICS)
    llama = pd.read_csv(HOLDOUT_LLAMA)
    llama["model"] = "llama3_1_70b_instruct_fp8_dynamic_full34"
    metrics = pd.concat([original, extension, llama], ignore_index=True)
    metrics = metrics.loc[metrics["task"].isin(holdout)].copy()
    if not (metrics.groupby("model")["task"].nunique() == 16).all():
        raise ValueError("holdout model coverage is incomplete")
    metrics["plot_date"] = pd.to_datetime(metrics["model"].map(DATES))
    means = metrics.groupby(["model", "plot_date"], as_index=False)["headline_f1"].mean()
    families = {name: str(definitions[name]["family"]) for name in holdout}
    return means, metrics, families


def task_deltas() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, broad = broad_data()
    manifest = yaml.safe_load(MANIFEST.read_text())
    broad_family = {
        row["name"]: {
            "claims": "Claims",
            "issues": "Policy topics",
            "events": "Events",
            "position": "Position / stance",
            "relevance": "Relevance / incivility",
        }[row["annotation_type"]]
        for row in manifest["tasks"]
    }
    bp = broad.pivot(index="task", columns="checkpoint_id", values="headline_f1")
    broad_delta = (bp[QWEN_ID] - bp[LLAMA_ID]).rename("delta_f1").reset_index()
    broad_delta["set"] = "Broad18"
    broad_delta["family"] = broad_delta["task"].map(broad_family)

    _, holdout, families = holdout_data()
    hp = holdout.pivot(index="task", columns="model", values="headline_f1")
    hold_delta = (
        hp["qwen3_6_27b_fp8"] - hp["llama3_1_70b_instruct_fp8_dynamic_full34"]
    ).rename("delta_f1").reset_index()
    hold_delta["set"] = "Held-out 16"
    family_map = {
        "Event coding": "Events",
        "Policy-topic coding": "Policy topics",
        "Relevance / Incivility": "Relevance / incivility",
        "Sentiment / Stance / Tone": "Position / stance",
        "Causal relation detection": "Causal relation",
        "Rhetoric / Discourse Function": "Discourse function",
    }
    hold_delta["family"] = hold_delta["task"].map(families).map(family_map)
    deltas = pd.concat([broad_delta, hold_delta], ignore_index=True)

    rng = np.random.default_rng(20260820)
    rows = []
    for set_name, group in deltas.groupby("set", sort=False):
        values = group["delta_f1"].to_numpy()
        draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
        rows.append({
            "set": set_name,
            "tasks": len(values),
            "mean_delta_f1": values.mean(),
            "ci_low": np.percentile(draws, 2.5),
            "ci_high": np.percentile(draws, 97.5),
        })
    return deltas, pd.DataFrame(rows)


def _short_label(name: str) -> str:
    return name.replace("_", " ").replace("polnli", "PolNLI").replace("qwen", "Qwen")


def render(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    broad_means, _ = broad_data()
    hold_means, _, _ = holdout_data()
    deltas, aggregate = task_deltas()
    deltas.to_csv(output_dir / "holdout_task_deltas.csv", index=False)
    pd.concat([
        broad_means.assign(task_set="Broad18"),
        hold_means.rename(columns={"model": "checkpoint_id", "headline_f1": "mean_f1"}).assign(task_set="Held-out 16"),
    ], ignore_index=True, sort=False).to_csv(output_dir / "holdout_checkpoint_means.csv", index=False)

    fig = plt.figure(figsize=(13.2, 7.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.94))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.65])
    left_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    right = fig.add_subplot(grid[:, 1])

    panels = [
        (left_axes[0], broad_means, "Broad18 tasks"),
        (left_axes[1], hold_means.rename(columns={"model": "checkpoint_id", "headline_f1": "mean_f1"}), "Held-out 16 tasks"),
    ]
    for axis, frame, title in panels:
        frame = frame.sort_values("plot_date")
        axis.scatter(frame["plot_date"], frame["mean_f1"], s=25, color="#555555", zorder=3)
        x = (frame["plot_date"] - frame["plot_date"].min()).dt.days.to_numpy() / 365.25
        if len(frame) > 2:
            fit = np.polyfit(x, frame["mean_f1"], 1)
            axis.plot(frame["plot_date"], np.polyval(fit, x), color="#999999", lw=1)
        axis.axhline(0.65, color="#DDDDDD", lw=0.7, zorder=0)
        axis.set_ylim(0.56, 0.70)
        axis.set_title(title, loc="left", fontsize=11, weight="bold")
        axis.set_ylabel("Mean task F1")
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        for _, row in frame.iterrows():
            cid = str(row["checkpoint_id"])
            if any(key in cid for key in ["llama3_1", "qwen3_6_27b", "gemma4_31b"]):
                label = "Llama 3.1" if "llama3_1" in cid else ("Qwen3.6" if "qwen3_6" in cid else "Gemma 4")
                axis.annotate(label, (row["plot_date"], row["mean_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    family_order = list(FAMILY_COLORS)
    deltas["family"] = pd.Categorical(deltas["family"], family_order, ordered=True)
    deltas = deltas.sort_values(["family", "set", "delta_f1"])
    y = np.arange(len(family_order))
    right.axvline(0, color="#555555", lw=0.9)
    offsets = {"Broad18": -0.16, "Held-out 16": 0.16}
    for family_index, family in enumerate(family_order):
        family_rows = deltas.loc[deltas["family"] == family]
        for set_name, set_rows in family_rows.groupby("set", sort=False):
            color = FAMILY_COLORS[family]
            center = family_index + offsets[set_name]
            jitter = np.linspace(-0.07, 0.07, len(set_rows)) if len(set_rows) > 1 else np.array([0.0])
            face = color if set_name == "Broad18" else "white"
            right.scatter(set_rows["delta_f1"] * 100, center + jitter, s=20,
                          facecolor=face, edgecolor=color, alpha=0.48, lw=0.9, zorder=2)
            right.scatter(set_rows["delta_f1"].mean() * 100, center, s=64, marker="D",
                          facecolor=face, edgecolor=color, lw=1.5, zorder=4)
    family_counts = deltas.groupby("family", observed=True).size().reindex(family_order)
    right.set_yticks(y, [f"{family}  (n={family_counts[family]})" for family in family_order], fontsize=8)
    right.invert_yaxis()
    right.set_xlabel("Qwen3.6 minus Llama 3.1 (F1 points)")
    right.set_title("Family averages conceal substantial task-level variation", loc="left",
                    fontsize=11, weight="bold")
    right.spines[["top", "right", "left"]].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.grid(False)
    for boundary in np.arange(len(family_order) - 1) + 0.5:
        right.axhline(boundary, color="#E3E3E3", lw=0.7)
    handles = [
        Line2D([0], [0], marker="D", linestyle="none", markersize=6,
               markerfacecolor="#555555", markeredgecolor="#555555", label="Broad18 family mean"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=6,
               markerfacecolor="white", markeredgecolor="#555555", label="Held-out family mean"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=4, alpha=0.48,
               markerfacecolor="#777777", markeredgecolor="#777777", label="Individual task"),
    ]
    right.legend(handles=handles, loc="upper right", ncol=1, frameon=False,
                 fontsize=7, labelspacing=0.35, handletextpad=0.35)

    notes = "; ".join(
        f"{row['set']}: {row['mean_delta_f1']*100:+.1f} [{row['ci_low']*100:+.1f}, {row['ci_high']*100:+.1f}]"
        for _, row in aggregate.iterrows()
    )
    fig.suptitle("Average accuracy changed little; which tasks models handle well changed substantially", fontsize=13, weight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.018, f"Diamonds: family means; small circles: tasks. Filled: Broad18; hollow: held-out. Mean task-bootstrap differences (95% intervals): {notes}.", fontsize=8)

    png = output_dir / "holdout_diagnostic.png"
    pdf = output_dir / "holdout_diagnostic.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
