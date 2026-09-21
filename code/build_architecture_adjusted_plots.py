#!/usr/bin/env python3
"""Build architecture-adjusted Twitter figures from audited Broad18 scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "output" / "sidecar" / "frontier_2026"
BLUE, DARK, GRAY, RED = "#2f6f9f", "#1d4f77", "#8c8c8c", "#a4443a"

# Parameter counts are nominal model-card counts, not measured FLOPs. For dense
# models, active approximately equals total. A-values are provider-reported MoE
# active counts. The bins were fixed before regenerating these figures.
PARAMETERS = {
    "qwen1_5_72b_chat_awq_hive": (72, 72, "dense"),
    "qwen1_5_32b_chat_hive": (32, 32, "dense"),
    "qwen2_72b_instruct_awq_hive": (72, 72, "dense"),
    "llama3_70b_instruct_fp8_hive": (70, 70, "dense"),
    "llama3_1_70b_instruct_fp8_dynamic_hive": (70, 70, "dense"),
    "qwen2_5_32b_instruct_bf16_hive": (32, 32, "dense"),
    "llama3_1_nemotron_70b_fp8_dynamic_hive": (70, 70, "dense"),
    "qwen2_5_72b_instruct_fp8_dynamic_hive": (72, 72, "dense"),
    "llama3_3_70b_instruct_fp8_dynamic_hive": (70, 70, "dense"),
    "deepseek_r1_distill_qwen_32b_bf16_hive": (32, 32, "dense"),
    "mistral_small_3_1_24b_bf16_hive": (24, 24, "dense"),
    "qwen3_32b_bf16_hive": (32, 32, "dense"),
    "qwen3_30b_a3b_bf16_hive": (30, 3, "moe"),
    "gemma3_27b_it_fp8_dynamic_hive": (27, 27, "dense"),
    "gpt_oss_120b_mxfp4_hive": (117, 5.1, "moe"),
    "qwen3_next_80b_a3b_fp8_hive": (80, 3, "moe"),
    "glm4_7_flash_hive": (30, 3, "moe"),
    "qwen3_5_35b_a3b_fp8_hive": (35, 3, "moe"),
    "mistral_small_4_119b_nvfp4_hive": (119, 6.5, "moe"),
    "qwen3_6_27b_fp8_hive": (27, 27, "dense"),
    "gemma4_31b_it_qat_w4a16_hive": (31, 31, "dense"),
}
MOE_PARAMETER_SOURCES = {
    "qwen3_30b_a3b_bf16_hive": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
    "gpt_oss_120b_mxfp4_hive": "https://huggingface.co/openai/gpt-oss-120b",
    "qwen3_next_80b_a3b_fp8_hive": "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    "glm4_7_flash_hive": "https://huggingface.co/zai-org/GLM-4.7-Flash",
    "qwen3_5_35b_a3b_fp8_hive": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8",
    "mistral_small_4_119b_nvfp4_hive": "https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4",
}

LINEAGES = {
    "Qwen dense 27–32B": (
        "qwen1_5_32b_chat_hive", "qwen2_5_32b_instruct_bf16_hive",
        "qwen3_32b_bf16_hive", "qwen3_6_27b_fp8_hive",
    ),
    "Qwen dense 72B": (
        "qwen1_5_72b_chat_awq_hive", "qwen2_72b_instruct_awq_hive",
        "qwen2_5_72b_instruct_fp8_dynamic_hive",
    ),
    "Llama dense 70B": (
        "llama3_70b_instruct_fp8_hive", "llama3_1_70b_instruct_fp8_dynamic_hive",
        "llama3_3_70b_instruct_fp8_dynamic_hive",
    ),
}


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    canvas = fig.bbox
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > canvas.x1 + 1 or bounds.y1 > canvas.y1 + 1:
            raise ValueError(f"text clips figure canvas: {artist.get_text()!r}")
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _candidate_style(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b3b3b3")
        spine.set_linewidth(.8)


def architecture_data(checkpoints: pd.DataFrame) -> pd.DataFrame:
    data = checkpoints.loc[
        checkpoints.series.eq("open_hive") & checkpoints.result_status.eq("completed")
        & checkpoints.frontier_eligible.eq(True)
    ].copy()
    if set(data.checkpoint_id) != set(PARAMETERS):
        raise ValueError("architecture map must cover exactly the 21 eligible open checkpoints")
    mapped = data.checkpoint_id.map(PARAMETERS)
    data[["total_parameters_b", "active_parameters_b", "architecture"]] = pd.DataFrame(
        mapped.tolist(), index=data.index
    )
    data["compute_class"] = pd.cut(
        data.active_parameters_b, bins=[0, 10, 40, float("inf")],
        labels=["Low (≤10B active)", "Medium (10–40B active)", "Large (>40B active)"],
        right=True,
    ).astype(str)
    data["parameter_source"] = data.apply(
        lambda row: MOE_PARAMETER_SOURCES.get(row.checkpoint_id, row.release_source), axis=1
    )
    data["plot_date"] = pd.to_datetime(data.plot_date)
    return data.sort_values("plot_date").reset_index(drop=True)


def lineage_plot(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    records = []
    colors = [DARK, BLUE, GRAY]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.8), sharey=True)
    for ax, (name, ids), color in zip(axes, LINEAGES.items(), colors):
        part = data.loc[data.checkpoint_id.isin(ids)].sort_values("plot_date").copy()
        if tuple(part.checkpoint_id) != tuple(sorted(ids, key=lambda x: part.set_index("checkpoint_id").loc[x, "plot_date"])):
            raise ValueError(f"unexpected chronology for {name}")
        part["lineage"] = name
        records.append(part)
        xpos = list(range(1, len(part) + 1))
        ax.plot(xpos, 100 * part.mean_f1, color=color, lw=1.6, marker="o", ms=4.5)
        label_offsets = {
            "qwen1_5_32b_chat_hive": (7, 8, "left"),
            "qwen3_6_27b_fp8_hive": (-5, 8, "right"),
            "qwen1_5_72b_chat_awq_hive": (5, 8, "left"),
            "qwen2_5_72b_instruct_fp8_dynamic_hive": (-4, 8, "right"),
            "llama3_3_70b_instruct_fp8_dynamic_hive": (-3, 8, "right"),
        }
        for x, row in zip(xpos, part.itertuples()):
            label = row.display_name.split(" Instruct")[0].replace(" Chat", "")
            label = label.replace(" BF16", "").replace(" AWQ", "").replace(" FP8 Dynamic", "")
            label += "\n" + row.plot_date.strftime("%b %Y")
            dx, dy, align = label_offsets.get(row.checkpoint_id, (0, 8, "center"))
            ax.annotate(label, (x, 100 * row.mean_f1), xytext=(dx, dy),
                        textcoords="offset points", ha=align, fontsize=6.5, color=color)
        delta = 100 * (part.mean_f1.iloc[-1] - part.mean_f1.iloc[0])
        ax.text(.03, .05, f"First to latest: {delta:+.1f} points", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=color)
        ax.set_title(name, fontsize=9.5, fontweight="bold")
        ax.set_xlim(.65, len(part) + .35)
        ax.set_xticks([])
        _candidate_style(ax)
        ax.set_ylim(61, 69.5)
    axes[0].set_ylabel("Mean F1 across 18 tasks", fontsize=9)
    fig.suptitle("Only the smaller Qwen line improved consistently", x=.075,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.075, .90, "Every tested same-developer dense lineage with ≥3 generations at roughly fixed size",
             fontsize=8, color="0.42")
    fig.tight_layout(rect=(.02, .02, .98, .86)); _save(fig, output)
    return pd.concat(records, ignore_index=True)


def compute_staircase_plot(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    styles = {
        "Low (≤10B active)": (RED, "^", "LOW  ≤10B active (MoE)"),
        "Medium (10–40B active)": (DARK, "o", "MEDIUM  10–40B active"),
        "Large (>40B active)": (GRAY, "o", "LARGE  >40B active"),
    }
    end_date = data.plot_date.max()
    frames = []
    for compute_class, (color, marker, label) in styles.items():
        part = data.loc[data.compute_class.eq(compute_class)].sort_values("plot_date").copy()
        part["class_frontier_f1"] = part.mean_f1.cummax()
        part["class_frontier_setter"] = part.mean_f1.gt(part.class_frontier_f1.shift(fill_value=-1))
        frames.append(part)
        dates = part.plot_date.tolist() + [end_date]
        frontier = (100 * part.class_frontier_f1).tolist() + [100 * part.class_frontier_f1.iloc[-1]]
        ax.step(dates, frontier, where="post", color=color, lw=1.7, zorder=2)
        ax.scatter(part.plot_date, 100 * part.mean_f1, marker=marker, s=25 if marker == "o" else 38,
                   facecolors="white" if marker == "^" else color, edgecolors=color,
                   linewidths=1.2, alpha=.9, zorder=3)
        final = 100 * part.class_frontier_f1.iloc[-1]
        y_nudge = {"Medium (10–40B active)": .24, "Large (>40B active)": -.18}.get(
            compute_class, 0
        )
        ax.text(end_date + pd.Timedelta(days=18), final + y_nudge,
                f"{label}  {final:.1f}", color=color, va="center", fontsize=8,
                fontweight="bold")
    result = pd.concat(frames, ignore_index=True).sort_values("plot_date").reset_index(drop=True)
    low_start = result.loc[result.compute_class.eq("Low (≤10B active)"), "plot_date"].min()
    ax.annotate("First low-active\nmodel tested", (low_start, 59.05), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7, color=RED)
    ax.set_xlim(data.plot_date.min() - pd.Timedelta(days=25), end_date + pd.Timedelta(days=155))
    ax.set_ylim(58.3, 69.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Artifact publication date")
    ax.set_ylabel("Mean F1 across 18 tasks", fontsize=9)
    ax.set_title("Medium-compute models caught the large-model frontier",
                 loc="left", fontsize=15, fontweight="bold", pad=10)
    ax.text(0, 1.01, "Cumulative best among tested checkpoints in each active-parameter class",
            transform=ax.transAxes, fontsize=8, color="0.42")
    _candidate_style(ax)
    fig.tight_layout(rect=(.02, .02, .98, .98)); _save(fig, output)
    return result[["checkpoint_id", "display_name", "plot_date", "mean_f1", "total_parameters_b",
                   "active_parameters_b", "architecture", "compute_class", "class_frontier_f1",
                   "class_frontier_setter", "parameter_source"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "twitter_claims")
    args = parser.parse_args()
    data = architecture_data(pd.read_csv(args.root / "broad18" / "checkpoint_summary.csv"))
    lineage_plot(data, args.output_dir / "04_dense_lineages").to_csv(
        args.output_dir / "04_dense_lineages.csv", index=False)
    compute_staircase_plot(data, args.output_dir / "05_compute_class_staircase").to_csv(
        args.output_dir / "05_compute_class_staircase.csv", index=False)


if __name__ == "__main__":
    main()
