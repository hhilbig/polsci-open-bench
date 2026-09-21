#!/usr/bin/env python3
"""Build three single-claim figures for the benchmark Twitter thread."""

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
BLUE, RED, GRAY = "#1d4f77", "#a4443a", "#8c8c8c"
GOLD = RED
QWEN = {
    "qwen1_5_32b_chat_hive": "Qwen1.5 32B",
    "qwen2_5_32b_instruct_bf16_hive": "Qwen2.5 32B",
    "qwen3_32b_bf16_hive": "Qwen3 32B",
    "qwen3_6_27b_fp8_hive": "Qwen3.6 27B",
}
SPARSE = {
    "qwen3_30b_a3b_bf16_hive",
    "gpt_oss_120b_mxfp4_hive",
    "qwen3_next_80b_a3b_fp8_hive",
    "glm4_7_flash_hive",
    "qwen3_5_35b_a3b_fp8_hive",
    "mistral_small_4_119b_nvfp4_hive",
}
CATEGORY_QUESTIONS = {
    "Events and protest": "What happened?\nattack, protest, event type",
    "Policy and topics": "What is it about?\npolicy area or topic",
    "Stance and sentiment": "Which side does the author take?\nstance or sentiment",
    "Claims and relations": "What is being asserted or linked?\nclaim, cause, entailment",
    "Relevance and tone": "Is it relevant, civil, or critical?\nfiltering and tone",
}
EXTERNAL_LABELS = {
    "alia_civic_stance_votes": "Civic stance\nWhich side does a post take?",
    "hatexplain_votes": "HateXplain\nHate, offensive, or normal?",
    "measuring_hate_speech_votes": "Measuring Hate Speech\nHate, unclear, or not hate?",
    "mfrc_votes": "Moral Foundations Reddit\nWhich moral concerns appear?",
}


def _style(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b3b3b3")
        spine.set_linewidth(.8)


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
            raise ValueError(
                f"text clips figure canvas: {artist.get_text()!r}; "
                f"bounds={tuple(round(value, 1) for value in bounds.bounds)}, "
                f"canvas={tuple(round(value, 1) for value in canvas.bounds)}"
            )
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def progress_plot(checkpoints: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = checkpoints.loc[
        checkpoints.series.eq("open_hive")
        & checkpoints.result_status.eq("completed")
        & checkpoints.frontier_eligible.eq(True)
    ].copy()
    if len(data) != 21 or not set(QWEN).issubset(set(data.checkpoint_id)):
        raise ValueError("progress plot requires 21 eligible checkpoints and the full Qwen series")
    data["plot_date"] = pd.to_datetime(data.plot_date)
    data = data.sort_values("plot_date").reset_index(drop=True)
    data["frontier_f1"] = data.mean_f1.cummax()
    qwen = data.loc[data.checkpoint_id.isin(QWEN)].copy()
    sparse = data.loc[data.checkpoint_id.isin(SPARSE)].copy()
    if len(sparse) != 6:
        raise ValueError("progress plot requires the six tested sparse/MoE checkpoints")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.scatter(data.plot_date, 100 * data.mean_f1, color="0.78", s=34, zorder=1)
    ax.scatter(sparse.plot_date, 100 * sparse.mean_f1, facecolors="white", edgecolors=GOLD,
               marker="^", linewidths=1.2, s=56, label="Tested sparse/MoE checkpoints", zorder=2)
    ax.step(data.plot_date, 100 * data.frontier_f1, where="post", color="0.25", lw=2.2,
            label="Best tested score by that date", zorder=2)
    ax.plot(qwen.plot_date, 100 * qwen.mean_f1, color=BLUE, lw=2, marker="o", ms=6,
            label="Dense Qwen 27–32B", zorder=3)
    for row in qwen.itertuples():
        offset = (5, 7) if row.checkpoint_id != "qwen3_32b_bf16_hive" else (5, -16)
        ax.annotate(QWEN[row.checkpoint_id], (row.plot_date, 100 * row.mean_f1),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=BLUE)
    l3 = data.loc[data.checkpoint_id.eq("llama3_70b_instruct_fp8_hive")].iloc[0]
    q36 = data.loc[data.checkpoint_id.eq("qwen3_6_27b_fp8_hive")].iloc[0]
    ax.annotate("Llama 3 70B\n67.6", (l3.plot_date, 100 * l3.mean_f1), xytext=(-64, 18),
                textcoords="offset points", fontsize=9, color="0.2", fontweight="bold")
    ax.text(pd.Timestamp("2024-12-20"), 69.05,
            "Best tested score: +0.5 points after June 2024\nDense Qwen 27–32B: +4.8 points from 2024 to 2026",
            fontsize=10, color="0.25", va="top")
    ax.set_ylabel("Mean F1 across 18 tasks")
    ax.set_xlabel("Artifact publication date")
    ax.set_ylim(58, 70)
    ax.set_title("The best score barely moved. Small models caught up.", loc="left",
                 fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "Gray points are every qualified checkpoint; lines answer two different questions",
            transform=ax.transAxes, fontsize=9, color="0.42")
    ax.legend(frameon=False, loc="lower right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    _style(ax); fig.tight_layout(); _save(fig, output)
    data["architecture_group"] = data.checkpoint_id.map(
        lambda value: "sparse_or_moe" if value in SPARSE else "dense"
    )
    return data[["checkpoint_id", "plot_date", "mean_f1", "frontier_f1", "architecture_group"]]


def category_plot(summary: pd.DataFrame, output: Path) -> pd.DataFrame:
    expected = {"Claims and relations", "Policy and topics", "Events and protest",
                "Stance and sentiment", "Relevance and tone"}
    if set(summary.category) != expected or int(summary.task_count.sum()) != 34:
        raise ValueError("category plot requires the frozen five-category, 34-task partition")
    shown = summary.sort_values("delta_f1_points").reset_index(drop=True)
    labels = [f"{CATEGORY_QUESTIONS[r.category]}\n{r.tasks_improved}/{r.task_count} tasks improved"
              for r in shown.itertuples()]
    colors = [BLUE if x > 0 else RED for x in shown.delta_f1_points]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvline(0, color="0.7", lw=.8)
    bars = ax.barh(range(len(shown)), shown.delta_f1_points, color=colors, height=.56)
    ax.set_yticks(range(len(shown)), labels)
    for bar, value in zip(bars, shown.delta_f1_points):
        ax.text(value + (.18 if value > 0 else -.18), bar.get_y() + bar.get_height()/2,
                f"{value:+.1f}", ha="left" if value > 0 else "right", va="center",
                fontsize=9, fontweight="bold", color=BLUE if value > 0 else RED)
    ax.set_xlim(-3.4, 5.2)
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_xlabel("Qwen3.6 27B minus Llama 3.1 70B (F1 points)")
    ax.set_title("Event coding improved most", loc="left",
                 fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "All 34 tasks, grouped by what the coder must identify",
            transform=ax.transAxes, fontsize=8, color="0.42")
    _style(ax); fig.tight_layout(rect=(.02, .02, .98, .98)); _save(fig, output)
    return shown


def external_task_deltas(root: Path) -> pd.DataFrame:
    rows = []
    paths = {
        "Llama 3.1": [
            root / "human_votes" / "external_runs" / "llama3_1_70b_external_votes" / "task_metrics.csv",
            root / "human_votes" / "alia" / "llama3_1_70b_human_votes" / "task_metrics.csv",
        ],
        "Qwen3.6": [
            root / "human_votes" / "external_runs" / "qwen3_6_27b_external_votes" / "task_metrics.csv",
            root / "human_votes" / "alia" / "qwen3_6_27b_human_votes" / "task_metrics.csv",
        ],
    }
    for model, model_paths in paths.items():
        for path in model_paths:
            frame = pd.read_csv(path)[["task", "headline_f1"]].copy()
            frame["model"] = model
            rows.append(frame)
    paired = pd.concat(rows, ignore_index=True).pivot(
        index="task", columns="model", values="headline_f1"
    ).reset_index()
    if set(paired.task) != set(EXTERNAL_LABELS) or paired.isna().any().any():
        raise ValueError("external detail requires the four audited paired tasks")
    paired["delta_f1_points"] = 100 * (paired["Qwen3.6"] - paired["Llama 3.1"])
    paired["display_label"] = paired.task.map(EXTERNAL_LABELS)
    return paired.sort_values("delta_f1_points", ascending=False).reset_index(drop=True)


def task_set_plot(task_sets: pd.DataFrame, external: pd.DataFrame, output: Path) -> pd.DataFrame:
    if task_sets.task_count.tolist() != [18, 16, 4]:
        raise ValueError("task-set plot requires Broad18, held-out 16, and external 4")
    external_mean = task_sets.loc[task_sets.evidence_set.eq("External 4"), "delta_f1_points"].iloc[0]
    if abs(external.delta_f1_points.mean() - external_mean) > 1e-9:
        raise ValueError("external task details do not reproduce the pooled result")
    repository = task_sets.iloc[:2].copy()
    labels = ["Selected benchmark\n18 repository tasks", "Excluded beforehand\n16 repository tasks"]
    labels += external.display_label.tolist()
    values = repository.delta_f1_points.tolist() + external.delta_f1_points.tolist()
    y = list(range(len(values)))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvline(0, color="0.65", lw=.8)
    colors = [BLUE, BLUE] + [RED if value < 0 else BLUE for value in external.delta_f1_points]
    ax.scatter(values, y, s=48, c=colors, zorder=3)
    ax.set_yticks(y, labels)
    for xpos, ypos in zip(values, y):
        ax.text(xpos + .18, ypos, f"{xpos:+.1f}", va="center", fontsize=9,
                fontweight="bold", color=BLUE if xpos >= 0 else RED)
    ax.axhline(y[1] - .55, color="0.82", lw=.8)
    ax.text(-5.8, y[1] - .75, "External tasks selected because they retain original annotation votes",
            fontsize=7.2, color="0.42", va="top")
    ax.set_xlim(-6.0, 1.5)
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_xlabel("Qwen3.6 27B minus Llama 3.1 70B (F1 points)")
    ax.set_title("Interpretive labels regressed", loc="left",
                 fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "Qwen3.6 27B versus Llama 3.1 70B; external mean = -2.6 points",
            transform=ax.transAxes, fontsize=8, color="0.42")
    _style(ax); fig.tight_layout(rect=(.02, .02, .98, .98)); _save(fig, output)
    result = pd.DataFrame({"display_label": labels, "delta_f1_points": values})
    result["set_type"] = ["repository", "repository"] + ["external"] * len(external)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "twitter_claims")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_plot(pd.read_csv(args.root / "broad18" / "checkpoint_summary.csv"),
                  args.output_dir / "01_frontier_and_efficiency").to_csv(
        args.output_dir / "01_frontier_and_efficiency.csv", index=False)
    category_plot(pd.read_csv(args.root / "twitter_figures" / "full34_category_summary.csv"),
                  args.output_dir / "02_capability_reallocation").to_csv(
        args.output_dir / "02_capability_reallocation.csv", index=False)
    task_set_plot(pd.read_csv(args.root / "validity_checks" / "task_set_sensitivity.csv"),
                  external_task_deltas(args.root),
                  args.output_dir / "03_task_set_sensitivity").to_csv(
        args.output_dir / "03_task_set_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
