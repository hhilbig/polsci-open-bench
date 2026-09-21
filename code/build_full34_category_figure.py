#!/usr/bin/env python3
"""Compare the 2024 and 2026 anchors across all 34 benchmark tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_regression_mechanisms import _read_predictions, _score  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

LLAMA_REV = "019d944e8e566c43939ea83775a27197ebb9b559"
QWEN_REV = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"

CATEGORIES = {
    "Claims and relations": {
        "agoraspeech_criticism_agenda", "burnham_covid_threat_minimization",
        "burnham_polnli_entailment", "burnham_polnli_event_entailment",
        "dicocco_manifesto_populism", "haunss_papea_claims",
        "politicause_causal_relation",
    },
    "Policy and topics": {
        "cap_crs_policy_topic", "cap_party_platform_policy_topic", "erlich_ati_topics",
        "mellon_bes_mii_2024", "muller_fujimura_campaign_policy_area",
        "osnabruegge_cross_domain_topic",
    },
    "Events and protest": {
        "brandt_gtd_attack_type", "douglass_icbe_sentence_event_type",
        "halterman_ccc_protest", "halterman_keith_bfrs", "halterman_keith_cmp",
        "haunss_papea_fgz_forms", "plover_cameo_event",
    },
    "Stance and sentiment": {
        "bestvater_kavanaugh_stance", "bestvater_wm_stance", "burnham_trump_stance",
        "chae_semeval_stance", "gilardi_stance", "ornstein_scotus_sentiment",
    },
    "Relevance and tone": {
        "ballard_incivility", "brandt_political_relevance", "gilardi_relevance",
        "rheault_line_of_fire_incivility", "theocharis_dynamics_incivility",
        "toxicity_protests_es", "twitcivility_impoliteness", "wesleyan_creative_ads_2022",
    },
}


def _audit(path: Path, revision: str) -> None:
    metadata = json.loads((path.parent / "run_metadata.json").read_text())
    if metadata.get("revision") != revision:
        raise ValueError(f"immutable revision mismatch for {path}")
    if int(metadata.get("rows_written", -1)) != 16_425:
        raise ValueError(f"full34 row count mismatch for {path}")
    if len(metadata.get("task_counts", {})) != 34 or set(metadata["task_counts"].values()) == {0}:
        raise ValueError(f"full34 task coverage mismatch for {path}")


def category_frame(llama_path: Path, qwen_path: Path, tasks_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _audit(llama_path, LLAMA_REV); _audit(qwen_path, QWEN_REV)
    tasks = {task["name"]: task for task in load_task_definitions(tasks_dir=tasks_dir)}
    llama, qwen = _read_predictions(llama_path), _read_predictions(qwen_path)
    expected = set(tasks)
    assigned = set().union(*CATEGORIES.values())
    if len(expected) != 34 or assigned != expected or sum(map(len, CATEGORIES.values())) != 34:
        raise ValueError("category taxonomy must partition exactly 34 tasks")
    if set(llama.task) != expected or set(qwen.task) != expected:
        raise ValueError("predictions lack exact 34-task coverage")
    if len(llama) != 16_425 or len(qwen) != 16_425:
        raise ValueError("predictions lack exact 16,425-row coverage")
    if set(zip(llama.task, llama.item_id)) != set(zip(qwen.task, qwen.item_id)):
        raise ValueError("model task/item keys differ")
    rows = []
    for task in sorted(expected):
        group = next(name for name, members in CATEGORIES.items() if task in members)
        rows.append({"task": task, "category": group,
                     "llama3_1_f1": _score(llama.loc[llama.task == task], tasks[task]),
                     "qwen3_6_f1": _score(qwen.loc[qwen.task == task], tasks[task])})
    detail = pd.DataFrame(rows)
    detail["delta_f1_points"] = 100 * (detail.qwen3_6_f1 - detail.llama3_1_f1)
    summary = detail.groupby("category", sort=False).agg(
        task_count=("task", "size"), llama3_1_f1=("llama3_1_f1", "mean"),
        qwen3_6_f1=("qwen3_6_f1", "mean"), tasks_improved=("delta_f1_points", lambda x: int((x > 0).sum()))
    ).reset_index()
    summary["delta_f1_points"] = 100 * (summary.qwen3_6_f1 - summary.llama3_1_f1)
    summary["category"] = pd.Categorical(summary.category, list(CATEGORIES), ordered=True)
    return detail, summary.sort_values("category").reset_index(drop=True)


def render(summary: pd.DataFrame, output_base: Path) -> None:
    blue, gray = "#1676A3", "#777777"
    y = list(range(len(summary)))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for ypos, row in zip(y, summary.itertuples()):
        old, new = row.llama3_1_f1 * 100, row.qwen3_6_f1 * 100
        ax.plot([old, new], [ypos, ypos], color="0.82", lw=2, zorder=1)
        ax.scatter(old, ypos, color=gray, s=55, zorder=2)
        ax.scatter(new, ypos, color=blue, s=65, zorder=3)
        ax.text(max(old, new) + .35, ypos, f"{row.delta_f1_points:+.1f}", va="center", fontsize=9)
    ax.set_yticks(y, [f"{r.category}\n{r.tasks_improved}/{r.task_count} tasks improved" for r in summary.itertuples()])
    ax.set_xlabel("Mean F1 within category")
    ax.set_title("Newer models improved some kinds of coding, not all", loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(0, 1.01, "All 34 tasks; gray = Llama 3.1 70B (2024), blue = Qwen3.6 27B (2026); labels show change", transform=ax.transAxes, fontsize=9, color="0.4")
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(False)
    fig.tight_layout(); output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llama", type=Path, required=True); p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--tasks-dir", type=Path, default=REPO / "tasks"); p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); detail, summary = category_frame(a.llama, a.qwen, a.tasks_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.output_dir / "full34_task_categories.csv", index=False)
    summary.to_csv(a.output_dir / "full34_category_summary.csv", index=False)
    render(summary, a.output_dir / "02_full34_categories")

if __name__ == "__main__": main()
