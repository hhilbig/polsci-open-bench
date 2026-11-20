#!/usr/bin/env python3
"""Timing check for training-data exposure across the open models of a release.

Compares each model's task F1 on tasks whose source was published before the
model's release year with its F1 on tasks published after, net of model and
task fixed effects. If models had seen the gold labels of older tasks, they
should do relatively better on tasks that predate them.

Inputs: the release task metrics, task source citations (tasks/*.yaml), and
model release dates (experiments/model_release_dates.csv; open models only,
dated by Hugging Face repository creation).

Default: the November release (33 tasks). For the 46-task release:
    python3 code/analyze_contamination_timing.py \
        --release-dir output/sidecar/release_20261119_tasks46 --tasks-dir tasks_ext_v2_20261114
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "output" / "sidecar" / "refresh_20261106_release_33"
TASKS = REPO / "tasks"
SEED = 20261114
N_BOOT = 2000

# Gold labels taken directly from long-running public labeled datasets that
# predate the cited source paper (see docs/task_source_fidelity_audit.md).
LONG_PUBLIC_LABELS = {
    "brandt_gtd_attack_type": "Global Terrorism Database",
    "chae_semeval_stance": "SemEval-2016 stance",
    "halterman_keith_cmp": "Manifesto Project codes",
    "cap_crs_policy_topic": "Policy Agendas Project",
    "cap_party_platform_policy_topic": "Policy Agendas Project",
    "plover_cameo_event": "CAMEO gold examples",
    # Tasks of the 46-task pool that follow the same rule.
    "bevan_uk_pmqs_policy_topic": "UK Policy Agendas Project",
    "cap_executive_order_policy_topic": "Policy Agendas Project",
    "scdb_opinion_issue_area": "Supreme Court Database",
}


def source_year(task: str, tasks_dir: Path) -> int:
    citation = str(yaml.safe_load((tasks_dir / f"{task}.yaml").read_text())["source"])
    years = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", citation)
    if not years:
        raise ValueError(f"{task}: no year in source citation {citation!r}")
    return int(years[0])  # first-listed (primary) source


def build_panel(release: Path, tasks_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(release / "tasks.csv")[["model", "task", "headline_f1"]]
    dates = pd.read_csv(REPO / "experiments" / "model_release_dates.csv")
    models = pd.read_csv(release / "models.csv")
    open_models = set(models.loc[models["access"].eq("open"), "model"]) if "access" in models else set()
    undated = open_models - set(dates["model"])
    if undated:
        raise ValueError(f"open models in the release without a release date: {sorted(undated)}")
    panel = metrics.merge(dates[["model", "release_date"]], on="model", how="inner")
    n_tasks = metrics["task"].nunique()
    if panel.groupby("model").size().nunique() != 1 or panel["task"].nunique() != n_tasks:
        raise ValueError(f"expected a balanced model x {n_tasks}-task panel")
    years = {t: source_year(t, tasks_dir) for t in panel["task"].unique()}
    panel["source_year"] = panel["task"].map(years)
    panel["release_year"] = pd.to_datetime(panel["release_date"]).dt.year
    panel["gap"] = panel["release_year"] - panel["source_year"]
    panel["pre"] = (panel["gap"] >= 1).astype(int)
    panel["same"] = (panel["gap"] == 0).astype(int)
    panel["long_public"] = panel["task"].isin(LONG_PUBLIC_LABELS).astype(int)
    bins = pd.cut(panel["gap"], [-99, -1, 0, 1, 99], labels=["post", "same", "pre1", "pre2plus"])
    panel["gap_bin"] = bins.astype(str)
    return panel


def fit(panel: pd.DataFrame, formula: str) -> pd.DataFrame:
    model = smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(panel["task"])[0]}
    )
    terms = [t for t in model.params.index if not t.startswith(("C(model)", "C(task)", "Intercept"))]
    ci = model.conf_int().loc[terms]
    return pd.DataFrame({
        "term": terms,
        "estimate": model.params[terms].values,
        "se_cluster_task": model.bse[terms].values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })


def bootstrap_pre(panel: pd.DataFrame, formula: str, rng: np.random.Generator) -> tuple[float, float]:
    tasks = panel["task"].unique()
    groups = {t: g for t, g in panel.groupby("task")}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(tasks, size=len(tasks), replace=True)
        sample = pd.concat(
            [groups[t].assign(task=f"{t}#{i}") for i, t in enumerate(pick)], ignore_index=True
        )
        if sample["pre"].std() == 0:
            continue
        draws.append(smf.ols(formula, data=sample).fit().params["pre"])
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, default=RELEASE)
    parser.add_argument("--tasks-dir", type=Path, default=TASKS)
    args = parser.parse_args()
    release = args.release_dir if args.release_dir.is_absolute() else REPO / args.release_dir
    tasks_dir = args.tasks_dir if args.tasks_dir.is_absolute() else REPO / args.tasks_dir
    out = release / "robustness"
    rng = np.random.default_rng(SEED)
    panel = build_panel(release, tasks_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_tasks = panel["task"].nunique()
    all_name = f"all_{n_tasks}_tasks"
    n_long = panel.loc[panel["long_public"].eq(1), "task"].nunique()

    varying = panel.groupby("task")["gap_bin"].nunique()
    print(f"models={panel['model'].nunique()} tasks={panel['task'].nunique()} rows={len(panel)}")
    print(f"tasks whose timing status varies across models: {(varying > 1).sum()}")

    main_f = "headline_f1 ~ pre + same + C(model) + C(task)"
    event_f = ("headline_f1 ~ C(gap_bin, Treatment('post')) + C(model) + C(task)")
    rows = []
    for sample_name, data in [
        (all_name, panel),
        ("drop_long_public_labels", panel[panel["long_public"].eq(0)]),
    ]:
        est = fit(data, main_f).assign(sample=sample_name, spec="pre_same_vs_post")
        lo, hi = bootstrap_pre(data, main_f, rng)
        est.loc[est["term"].eq("pre"), ["boot_ci_low", "boot_ci_high"]] = [lo, hi]
        ev = fit(data, event_f).assign(sample=sample_name, spec="event_bins_vs_post")
        rows += [est, ev]
        for name, frame in [("main", est), ("event", ev)]:
            print(sample_name, name)
            print(frame[["term", "estimate", "ci_low", "ci_high"]].round(4).to_string(index=False))
    results = pd.concat(rows, ignore_index=True)
    results["n_tasks"] = results["sample"].map(
        {all_name: n_tasks, "drop_long_public_labels": n_tasks - n_long}
    )
    results["n_models"] = panel["model"].nunique()
    results.to_csv(out / "contamination_timing.csv", index=False)

    # Raw descriptive table: mean F1 by task source year x model release year.
    cells = (panel.groupby(["source_year", "release_year"])["headline_f1"].mean().unstack().round(3))
    cells.to_csv(out / "contamination_timing_cells.csv")
    print(cells.to_string())

    # Figure: event-bin coefficients relative to tasks published after release.
    ev = results[results["spec"].eq("event_bins_vs_post")].copy()
    ev["bin"] = ev["term"].str.extract(r"\[T\.(\w+)\]")[0]
    order = ["same", "pre1", "pre2plus"]
    labels = {"same": "Same year", "pre1": "1 year before", "pre2plus": "2+ years before"}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for offset, (sample, color) in zip(
        [-0.08, 0.08], [(all_name, "#3b5b7a"), ("drop_long_public_labels", "#9aa9b8")]
    ):
        s = ev[ev["sample"].eq(sample)].set_index("bin").loc[order]
        x = np.arange(len(order)) + offset
        ax.errorbar(x, 100 * s["estimate"], yerr=[100 * (s["estimate"] - s["ci_low"]), 100 * (s["ci_high"] - s["estimate"])],
                    fmt="o", color=color, capsize=3,
                    label={all_name: f"All {n_tasks} tasks",
                           "drop_long_public_labels": f"Without {n_long} long-public-label tasks"}[sample])
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(len(order)), [labels[b] for b in order])
    ax.set_xlabel("Task source published, relative to model release year")
    ax.set_ylabel("F1 difference vs. tasks\npublished after release (points)")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "contamination_timing.pdf")
    fig.savefig(out / "contamination_timing.png", dpi=200)


if __name__ == "__main__":
    main()
