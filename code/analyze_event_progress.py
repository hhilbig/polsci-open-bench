#!/usr/bin/env python3
"""Stress-test whether progress is concentrated in event-coding tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_architecture_adjusted_plots import LINEAGES, architecture_data  # noqa: E402
from build_broad_robustness import TASK_TO_SOURCE  # noqa: E402

SEED = 20260820
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "output/sidecar/frontier_2026"


def analyze(root: Path = ROOT, draws: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(root / "twitter_figures/full34_task_categories.csv")
    split = pd.read_csv(root / "broad18/holdout_task_deltas.csv")[["task", "set"]]
    data = detail.merge(split, on="task", validate="one_to_one")
    rng = np.random.default_rng(SEED)
    rows = []
    for (subset, category), group in data.groupby(["set", "category"], sort=True):
        values = group.delta_f1_points.to_numpy()
        sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
        loo = [(values.sum() - value) / (len(values) - 1) for value in values] if len(values) > 1 else [values[0]]
        rows.append({"set": subset, "category": category, "task_count": len(values),
                     "mean_delta_points": values.mean(), "median_delta_points": np.median(values),
                     "tasks_improved": int((values > 0).sum()), "ci_low": np.percentile(sampled, 2.5),
                     "ci_high": np.percentile(sampled, 97.5), "loo_min": min(loo), "loo_max": max(loo)})
    sensitivity = pd.DataFrame(rows)
    y = data.delta_f1_points.to_numpy(); overall = y.mean()
    observed = sum(len(g) * (g.delta_f1_points.mean() - overall) ** 2
                   for _, g in data.groupby("category")) / ((y - overall) ** 2).sum()
    sizes = data.groupby("category", sort=True).size().tolist()
    null = np.empty(draws)
    for index in range(draws):
        shuffled = rng.permutation(y); start = 0; between = 0.0
        for size in sizes:
            between += size * (shuffled[start:start + size].mean() - shuffled.mean()) ** 2
            start += size
        null[index] = between / ((shuffled - shuffled.mean()) ** 2).sum()
    omnibus = pd.DataFrame([{"statistic": "category_eta_squared", "estimate": observed,
                             "permutation_p": float((null >= observed).mean()),
                             "draws": draws, "seed": SEED}])
    return sensitivity, omnibus


def longitudinal(root: Path = ROOT, draws: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate task trends, then contrast event with other tasks by source."""
    categories = pd.read_csv(root / "twitter_figures/full34_task_categories.csv")[["task", "category"]]
    scores = pd.read_csv(root / "broad18/model_by_task.csv").merge(
        categories, on="task", validate="many_to_one"
    )
    checkpoints = architecture_data(pd.read_csv(root / "broad18/checkpoint_summary.csv"))[
        ["checkpoint_id", "compute_class"]
    ]
    scores = scores.merge(checkpoints, on="checkpoint_id", validate="many_to_one")
    scores["plot_date"] = pd.to_datetime(scores.plot_date)
    scores["years"] = (scores.plot_date - scores.plot_date.min()).dt.days / 365.25
    scores["is_event"] = scores.category.eq("Events and protest")
    task_rows = []
    for (task, category), group in scores.groupby(["task", "category"]):
        slope = np.polyfit(group.years, group.headline_f1, 1)[0] * 100
        task_rows.append({"task": task, "category": category, "is_event": category == "Events and protest",
                          "source_cluster_id": TASK_TO_SOURCE[task], "slope_points_per_year": slope})
    tasks = pd.DataFrame(task_rows)
    source = tasks.groupby(["source_cluster_id", "is_event"], as_index=False).slope_points_per_year.mean()
    # Sources cannot contribute to both sides in the frozen Broad18 mapping.
    source_means = source.groupby("is_event").slope_points_per_year.mean()
    estimate = float(source_means[True] - source_means[False])
    event_values = source.loc[source.is_event, "slope_points_per_year"].to_numpy()
    other_values = source.loc[~source.is_event, "slope_points_per_year"].to_numpy()
    rng = np.random.default_rng(SEED)
    boot = event_values[rng.integers(0, len(event_values), (draws, len(event_values)))].mean(1) - other_values[
        rng.integers(0, len(other_values), (draws, len(other_values)))
    ].mean(1)
    rows = [{"scope": "all_21_source_equal", "models": scores.checkpoint_id.nunique(),
             "event_minus_other_points_per_year": estimate, "ci_low": np.percentile(boot, 2.5),
             "ci_high": np.percentile(boot, 97.5), "probability_positive": float((boot > 0).mean())}]
    for compute_class, group in scores.groupby("compute_class"):
        means = group.groupby(["checkpoint_id", "plot_date", "years", "is_event"], as_index=False).headline_f1.mean()
        slopes = means.groupby("is_event").apply(
            lambda frame: np.polyfit(frame.years, frame.headline_f1, 1)[0] * 100,
            include_groups=False,
        )
        rows.append({"scope": f"compute:{compute_class}", "models": group.checkpoint_id.nunique(),
                     "event_minus_other_points_per_year": float(slopes[True] - slopes[False]),
                     "ci_low": np.nan, "ci_high": np.nan, "probability_positive": np.nan})
    for lineage, ids in LINEAGES.items():
        group = scores[scores.checkpoint_id.isin(ids)]
        means = group.groupby(["checkpoint_id", "plot_date", "is_event"], as_index=False).headline_f1.mean()
        pivot = means.pivot(index=["checkpoint_id", "plot_date"], columns="is_event", values="headline_f1").reset_index().sort_values("plot_date")
        contrast = 100 * ((pivot[True].iloc[-1] - pivot[True].iloc[0]) - (pivot[False].iloc[-1] - pivot[False].iloc[0]))
        rows.append({"scope": f"lineage:{lineage}", "models": len(pivot),
                     "event_minus_other_points_per_year": float(contrast), "ci_low": np.nan,
                     "ci_high": np.nan, "probability_positive": np.nan})
    return tasks, pd.DataFrame(rows)


def _macro_f1(gold: np.ndarray, pred: np.ndarray, labels: list[str]) -> float:
    values=[]
    for label in labels:
        g=gold==label; p=pred==label; tp=(g&p).sum(); fp=(~g&p).sum(); fn=(g&~p).sum()
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return float(np.mean(values))


def external_confirmation(root: Path = ROOT, draws: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    base=root/"event_progress/external"; task_dir=REPO/"experiments/external_event_tasks"
    model_dirs={"llama":base/"llama3_1_70b_external_event","qwen":base/"qwen3_6_27b_external_event"}
    expected={"llama":"019d944e8e566c43939ea83775a27197ebb9b559","qwen":"e89b16ebf1988b3d6befa7de50abc2d76f26eb09"}
    frames={}
    for model,path in model_dirs.items():
        meta=json.loads((path/"run_metadata.json").read_text()); frame=pd.read_csv(path/"predictions.csv",dtype={"item_id":str})
        if meta["revision"]!=expected[model] or meta.get("config_sha256")!="8627158119094091f51e8255f62dbbd85d992c9f17f8cb4a44908f1b232f3e62" or len(frame)!=2000 or frame.duplicated(["task","item_id"]).any(): raise ValueError("external evidence identity/coverage failure")
        gpu=meta["gpu_before_load"]["gpus"]; after=meta["gpu_after_load"]["gpus"]
        if len(gpu)!=1 or gpu[0]["memory_total_mib"]!=97887 or gpu[0]["uuid"]!=after[0]["uuid"] or meta["observed_peak_gpu_memory_used_mib"]>97887: raise ValueError("external GPU gate failure")
        frames[model]=frame
    merged=frames["llama"].merge(frames["qwen"],on=["task","item_id"],suffixes=("_llama","_qwen"),validate="one_to_one")
    rng=np.random.default_rng(SEED); rows=[]
    source={"arabic_gsr_assault_presence":"arabic_gsr","arabic_gsr_protest_presence":"arabic_gsr","maven_event_presence":"maven","rams_event_type":"rams"}
    for task,g in merged.groupby("task",sort=True):
        spec=yaml.safe_load((task_dir/f"{task}.yaml").read_text()); key=spec["label_key"]; labels=spec["labels"]
        frozen=pd.read_csv(task_dir/spec["data_file"],dtype={"source_id":str}).set_index("source_id")
        if set(g.item_id)!=set(frozen.index) or len(frozen)!=500: raise ValueError("frozen external item coverage failure")
        gold=g[f"gt_{key}_llama"].astype(str).to_numpy();
        if not np.array_equal(gold,g[f"gt_{key}_qwen"].astype(str).to_numpy()): raise ValueError("gold mismatch")
        expected_gold=g.item_id.map(frozen[spec["ground_truth"]["column"]]).astype(str).to_numpy()
        if not np.array_equal(gold,expected_gold): raise ValueError("frozen external gold mismatch")
        pred={m:g[f"pred_{key}_{m}"].fillna("__MALFORMED__").astype(str).to_numpy() for m in ["llama","qwen"]}
        scores={m:_macro_f1(gold,pred[m],labels) for m in pred}; idx=rng.integers(0,len(g),(draws,len(g)))
        boot=np.array([_macro_f1(gold[ii],pred["qwen"][ii],labels)-_macro_f1(gold[ii],pred["llama"][ii],labels) for ii in idx])
        rows.append({"task":task,"source_family":source[task],"items":len(g),"llama_f1":scores["llama"],"qwen_f1":scores["qwen"],"delta_f1_points":100*(scores["qwen"]-scores["llama"]),"ci_low":100*np.percentile(boot,2.5),"ci_high":100*np.percentile(boot,97.5),"qwen_better_probability":float((boot>0).mean())})
    detail=pd.DataFrame(rows); family=detail.groupby("source_family").delta_f1_points.mean()
    aggregate=pd.DataFrame([{"aggregation":"equal_task","delta_f1_points":detail.delta_f1_points.mean()},
                            {"aggregation":"equal_source_family","delta_f1_points":family.mean()}])
    return detail,aggregate


def render_confirmation(root: Path, external: pd.DataFrame, output: Path) -> None:
    discovery=pd.read_csv(root/"twitter_figures/full34_task_categories.csv")
    discovery=discovery[discovery.category.eq("Events and protest")]
    groups=[("Existing repository tasks",discovery.delta_f1_points.to_numpy(),"#1d4f77"),
            ("Outcome-blind external tasks",external.delta_f1_points.to_numpy(),"#a4443a")]
    fig,ax=plt.subplots(figsize=(7.6,4.8)); ax.axhline(0,color="#b3b3b3",lw=.8)
    for x,(label,values,color) in enumerate(groups):
        jitter=np.linspace(-.12,.12,len(values)); ax.scatter(x+jitter,values,s=42,color=color,zorder=3)
        mean=values.mean(); ax.scatter(x,mean,s=100,marker="D",facecolor="white",edgecolor=color,lw=2,zorder=4)
        ax.text(x+.18,mean,f"mean {mean:+.1f}",va="center",fontsize=9,fontweight="bold",color=color)
    ax.set_xticks([0,1],[f"{groups[0][0]}\n7 tasks",f"{groups[1][0]}\n4 tasks"])
    ax.set_ylabel("Qwen3.6 27B minus Llama 3.1 70B (F1 points)")
    ax.set_title("The event-coding gain did not generalize",loc="left",fontsize=15,fontweight="bold",pad=10)
    ax.text(0,1.01,"Dots are tasks; diamonds are equal-task means",transform=ax.transAxes,fontsize=8,color=".42")
    ax.grid(False); ax.tick_params(length=0); ax.set_xlim(-.45,1.55)
    for spine in ax.spines.values(): spine.set_color("#b3b3b3"); spine.set_linewidth(.8)
    fig.tight_layout(rect=(.02,.02,.98,.98)); output.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text().strip():
            b=artist.get_window_extent(renderer); c=fig.bbox
            if b.x0 < -1 or b.y0 < -1 or b.x1 > c.x1+1 or b.y1 > c.y1+1: raise ValueError(f"clipped text: {artist.get_text()}")
    fig.savefig(output.with_suffix(".png"),dpi=450,facecolor="white"); fig.savefig(output.with_suffix(".pdf"),facecolor="white"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "event_progress")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity, omnibus = analyze(args.root)
    tasks, trends = longitudinal(args.root)
    external, external_summary = external_confirmation(args.root) if (args.root/"event_progress/external").exists() else (None,None)
    sensitivity.to_csv(args.output_dir / "split_category_sensitivity.csv", index=False)
    omnibus.to_csv(args.output_dir / "category_partition_test.csv", index=False)
    tasks.to_csv(args.output_dir / "task_time_trends.csv", index=False)
    trends.to_csv(args.output_dir / "event_progress_sensitivity.csv", index=False)
    if external is not None:
        external.to_csv(args.output_dir / "external_confirmation.csv", index=False)
        external_summary.to_csv(args.output_dir / "external_confirmation_summary.csv", index=False)
        render_confirmation(args.root,external,args.output_dir/"external_event_confirmation")


if __name__ == "__main__": main()
