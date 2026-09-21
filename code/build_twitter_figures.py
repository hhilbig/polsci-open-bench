#!/usr/bin/env python3
"""Build three plain-language figures for the Broad18 Twitter thread."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#1676A3"
RED = "#B14A42"
GRAY = "#9A9A9A"

PARAMETERS = {
    "qwen1_5_72b_chat_awq_hive": 72,
    "qwen1_5_32b_chat_hive": 32,
    "qwen2_72b_instruct_awq_hive": 72,
    "llama3_70b_instruct_fp8_hive": 70,
    "llama3_1_nemotron_70b_fp8_dynamic_hive": 70,
    "llama3_1_70b_instruct_fp8_dynamic_hive": 70,
    "qwen2_5_32b_instruct_bf16_hive": 32,
    "qwen2_5_72b_instruct_fp8_dynamic_hive": 72,
    "llama3_3_70b_instruct_fp8_dynamic_hive": 70,
    "deepseek_r1_distill_qwen_32b_bf16_hive": 32,
    "mistral_small_3_1_24b_bf16_hive": 24,
    "qwen3_32b_bf16_hive": 32,
    "qwen3_30b_a3b_bf16_hive": 30,
    "gemma3_27b_it_fp8_dynamic_hive": 27,
    "gpt_oss_120b_mxfp4_hive": 120,
    "qwen3_next_80b_a3b_fp8_hive": 80,
    "glm4_7_flash_hive": 30,
    "qwen3_5_35b_a3b_fp8_hive": 35,
    "mistral_small_4_119b_nvfp4_hive": 119,
    "qwen3_6_27b_fp8_hive": 27,
    "gemma4_31b_it_qat_w4a16_hive": 31,
}

SHORT_MODELS = {
    "qwen1_5_72b_chat_awq_hive": "Qwen1.5 72B",
    "qwen1_5_32b_chat_hive": "Qwen1.5 32B",
    "qwen2_72b_instruct_awq_hive": "Qwen2 72B",
    "llama3_70b_instruct_fp8_hive": "Llama 3 70B",
    "llama3_1_nemotron_70b_fp8_dynamic_hive": "Nemotron 70B",
    "llama3_1_70b_instruct_fp8_dynamic_hive": "Llama 3.1 70B",
    "qwen2_5_32b_instruct_bf16_hive": "Qwen2.5 32B",
    "qwen2_5_72b_instruct_fp8_dynamic_hive": "Qwen2.5 72B",
    "llama3_3_70b_instruct_fp8_dynamic_hive": "Llama 3.3 70B",
    "deepseek_r1_distill_qwen_32b_bf16_hive": "DeepSeek R1 32B",
    "mistral_small_3_1_24b_bf16_hive": "Mistral Small 3.1 24B",
    "qwen3_32b_bf16_hive": "Qwen3 32B",
    "qwen3_30b_a3b_bf16_hive": "Qwen3 MoE 30B",
    "gemma3_27b_it_fp8_dynamic_hive": "Gemma 3 27B",
    "gpt_oss_120b_mxfp4_hive": "GPT-OSS 120B",
    "qwen3_next_80b_a3b_fp8_hive": "Qwen3-Next 80B",
    "glm4_7_flash_hive": "GLM-4.7 Flash 30B",
    "qwen3_5_35b_a3b_fp8_hive": "Qwen3.5 MoE 35B",
    "mistral_small_4_119b_nvfp4_hive": "Mistral Small 4 119B",
    "qwen3_6_27b_fp8_hive": "Qwen3.6 27B",
    "gemma4_31b_it_qat_w4a16_hive": "Gemma 4 31B",
}

TASK_LABELS = {
    "burnham_covid_threat_minimization": "Minimizes COVID threat?",
    "burnham_polnli_entailment": "Does text support claim?",
    "burnham_polnli_event_entailment": "Does text report event?",
    "dicocco_manifesto_populism": "Populist claim?",
    "cap_crs_policy_topic": "Policy area: reports",
    "cap_party_platform_policy_topic": "Policy area: platforms",
    "erlich_ati_topics": "Attitude topic",
    "muller_fujimura_campaign_policy_area": "Campaign policy area",
    "douglass_icbe_sentence_event_type": "Event type",
    "halterman_keith_bfrs": "Event report?",
    "haunss_papea_fgz_forms": "Protest form",
    "burnham_trump_stance": "Position toward Trump",
    "chae_semeval_stance": "Position toward target",
    "gilardi_stance": "Position in advocacy text",
    "gilardi_relevance": "Relevant to advocacy?",
    "rheault_line_of_fire_incivility": "Uncivil language: speeches",
    "theocharis_dynamics_incivility": "Uncivil language: tweets",
    "twitcivility_impoliteness": "Impolite language",
}

GROUP_LABELS = {
    "claims": "CLAIM JUDGMENTS",
    "issues": "POLICY AND TOPIC LABELS",
    "events": "EVENT LABELS",
    "position": "STANCE LABELS",
    "relevance": "RELEVANCE AND INCIVILITY",
}


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)


def save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overall(checkpoints: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = checkpoints.loc[(checkpoints.series == "open_hive") & (checkpoints.result_status == "completed")].copy()
    if set(data.checkpoint_id) != set(PARAMETERS):
        raise ValueError("completed checkpoint roster does not match frozen parameter map")
    data["parameters_b"] = data.checkpoint_id.map(PARAMETERS)
    data["short_name"] = data.checkpoint_id.map(SHORT_MODELS)
    data["plot_date"] = pd.to_datetime(data.plot_date)
    data = data.sort_values("plot_date").reset_index(drop=True)
    colors = [BLUE if x == "qwen3_6_27b_fp8_hive" else ("0.2" if x == "llama3_70b_instruct_fp8_hive" else "0.68") for x in data.checkpoint_id]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.vlines(data.plot_date, data.ci_low * 100, data.ci_high * 100, color="0.84", lw=1)
    ax.scatter(data.plot_date, data.mean_f1 * 100, s=20 + data.parameters_b * 1.1, c=colors, edgecolor="white", lw=.6, zorder=3)
    labels={
        "qwen1_5_72b_chat_awq_hive":("Qwen1.5 72B",(8,-16)),
        "llama3_70b_instruct_fp8_hive":("Llama 3 70B",(8,8)),
        "gpt_oss_120b_mxfp4_hive":("GPT-OSS 120B",(-6,10)),
        "qwen3_next_80b_a3b_fp8_hive":("Qwen3-Next 80B",(8,-17)),
        "qwen3_6_27b_fp8_hive":("Qwen3.6 27B",(-88,10)),
    }
    for _,row in data.loc[data.checkpoint_id.isin(labels)].iterrows():
        label,offset=labels[row.checkpoint_id]
        ax.annotate(label,(row.plot_date,row.mean_f1*100),xytext=offset,textcoords="offset points",fontsize=9,fontweight="bold" if row.checkpoint_id in {"llama3_70b_instruct_fp8_hive","qwen3_6_27b_fp8_hive"} else "normal")
    ax.set_ylabel("Mean F1 across 18 tasks")
    ax.set_xlabel("Artifact publication date")
    ax.set_title("Most average gain occurred by June 2024", loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(.0, 1.005, "Every completed model; point size shows parameters and lines show 95% item-bootstrap intervals", transform=ax.transAxes, fontsize=9, color="0.4")
    early_frontier = data.loc[data.checkpoint_id.eq("llama3_70b_instruct_fp8_hive"), "mean_f1"].iloc[0]
    ax.axhline(early_frontier * 100, color="0.82", lw=.8, ls="--")
    ax.spines[["top","right"]].set_visible(False); ax.grid(False); fig.autofmt_xdate(rotation=0); fig.tight_layout(); save(fig, output)
    return data[["checkpoint_id", "plot_date", "short_name", "parameters_b", "mean_f1", "ci_low", "ci_high"]]


def task_changes(model_by_task: pd.DataFrame, output: Path) -> pd.DataFrame:
    old_id = "llama3_1_70b_instruct_fp8_dynamic_hive"; new_id = "qwen3_6_27b_fp8_hive"
    pair = model_by_task.loc[model_by_task.checkpoint_id.isin([old_id, new_id])].pivot(index="task", columns="checkpoint_id", values="headline_f1")
    if set(pair.index) != set(TASK_LABELS) or pair.isna().any().any():
        raise ValueError("task comparison does not have exact 18-task paired coverage")
    pair["delta_points"] = 100 * (pair[new_id] - pair[old_id])
    # Frozen ordering keeps related judgments together while sorting within groups by change.
    group_for = {t: g for g, tasks in {
        "claims": ["burnham_covid_threat_minimization","burnham_polnli_entailment","burnham_polnli_event_entailment","dicocco_manifesto_populism"],
        "issues": ["cap_crs_policy_topic","cap_party_platform_policy_topic","erlich_ati_topics","muller_fujimura_campaign_policy_area"],
        "events": ["douglass_icbe_sentence_event_type","halterman_keith_bfrs","haunss_papea_fgz_forms"],
        "position": ["burnham_trump_stance","chae_semeval_stance","gilardi_stance"],
        "relevance": ["gilardi_relevance","rheault_line_of_fire_incivility","theocharis_dynamics_incivility","twitcivility_impoliteness"],
    }.items() for t in tasks}
    pair["group"] = [group_for[t] for t in pair.index]; pair["label"] = [TASK_LABELS[t] for t in pair.index]
    rows=[]
    for g,label in GROUP_LABELS.items():
        values=pair.loc[pair.group.eq(g),"delta_points"]
        rows.append({"group":g,"group_label":label.title(),"task_count":len(values),"mean_delta_points":values.mean(),"tasks_better":int((values>0).sum()),"tasks_worse":int((values<0).sum())})
    shown=pd.DataFrame(rows); y=np.arange(len(shown))[::-1]
    fig,ax=plt.subplots(figsize=(9.5,5.5)); ax.axvline(0,color="0.65",lw=.8)
    colors=np.where(shown.mean_delta_points>=0,BLUE,RED); ax.hlines(y,0,shown.mean_delta_points,color=colors,lw=2); ax.scatter(shown.mean_delta_points,y,c=colors,s=65,zorder=3)
    labels=[f"{r.group_label}\n{r.tasks_better} of {r.task_count} tasks improved" for r in shown.itertuples()]
    ax.set_yticks(y,labels,fontsize=10); ax.set_xlabel("Average Qwen3.6 27B minus Llama 3.1 70B (F1 points)")
    ax.set_title("Qwen3.6 gains on some labels and regresses on others",loc="left",fontsize=14,fontweight="bold",pad=14)
    mean=pair.delta_points.mean(); ax.text(.0,1.005,f"Category averages; across all 18 tasks: {mean:+.1f} points (11 gains, 7 losses)",transform=ax.transAxes,fontsize=9,color="0.4")
    style(ax); fig.tight_layout(); save(fig,output)
    return shown


def disagreement(items: pd.DataFrame, output: Path, draws: int = 10_000, seed: int = 20260822) -> pd.DataFrame:
    required={"dataset","agreement","delta"}
    if not required.issubset(items.columns) or items.dataset.nunique()!=5:
        raise ValueError("disagreement items must contain five complete datasets")
    work=items.copy(); labels=["Lower human agreement","Higher human agreement"]
    work["agreement_group"]=""
    for dataset,group in work.groupby("dataset"):
        q1,q3=group.agreement.quantile([.25,.75])
        if q1>=q3: raise ValueError(f"agreement quartiles do not separate for {dataset}")
        selected=group.index[(group.agreement<=q1)|(group.agreement>=q3)]
        work.loc[selected,"agreement_group"]=np.where(group.loc[selected,"agreement"]<=q1,labels[0],labels[1])
    work=work.loc[work.agreement_group.ne("")].copy()
    by_dataset=work.groupby(["dataset","agreement_group"]).delta.mean().unstack()
    if list(by_dataset.columns)!=labels: by_dataset=by_dataset.reindex(columns=labels)
    if by_dataset.isna().any().any(): raise ValueError("a dataset lacks an agreement group")
    point=by_dataset.mean(axis=0); rows=[]
    for dataset,row in by_dataset.iterrows():
        for label in labels: rows.append({"dataset":dataset,"agreement_group":label,"mean_delta_points":100*row[label],"is_pooled":False})
    for label in labels: rows.append({"dataset":"Equal-dataset average","agreement_group":label,"mean_delta_points":100*point[label],"is_pooled":True})
    shown=pd.DataFrame(rows); x=np.arange(2); fig,ax=plt.subplots(figsize=(8.8,5.7)); ax.axhline(0,color="0.7",lw=.8)
    offsets={"ALIA civic stance":.10,"HateXplain":-.06,"Measuring Hate Speech":.16,"Moral Foundations Reddit":-.16,"SemEval stance":.03}
    for dataset,row in by_dataset.iterrows():
        values=100*row[labels].to_numpy(); ax.plot(x,values,color="0.67",lw=1.2); ax.scatter(x,values,color="0.55",s=28,zorder=2)
        ax.text(1.025,values[1]+offsets[dataset],dataset,fontsize=8,color="0.38",va="center")
    values=100*point[labels].to_numpy(); ax.plot(x,values,color=BLUE,lw=3); ax.scatter(x,values,color=BLUE,s=80,zorder=3)
    ax.text(1.025,values[1]-.23,"Equal-dataset average",fontsize=9,color=BLUE,fontweight="bold",va="center")
    ax.set_xticks(x,labels); ax.set_ylabel("Qwen3.6 minus Llama 3.1 score (points)")
    ax.set_title("Human disagreement does not consistently explain the regressions",loc="left",fontsize=14,fontweight="bold",pad=14)
    ax.text(.0,1.01,"Three datasets slope upward; two slope downward",transform=ax.transAxes,fontsize=9,color="0.4")
    ax.set_xlim(-.05,1.34)
    ax.spines[["top","right"]].set_visible(False); ax.grid(False); fig.tight_layout(); save(fig,output); return shown


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--broad-dir",type=Path,required=True); p.add_argument("--disagreement-items",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    overall(pd.read_csv(a.broad_dir/"checkpoint_summary.csv"),a.output_dir/"01_overall_progress").to_csv(a.output_dir/"01_overall_progress.csv",index=False)
    task_changes(pd.read_csv(a.broad_dir/"model_by_task.csv"),a.output_dir/"02_task_changes").to_csv(a.output_dir/"02_task_changes.csv",index=False)
    disagreement(pd.read_csv(a.disagreement_items),a.output_dir/"03_human_disagreement").to_csv(a.output_dir/"03_human_disagreement.csv",index=False)

if __name__=="__main__": main()
