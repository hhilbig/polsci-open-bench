#!/usr/bin/env python3
"""Do the model rankings change across subgroups of tasks or texts?

Three splits, fixed before looking at results (coding in task_subgroups.csv):
1. language: the 36 English tasks vs the 10 tasks that are not English only
   (8 non-English, 2 mixed).
2. label_structure: yes/no decisions (binary and multi-label, 18 tasks), 3 to 5
   classes (16), 7 or more classes (12).
3. text length within task, open-weight models only: each task's 500 texts
   split at the task's median input length, so short and long texts share the
   codebook and the gold-label process. Length is the prompt token count of each
   text under the Qwen3.6 27B tokenizer (from its run); the prompt template is the
   same for every text of a task, so within a task this orders texts by length.

The question is about relative performance, not levels: a group of tasks can be
harder for every model, and that says nothing about which model to pick. The
statistic is therefore the model x group interaction after removing model and
task averages (task-level splits) or the common short/long gap (length split).
Its p-value comes from randomization inference: reassign the group labels at
random (across tasks, keeping group sizes; within task for the length split) and
ask how often the interaction is as large as observed.

Outputs go to output/sidecar/release_20261119_tasks46/robustness/subgroups/.

    python3 experiments/task_ext_20261114/subgroup_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import build_refresh_release as brr  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

RELEASE = ROOT / "output/sidecar/release_20261119_tasks46"
OPEN = ROOT / "output/sidecar/task_ext_20261114_r2"
SUBGROUPS = Path(__file__).resolve().parent / "task_subgroups.csv"
TASK_DIR = ROOT / "tasks_ext_v2_20261114"
OUT = RELEASE / "robustness" / "subgroups"
LENGTH_MODEL = "qwen3_6_27b_fp8"
SEED = 20261120
N_PERM = 5000
N_BOOT = 2000


def interaction_stat(matrix: np.ndarray, groups: np.ndarray) -> float:
    """Size of the model x group interaction in a model x task F1 matrix.

    Double-centre the matrix (remove model and task means), average the residuals
    over the tasks of each group, and sum the squares weighted by group size.
    Zero when every model's advantage is the same in every group.
    """
    resid = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + matrix.mean()
    total = 0.0
    for g in np.unique(groups):
        cols = groups == g
        total += cols.sum() * (resid[:, cols].mean(axis=1) ** 2).sum()
    return float(total)


def stratified_language_test(f1: pd.DataFrame, sub: pd.DataFrame, rng) -> dict:
    """Language interaction net of label structure.

    Six of the ten tasks that are not English only are yes/no tasks, so a language
    pattern could be a label-structure pattern. Remove each model's advantage
    within each label-structure group first, then measure the language
    interaction on what is left, permuting language only within label-structure
    groups.
    """
    matrix = f1[sub.index].to_numpy()
    strata = sub["label_structure"].to_numpy()
    lang = sub["language"].to_numpy()
    resid = matrix - matrix.mean(axis=0, keepdims=True)
    for s in np.unique(strata):
        cols = strata == s
        resid[:, cols] -= resid[:, cols].mean(axis=1, keepdims=True)

    def stat(groups):
        total = 0.0
        for g in np.unique(groups):
            cols = groups == g
            total += cols.sum() * (resid[:, cols].mean(axis=1) ** 2).sum()
        return total

    observed = stat(lang)
    perms = []
    for _ in range(N_PERM):
        shuffled = lang.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            shuffled[idx] = rng.permutation(lang[idx])
        perms.append(stat(shuffled))
    perms = np.array(perms)
    return dict(split="language_within_label_structure", statistic=float(observed),
                p_value=float((1 + (perms >= observed).sum()) / (1 + N_PERM)), permutations=N_PERM,
                null_median=float(np.median(perms)))


def task_split(f1: pd.DataFrame, labels: pd.Series, name: str, rng) -> tuple[pd.DataFrame, dict]:
    """Ranks and task-resampling intervals within each group, plus the permutation test."""
    matrix = f1[labels.index].to_numpy()
    groups = labels.to_numpy()
    observed = interaction_stat(matrix, groups)
    perms = np.array([interaction_stat(matrix, rng.permutation(groups)) for _ in range(N_PERM)])
    p = float((1 + (perms >= observed).sum()) / (1 + N_PERM))
    overall = f1.mean(axis=1)
    rows = []
    for g in sorted(set(groups)):
        cols = labels.index[groups == g]
        sub = f1[cols].to_numpy()
        means = sub.mean(axis=1)
        draws = sub[:, rng.integers(len(cols), size=(N_BOOT, len(cols)))].mean(axis=2)
        # Relative to the average model in the group: positive = better than the
        # field here. Its interval comes from the same task resampling.
        rel = means - means.mean()
        rel_draws = draws - draws.mean(axis=0, keepdims=True)
        for i, model in enumerate(f1.index):
            rows.append(dict(split=name, group=g, n_tasks=len(cols), model=model, mean_f1=means[i],
                             ci_low=np.percentile(draws[i], 2.5), ci_high=np.percentile(draws[i], 97.5),
                             vs_field=rel[i], vs_field_ci_low=np.percentile(rel_draws[i], 2.5),
                             vs_field_ci_high=np.percentile(rel_draws[i], 97.5)))
    out = pd.DataFrame(rows)
    out["rank_in_group"] = out.groupby("group")["mean_f1"].rank(ascending=False, method="min").astype(int)
    out["rank_overall"] = out["model"].map(overall.rank(ascending=False, method="min").astype(int))
    rho = {g: float(out[out.group == g][["rank_in_group", "rank_overall"]].corr(method="spearman").iloc[0, 1])
           for g in sorted(set(groups))}
    test = dict(split=name, statistic=observed, p_value=p, permutations=N_PERM,
                null_median=float(np.median(perms)), spearman_with_overall=rho,
                n_models=int(f1.shape[0]), group_sizes={g: int((groups == g).sum()) for g in sorted(set(groups))})
    return out, test


def length_split(task_defs, rng) -> tuple[pd.DataFrame, dict]:
    """Short vs long texts within each task, open-weight models, full 500-text pool."""
    released = pd.read_csv(RELEASE / "models.csv")
    models = sorted(released.loc[released["access"].eq("open"), "model"])
    preds = {m: pd.read_csv(OPEN / m / "predictions.csv", low_memory=False, dtype={"item_id": str}) for m in models}
    ref = preds[LENGTH_MODEL]
    gaps = np.zeros((len(models), len(task_defs)))
    short_f1, long_f1 = np.zeros_like(gaps), np.zeros_like(gaps)
    perm_gaps = np.zeros((N_PERM, len(models), len(task_defs)))
    meta = []
    for j, name in enumerate(sorted(task_defs)):
        task = task_defs[name]
        frames = [preds[m][preds[m].task == name].sort_values("item_id").reset_index(drop=True) for m in models]
        ids = frames[0]["item_id"].tolist()
        if any(f["item_id"].tolist() != ids for f in frames):
            raise ValueError(f"{name}: models do not share the same texts")
        chars = ref[ref.task == name].set_index("item_id").loc[ids, "prompt_tokens"].to_numpy()
        n = len(ids)
        # Median split; ties broken by item order so both halves have n//2 texts.
        order = np.lexsort((np.arange(n), chars))
        is_long = np.zeros(n, dtype=bool)
        is_long[order[n - n // 2:]] = True
        masks = np.vstack([~is_long, is_long]).astype(float)
        # Random halves of the same sizes, shared by all models (paired).
        perm = np.zeros((N_PERM, n))
        for k in range(N_PERM):
            perm[k, rng.permutation(n)[: n // 2]] = 1.0
        for i, frame in enumerate(frames):
            s, l = brr.headline_replicates(frame, task, masks, support_only=True)
            short_f1[i, j], long_f1[i, j], gaps[i, j] = s, l, l - s
            a = brr.headline_replicates(frame, task, perm, support_only=True)
            b = brr.headline_replicates(frame, task, 1.0 - perm, support_only=True)
            perm_gaps[:, i, j] = a - b
        meta.append(dict(task=name, n=n, median_tokens=float(np.median(chars)),
                         short_max_tokens=int(chars[~is_long].max()), long_min_tokens=int(chars[is_long].min())))
    def dispersion(g):  # models' mean long-minus-short gaps around their average
        m = g.mean(axis=-1)
        return ((m - m.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    observed = float(dispersion(gaps))
    null = dispersion(perm_gaps)
    p = float((1 + (null >= observed).sum()) / (1 + N_PERM))
    boot = rng.integers(gaps.shape[1], size=(N_BOOT, gaps.shape[1]))
    gap_draws = gaps[:, boot].mean(axis=2)
    rel_draws = gap_draws - gap_draws.mean(axis=0, keepdims=True)
    mean_gap = gaps.mean(axis=1)
    rows = [dict(split="length_within_task", model=m, short_f1=short_f1[i].mean(), long_f1=long_f1[i].mean(),
                 long_minus_short=mean_gap[i],
                 ci_low=np.percentile(gap_draws[i], 2.5), ci_high=np.percentile(gap_draws[i], 97.5),
                 vs_field=mean_gap[i] - mean_gap.mean(),
                 vs_field_ci_low=np.percentile(rel_draws[i], 2.5), vs_field_ci_high=np.percentile(rel_draws[i], 97.5))
            for i, m in enumerate(models)]
    out = pd.DataFrame(rows)
    for col in ("short_f1", "long_f1"):
        out[f"rank_{col[:-3]}"] = out[col].rank(ascending=False, method="min").astype(int)
    test = dict(split="length_within_task", statistic=observed, p_value=p, permutations=N_PERM,
                null_median=float(np.median(null)), common_gap=float(mean_gap.mean()),
                spearman_short_long=float(out[["rank_short", "rank_long"]].corr(method="spearman").iloc[0, 1]),
                n_models=len(models), n_tasks=int(gaps.shape[1]))
    pd.DataFrame(meta).to_csv(OUT / "length_split_tasks.csv", index=False)
    return out, test


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(RELEASE / "tasks.csv")
    f1 = tasks.pivot(index="model", columns="task", values="headline_f1")
    sub = pd.read_csv(SUBGROUPS).set_index("task")
    if set(sub.index) != set(f1.columns):
        raise ValueError("task_subgroups.csv must list the release tasks exactly")
    results, tests = [], []
    for name in ("language", "label_structure"):
        out, test = task_split(f1, sub[name], name, rng)
        results.append(out)
        tests.append(test)
    tests.append(stratified_language_test(f1, sub, rng))
    # Leave one task out within the smaller language group: best and worst rank.
    small = sub.index[sub["language"].ne("English")]
    loo = pd.DataFrame({drop: f1[[c for c in small if c != drop]].mean(axis=1).rank(ascending=False, method="min")
                        for drop in small})
    pd.DataFrame(dict(model=f1.index, rank_all10=f1[small].mean(axis=1).rank(ascending=False, method="min").values,
                      loo_rank_best=loo.min(axis=1).values, loo_rank_worst=loo.max(axis=1).values)
                 ).to_csv(OUT / "language_leave_one_out.csv", index=False)
    pd.concat(results).to_csv(OUT / "task_splits.csv", index=False, float_format="%.6g")

    task_defs = {t["name"]: t for t in load_task_definitions(tasks_dir=TASK_DIR) if t["name"] in set(f1.columns)}
    for t in task_defs.values():
        t.pop("loader")
    out, test = length_split(task_defs, rng)
    out.to_csv(OUT / "length_split.csv", index=False, float_format="%.6g")
    tests.append(test)
    (OUT / "tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    for t in tests:
        print(json.dumps(t))


if __name__ == "__main__":
    main()
