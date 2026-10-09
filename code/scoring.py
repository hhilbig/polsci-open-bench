"""Shared scoring for benchmark summaries.

Single source of truth for per-class F1 and the headline metric, so that
`build_summary.py`, `build_summary_batched.py`, `hive_vllm_benchmark.py`,
`summarize_hive_bakeoff.py` and the analysis scripts cannot drift apart.

Background: the original per-task macro F1 averaged over every label in the task
manifest, including labels with zero gold support in the sampled items. Those
labels can only score 0, so they mechanically depress the metric in proportion to
how much unused slack a manifest's label list carries. `mellon_bes_mii_2024`
declares 50 labels and only 35 occur in its sample, so 30% of the macro-F1
denominator was structurally zero (accuracy 0.936 against headline F1 0.505).

`support_only=True` restricts the average to labels that actually occur in the
gold column. `support_only=False` reproduces the original behaviour and is kept
so published v1 numbers stay reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# scipy and scikit-learn are not in the pinned Hive runtime lock, and
# `hive_vllm_benchmark.py` imports the score and ordinal metrics from here. The
# correlation and kappa helpers below are therefore numpy-only, and f1_score is
# imported inside the F1 functions that need it.


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    """Spearman rho as Pearson r of average ranks, which is how scipy handles ties."""
    return _pearson(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))


def _quadratic_weighted_kappa(gt, pred, n_labels):
    """Cohen's kappa with quadratic weights over the fixed scale 0..n_labels-1.

    Matches sklearn's cohen_kappa_score(weights="quadratic", labels=range(n_labels))."""
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)
    observed = np.zeros((n_labels, n_labels))
    np.add.at(observed, (gt, pred), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    idx = np.arange(n_labels)
    weights = (idx[:, None] - idx[None, :]) ** 2
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def _gt_pred_columns(task_def):
    """Return (gt_col, pred_col) for binary/categorical tasks."""
    key = task_def["label_key"]
    return f"gt_{key}", f"pred_{key}"


def scored_labels(task_def, frame, support_only=True):
    """Labels that enter the macro average.

    With `support_only=True`, a label is included only when it actually occurs in
    the gold column (categorical) or has at least one positive gold value
    (multi-binary). Binary tasks have a single positive class and are unaffected.

    The caller should compute this ONCE over the full sample and pass the result
    back in as `label_subset` for any resampling (e.g. the bootstrap), so the
    label set does not fluctuate between replicates.
    """
    labels = list(task_def["labels"])
    if not support_only:
        return labels

    kind = task_def["label_kind"]
    if kind == "binary":
        return labels
    if kind == "score":
        return []
    if kind == "categorical":
        gt_col, _ = _gt_pred_columns(task_def)
        if gt_col not in frame:
            return labels
        present = set(frame[gt_col].dropna().astype(str))
        return [lbl for lbl in labels if str(lbl) in present]
    if kind == "multi_binary":
        kept = []
        for lbl in labels:
            gt_col = f"gt_{lbl}"
            if gt_col not in frame:
                continue
            col = frame[gt_col].dropna()
            if len(col) and col.astype(float).sum() > 0:
                kept.append(lbl)
        return kept
    raise ValueError(f"Unknown label_kind: {kind}")


def per_class_f1(task_def, frame, support_only=True, label_subset=None):
    """Per-class F1 keyed by label.

    Labels excluded by `support_only` (or absent from `label_subset`) map to NaN
    rather than 0, so they are visibly not-scored instead of silently counting as
    a failure. Callers that write `f1_<label>` columns should keep emitting a
    column per manifest label and let NaN mark the unsupported ones.
    """
    from sklearn.metrics import f1_score

    kind = task_def["label_kind"]
    labels = list(task_def["labels"])
    keep = set(
        label_subset
        if label_subset is not None
        else scored_labels(task_def, frame, support_only=support_only)
    )

    out = {}
    if kind == "score":
        return out
    if kind == "multi_binary":
        for lbl in labels:
            if lbl not in keep:
                out[lbl] = np.nan
                continue
            gt_col, pred_col = f"gt_{lbl}", f"pred_{lbl}"
            sub = frame[[gt_col, pred_col]].dropna()
            out[lbl] = (
                f1_score(
                    sub[gt_col].astype(int),
                    sub[pred_col].astype(int),
                    pos_label=1,
                    zero_division=0,
                )
                if len(sub)
                else np.nan
            )
        return out

    gt_col, pred_col = _gt_pred_columns(task_def)
    sub = frame[[gt_col, pred_col]].dropna()

    if kind == "binary":
        key = task_def["label_key"]
        out[key] = (
            f1_score(
                sub[gt_col].astype(int),
                sub[pred_col].astype(int),
                pos_label=1,
                zero_division=0,
            )
            if len(sub)
            else np.nan
        )
        return out

    if kind == "categorical":
        for lbl in labels:
            if lbl not in keep:
                out[lbl] = np.nan
                continue
            out[lbl] = (
                f1_score(
                    sub[gt_col], sub[pred_col],
                    labels=[lbl], average="macro", zero_division=0,
                )
                if len(sub)
                else np.nan
            )
        return out

    raise ValueError(f"Unknown label_kind: {kind}")


def headline_f1(task_def, frame, support_only=True, label_subset=None):
    """The benchmark's headline metric for one (task, model) cell.

    Binary: positive-class F1. Categorical: macro F1 over scored labels.
    Multi-binary: mean positive-class F1 over scored labels.

    `frame` must already be restricted to scoreable rows (the caller decides how
    unparseable outputs are handled; see `mask_unparsed`).
    """
    vals = [
        v
        for v in per_class_f1(
            task_def, frame, support_only=support_only, label_subset=label_subset
        ).values()
        if v is not None and not np.isnan(v)
    ]
    return float(np.mean(vals)) if vals else np.nan


def score_metrics(task_def, frame):
    """Agreement with a numeric gold value, for `label_kind: score` tasks.

    Pearson and Spearman correlation and mean absolute error with the gold value.
    These replace F1, which is undefined for a continuous target.

    A "not applicable" answer (`parse_error == "not_applicable"`, which the prompt
    allows for text with no political content) is scored as the midpoint of
    `score_range`, so a model gains nothing by abstaining on hard, near-centre
    items. Pass the not-applicable rows in; drop only genuinely malformed rows.
    `pearson_r_answered`, `spearman_rho_answered` and `mae_answered` repeat the
    measures over the numerically answered rows only, for reference."""
    gt_col, pred_col = _gt_pred_columns(task_def)
    lo, hi = task_def["score_range"]
    sub = frame[[gt_col, pred_col]].copy()
    if "parse_error" in frame.columns:
        not_applicable = frame["parse_error"].astype(str).eq("not_applicable")
        sub.loc[not_applicable, pred_col] = (float(lo) + float(hi)) / 2
    answered = frame[[gt_col, pred_col]].dropna().astype(float)
    sub = sub.dropna().astype(float)
    out = {"n_scored": len(sub)}
    for suffix, rows in (("", sub), ("_answered", answered)):
        out.update({f"pearson_r{suffix}": np.nan, f"spearman_rho{suffix}": np.nan,
                    f"mae{suffix}": np.nan})
        if len(rows) == 0:
            continue
        out[f"mae{suffix}"] = float((rows[pred_col] - rows[gt_col]).abs().mean())
        if len(rows) >= 3 and rows[pred_col].nunique() > 1 and rows[gt_col].nunique() > 1:
            out[f"pearson_r{suffix}"] = _pearson(rows[gt_col], rows[pred_col])
            out[f"spearman_rho{suffix}"] = _spearman(rows[gt_col], rows[pred_col])
    return out


def ordinal_metrics(task_def, frame):
    """Ordinal agreement for categorical tasks flagged `ordinal: true`.

    Labels are mapped to their position in the manifest list, which is the scale
    order. Quadratic weighted kappa and Spearman correlation give partial credit
    for near misses, which macro F1 does not; mean absolute error is in scale
    steps."""
    gt_col, pred_col = _gt_pred_columns(task_def)
    position = {str(lbl): i for i, lbl in enumerate(task_def["labels"])}
    sub = frame[[gt_col, pred_col]].dropna()
    gt = sub[gt_col].astype(str).map(position)
    pred = sub[pred_col].astype(str).map(position)
    ok = gt.notna() & pred.notna()
    gt, pred = gt[ok].astype(int), pred[ok].astype(int)
    out = {"weighted_kappa": np.nan, "spearman_rho": np.nan, "mae_steps": np.nan}
    if len(gt) == 0:
        return out
    out["mae_steps"] = float((pred - gt).abs().mean())
    if gt.nunique() > 1 or pred.nunique() > 1:
        out["weighted_kappa"] = _quadratic_weighted_kappa(gt, pred, len(position))
    if len(gt) >= 3 and gt.nunique() > 1 and pred.nunique() > 1:
        out["spearman_rho"] = _spearman(gt, pred)
    return out


def macro_f1_arrays(gt_arr, pred_arr, labels):
    """Macro F1 over a fixed label list, for array-based inner loops.

    Used by the bootstrap, which pre-extracts numpy arrays once per cell. Pass
    the label list from `scored_labels` computed on the FULL sample so every
    replicate averages over the same classes.
    """
    from sklearn.metrics import f1_score

    if len(gt_arr) == 0 or not labels:
        return np.nan
    return f1_score(gt_arr, pred_arr, labels=list(labels),
                    average="macro", zero_division=0)


def mask_unparsed(frame, unparsed_as_wrong=False, pred_columns=()):
    """Split rows on parse success.

    Default (`unparsed_as_wrong=False`) drops unparseable rows before scoring,
    which is what the published summaries do. That lets a model's score be
    computed on a subset it selected by failing, so the alternative is offered
    explicitly: with `unparsed_as_wrong=True` the rows are retained and their
    predictions replaced with a sentinel that cannot match any gold label.

    Returns the frame to score.
    """
    if "parse_error" not in frame:
        return frame
    ok = frame["parse_error"].isna() | (
        frame["parse_error"].astype(str).str.strip() == ""
    )
    if not unparsed_as_wrong:
        return frame[ok]
    out = frame.copy()
    for col in pred_columns:
        if col in out:
            out.loc[~ok, col] = "__UNPARSED__"
    return out
