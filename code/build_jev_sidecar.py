#!/usr/bin/env python3
"""Lightweight cost/performance sidecar for the refresh panel, centred on Jev 1.13.

Places every panel model on the two axes that the stored evidence actually
supports, and says plainly why the third one is missing.

Performance: mean task F1 from the 33-task rescored release.
Cost: the run's usage-based cost divided by items, a conservative estimate
rather than an invoice.
Speed: median per-request latency from code/build_latency_sample.py.

The speed figures come from a separate timed run, not from the panel. The eight
original panel models went through provider batch endpoints over one to three
days and share a single collection timestamp, so their wall-clock reflects batch
scheduling rather than model speed, and no per-request latency was recorded for
any of them. The timed run calls every model the same way, sequentially and
unbatched, over 99 stratified panel items.

That measurement is a single sample from one machine at one moment. Provider
load, time of day and network location all move it. It supports the ordering and
the order of magnitude, not a reproducible speed benchmark.

Usage:
  python3 code/build_jev_sidecar.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from page_config import load_page_config

CONFIG = load_page_config(check_paths=False)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RELEASE = CONFIG.release_dir
LATENCY = REPO / "output/sidecar/latency_sample/latency_summary.csv"
LEDGERS = (REPO / "output/sidecar/latency_sample/openai_ledger.json",
           REPO / "output/sidecar/latency_sample/deepseek_ledger.json")
JEV_RESPONSES = REPO / "output/sidecar/refresh_jev_20260918"
# Google was absent from the original panel. These two ran later on the same
# frozen items through the same request builder, so they join the panel rather
# than sitting beside it: the task-demeaned axis is a within-task comparison
# across models, so every model present changes it.
GEMINI = REPO / Path(CONFIG.inputs["api_manifests"][-1]).parent
GEMINI_COST_USD = {"gemini-3.8-flash": 1.803113, "gemini-3.1-flash-lite": 0.474220}
GEMINI_ITEMS = 3400
OUT = CONFIG.resolve("cost_table").parent
PANEL_ITEMS = 3400   # token and cost totals are run-level, over the full panel
LATENCY_ITEMS = 99   # the sequential timed sample, used for ledger-based cost
FOCUS = "jev-1.13.0"

# How each model's cost figure was obtained, best evidence first.
#   provider_ledger   the provider's own billing records for an isolated run
#   provider_reported the provider returns a cost with every response
#   provider_tokens   the provider's own token counts times its published price;
#                     our arithmetic, but on counts we did not have to guess
#   token_estimate    our token counts times published price, never checked
#                     against a bill
COST_BASIS_RANK = {"provider_ledger": 0, "provider_reported": 1,
                   "provider_tokens": 2, "token_estimate": 3}

# Published batch discount, identical across OpenAI, Anthropic and Google.
# Verified for Anthropic at platform.claude.com/docs/en/build-with-claude/
# batch-processing on 2026-09-21: "reducing costs by 50%".
BATCH_DISCOUNT = 0.5


def pareto_frontier(frame, criteria):
    """Models no other model beats on every criterion at once.

    `criteria` maps a column to 'max' or 'min'. A model is dominated when some
    other model is at least as good on all criteria and strictly better on one.
    Rows missing a criterion are skipped for that criterion, so the frontier
    stays computable before the timed run exists.
    """
    usable = [c for c in criteria if c in frame and frame[c].notna().all()]
    keep = []
    for _, row in frame.iterrows():
        dominated = False
        for _, other in frame.iterrows():
            if other.model == row.model:
                continue
            at_least_as_good = all(
                other[c] >= row[c] if criteria[c] == "max" else other[c] <= row[c]
                for c in usable)
            strictly_better = any(
                other[c] > row[c] if criteria[c] == "max" else other[c] < row[c]
                for c in usable)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return keep


def jev_reported_cost_per_1k(root=JEV_RESPONSES):
    """Jev's own per-request costs, summed. OpenRouter returns usage.cost."""
    costs = []
    for path in Path(root).glob("*/responses/*.json"):
        usage = (json.loads(path.read_text()).get("response") or {}).get("usage") or {}
        if "cost" in usage:
            costs.append(float(usage["cost"]))
    if not costs:
        return None, 0
    return sum(costs) / len(costs) * 1000, len(costs)


def ledger_cost_per_1k(paths=LEDGERS, items=LATENCY_ITEMS):
    """Per-model invoiced cost from the provider ledgers for the timed run.

    Ledgers lag: an OpenAI query made immediately after the run returned a
    partial total, $0.2659 against a settled $0.7717. Each file must therefore
    be captured after its ledger settled, which is why this reads snapshots
    rather than calling the APIs.

    Returns (cost per 1,000 items, caveat, peak-rate cost) per model. The caveat
    is a short string where the invoiced figure needs qualifying, empty where it
    does not. The peak figure exists only for a provider whose ledger records a
    `peak_rate_multiplier`, meaning the run was billed at a discounted rate and
    the undiscounted price is a defined multiple of it.
    """
    costs, caveats, peak = {}, {}, {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        snapshot = json.loads(path.read_text())
        n = snapshot.get("items_per_model", items)
        caveat = "; ".join(snapshot.get("caveats", {}).values())
        multiplier = snapshot.get("peak_rate_multiplier")
        for model, total in snapshot.get("cost_usd_by_model", {}).items():
            costs[model] = total / n * 1000
            caveats[model] = caveat
            if multiplier:
                peak[model] = costs[model] * multiplier
    return costs, caveats, peak


def api_model_names(release=RELEASE):
    models = pd.read_csv(Path(release) / "models.csv")
    if "access" not in models:
        return set(models.model)
    return set(models.model[models.access == "api"])


def panel_task_f1(release=RELEASE):
    """Per-task F1 for every model on the panel, Gemini included."""
    # The release also carries the open-weight checkpoints; this sidecar is the
    # API panel only. Once Gemini is in the release it is read from there, so it
    # is scored under the release rule (malformed rows count as incorrect)
    # rather than the Gemini run's own summary, which dropped them.
    tasks = pd.read_csv(Path(release) / "tasks.csv")
    tasks = tasks[tasks.model.isin(api_model_names(release))]
    frames = [tasks[["model", "task", "headline_f1"]]]
    extra = Path(GEMINI) / "tasks.csv"
    if extra.exists() and not tasks.model.str.startswith("gemini").any():
        frames.append(pd.read_csv(extra)[["model", "task", "headline_f1"]])
    combined = pd.concat(frames, ignore_index=True)
    return combined.pivot(index="task", columns="model", values="headline_f1").dropna()


def task_demeaned(release=RELEASE, out=OUT, draws=20000, seed=20260919):
    """Each model's F1 minus the panel mean on the same task, with intervals.

    The per-model intervals on absolute F1 are wide because tasks differ
    enormously in difficulty, from 0.36 to 0.99 F1, and that difficulty moves
    every model together. Subtracting the across-model mean within each task
    removes it, leaving only how far a model sits above or below the panel on
    the same material. On this panel that is about 3.6 times tighter.

    Intervals are a task bootstrap, resampling tasks rather than items, so they
    answer what would happen on a different draw of 33 tasks. They are still not
    a substitute for the pairwise comparisons in the release's pairs.csv: two
    particular models stay correlated after demeaning, so reading a specific
    pair off these intervals understates the evidence.
    """
    wide = panel_task_f1(release)
    centred = wide.sub(wide.mean(axis=1), axis=0)
    values = centred.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, n, (draws, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    frame = pd.DataFrame({
        "model": centred.columns,
        "demeaned_f1": values.mean(axis=0),
        "demeaned_ci_low": lo,
        "demeaned_ci_high": hi,
        "n_tasks": n,
    }).sort_values("demeaned_f1", ascending=False).reset_index(drop=True)
    frame["differs_from_panel_mean"] = (frame.demeaned_ci_low > 0) | (frame.demeaned_ci_high < 0)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "task_demeaned.csv", index=False)
    return frame


def complexity_curve(release=RELEASE, out=OUT, draws=5000, seed=20260920):
    """Does Jev fall further behind as tasks get harder? Slope against the panel.

    The test regresses Jev's per-task F1 on the mean of the other eight models
    on the same task and asks whether the slope is 1. Slope 1 means Jev moves
    one for one with the panel and its gap is a constant offset. Above 1 means
    it falls further behind on hard tasks, below 1 the reverse.

    The comparator leaves Jev out, so the predictor does not contain the
    outcome. An earlier version regressed the Jev-minus-panel gap on the
    all-model mean, which contains Jev's own score at one ninth; simulation put
    that bias at -0.0003, negligible here, but leave-one-out removes it by
    construction and makes the null exactly 1.

    Both scales are reported because they disagree. F1 is bounded, and the
    across-model spread on this panel is about twice as large on hard tasks as
    on easy ones, so the raw scale understates differences at the ceiling. The
    logit scale corrects that and moves the slope away from 1.

    Prompt length is carried as a second complexity axis. It is close to
    uncorrelated with difficulty, so it is a separate question rather than a
    restatement.
    """
    classes = pd.read_csv(Path(release) / "classes.csv")
    wide = panel_task_f1(release)
    focus = wide[FOCUS]
    others = wide.drop(columns=FOCUS).mean(axis=1)
    one = classes[classes.model == classes.model.iloc[0]]

    points = pd.DataFrame({
        "task": wide.index,
        "focus_f1": focus.values,
        "others_mean_f1": others.values,
        "gap": (focus - others).values,
        "n_classes": one.groupby("task").size().reindex(wide.index).values,
    })
    lengths = _median_prompt_chars(wide.index)
    points["median_prompt_chars"] = points.task.map(lengths)

    def logit(p, eps=1e-3):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(points), (draws, len(points)))
    rows = []
    for name, x, y, null in (
            ("slope_vs_panel_raw", others.values, focus.values, 1.0),
            ("slope_vs_panel_logit", logit(others.values), logit(focus.values), 1.0),
            ("gap_vs_log_prompt_chars", np.log(points.median_prompt_chars.values),
             points.gap.values, 0.0),
            ("gap_vs_log_n_classes", np.log(points.n_classes.values),
             points.gap.values, 0.0)):
        boot = np.array([np.polyfit(x[i], y[i], 1)[0] for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"test": name, "null": null, "slope": np.polyfit(x, y, 1)[0],
                     "ci_low": lo, "ci_high": hi,
                     "differs_from_null": bool(lo > null or hi < null)})
    slopes = pd.DataFrame(rows)

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    points.to_csv(out / "complexity_points.csv", index=False)
    slopes.to_csv(out / "complexity_slopes.csv", index=False)
    return points, slopes


def _median_prompt_chars(task_index, panel=REPO / "output/sidecar/refresh_20260910/panel.json"):
    """Median rendered prompt length per task, from the frozen panel."""
    rows = json.loads(Path(panel).read_text())["rows"]
    lengths = {}
    for row in rows:
        lengths.setdefault(row["task"], []).append(len(row["item"]["user_content"]))
    return {task: float(np.median(v)) for task, v in lengths.items()}


def gemini_models(release=RELEASE, draws=20000, seed=20260919):
    """Gemini rows in the release models.csv shape, so build() can concat them.

    Cost is the invoiced total from the run divided by items, which is the
    provider's own billing figure rather than a token estimate.
    """
    tasks_path = Path(GEMINI) / "tasks.csv"
    summary_path = Path(GEMINI) / "run_summary.csv"
    if not (tasks_path.exists() and summary_path.exists()):
        return pd.DataFrame()
    per_task = pd.read_csv(tasks_path)
    summary = pd.read_csv(summary_path).set_index("model")
    rng = np.random.default_rng(seed)
    rows = []
    for model, group in per_task.groupby("model"):
        values = group.headline_f1.to_numpy()
        boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "model": model, "mean_task_f1": values.mean(),
            "task_ci_low": lo, "task_ci_high": hi,
            "cost_usd_upper": GEMINI_COST_USD[model],
            "input_tokens": summary.loc[model, "input_tokens"],
            "output_tokens": summary.loc[model, "output_tokens"],
            "malformed_rate": summary.loc[model, "malformed_rate"],
        })
    return pd.DataFrame(rows)


def build(release=RELEASE, out=OUT, latency=LATENCY):
    models = pd.read_csv(Path(release) / "models.csv")
    models = models[models.model.isin(api_model_names(release))].reset_index(drop=True)
    extra = pd.DataFrame() if models.model.str.startswith("gemini").any() else gemini_models(release)
    if not extra.empty:
        models = pd.concat([models, extra], ignore_index=True)
    timing = pd.read_csv(latency) if Path(latency).exists() else None
    ledger, ledger_caveats, ledger_peak = ledger_cost_per_1k()
    jev_cost, jev_n = jev_reported_cost_per_1k()
    frame = pd.DataFrame({
        "model": models.model,
        "mean_task_f1": models.mean_task_f1,
        "task_ci_low": models.task_ci_low,
        "task_ci_high": models.task_ci_high,
        "cost_usd_run": models.cost_usd_upper,
        # Named "batch_estimate" historically, but the rate basis differs by
        # provider. refresh_pilot.py:28 states it: batch prices for OpenAI and
        # Anthropic, peak uncached direct prices for DeepSeek. Anything that
        # plots this column as a batch rate misrepresents the DeepSeek models.
        "cost_per_1k_items_estimate": models.cost_usd_upper / PANEL_ITEMS * 1000,
        "cost_per_1k_items_batch_estimate": models.cost_usd_upper / PANEL_ITEMS * 1000,
        "estimate_rate_basis": models.model.map(
            lambda m: "peak_direct" if m.startswith("deepseek")
            else ("batch" if m.startswith(("gpt", "claude")) else "not_applicable")),
        "output_tokens_per_item": models.output_tokens / PANEL_ITEMS,
        "input_tokens_per_item": models.input_tokens / PANEL_ITEMS,
        "malformed_rate": models.malformed_rate,
    }).sort_values("mean_task_f1", ascending=False).reset_index(drop=True)

    # Resolve each model to the best cost evidence available, and record which.
    basis, per_1k, mode, caveat = [], [], [], []
    for row in frame.itertuples():
        if row.model.startswith("gemini"):
            # Google returns token counts but no cost, so this is published
            # price times their counts. Tighter than the Anthropic estimate,
            # which uses token counts we produced ourselves, but it is still
            # not a bill and is marked as an estimate on the plots.
            basis.append("provider_tokens")
            per_1k.append(GEMINI_COST_USD[row.model] / GEMINI_ITEMS * 1000)
            mode.append("sequential")
            caveat.append("published price times provider-reported tokens; "
                          "no invoice was read for this provider")
        elif row.model in ledger:
            basis.append("provider_ledger"); per_1k.append(ledger[row.model])
            mode.append("sequential"); caveat.append(ledger_caveats[row.model])
        elif row.model == FOCUS and jev_cost is not None:
            basis.append("provider_reported"); per_1k.append(jev_cost)
            mode.append("sequential"); caveat.append("")
        else:
            # The only models left here are Anthropic's, whose estimate was
            # priced at batch rates (refresh_pilot.py:28) while every other
            # model on the plot carries a standard-rate sequential figure.
            # Anthropic's Message Batches API is a flat 50% discount, so
            # doubling puts them on the same basis. Still an estimate, not an
            # invoice, so they stay marked as such.
            basis.append("token_estimate")
            per_1k.append(row.cost_per_1k_items_batch_estimate / BATCH_DISCOUNT)
            mode.append("sequential")
            caveat.append("batch-rate token estimate rescaled to standard rates "
                          "by the published 50% batch discount; never invoiced")
    frame["cost_per_1k_items"] = per_1k
    frame["cost_basis"] = basis
    frame["cost_mode"] = mode
    frame["cost_caveat"] = caveat
    frame["cost_is_invoiced"] = [b in ("provider_ledger", "provider_reported")
                                 for b in basis]
    # Undiscounted price where the run was billed at a discount. NaN elsewhere,
    # which is most models: only DeepSeek has a peak/off-peak split.
    frame["cost_per_1k_items_peak"] = frame.model.map(ledger_peak)
    # Cost if the same work went through the provider's batch endpoint. OpenAI,
    # Anthropic and Google all publish a flat 50% batch discount, so this is the
    # standard-rate figure halved. DeepSeek and OpenRouter/Jev have no batch
    # endpoint, so they get NaN rather than a discount they cannot obtain.
    has_batch = frame.model.str.startswith(("gpt", "claude", "gemini"))
    frame["cost_per_1k_items_batch"] = np.where(
        has_batch, frame.cost_per_1k_items * BATCH_DISCOUNT, np.nan)
    frame["has_batch_endpoint"] = has_batch
    # How far the batch-rate estimate sits from the invoiced sequential figure,
    # where both exist. Reported so the remaining estimates can be read with the
    # right scepticism rather than at face value.
    frame["estimate_to_actual_ratio"] = [
        (a / e if b != "token_estimate" and e else float("nan"))
        for a, e, b in zip(frame.cost_per_1k_items,
                           frame.cost_per_1k_items_batch_estimate, basis)]

    if timing is not None:
        t = timing.set_index("model")
        frame["median_latency_s"] = frame.model.map(t.median_latency_s)
        frame["p90_latency_s"] = frame.model.map(t.p90_latency_s)
        frame["items_per_minute_sequential"] = frame.model.map(t.items_per_minute_sequential)
        frame["latency_timed_n"] = frame.model.map(t.n)

    frame["on_cost_performance_frontier"] = pareto_frontier(
        frame, {"cost_per_1k_items": "min", "mean_task_f1": "max"})
    # Sensitivity: the same frontier priced as if the discounted run had been
    # billed at standard rates. Reported rather than substituted, because the
    # invoiced figure is what was actually paid.
    at_peak = frame.assign(
        cost_per_1k_items=frame.cost_per_1k_items_peak.fillna(frame.cost_per_1k_items))
    at_peak["on"] = pareto_frontier(
        at_peak, {"cost_per_1k_items": "min", "mean_task_f1": "max"})
    frame["on_three_way_frontier"] = pareto_frontier(
        frame, {"cost_per_1k_items": "min", "mean_task_f1": "max",
                "median_latency_s": "min"})
    focus = frame[frame.model == FOCUS].iloc[0]
    frame["f1_vs_jev"] = frame.mean_task_f1 - focus.mean_task_f1
    frame["cost_multiple_of_jev"] = frame.cost_per_1k_items / focus.cost_per_1k_items

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "cost_performance.csv", index=False)

    dominated = frame.loc[~frame.on_cost_performance_frontier, "model"].tolist()
    summary = {
        "focus": FOCUS,
        "panel": "refresh_20260910, 33 tasks x 100 items, corrected metric",
        "items_per_model": PANEL_ITEMS,
        "jev_mean_task_f1": float(focus.mean_task_f1),
        "jev_cost_per_1k_items_usd": float(focus.cost_per_1k_items),
        "cost_basis_counts": frame.cost_basis.value_counts().to_dict(),
        "cost_note": (
            "cost_per_1k_items uses the best evidence per model: provider_ledger "
            "is the provider's own billing record for the sequential timed run "
            "(OpenAI's costs endpoint, DeepSeek's usage console), "
            "provider_reported is OpenRouter's per-request usage.cost summed over "
            "Jev's own run, and token_estimate is tokens times published price "
            "from the panel's batch run. The estimate does not err in one "
            "direction: against OpenAI's ledger it understates sequential cost by "
            "1.2 to 2.3 times, while against DeepSeek's it overstates by roughly "
            "half, because the DeepSeek run fell on a weekend and was billed at "
            "off-peak rates. The two Anthropic models remain estimates: their "
            "cost data had not appeared in the Claude Console for the run day at "
            "the time of writing, and the Admin API that would return it "
            "programmatically is unavailable for individual accounts."
        ),
        "estimate_to_actual_ratio_by_model": {
            m: round(float(r), 3) for m, r in
            zip(frame.model, frame.estimate_to_actual_ratio) if pd.notna(r)},
        "cost_caveats": {m: c for m, c in zip(frame.model, frame.cost_caveat) if c},
        "jev_rank_by_f1": int(frame.index[frame.model == FOCUS][0]) + 1,
        "jev_is_cheapest": bool(focus.cost_per_1k_items == frame.cost_per_1k_items.min()),
        "cheapest_alternative_above_jev_f1": (
            frame[frame.mean_task_f1 > focus.mean_task_f1]
            .nsmallest(1, "cost_per_1k_items")[["model", "cost_per_1k_items", "mean_task_f1"]]
            .to_dict("records")
        ),
        "cost_performance_frontier": frame.loc[
            frame.on_cost_performance_frontier, "model"].tolist(),
        "cost_performance_frontier_at_peak_rates": at_peak.loc[
            at_peak.on, "model"].tolist(),
        "peak_rate_note": (
            "The DeepSeek run was billed at off-peak rates. DeepSeek's published "
            "off-peak price is exactly half its peak price in every token "
            "category, so the standard-rate figure is the invoiced one doubled. "
            "The peak window is 01:00-04:00 and 06:00-10:00 UTC on weekdays, so "
            "a rerun would have to be scheduled inside it to be billed at the "
            "higher rate."
        ),
        "dominated_on_cost_and_performance": dominated,
        "three_way_frontier": frame.loc[
            frame.on_three_way_frontier, "model"].tolist() if timing is not None else None,
        "jev_median_latency_s": float(focus.median_latency_s) if timing is not None else None,
        "jev_is_fastest": (
            bool(focus.median_latency_s == frame.median_latency_s.min())
            if timing is not None else None),
        "speed_axis": (
            "median per-request latency over 99 stratified panel items, every "
            "model called sequentially and unbatched. Measured separately from "
            "the panel because the eight original models ran through provider "
            "batch endpoints and no per-request latency was recorded for them. "
            "One sample, one machine, one moment: good for ordering and "
            "magnitude, not a reproducible benchmark."
            if timing is not None else "not measured; run code/build_latency_sample.py"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return frame, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default=str(RELEASE))
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    frame, summary = build(args.release, args.output)
    centred = task_demeaned(args.release, args.output)
    complexity_curve(args.release, args.output)
    cols = [c for c in ["model", "mean_task_f1", "cost_per_1k_items", "cost_basis",
                        "cost_mode", "median_latency_s", "cost_multiple_of_jev",
                        "on_three_way_frontier"]
            if c in frame]
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()
    print(json.dumps({k: summary[k] for k in
                      ("jev_rank_by_f1", "jev_is_cheapest", "jev_is_fastest",
                       "three_way_frontier",
                       "dominated_on_cost_and_performance")}, indent=2))
    print(f"\nwrote {args.output}/cost_performance.csv and summary.json")


if __name__ == "__main__":
    main()
