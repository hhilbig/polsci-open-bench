#!/usr/bin/env python3
"""Build the page release for the 25 models on the November task pool.

All 25 models (12 API, 13 open-weight) are scored on the same 4,398 texts: the
stratified API draw (100 texts per task, 50 for the six longest tasks) minus the
two texts OpenRouter's moderation kept out of the Anthropic batches. Predictions
come from score_api.load_predictions, so the page and the draft table use the
same rows.

Scoring is the release scorer (build_refresh_release.compute with the
support-only metric), on the 46 tasks with categorical or binary labels. The
numeric-score task (lemens_gallego_tweet_position) has no F1 and is left out of
the release; its categorical version (lemens_gallego_tweet_position3) is in.

Writes the release in the same layout as the November release, so the page and
figure builders read it unchanged, plus the panel (item ids and gold labels,
used by the figure builder; not published) and the cost table.

Run from the repository root:
    python3 experiments/task_ext_20261114/build_ext_release.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_refresh_release as brr  # noqa: E402
import ext_api_batch as eab  # noqa: E402
import score_api  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

RELEASE_ID = "tasks46_20261119"
OUT = ROOT / "output/sidecar/release_20261119_tasks46"
API = ROOT / "output/sidecar/task_ext_20261114_api"
OPEN = ROOT / "output/sidecar/task_ext_20261114_r2"
PANEL = API / "panel_f1_tasks.json"
COSTS = API / "cost_table.csv"
TAXONOMY = Path(__file__).resolve().parent / "task_categories.csv"
TASK_DIR = ROOT / "tasks_ext_v2_20261114"
SEED = 20261114
ITERATIONS = 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api_provenance(model, rows):
    """Settings, tokens, cost and dates of one API model, from its batch state files."""
    spec = eab.MODELS[model]
    states = [json.loads((API / model / s / "state.json").read_text()) for s in ("pilot", "remainder")]
    dates = sorted({d[:10] for s in states for d in (s.get("submitted") or s.get("created"), s.get("finished")) if d})
    max_tokens = sorted({eab.spec_for(model, s)["max_tokens"] for s in ("pilot", "remainder")})
    direct = any(s.get("mode") == "anthropic_direct" for s in states)
    settings = dict(model=spec["id"], max_tokens=max_tokens[-1],
                    reasoning_effort=(spec.get("reasoning") or {}).get("effort"))
    if spec.get("temperature") is not None:
        settings["temperature"] = spec["temperature"]
    if len(max_tokens) > 1:
        settings["max_tokens_pilot"] = max_tokens[-1]
        settings["max_tokens"] = max_tokens[0]
    providers = sorted({str(p) for p in rows.provider.dropna()})
    return dict(model=model, model_id=spec["id"], provider=", ".join(providers),
                revision=spec["id"], hardware_tier="api", access="api",
                run_dates=dates, input_tokens=int(rows.input_tokens.fillna(0).sum()),
                output_tokens=int(rows.output_tokens.fillna(0).sum()),
                cost_usd_upper=round(sum(s.get("cost_usd") or 0 for s in states), 6),
                cost_note=("Anthropic Message Batches API directly; cost is our token counts times the "
                           "published batch price" if direct else
                           "OpenRouter batch endpoint; cost as reported by OpenRouter per request"
                           if spec.get("mode") == "batch" else
                           "OpenRouter, one request per text; cost as reported by OpenRouter"),
                returned_models=sorted({str(m) for m in rows.returned_model.dropna()}),
                documented_versions=[spec["id"]],
                settings=brr.public_settings(settings) | ({"max_tokens_pilot": settings["max_tokens_pilot"]}
                                                          if "max_tokens_pilot" in settings else {}),
                route="anthropic_direct" if direct else spec.get("mode"))


def open_provenance(model, tasks):
    meta = json.loads((OPEN / model / "run_metadata.json").read_text())
    kwargs = meta["llm_kwargs"]
    tp = int(kwargs.get("tensor_parallel_size", 1))
    gpu = meta["gpu_before_load"]["gpus"][0]["name"].replace(" Max-Q Workstation Edition", "")
    results = {k: v for k, v in meta["task_results"].items() if k in tasks}
    items = sum(v["rows"] for v in results.values())
    seconds = sum(v["generation_seconds"] for v in results.values())
    settings = dict(meta["generation"], **{k: kwargs[k] for k in kwargs if k in
                    ("max_model_len", "tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs",
                     "kernel_config")}, vllm_version=meta["runtime"]["vllm_version"], model=meta["model_id"])
    if "max_num_batched_tokens" in kwargs:
        settings["max_num_batched_tokens"] = kwargs["max_num_batched_tokens"]
    return dict(model=model, model_id=meta["model_id"], revision=meta["revision"],
                quantization=meta.get("quantization"), hardware=f"{tp} × {gpu}",
                hardware_tier="multi-gpu" if tp > 1 else "single-gpu", access="open",
                run_dates=sorted({meta["started_at"][:10], meta["completed_at"][:10]}),
                settings=brr.public_settings(settings) | ({"max_num_batched_tokens": settings["max_num_batched_tokens"]}
                                                          if "max_num_batched_tokens" in settings else {}),
                throughput_items=items, generation_items_per_second=items / seconds,
                throughput_hardware=gpu, prediction_sha256=sha(OPEN / model / "predictions.csv"))


def cost_table(models, n_items):
    """Cost per 1,000 texts at the batch price paid and at the standard rate (twice that)."""
    rows = []
    for m in models:
        p = m["provenance"]
        if p["access"] != "api":
            continue
        paid = p["cost_usd_upper"] / n_items * 1000
        batch = p["route"] in ("batch", "anthropic_direct")
        rows.append(dict(model=m["model"], mean_task_f1=m["mean_task_f1"],
                         cost_usd_run=p["cost_usd_upper"],
                         cost_per_1k_items=paid * 2 if batch else paid,
                         cost_per_1k_items_batch=paid if batch else None,
                         cost_basis="provider_tokens" if p["route"] == "anthropic_direct" else "provider_reported",
                         has_batch_endpoint=batch))
    pd.DataFrame(rows).to_csv(COSTS, index=False)


def main():
    predictions, keep = score_api.load_predictions()
    task_defs = {t["name"]: t for t in load_task_definitions(tasks_dir=TASK_DIR)}
    f1_tasks = sorted(n for n, t in task_defs.items() if t["label_kind"] != "score")
    predictions = predictions[predictions.task.isin(f1_tasks)].reset_index(drop=True)
    taxonomy = pd.read_csv(TAXONOMY)
    if set(taxonomy.task) != set(f1_tasks) or taxonomy.task.duplicated().any():
        raise ValueError("taxonomy must list each of the F1 tasks once")
    categories = dict(zip(taxonomy.task, taxonomy.category))

    # Panel: item ids and gold labels of the scored texts, from the first open model's rows.
    first = predictions[predictions.model == sorted(set(predictions.model))[0]]
    rows = []
    for r in first.sort_values(["task", "item_id"]).to_dict("records"):
        t = task_defs[r["task"]]
        keys = t["labels"] if t["label_kind"] == "multi_binary" else [t["label_key"]]
        gt = {k: (r["gt_" + k] if t["label_kind"] == "categorical" else float(r["gt_" + k])) for k in keys}
        rows.append(dict(task=r["task"], item=dict(item_id=str(r["item_id"]), gt=gt)))
    panel = dict(seed=SEED, source="output/sidecar/task_ext_20261114_api/draw.json minus moderation exclusions",
                 rows=rows)
    PANEL.write_text(json.dumps(panel, ensure_ascii=False) + "\n")
    for name, group in predictions.groupby("model"):
        brr.validate_panel(group, panel, task_defs)

    result = brr.compute(predictions, task_defs, categories, ITERATIONS, active_only=True)
    provenance = {}
    for model in sorted(set(predictions.model)):
        if model in eab.MODELS:
            provenance[model] = api_provenance(model, predictions[predictions.model == model])
        else:
            provenance[model] = open_provenance(model, set(f1_tasks))
    for row in result["models"]:
        row["provenance"] = provenance[row["model"]]
    result["task_definitions"] = [dict(task=n, category=categories[n], source=task_defs[n].get("source"),
                                       family=task_defs[n].get("family"), label_kind=task_defs[n]["label_kind"],
                                       labels=task_defs[n]["labels"], label_key=task_defs[n].get("label_key"))
                                  for n in f1_tasks]
    sizes = pd.Series([r["task"] for r in rows]).value_counts()
    api_dates = [d for p in provenance.values() if p["access"] == "api" for d in p["run_dates"]]
    result["manifest"] = dict(
        release=RELEASE_ID, status="validated_local_preview_publication_approval_required",
        seed=SEED, bootstrap_replicates=ITERATIONS, panel_sha256=brr.canonical_sha(panel),
        taxonomy_sha256=sha(TAXONOMY), generator_sha256=sha(__file__),
        scorer_sha256=sha(ROOT / "code/build_summary.py"),
        malformed_scorer_sha256=sha(ROOT / "code/build_frontier_2026.py"),
        numpy_version=np.__version__, pandas_version=pd.__version__,
        api_completed_at=max(api_dates), baseline="Qwen3.6 27B",
        api_cost_usd_upper=round(sum(p.get("cost_usd_upper") or 0 for p in provenance.values()), 4),
        texts=len(rows), texts_per_task={str(k): int(v) for k, v in sorted(sizes.items())},
        uncertainty=("95% percentile intervals; paired item resampling within each fixed task versus "
                     "resampling tasks. Not simultaneous intervals."),
        scoring=("Equal-task mean F1. Binary positive-class F1; multi-binary mean positive-class F1; "
                 "categorical headline macro F1 over declared classes present in the gold sample of the "
                 "task (classes with no gold item are left out). Malformed decisions and requests the "
                 "provider failed are incorrect. Undefined MCC is null."),
        public_data="Aggregate metrics only; no raw texts, responses, or row-level predictions.")
    result = brr.clean_json(result)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "release.json").write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2,
                                                 allow_nan=False) + "\n")
    for key in ["models", "tasks", "classes", "categories", "pairs"]:
        out_rows = [{k: v for k, v in row.items() if k != "provenance"} for row in result[key]]
        if key == "models":
            for row, original in zip(out_rows, result[key]):
                for field, value in original["provenance"].items():
                    if field not in row:
                        row[field] = (json.dumps(value, sort_keys=True, ensure_ascii=False)
                                      if isinstance(value, (list, dict)) else value)
        pd.DataFrame(out_rows).to_csv(OUT / f"{key}.csv", index=False, float_format="%.12g")
    cost_table(result["models"], len(rows))
    ranked = sorted(result["models"], key=lambda m: -m["mean_task_f1"])
    print(json.dumps(dict(models=len(ranked), tasks=len(f1_tasks), texts=len(rows), output=str(OUT))))
    for m in ranked:
        print(f"{m['model']:36s} {m['mean_task_f1']:.3f}")


if __name__ == "__main__":
    main()
