#!/usr/bin/env python3
"""GEPA prompt optimisation for Jev 1.13, scored on data the optimiser never saw.

Why this exists. The benchmark runs every model on one fixed prompt per task, so
it measures zero-shot behaviour. The obvious objection is that a cheap model
might close the gap if you let an optimiser rewrite the prompt for it. GEPA
(reflective prompt evolution, the `gepa` package that dspy.GEPA wraps) is the
method that was suggested, so this runs it on Jev and reports the result in a
way that separates real gain from selection noise.

The honest-scoring design. Every task's frame splits three ways:

  pool      rows the benchmark never sampled. GEPA gets these, split again into
            a reflection set (what the teacher model reads) and a selection set
            (what candidates are ranked on).
  panel     the frozen 100 items every model in the release was scored on. GEPA
            never sees these, and they are what the reported number is computed
            from.

That separation is the point. GEPA's default selection keeps candidates that win
on ANY selection instance (`frontier_type='instance'`), which is a per-instance
selection rule over many candidates, so the selection-set score is optimistically
biased essentially by construction. Reporting it alone would overstate the gain.
This script reports the selection-set score and the frozen-panel score side by
side; the difference between them is the overfitting estimate.

One metric mismatch is unavoidable and is disclosed rather than hidden. GEPA
scores per example, so it optimises exact-match accuracy, while the benchmark
headline is macro-F1 over the scored labels. On an imbalanced task those can
move in opposite directions. The final numbers are always the real headline
metric via code/scoring.py, so a gain in accuracy that does not show up in F1
will be visible as exactly that.

Run with the dedicated venv, which holds the dspy/gepa dependency tree:
  .venv-gepa/bin/python code/gepa_jev.py preflight --task gilardi_stance
  .venv-gepa/bin/python code/gepa_jev.py run --budget 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_supervised_baseline as sb          # noqa: E402
import refresh_jev                              # noqa: E402
import scoring                                  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402

ROOT = Path("output/sidecar/gepa_jev_20260920")
COMPONENT = "benchmark_instructions"
SEED = 20260920

# Reflection model. GEPA needs a capable model to read failures and rewrite the
# instruction; Jev cannot do this for itself. Priced at standard (non-batch)
# rates because these are interactive calls.
REFLECTION_MODEL = "gpt-5.6-terra"
REFLECTION_INPUT_USD_PER_M = 2.0
REFLECTION_OUTPUT_USD_PER_M = 12.0

WORKERS = 8            # Jev requests in flight; the optimiser makes thousands
SELECTION_SIZE = 40     # candidates are ranked on these
REFLECTION_SIZE = 60    # the teacher model reads minibatches from these


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Meter:
    """Running spend, so a run can be stopped before it exceeds what was approved."""
    cap_usd: float
    # Jev bills to OpenRouter and the reflection model to OpenAI, so the two
    # have independent limits. The OpenRouter key carries a hard total cap and
    # returns 403 once it is hit, which aborts a run mid-task, so the Jev share
    # is metered separately and stopped just short of it.
    jev_cap_usd: float = float("inf")
    jev_usd: float = 0.0
    reflection_usd: float = 0.0
    jev_calls: int = 0
    reflection_calls: int = 0
    notes: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total(self) -> float:
        return self.jev_usd + self.reflection_usd

    def check(self) -> None:
        if self.total >= self.cap_usd:
            raise RuntimeError(f"cost cap reached: ${self.total:.4f} >= ${self.cap_usd:.2f}")

    def check_jev(self) -> None:
        if self.jev_usd >= self.jev_cap_usd:
            raise RuntimeError(
                f"openrouter cap reached: ${self.jev_usd:.4f} >= ${self.jev_cap_usd:.2f}")


def jev_key() -> str:
    return refresh_jev.load_key()


def call_jev(task: dict, item: dict, instructions: str, key: str, meter: Meter) -> tuple[dict, float]:
    """One Jev Decisions call with the candidate instruction text substituted in.

    The payload comes from the frozen adapter in refresh_jev so the request is
    identical in shape to the run being compared against; only the instruction
    text differs, which is the thing GEPA is allowed to change.
    """
    meter.check()
    meter.check_jev()
    payload = refresh_jev.build_payload(task, item)
    payload["state"]["benchmark_instructions"] = instructions
    body = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://hhilbig.github.io/llm-benchmark/",
               "X-OpenRouter-Title": "Political Science LLM Benchmark"}
    request = urllib.request.Request(refresh_jev.ENDPOINT, data=body,
                                     headers=headers, method="POST")
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 529) or exc.code >= 500:
                if attempt == 4:
                    raise RuntimeError(f"jev http {exc.code} after {attempt} tries")
                time.sleep(min(2 ** attempt, 20))
                continue
            raise RuntimeError(f"jev http {exc.code}: {exc.read()[:200]!r}")
    usage = (result.get("response") or result).get("usage") or {}
    cost = float(usage.get("cost", 0.0))
    with meter.lock:
        meter.jev_usd += cost
        meter.jev_calls += 1
    return result, cost


def jev_prediction(task: dict, item_id, gold: dict, result: dict) -> tuple[dict, str]:
    """Decode a Decisions response with the frozen decoder from the scored run.

    Reusing refresh_jev.decode rather than re-implementing it is deliberate: a
    prompt that scores well only because this script parses responses more
    leniently than the benchmark did would be an artefact, not a gain.
    """
    record = refresh_jev.decode({"item_id": item_id, "gold": gold}, task, result)
    predictions = {key[5:]: value for key, value in record.items()
                   if key.startswith("pred_")}
    return predictions, record.get("parse_error", "")


def reflection_lm(meter: Meter):
    """OpenAI-backed teacher model as a plain callable, metered per call."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        path = Path("~/.openai_api_key").expanduser()
        key = path.read_text().strip() if path.exists() else ""
    if not key:
        raise RuntimeError("no OpenAI credential for the reflection model")

    def call(prompt: str) -> str:
        meter.check()
        body = {"model": REFLECTION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 4000}
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
        usage = result.get("usage", {})
        with meter.lock:
            meter.reflection_calls += 1
        meter.reflection_usd += (
            usage.get("prompt_tokens", 0) * REFLECTION_INPUT_USD_PER_M
            + usage.get("completion_tokens", 0) * REFLECTION_OUTPUT_USD_PER_M) / 1e6
        return result["choices"][0]["message"]["content"] or ""

    return call


def split_for(task: dict, rng: np.random.Generator) -> dict:
    """pool -> (reflection, selection); panel 100 -> the untouched test set."""
    frame, texts, test_idx, pool_idx, gt_spec = sb.task_split(task)
    gold = sb.gold_matrix(task, frame, gt_spec, test_idx)
    if len(pool_idx) < SELECTION_SIZE + 10:
        return {}
    pool = rng.permutation(np.asarray(pool_idx))
    selection = pool[:SELECTION_SIZE]
    reflection = pool[SELECTION_SIZE:SELECTION_SIZE + REFLECTION_SIZE]
    pool_gold = sb.gold_matrix(task, frame, gt_spec, np.concatenate([selection, reflection]))
    return {"texts": texts, "test_idx": np.asarray(test_idx), "test_gold": gold,
            "selection": selection, "reflection": reflection, "pool_gold": pool_gold}


def gold_dict(task: dict, gold_row) -> dict:
    """Gold in the {field: value} shape refresh_jev.decode expects."""
    if task["label_kind"] == "multi_binary":
        return {label: int(v) for label, v in zip(task["labels"], np.atleast_1d(gold_row))}
    return {task["label_key"]: np.atleast_1d(gold_row)[0]}


def exact_match(task: dict, predictions: dict, gold: dict) -> float:
    if not predictions:
        return 0.0
    if task["label_kind"] == "multi_binary":
        return float(np.mean([int(predictions.get(k, -1)) == int(v) for k, v in gold.items()]))
    key = task["label_key"]
    return float(str(predictions.get(key)) == str(gold[key]))


class JevAdapter:
    """Runs Jev under a candidate instruction and reports per-example scores.

    GEPA scores per example and aggregates by sum or mean, so there is no way to
    hand it a set-level metric like macro-F1. The score here is exact match,
    and the reported headline numbers are recomputed with the real metric.
    """

    # gepa probes `adapter.propose_new_texts is not None` without a hasattr
    # guard, so an adapter that simply omits this optional hook raises
    # AttributeError on every proposal and the run silently produces no
    # candidates at all. Declaring it None selects the default proposer.
    propose_new_texts = None

    def __init__(self, task, texts, gold_lookup, key, meter):
        self.task, self.texts = task, texts
        self.gold_lookup, self.key, self.meter = gold_lookup, key, meter

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa import EvaluationBatch
        instructions = candidate[COMPONENT]
        def one(row_index):
            gold = self.gold_lookup[int(row_index)]
            item = {"user_content": self.texts[int(row_index)], "item_id": int(row_index)}
            try:
                result, _ = call_jev(self.task, item, instructions, self.key, self.meter)
                predictions, parse_error = jev_prediction(
                    self.task, int(row_index), gold, result)
            except RuntimeError as exc:
                if "cost cap" in str(exc):
                    raise
                predictions, parse_error = {}, str(exc)[:160]
            score = exact_match(self.task, predictions, gold)
            return predictions, score, {"text": self.texts[int(row_index)],
                                        "gold": gold, "predicted": predictions,
                                        "parse_error": parse_error, "score": score}

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(one, batch))
        outputs = [r[0] for r in results]
        scores = [r[1] for r in results]
        trajectories = [r[2] for r in results]
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for trace in eval_batch.trajectories or []:
            records.append({
                "Inputs": {"text_to_code": trace["text"][:2000]},
                "Generated Outputs": trace["predicted"] or {"error": trace["parse_error"]},
                "Feedback": ("Correct." if trace["score"] >= 1.0 else
                             f"Incorrect. The correct labelling is {trace['gold']}."
                             + (f" Response problem: {trace['parse_error']}."
                                if trace["parse_error"] else "")),
            })
        return {component: records for component in components_to_update}


def headline_on_panel(task, texts, idx, gold, instructions, key, meter):
    """Score a candidate instruction on the frozen panel with the real metric."""
    def one(pair):
        position, row_index = pair
        item = {"user_content": texts[int(row_index)], "item_id": int(row_index)}
        gold_row = gold_dict(task, gold[position])
        result, _ = call_jev(task, item, instructions, key, meter)
        predictions, _ = jev_prediction(task, int(row_index), gold_row, result)
        record = {f"gt_{k}": v for k, v in gold_row.items()}
        record.update({f"pred_{k}": predictions.get(k) for k in gold_row})
        return record

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, enumerate(idx)))
    return scoring.headline_f1(task, pd.DataFrame(rows), support_only=True)


def optimise_task(task, budget, meter, key, rng):
    """Run GEPA on one task and score the winner on the untouched frozen panel."""
    import gepa

    parts = split_for(task, rng)
    if not parts:
        return {"task": task["name"], "status": "skipped: training pool too small"}

    texts, test_idx, test_gold = parts["texts"], parts["test_idx"], parts["test_gold"]
    selection, reflection = parts["selection"], parts["reflection"]
    order = np.concatenate([selection, reflection])
    gold_lookup = {int(r): gold_dict(task, parts["pool_gold"][i])
                   for i, r in enumerate(order)}

    seed_text = Path(task["prompt_path"]).read_text()
    adapter = JevAdapter(task, texts, gold_lookup, key, meter)
    started = meter.total

    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_text},
        trainset=[int(r) for r in reflection],
        valset=[int(r) for r in selection],
        adapter=adapter,
        reflection_lm=reflection_lm(meter),
        max_metric_calls=budget,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
    )
    best = result.best_candidate[COMPONENT]

    baseline_panel = headline_on_panel(task, texts, test_idx, test_gold, seed_text, key, meter)
    # The seed prompt again, on the same items. Jev is not deterministic, so
    # this is the noise floor the GEPA delta has to clear to mean anything.
    replicate_panel = headline_on_panel(task, texts, test_idx, test_gold, seed_text, key, meter)
    gepa_panel = headline_on_panel(task, texts, test_idx, test_gold, best, key, meter)
    return {
        "task": task["name"], "status": "ok",
        "selection_score_seed": float(result.val_aggregate_scores[0]),
        "selection_score_best": float(max(result.val_aggregate_scores)),
        "panel_f1_seed": float(baseline_panel),
        "panel_f1_seed_replicate": float(replicate_panel),
        "same_prompt_delta": float(replicate_panel - baseline_panel),
        "panel_f1_gepa": float(gepa_panel),
        "panel_f1_delta": float(gepa_panel - baseline_panel),
        "candidates": len(result.val_aggregate_scores),
        "usd": round(meter.total - started, 4),
        "prompt_chars_seed": len(seed_text), "prompt_chars_gepa": len(best),
        "best_prompt": best,
    }


def preflight(task_name, budget, cap):
    """Measure real per-call cost on one task so the full run can be priced."""
    tasks = {t["name"]: t for t in load_task_definitions(active_only=True)}
    task = tasks[task_name]
    meter = Meter(cap_usd=cap)
    rng = np.random.default_rng(SEED)
    record = optimise_task(task, budget, meter, jev_key(), rng)
    n_usable = sum(1 for t in tasks.values() if split_for(t, np.random.default_rng(SEED)))
    print(json.dumps({k: v for k, v in record.items() if k != "best_prompt"}, indent=2))
    print(f"\nmetered: {meter.jev_calls} jev calls ${meter.jev_usd:.4f}, "
          f"{meter.reflection_calls} reflection calls ${meter.reflection_usd:.4f}")
    print(f"total ${meter.total:.4f} for one task at budget {budget}")
    print(f"\nusable tasks: {n_usable}")
    print(f"projected full run: ${meter.total * n_usable:.2f}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "preflight.json").write_text(json.dumps(
        {"task": task_name, "budget": budget, "record": record,
         "jev_calls": meter.jev_calls, "jev_usd": meter.jev_usd,
         "reflection_calls": meter.reflection_calls,
         "reflection_usd": meter.reflection_usd,
         "usable_tasks": n_usable,
         "projected_full_run_usd": meter.total * n_usable}, indent=2))


def run(budget, cap, jev_cap):
    tasks = [t for t in load_task_definitions(active_only=True)]
    meter = Meter(cap_usd=cap, jev_cap_usd=jev_cap)
    rng = np.random.default_rng(SEED)
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        try:
            record = optimise_task(task, budget, meter, jev_key(), rng)
        except RuntimeError as exc:
            print(f"  stopped at {task['name']}: {exc}")
            break
        records.append(record)
        print(f"  {record['task']:36s} {record.get('status')}  "
              f"delta {record.get('panel_f1_delta', float('nan')):+.4f}  "
              f"${meter.total:.3f}", flush=True)
        pd.DataFrame(records).to_csv(ROOT / "gepa_results.csv", index=False)
    pd.DataFrame(records).to_csv(ROOT / "gepa_results.csv", index=False)
    print(f"\n{len(records)} tasks completed")
    print(f"jev ${meter.jev_usd:.4f} (cap ${jev_cap:.2f}), "
          f"reflection ${meter.reflection_usd:.4f}, total ${meter.total:.4f}")
    print(f"wrote {ROOT}/gepa_results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["preflight", "run"])
    ap.add_argument("--task", default="gilardi_stance")
    ap.add_argument("--budget", type=int, default=150,
                    help="max_metric_calls per task, i.e. Jev calls the optimiser may spend")
    ap.add_argument("--cap", type=float, default=2.0, help="hard USD cap for the whole invocation")
    ap.add_argument("--jev-cap", type=float, default=float("inf"),
                    help="separate cap on the OpenRouter/Jev share, which has its own key limit")
    args = ap.parse_args()
    if args.command == "preflight":
        preflight(args.task, args.budget, args.cap)
    else:
        run(args.budget, args.cap, args.jev_cap)


if __name__ == "__main__":
    main()
