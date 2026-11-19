#!/usr/bin/env python3
"""Draw the API subset of the 2026-11-14 task pool.

The open models answered 500 items per task (fewer where the pool is smaller).
API models get a fixed subset of those items: 100 per task, and 50 for the six
tasks with the most input tokens per item, so the two groups can be scored on
exactly the same items.

The draw is stratified by gold label with proportional (largest-remainder)
allocation, so class shares match the 500-item pool. For categorical tasks,
every class present in the pool keeps at least one item. Within a stratum,
items are ranked by sha256(seed|task|item_id), so the draw is reproducible and
does not depend on any model's answers. Multi-label tasks are stratified by the
number of positive labels; the numeric-score task by gold-score quintile.

The pilot is the first 20 drawn items (in hash order) of two tasks whose fixed
instructions are long, which is where prompt caching would save money.

Item ids come from the local task loader. For theocharis_dynamics_incivility
the Hive loader wrote float-string ids whose last digit can differ, so those are
matched to the open-model rows by numeric value; the mapping is stored.

Run from the repository root:
    python3 experiments/task_ext_20261114/build_api_draw.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from task_registry import load_task_definitions  # noqa: E402

TASK_DIR = ROOT / "tasks_ext_v2_20261114"
OPEN_RUN = ROOT / "output/sidecar/task_ext_20261114_r2/qwen3_6_27b_fp8/predictions.csv"
OUT = ROOT / "output/sidecar/task_ext_20261114_api/draw.json"
SEED = 20261115
PER_TASK = 100
LONG_TASK_ITEMS = 50
N_LONG = 6
PILOT_TASKS = ["chalkidis_trump_populism", "halterman_keith_bfrs"]
PILOT_ITEMS = 20


def rank(task, item_id):
    return hashlib.sha256(f"{SEED}|{task}|{item_id}".encode()).hexdigest()


def stratum(task, gt):
    kind = task["label_kind"]
    if kind == "multi_binary":
        return str(sum(int(v) for v in gt.values()))
    if kind == "score":
        return None  # filled in below from quintiles
    return str(gt[task["label_key"]])


def allocate(sizes: dict[str, int], k: int, keep_all: bool) -> dict[str, int]:
    total = sum(sizes.values())
    if k >= total:
        return dict(sizes)
    quota = {s: k * n / total for s, n in sizes.items()}
    take = {s: int(q) for s, q in quota.items()}
    if keep_all:
        for s in take:
            take[s] = max(take[s], 1)
    left = k - sum(take.values())
    order = sorted(sizes, key=lambda s: (-(quota[s] - int(quota[s])), s))
    while left > 0:
        for s in order:
            if left and take[s] < sizes[s]:
                take[s] += 1
                left -= 1
    while left < 0:  # keep_all overshot: trim the largest strata
        s = max((s for s in take if take[s] > 1), key=lambda s: (take[s], s))
        take[s] -= 1
        left += 1
    return take


def main() -> None:
    tasks = {t["name"]: t for t in load_task_definitions(tasks_dir=TASK_DIR)}
    run = pd.read_csv(OPEN_RUN, low_memory=False, dtype={"item_id": str},
                      usecols=["task", "item_id", "prompt_tokens"])
    per_item = run.groupby("task").prompt_tokens.mean().sort_values(ascending=False)
    long_tasks = list(per_item.index[:N_LONG])
    rows, id_map, summary = [], {}, []
    for name in sorted(tasks):
        task = tasks[name]
        run_ids = set(run.loc[run.task == name, "item_id"])
        items = task["loader"]()
        local = {str(i["item_id"]): i for i in items}
        if set(local) != run_ids:
            # Match float-string ids by value (theocharis); refuse anything else.
            run_by_value = {float(x): x for x in run_ids}
            mapping = {}
            for lid in local:
                hits = [rid for v, rid in run_by_value.items() if abs(v - float(lid)) <= abs(v) * 1e-15]
                if len(hits) != 1:
                    raise ValueError(f"{name}: cannot match local id {lid} to the open-model run")
                mapping[lid] = hits[0]
            if set(mapping.values()) != run_ids:
                raise ValueError(f"{name}: id matching is not one-to-one")
            id_map[name] = {k: v for k, v in mapping.items() if k != v}
        if len(local) != len(run_ids):
            raise ValueError(f"{name}: {len(local)} loader items vs {len(run_ids)} run items")
        frame = pd.DataFrame([{"item_id": k, "gt": v["gt"]} for k, v in local.items()])
        if task["label_kind"] == "score":
            scores = frame["gt"].map(lambda g: float(g["score"]))
            frame["stratum"] = pd.qcut(scores.rank(method="first"), 5, labels=False).astype(str)
        else:
            frame["stratum"] = frame["gt"].map(lambda g: stratum(task, g))
        frame["rank"] = frame.item_id.map(lambda i: rank(name, i))
        k = LONG_TASK_ITEMS if name in long_tasks else PER_TASK
        take = allocate(frame.stratum.value_counts().to_dict(), k,
                        keep_all=task["label_kind"] == "categorical")
        chosen = pd.concat([g.sort_values("rank").head(take[s]) for s, g in frame.groupby("stratum")])
        chosen = chosen.sort_values("rank")
        pilot_ids = set(chosen.item_id.head(PILOT_ITEMS)) if name in PILOT_TASKS else set()
        for item_id in chosen.item_id:
            rows.append({"task": name, "item": local[item_id], "pilot": item_id in pilot_ids,
                         "open_item_id": id_map.get(name, {}).get(item_id, item_id)})
        summary.append({"task": name, "pool": len(frame), "drawn": len(chosen), "long": name in long_tasks,
                        "strata": len(take)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": SEED, "per_task": PER_TASK, "long_task_items": LONG_TASK_ITEMS,
               "long_tasks": long_tasks, "pilot_tasks": PILOT_TASKS, "pilot_items": PILOT_ITEMS,
               "rows": rows}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    print(f"{len(rows)} items, {sum(r['pilot'] for r in rows)} in the pilot; long tasks: {long_tasks}")


if __name__ == "__main__":
    main()
