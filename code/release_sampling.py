"""Class-stratified item draw for releases after refresh_20261106.

The published releases draw the first 100 items of each task in hash order
(``refresh_pilot.select``). That draw is label-blind, so a class with 1 to 2
items in the sample can decide a task's macro-F1 (see
docs/task_quality_audit_20261114.md). This module keeps the hash order but
first reserves up to ``k`` items per class, then fills the rest of the sample
in hash order. ``refresh_pilot.select`` is left unchanged so the frozen
panels stay reproducible.

Rules:

1. Items are ranked by sha256 of ``[seed, task, item_id]``, as in
   ``refresh_pilot``.
2. For binary and categorical tasks, each class in the manifest's label list
   gets the first ``min(count, k)`` items of that class in hash order. For
   multi-binary tasks, each label gets at least ``min(positives, k)`` positive
   items; items already drawn for an earlier label count toward later floors.
3. If the reserved items would exceed ``max_reserved_share`` of ``n``, ``k``
   is lowered until they fit.
4. The remaining slots are filled with the next items in hash order.
5. Score tasks (no classes) fall back to the plain hash draw.

Because rare classes are over-represented, sample accuracy and positive-class
prevalence no longer equal their pool values. Macro-F1 over classes is the
headline, so this is the intended trade.
"""

from __future__ import annotations

import hashlib
import json
import math

DEFAULT_N = 100
DEFAULT_K = 5
MAX_RESERVED_SHARE = 0.6


def _digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _rank(items, seed, task):
    return sorted(items, key=lambda x: (_digest([seed, task, str(x["item_id"])]), str(x["item_id"])))


def _classes_of(item, label_kind, label_key, labels):
    gt = item["gt"]
    if label_kind == "multi_binary":
        return [lab for lab in labels if int(gt.get(lab, 0)) == 1]
    value = gt[label_key]
    if label_kind == "binary":
        value = int(value)
    return [value]


def stratified_select(items, task, *, label_kind, labels=None, label_key=None,
                      seed, n=DEFAULT_N, k=DEFAULT_K,
                      max_reserved_share=MAX_RESERVED_SHARE):
    """Return (selected items in hash order, effective k, per-class counts)."""
    ids = [str(x["item_id"]) for x in items]
    assert len(set(ids)) == len(ids), f"{task}: duplicate item ids"
    assert len(items) >= n, f"{task}: {len(items)} items < {n}"
    ranked = _rank(items, seed, task)
    if label_kind == "score":
        return ranked[:n], 0, {}
    if label_kind == "binary":
        labels = [0, 1]
    assert labels, f"{task}: label list required"

    members = {lab: [] for lab in labels}
    for pos, item in enumerate(ranked):
        for lab in _classes_of(item, label_kind, label_key, labels):
            if lab in members:
                members[lab].append(pos)

    cap = math.floor(max_reserved_share * n)
    k_eff = k
    while k_eff > 0 and sum(min(len(v), k_eff) for v in members.values()) > cap:
        k_eff -= 1

    chosen = set()
    for lab in labels:
        need = min(len(members[lab]), k_eff)
        have = sum(1 for pos in members[lab] if pos in chosen)
        for pos in members[lab]:
            if have >= need:
                break
            if pos not in chosen:
                chosen.add(pos)
                have += 1
    for pos in range(len(ranked)):
        if len(chosen) >= n:
            break
        chosen.add(pos)
    selected = [ranked[pos] for pos in sorted(chosen)]
    counts = {lab: sum(1 for pos in members[lab] if pos in chosen) for lab in labels}
    return selected, k_eff, counts


def select_with_pilots(items, task, **kwargs):
    """Stratified draw plus the two pilot items ``refresh_pilot`` uses:
    the lower-median-length item and the longest item."""
    selected, k_eff, counts = stratified_select(items, task, **kwargs)
    ordered = sorted(selected, key=lambda x: (len(x["user_content"]), str(x["item_id"])))
    pilots = {str(ordered[(len(ordered) - 1) // 2]["item_id"]), str(ordered[-1]["item_id"])}
    return selected, pilots, k_eff, counts


def select_for_task(task_def, *, seed, n=DEFAULT_N, k=DEFAULT_K):
    """Convenience wrapper taking a ``task_registry.load_task_definition`` dict."""
    items = task_def["loader"]()
    return select_with_pilots(
        items, task_def["name"], label_kind=task_def.get("label_kind", "categorical"),
        labels=task_def.get("labels"), label_key=task_def.get("label_key"),
        seed=seed, n=n, k=k)
