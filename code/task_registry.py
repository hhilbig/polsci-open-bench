from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_TASKS_DIR = REPO / "tasks"
DEFAULT_SAMPLE_N_V1 = 250
DEFAULT_SAMPLE_N_V2_NEW = 250
DEFAULT_SAMPLE_SEED = 20260422


class _StrictFormatDict(dict):
    def __missing__(self, key):
        raise KeyError(f"Template references missing field: {key}")


def add_task_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-manifest",
        help="Path to a single task manifest YAML file.",
    )
    parser.add_argument(
        "--task-dir",
        help="Path to a single task directory containing task.yaml plus local files.",
    )
    parser.add_argument(
        "--tasks-dir",
        help="Path to a directory of task manifest YAML files.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help=(
            "Load only the active benchmark, skipping manifests marked "
            "`status: excluded`. Frozen panels keep the full manifest set."
        ),
    )


def is_excluded(spec: dict) -> bool:
    """Whether a manifest is currently held out of the active benchmark.

    An excluded task keeps its manifest, prompt, data and past predictions, so the
    decision is reversible and the historical record stays intact. Set
    `status: excluded` plus `excluded_reason` in the manifest.

    Loading returns EVERY manifest by default, because most consumers in this repo
    are frozen panels whose scope is fixed by a config or a published artifact and
    which must keep resolving to the task set they were built on. Only the current
    release view -- the benchmark runners, the summary builders and the task
    inventory -- passes active_only=True.
    """
    return str(spec.get("status", "")).strip().lower() == "excluded"


def _resolve_manifest_paths(task_manifest=None, task_dir=None, tasks_dir=None):
    provided = [bool(task_manifest), bool(task_dir), bool(tasks_dir)]
    if sum(provided) > 1:
        raise ValueError("Use at most one of --task-manifest, --task-dir, or --tasks-dir.")

    if task_manifest:
        path = Path(task_manifest).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    if task_dir:
        path = Path(task_dir).resolve() / "task.yaml"
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    root = Path(tasks_dir).resolve() if tasks_dir else DEFAULT_TASKS_DIR
    if not root.exists():
        raise FileNotFoundError(root)
    manifests = sorted(root.glob("*.yaml"))
    if not manifests:
        raise FileNotFoundError(f"No *.yaml task manifests found in {root}")
    return manifests


def _manifest_default_path(manifest_path: Path, field_name: str, default_name: str) -> Path:
    raw = field_name and field_name.strip()
    path = Path(raw) if raw else Path(default_name)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return path


def _normalize_value(value):
    if pd.isna(value):
        return ""
    return value


def _v1_v2_indices(
    n_total: int,
    n_v1: int = DEFAULT_SAMPLE_N_V1,
    n_v2_new: int = DEFAULT_SAMPLE_N_V2_NEW,
    seed: int = DEFAULT_SAMPLE_SEED,
):
    """Return two sorted, disjoint index lists with the benchmark's fixed sampling."""
    rng_v1 = random.Random(seed)
    n_v1 = min(n_v1, n_total)
    v1_idxs = sorted(rng_v1.sample(range(n_total), n_v1))
    remaining = sorted(set(range(n_total)) - set(v1_idxs))
    n_v2 = min(n_v2_new, len(remaining))
    rng_v2 = random.Random(seed + 1)
    v2_idxs = sorted(rng_v2.sample(remaining, n_v2)) if n_v2 > 0 else []
    return v1_idxs, v2_idxs


def ensure_unique_item_ids(items):
    """Make duplicate sampled item_ids unique while preserving sample order."""
    seen = {}
    out = []
    for item in items:
        item = item.copy()
        base = str(item["item_id"])
        dup_idx = seen.get(base, 0)
        if dup_idx:
            item["item_id"] = f"{base}__dup{dup_idx}"
        seen[base] = dup_idx + 1
        out.append(item)
    return out


def _build_json_schema(label_kind, label_key, labels):
    if label_kind == "binary":
        return {
            "type": "object",
            "properties": {label_key: {"type": "integer", "enum": [0, 1]}},
            "required": [label_key],
            "additionalProperties": False,
        }
    if label_kind == "categorical":
        return {
            "type": "object",
            "properties": {label_key: {"type": "string", "enum": labels}},
            "required": [label_key],
            "additionalProperties": False,
        }
    if label_kind == "multi_binary":
        return {
            "type": "object",
            "properties": {label: {"type": "integer", "enum": [0, 1]} for label in labels},
            "required": labels,
            "additionalProperties": False,
        }
    raise ValueError(f"Unsupported label_kind: {label_kind}")


def _validate_manifest(spec: dict, manifest_path: Path) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{manifest_path} did not parse to a mapping")
    for key in ["label_kind", "text", "ground_truth"]:
        if key not in spec:
            raise ValueError(f"{manifest_path} missing required field: {key}")
    if "template" not in spec["text"]:
        raise ValueError(f"{manifest_path} missing text.template")


def _sampling_config(spec: dict):
    sampling = spec.get("sampling", {}) or {}
    return {
        "n_v1": int(sampling.get("n_v1", DEFAULT_SAMPLE_N_V1)),
        "n_v2_new": int(sampling.get("n_v2_new", DEFAULT_SAMPLE_N_V2_NEW)),
        "seed": int(sampling.get("seed", DEFAULT_SAMPLE_SEED)),
    }


def _render_user_content(row, text_spec):
    row_dict = {k: _normalize_value(v) for k, v in row.to_dict().items()}
    for field, trunc_spec in (text_spec.get("truncations") or {}).items():
        raw = str(row_dict.get(field, ""))
        max_chars = int(trunc_spec["max_chars"])
        if len(raw) > max_chars:
            suffix = trunc_spec.get("suffix", "... [truncated at {max_chars} chars]")
            row_dict[field] = raw[:max_chars] + suffix.format(max_chars=max_chars, field=field)
    return text_spec["template"].format_map(_StrictFormatDict(row_dict))


def _build_gt(row, label_kind, label_key, labels, gt_spec):
    if label_kind == "binary":
        return {label_key: int(row[gt_spec["column"]])}
    if label_kind == "categorical":
        return {label_key: str(row[gt_spec["column"]])}
    columns = gt_spec.get("columns", {})
    if not columns:
        columns = {label: f"gt_{label}" for label in labels}
    return {label: int(row[columns[label]]) for label in labels}


def _build_item_id(row, id_spec, position):
    if not id_spec or id_spec.get("kind", "generated") == "generated":
        prefix = (id_spec or {}).get("prefix", "item")
        return f"{prefix}_{position:03}"
    return str(row[id_spec["column"]])


def _excluded_labels(gt_spec):
    """Gold labels a task declares unscoreable and drops before sampling.

    Used where a source label records coder uncertainty rather than a property of
    the text, so no annotator could recover it from the benchmark input. The GTD
    `Unknown` attack type is the motivating case: GTD assigns it when the source
    report does not specify the method, but the benchmark item is the GTD summary,
    which usually does describe the attack.

    Dropping rows re-draws the sample from a smaller frame, so a manifest using
    this is NOT item-paired with one that does not. Version the task rather than
    editing a published manifest in place.
    """
    return [str(x) for x in (gt_spec.get("exclude_labels") or [])]


def _label_map(gt_spec):
    """Rename gold label values at load time.

    Lets a task correct a label NAME without regenerating the cleaned CSV from a
    source archive. The motivating case is the CAP major-topic list, where topic 5
    carried the legacy name "Labor and Immigration" alongside a separate topic 9
    "Immigration", giving the model two plausible buckets for the same content.

    Renaming does not change which rows are sampled, so a manifest that only maps
    labels stays item-paired with the manifest it supersedes.
    """
    return {str(k): str(v) for k, v in (gt_spec.get("label_map") or {}).items()}


def _make_loader(data_path, id_spec, text_spec, gt_spec, label_kind, label_key, labels, sampling):
    exclude = _excluded_labels(gt_spec)
    label_map = _label_map(gt_spec)

    def loader():
        df = pd.read_csv(data_path, low_memory=False)
        if label_map or exclude:
            column = gt_spec["column"]
            if label_map:
                df[column] = df[column].astype(str).replace(label_map)
            if exclude:
                df = df[~df[column].astype(str).isin(exclude)].reset_index(drop=True)
        v1_idxs, v2_idxs = _v1_v2_indices(
            len(df),
            n_v1=sampling["n_v1"],
            n_v2_new=sampling["n_v2_new"],
            seed=sampling["seed"],
        )
        items = []
        for i, idx in enumerate(v1_idxs):
            row = df.iloc[idx]
            items.append(
                {
                    "item_id": _build_item_id(row, id_spec, i),
                    "user_content": _render_user_content(row, text_spec),
                    "gt": _build_gt(row, label_kind, label_key, labels, gt_spec),
                }
            )
        for j, idx in enumerate(v2_idxs):
            row = df.iloc[idx]
            pos = sampling["n_v1"] + j
            items.append(
                {
                    "item_id": _build_item_id(row, id_spec, pos),
                    "user_content": _render_user_content(row, text_spec),
                    "gt": _build_gt(row, label_kind, label_key, labels, gt_spec),
                }
            )
        return ensure_unique_item_ids(items)

    return loader


def load_task_definition(manifest_path: Path):
    spec = yaml.safe_load(manifest_path.read_text())
    _validate_manifest(spec, manifest_path)

    name = spec.get("name", manifest_path.stem)
    label_kind = spec["label_kind"]
    label_key = spec.get("label_key")
    labels = list(spec.get("labels", []))

    mapped = _label_map(spec["ground_truth"])
    if mapped:
        if label_kind != "categorical":
            raise ValueError(
                f"{manifest_path}: ground_truth.label_map is only supported for "
                f"categorical tasks, not {label_kind}"
            )
        unknown = [v for v in mapped.values() if v not in [str(x) for x in labels]]
        if unknown:
            raise ValueError(
                f"{manifest_path}: ground_truth.label_map maps onto labels that "
                f"are not in the task's label list: {sorted(set(unknown))}"
            )

    excluded = _excluded_labels(spec["ground_truth"])
    if excluded:
        if label_kind != "categorical":
            raise ValueError(
                f"{manifest_path}: ground_truth.exclude_labels is only supported "
                f"for categorical tasks, not {label_kind}"
            )
        unknown = [lbl for lbl in excluded if lbl not in [str(x) for x in labels]]
        if unknown:
            raise ValueError(
                f"{manifest_path}: ground_truth.exclude_labels names labels that "
                f"are not in the task's label list: {unknown}"
            )
        # Drop from the label list too, so the JSON schema and the macro average
        # do not offer a class the sample can no longer contain.
        labels = [lbl for lbl in labels if str(lbl) not in excluded]

    if label_kind == "binary":
        if not label_key:
            raise ValueError(f"{manifest_path} binary task missing label_key")
        labels = labels or [label_key]
    elif label_kind == "categorical":
        if not label_key or not labels:
            raise ValueError(f"{manifest_path} categorical task needs label_key and labels")
    elif label_kind == "multi_binary":
        if not labels:
            raise ValueError(f"{manifest_path} multi_binary task needs labels")
    else:
        raise ValueError(f"{manifest_path} unsupported label_kind: {label_kind}")

    data_path = _manifest_default_path(manifest_path, spec.get("data_file", ""), "data.csv")
    prompt_path = _manifest_default_path(manifest_path, spec.get("prompt_file", ""), "prompt.txt")
    sampling = _sampling_config(spec)

    task = {
        "name": name,
        "order": int(spec.get("order", 999)),
        "family": spec.get("family", "Custom"),
        "source": spec.get("source", "Unspecified"),
        "manifest_path": manifest_path,
        "data_path": data_path,
        "prompt_path": prompt_path,
        "prompt_file": prompt_path.name,
        "label_kind": label_kind,
        "labels": labels,
        "label_key": label_key,
        "json_schema": _build_json_schema(label_kind, label_key, labels),
        "sampling": sampling,
        "loader": _make_loader(
            data_path=data_path,
            id_spec=spec.get("id"),
            text_spec=spec["text"],
            gt_spec=spec["ground_truth"],
            label_kind=label_kind,
            label_key=label_key,
            labels=labels,
            sampling=sampling,
        ),
    }
    return task


def load_task_definitions(task_manifest=None, task_dir=None, tasks_dir=None, active_only=False):
    manifests = _resolve_manifest_paths(task_manifest=task_manifest, task_dir=task_dir, tasks_dir=tasks_dir)
    # Exclusions apply when scanning a directory. A manifest named explicitly is
    # always honoured, because silently returning nothing would look like a
    # missing file rather than a policy decision.
    named_explicitly = bool(task_manifest or task_dir)
    if active_only and not named_explicitly:
        manifests = [
            path
            for path in manifests
            if not is_excluded(yaml.safe_load(path.read_text()) or {})
        ]
    tasks = [load_task_definition(path) for path in manifests]
    return sorted(tasks, key=lambda t: (t["order"], t["name"]))


def load_task_definitions_from_args(args, active_only=False):
    return load_task_definitions(
        task_manifest=getattr(args, "task_manifest", None),
        task_dir=getattr(args, "task_dir", None),
        tasks_dir=getattr(args, "tasks_dir", None),
        active_only=active_only or getattr(args, "active_only", False),
    )
