from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from task_registry import DEFAULT_TASKS_DIR, load_task_definitions


class PanelManifestError(ValueError):
    """Raised when a frozen panel manifest does not match the benchmark tasks."""


@dataclass(frozen=True)
class PanelTaskSpec:
    name: str
    expected_items: int
    fingerprint_sha256: str
    annotation_type: str
    complexity: str
    manifest_sha256: str = ""
    prompt_sha256: str = ""
    schema_sha256: str = ""
    item_keys_sha256: str = ""
    gold_labels_sha256: str = ""
    rendered_inputs_sha256: str = ""
    source_fingerprint_sha256: str = ""
    selection_method: str = ""
    selection_seed: int | None = None

    @property
    def identity_hashes(self) -> dict[str, str]:
        return {
            field: str(getattr(self, field))
            for field in TASK_IDENTITY_HASH_FIELDS
            if getattr(self, field)
        }


@dataclass(frozen=True)
class BenchmarkRequest:
    """One frozen task/item request in manifest order."""

    ordinal: int
    task_name: str
    item_id: str
    task: Mapping[str, Any]
    item: Mapping[str, Any]


@dataclass(frozen=True)
class PanelSelection:
    schema_version: int
    manifest_path: Path
    panel_id: str
    benchmark_commit: str
    expected_tasks: int
    expected_items: int
    task_item_keys_sha256: str
    scorer_version: str
    scorer: Mapping[str, Any]
    calibration: Mapping[str, Any]
    promotion: Mapping[str, Any]
    bootstrap: Mapping[str, Any]
    execution: Mapping[str, Any]
    reliability_gate: Mapping[str, Any]
    task_specs: tuple[PanelTaskSpec, ...]
    tasks: tuple[dict[str, Any], ...]
    panel_sha256: str

    @property
    def benchmark_id(self) -> str:
        """Neutral alias retained alongside legacy ``panel_id`` metadata."""
        return self.panel_id

    @property
    def compact_manifest_sha256(self) -> str:
        return self.panel_sha256

    @property
    def pilot_items(self) -> int:
        return int(self.execution.get("pilot_items", 0))

    @property
    def remainder_items(self) -> int:
        return self.expected_items - self.pilot_items

    @property
    def task_fingerprints(self) -> dict[str, str]:
        return {spec.name: spec.fingerprint_sha256 for spec in self.task_specs}

    @property
    def task_counts(self) -> dict[str, int]:
        return {spec.name: spec.expected_items for spec in self.task_specs}

    @property
    def task_identity_hashes(self) -> dict[str, dict[str, str]]:
        return {spec.name: spec.identity_hashes for spec in self.task_specs}

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.task_specs)

    def row_metadata(self, task_name: str) -> dict[str, str]:
        try:
            fingerprint = self.task_fingerprints[task_name]
        except KeyError as exc:
            raise PanelManifestError(
                f"task {task_name!r} is not part of panel {self.panel_id!r}"
            ) from exc
        return {
            "panel_id": self.panel_id,
            "panel_sha256": self.panel_sha256,
            "panel_task_fingerprint": fingerprint,
        }


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TASK_IDENTITY_HASH_FIELDS = (
    "manifest_sha256",
    "prompt_sha256",
    "schema_sha256",
    "item_keys_sha256",
    "gold_labels_sha256",
    "rendered_inputs_sha256",
)


def task_identity_hashes(task: Mapping[str, Any]) -> dict[str, str]:
    """Hash each frozen input component and the legacy composite fingerprint."""
    items = task["loader"]()
    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise PanelManifestError(f"{task['name']}: duplicate item_id values")
    missing_gold = [
        item["item_id"]
        for item in items
        if any(pd.isna(value) for value in item["gt"].values())
    ]
    if missing_gold:
        raise PanelManifestError(
            f"{task['name']}: {len(missing_gold)} items have missing gold labels; "
            f"first={missing_gold[:5]}"
        )

    manifest_path = Path(task["manifest_path"])
    prompt_path = Path(task["prompt_path"])
    item_key_payload = {
        "schema_version": 1,
        "task": str(task["name"]),
        "item_ids": item_ids,
    }
    gold_payload = {
        "schema_version": 1,
        "task": str(task["name"]),
        "gold_labels": [
            {"item_id": str(item["item_id"]), "gold": item["gt"]}
            for item in items
        ],
    }
    rendered_input_payload = {
        "schema_version": 1,
        "task": str(task["name"]),
        "rendered_inputs": [
            {
                "item_id": str(item["item_id"]),
                "user_content": str(item["user_content"]),
            }
            for item in items
        ],
    }
    payload = {
        "fingerprint_version": 1,
        "task": str(task["name"]),
        "manifest_sha256": _file_sha256(manifest_path),
        "prompt": prompt_path.read_text(),
        "label_kind": task["label_kind"],
        "label_key": task.get("label_key"),
        "labels": list(task["labels"]),
        "json_schema": task["json_schema"],
        "items": [
            {
                "item_id": str(item["item_id"]),
                "user_content": str(item["user_content"]),
                "gold": item["gt"],
            }
            for item in items
        ],
    }
    return {
        "fingerprint_sha256": _canonical_json_sha256(payload),
        "manifest_sha256": _file_sha256(manifest_path),
        "prompt_sha256": _file_sha256(prompt_path),
        "schema_sha256": _canonical_json_sha256(task["json_schema"]),
        "item_keys_sha256": _canonical_json_sha256(item_key_payload),
        "gold_labels_sha256": _canonical_json_sha256(gold_payload),
        "rendered_inputs_sha256": _canonical_json_sha256(rendered_input_payload),
    }


def task_input_fingerprint(task: Mapping[str, Any]) -> str:
    """Hash the exact prompt, schema, rendered inputs, IDs, gold, and task manifest."""
    return task_identity_hashes(task)["fingerprint_sha256"]


def benchmark_item_keys_sha256(tasks: Iterable[Mapping[str, Any]]) -> str:
    """Hash the complete set of task/item keys independent of runner order."""
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        task_name = str(task["name"])
        for item in task["loader"]():
            key = (task_name, str(item["item_id"]))
            if key in seen:
                raise PanelManifestError(
                    f"duplicate benchmark task/item key: {key[0]} {key[1]}"
                )
            seen.add(key)
            records.append({"task": key[0], "item_id": key[1]})
    records.sort(key=lambda row: (row["task"], row["item_id"]))
    return _canonical_json_sha256(
        {"schema_version": 1, "task_item_keys": records}
    )


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PanelManifestError(f"{context} must be a mapping")
    return value


def _full_sha256(value: Any, context: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise PanelManifestError(f"{context} must be a 64-character SHA-256 hash")
    return text


def _full_git_commit(value: Any, context: str) -> str:
    text = str(value).lower()
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text):
        raise PanelManifestError(f"{context} must be a full 40-character Git commit")
    return text


def _task_specs(
    raw_tasks: Any,
    *,
    require_component_hashes: bool = False,
) -> tuple[PanelTaskSpec, ...]:
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise PanelManifestError("tasks must be a non-empty list")
    specs: list[PanelTaskSpec] = []
    for index, raw in enumerate(raw_tasks):
        entry = _mapping(raw, f"tasks[{index}]")
        missing = {
            "name",
            "expected_items",
            "fingerprint_sha256",
            "annotation_type",
            "complexity",
        } - set(entry)
        if missing:
            raise PanelManifestError(
                f"tasks[{index}] missing fields: {sorted(missing)}"
            )
        if require_component_hashes:
            missing_hashes = set(TASK_IDENTITY_HASH_FIELDS) - set(entry)
            if missing_hashes:
                raise PanelManifestError(
                    f"tasks[{index}] missing component hashes: {sorted(missing_hashes)}"
                )
        expected_items = int(entry["expected_items"])
        if expected_items < 1:
            raise PanelManifestError(f"tasks[{index}].expected_items must be positive")
        selection = entry.get("selection")
        selection_method = ""
        selection_seed: int | None = None
        source_fingerprint_sha256 = ""
        if selection is not None:
            if not require_component_hashes:
                raise PanelManifestError(
                    f"tasks[{index}].selection requires a direct schema-version-2 manifest"
                )
            selection = _mapping(selection, f"tasks[{index}].selection")
            selection_method = str(selection.get("method", ""))
            if selection_method != "hash_rank_sha256_v1":
                raise PanelManifestError(
                    f"tasks[{index}].selection.method must be hash_rank_sha256_v1"
                )
            seed = selection.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise PanelManifestError(
                    f"tasks[{index}].selection.seed must be an integer"
                )
            selection_seed = seed
            source_fingerprint_sha256 = _full_sha256(
                entry.get("source_fingerprint_sha256"),
                f"tasks[{index}].source_fingerprint_sha256",
            )
        specs.append(
            PanelTaskSpec(
                name=str(entry["name"]),
                expected_items=expected_items,
                fingerprint_sha256=_full_sha256(
                    entry["fingerprint_sha256"],
                    f"tasks[{index}].fingerprint_sha256",
                ),
                annotation_type=str(entry["annotation_type"]).lower(),
                complexity=str(entry["complexity"]).lower(),
                source_fingerprint_sha256=source_fingerprint_sha256,
                selection_method=selection_method,
                selection_seed=selection_seed,
                **{
                    field: (
                        _full_sha256(entry[field], f"tasks[{index}].{field}")
                        if field in entry
                        else ""
                    )
                    for field in TASK_IDENTITY_HASH_FIELDS
                },
            )
        )
    names = [spec.name for spec in specs]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise PanelManifestError(f"duplicate panel task names: {duplicates}")
    return tuple(specs)


def _selected_task(task: Mapping[str, Any], spec: PanelTaskSpec) -> dict[str, Any]:
    """Apply a manifest-declared, label-blind deterministic item sample."""
    if not spec.selection_method:
        return dict(task)
    source_identity = task_identity_hashes(task)
    if source_identity["fingerprint_sha256"] != spec.source_fingerprint_sha256:
        raise PanelManifestError(
            f"{spec.name}: source task fingerprint drift before deterministic sampling"
        )
    items = list(task["loader"]())
    if spec.expected_items > len(items):
        raise PanelManifestError(
            f"{spec.name}: cannot sample {spec.expected_items} from {len(items)} items"
        )
    seed = int(spec.selection_seed)

    def rank(item: Mapping[str, Any]) -> tuple[str, str]:
        item_id = str(item["item_id"])
        payload = f"{seed}:{spec.name}:{item_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), item_id

    selected_items = tuple(sorted(items, key=rank)[: spec.expected_items])
    selected = dict(task)
    selected["loader"] = lambda frozen=selected_items: list(frozen)
    return selected


def _validate_strata(raw: Mapping[str, Any], specs: Sequence[PanelTaskSpec]) -> None:
    strata = _mapping(raw.get("strata"), "strata")
    observed = {
        "annotation_type": Counter(spec.annotation_type for spec in specs),
        "complexity": Counter(spec.complexity for spec in specs),
    }
    for dimension, counts in observed.items():
        expected_raw = _mapping(strata.get(dimension), f"strata.{dimension}")
        expected = {str(key).lower(): int(value) for key, value in expected_raw.items()}
        if dict(counts) != expected:
            raise PanelManifestError(
                f"strata.{dimension} mismatch: observed {dict(counts)}, expected {expected}"
            )


def _resolve_task_definitions(
    *,
    tasks_dir: Path | str | None,
    task_definitions: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if tasks_dir is not None and task_definitions is not None:
        raise PanelManifestError("provide at most one of tasks_dir or task_definitions")
    if task_definitions is not None:
        return list(task_definitions)
    return load_task_definitions(tasks_dir=tasks_dir or DEFAULT_TASKS_DIR)


def _validate_compact_configuration(
    raw: Mapping[str, Any],
    *,
    expected_items: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = _mapping(raw.get("bootstrap"), "bootstrap")
    execution = _mapping(raw.get("execution"), "execution")

    if str(bootstrap.get("unit", "")) != "within_task_items":
        raise PanelManifestError("bootstrap.unit must be within_task_items")
    if bootstrap.get("paired") is not True:
        raise PanelManifestError("bootstrap.paired must be true")
    if int(bootstrap.get("iterations", -1)) < 1:
        raise PanelManifestError("bootstrap.iterations must be positive")
    seed = bootstrap.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PanelManifestError("bootstrap.seed must be an integer")
    if not 0 < float(bootstrap.get("confidence_level", -1)) < 1:
        raise PanelManifestError("bootstrap.confidence_level must lie between 0 and 1")
    if str(bootstrap.get("interval", "")) != "percentile":
        raise PanelManifestError("bootstrap.interval must be percentile")

    if str(execution.get("order", "")) != "manifest_task_then_item":
        raise PanelManifestError(
            "execution.order must be manifest_task_then_item"
        )
    if str(execution.get("pilot_selection", "")) != "first_requests":
        raise PanelManifestError(
            "execution.pilot_selection must be first_requests"
        )
    pilot_items = int(execution.get("pilot_items", -1))
    if pilot_items < 1 or pilot_items >= expected_items:
        raise PanelManifestError(
            "execution.pilot_items must be positive and smaller than expected_items"
        )
    return bootstrap, execution


def load_panel_manifest(
    path: Path | str,
    *,
    tasks_dir: Path | str | None = None,
    task_definitions: Iterable[dict[str, Any]] | None = None,
) -> PanelSelection:
    """Load a frozen panel and fail closed on task, count, or input drift."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text())
    raw = _mapping(raw, str(manifest_path))
    schema_version = int(raw.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise PanelManifestError("schema_version must be 1 or 2")

    panel_id = str(raw.get("panel_id", "")).strip()
    if not panel_id:
        raise PanelManifestError("panel_id is required")
    benchmark_commit = _full_git_commit(raw.get("benchmark_commit"), "benchmark_commit")
    expected_tasks = int(raw.get("expected_tasks", -1))
    expected_items = int(raw.get("expected_items", -1))
    if expected_tasks < 1 or expected_items < 1:
        raise PanelManifestError("expected_tasks and expected_items must be positive")
    declared_task_item_keys_sha256 = (
        _full_sha256(raw.get("task_item_keys_sha256"), "task_item_keys_sha256")
        if schema_version == 2
        else ""
    )

    scorer = _mapping(raw.get("scorer"), "scorer")
    scorer_version = str(scorer.get("version", "")).strip()
    if not scorer_version:
        raise PanelManifestError("scorer.version is required")
    if schema_version == 2:
        if scorer.get("task_weighting") != "equal":
            raise PanelManifestError("compact scorer.task_weighting must be equal")
        required_scorers = {"binary", "categorical", "multi_binary"}
        missing_scorers = required_scorers - set(scorer)
        if missing_scorers:
            raise PanelManifestError(
                f"compact scorer missing definitions: {sorted(missing_scorers)}"
            )
    if schema_version == 1:
        calibration = _mapping(raw.get("calibration"), "calibration")
        promotion = _mapping(raw.get("promotion"), "promotion")
        bootstrap: dict[str, Any] = {}
        execution: dict[str, Any] = {}
    else:
        if "calibration" in raw or "promotion" in raw:
            raise PanelManifestError(
                "schema_version 2 compact manifests cannot define calibration or promotion"
            )
        calibration = {}
        promotion = {}
        bootstrap, execution = _validate_compact_configuration(
            raw,
            expected_items=expected_items,
        )
    reliability_gate = _mapping(raw.get("reliability_gate"), "reliability_gate")
    if schema_version == 2:
        malformed_max = float(reliability_gate.get("max_malformed_rate", -1))
        if not 0 <= malformed_max <= 1:
            raise PanelManifestError(
                "reliability_gate.max_malformed_rate must lie between 0 and 1"
            )
        if reliability_gate.get("exactly_at_max_passes") is not True:
            raise PanelManifestError(
                "reliability_gate.exactly_at_max_passes must be true"
            )
        if reliability_gate.get("selective_retry_malformed") is not False:
            raise PanelManifestError(
                "reliability_gate.selective_retry_malformed must be false"
            )
    specs = _task_specs(
        raw.get("tasks"),
        require_component_hashes=schema_version == 2,
    )
    if len(specs) != expected_tasks:
        raise PanelManifestError(
            f"panel lists {len(specs)} tasks, expected {expected_tasks}"
        )
    if sum(spec.expected_items for spec in specs) != expected_items:
        raise PanelManifestError(
            "sum of per-task expected_items does not match panel expected_items"
        )
    _validate_strata(raw, specs)

    definitions = _resolve_task_definitions(
        tasks_dir=tasks_dir,
        task_definitions=task_definitions,
    )
    by_name: dict[str, dict[str, Any]] = {}
    duplicate_definitions: set[str] = set()
    for task in definitions:
        name = str(task["name"])
        if name in by_name:
            duplicate_definitions.add(name)
        by_name[name] = task
    if duplicate_definitions:
        raise PanelManifestError(
            f"duplicate loaded task definitions: {sorted(duplicate_definitions)}"
        )
    missing_tasks = [spec.name for spec in specs if spec.name not in by_name]
    if missing_tasks:
        raise PanelManifestError(f"panel tasks are missing from task definitions: {missing_tasks}")

    selected: list[dict[str, Any]] = []
    for spec in specs:
        task = _selected_task(by_name[spec.name], spec)
        observed_items = len(task["loader"]())
        if observed_items != spec.expected_items:
            raise PanelManifestError(
                f"{spec.name}: observed {observed_items} items, expected {spec.expected_items}"
            )
        observed_identity = task_identity_hashes(task)
        observed_fingerprint = observed_identity["fingerprint_sha256"]
        if observed_fingerprint != spec.fingerprint_sha256:
            raise PanelManifestError(
                f"{spec.name}: input fingerprint drift; observed {observed_fingerprint}, "
                f"expected {spec.fingerprint_sha256}"
            )
        for field, expected_hash in spec.identity_hashes.items():
            observed_hash = observed_identity[field]
            if observed_hash != expected_hash:
                raise PanelManifestError(
                    f"{spec.name}: {field} drift; observed {observed_hash}, "
                    f"expected {expected_hash}"
                )
        selected.append(task)
    if len(selected) != expected_tasks:
        raise PanelManifestError(
            f"resolved {len(selected)} panel tasks, expected {expected_tasks}"
        )
    observed_total = sum(len(task["loader"]()) for task in selected)
    if observed_total != expected_items:
        raise PanelManifestError(
            f"resolved {observed_total} panel items, expected {expected_items}"
        )
    observed_task_item_keys_sha256 = benchmark_item_keys_sha256(selected)
    if (
        declared_task_item_keys_sha256
        and observed_task_item_keys_sha256 != declared_task_item_keys_sha256
    ):
        raise PanelManifestError(
            "task_item_keys_sha256 drift; observed "
            f"{observed_task_item_keys_sha256}, expected "
            f"{declared_task_item_keys_sha256}"
        )

    panel = PanelSelection(
        schema_version=schema_version,
        manifest_path=manifest_path,
        panel_id=panel_id,
        benchmark_commit=benchmark_commit,
        expected_tasks=expected_tasks,
        expected_items=expected_items,
        task_item_keys_sha256=observed_task_item_keys_sha256,
        scorer_version=scorer_version,
        scorer=scorer,
        calibration=calibration,
        promotion=promotion,
        bootstrap=bootstrap,
        execution=execution,
        reliability_gate=reliability_gate,
        task_specs=specs,
        tasks=tuple(selected),
        panel_sha256=_canonical_json_sha256(raw),
    )
    requests = benchmark_requests(panel)
    if len(requests) != expected_items:
        raise PanelManifestError(
            f"resolved {len(requests)} unique benchmark requests, expected {expected_items}"
        )
    if schema_version == 2:
        pilot, remainder = split_pilot_remainder(panel)
        if len(pilot) + len(remainder) != expected_items:
            raise PanelManifestError("pilot and remainder do not partition the benchmark")
    return panel


def benchmark_requests(panel: PanelSelection) -> tuple[BenchmarkRequest, ...]:
    """Flatten a frozen benchmark in its only valid task/item request order."""
    requests: list[BenchmarkRequest] = []
    seen: set[tuple[str, str]] = set()
    for task in panel.tasks:
        task_name = str(task["name"])
        for item in task["loader"]():
            item_id = str(item["item_id"])
            key = (task_name, item_id)
            if key in seen:
                raise PanelManifestError(
                    f"duplicate benchmark task/item key: {task_name} {item_id}"
                )
            seen.add(key)
            requests.append(
                BenchmarkRequest(
                    ordinal=len(requests),
                    task_name=task_name,
                    item_id=item_id,
                    task=task,
                    item=item,
                )
            )
    return tuple(requests)


def split_pilot_remainder(
    panel: PanelSelection,
    *,
    pilot_items: int | None = None,
) -> tuple[tuple[BenchmarkRequest, ...], tuple[BenchmarkRequest, ...]]:
    """Return the fixed first-request pilot and its non-overlapping remainder."""
    requests = benchmark_requests(panel)
    n_pilot = panel.pilot_items if pilot_items is None else int(pilot_items)
    if n_pilot < 1 or n_pilot >= len(requests):
        raise PanelManifestError(
            "pilot_items must be positive and smaller than the benchmark request count"
        )
    return requests[:n_pilot], requests[n_pilot:]


def assert_panel_commit(panel: PanelSelection, observed_commit: str) -> None:
    if panel.benchmark_commit != observed_commit:
        raise PanelManifestError(
            f"panel requires benchmark commit {panel.benchmark_commit}, "
            f"but the runner requires {observed_commit}"
        )


def assert_panel_checkout(panel: PanelSelection, repo: Path | str) -> str:
    repo = Path(repo)
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PanelManifestError(
            "the benchmark checkout has no readable Git HEAD; "
            "the frozen panel commit cannot be verified"
        ) from exc
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "cat-file",
                "-e",
                f"{panel.benchmark_commit}^{{commit}}",
            ],
            check=True,
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PanelManifestError(
            f"panel benchmark commit {panel.benchmark_commit} is not available "
            "in this checkout"
        ) from exc
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                panel.benchmark_commit,
                "HEAD",
            ],
            check=True,
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PanelManifestError(
            f"panel benchmark commit {panel.benchmark_commit} is not an ancestor "
            f"of checkout HEAD {observed}"
        ) from exc
    return observed


def validate_panel_checkpoint_frame(
    frame: pd.DataFrame,
    panel: PanelSelection,
    *,
    cell_columns: Sequence[str],
) -> None:
    """Validate completed panel cells in a resumable CSV checkpoint."""
    required = {
        "task",
        "item_id",
        "panel_id",
        "panel_sha256",
        "panel_task_fingerprint",
        *cell_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PanelManifestError(f"panel checkpoint missing columns: {missing}")
    if frame.empty:
        return
    if set(frame["panel_id"].astype(str)) != {panel.panel_id}:
        raise PanelManifestError("panel checkpoint contains a different panel_id")
    if set(frame["panel_sha256"].astype(str)) != {panel.panel_sha256}:
        raise PanelManifestError("panel checkpoint contains a different panel_sha256")
    extra_tasks = sorted(set(frame["task"].astype(str)) - set(panel.task_names))
    if extra_tasks:
        raise PanelManifestError(f"panel checkpoint contains extra tasks: {extra_tasks}")

    expected_ids = {
        task["name"]: {str(item["item_id"]) for item in task["loader"]()}
        for task in panel.tasks
    }
    fingerprints = panel.task_fingerprints
    group_columns = ["task", *cell_columns]
    for keys, cell in frame.groupby(group_columns, dropna=False, sort=False):
        task_name = str(keys[0] if isinstance(keys, tuple) else keys)
        observed_ids = cell["item_id"].astype(str)
        if observed_ids.duplicated().any():
            raise PanelManifestError(f"{task_name}: panel checkpoint has duplicate item IDs")
        if set(observed_ids) != expected_ids[task_name]:
            raise PanelManifestError(
                f"{task_name}: panel checkpoint cell does not have exact item coverage"
            )
        if set(cell["panel_task_fingerprint"].astype(str)) != {fingerprints[task_name]}:
            raise PanelManifestError(
                f"{task_name}: panel checkpoint task fingerprint does not match"
            )
