from __future__ import annotations

import argparse
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MODELS_DIR = REPO / "models"
VALID_BACKENDS = {"ollama", "openai", "anthropic"}
VALID_COMPUTE_CLASSES = {"local", "api"}
VALID_OPENAI_RESPONSE_FORMATS = {"json_schema", "json_object"}
VALID_THINKING_MODES = {"enabled", "disabled"}


def add_model_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-manifest",
        help="Path to a single model manifest YAML file.",
    )
    parser.add_argument(
        "--models-dir",
        help="Path to a directory of model manifest YAML files.",
    )


def _resolve_model_manifest_paths(model_manifest=None, models_dir=None):
    provided = [bool(model_manifest), bool(models_dir)]
    if sum(provided) > 1:
        raise ValueError("Use at most one of --model-manifest or --models-dir.")

    if model_manifest:
        path = Path(model_manifest).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    root = Path(models_dir).resolve() if models_dir else DEFAULT_MODELS_DIR
    if not root.exists():
        raise FileNotFoundError(root)
    manifests = sorted(root.glob("*.yaml"))
    if not manifests:
        raise FileNotFoundError(f"No *.yaml model manifests found in {root}")
    return manifests


def _resolve_optional_path(raw_path: str | None):
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


def _validate_model_manifest(spec: dict, manifest_path: Path):
    if not isinstance(spec, dict):
        raise ValueError(f"{manifest_path} did not parse to a mapping")
    for key in ["name", "backend"]:
        if key not in spec:
            raise ValueError(f"{manifest_path} missing required field: {key}")
    backend = spec["backend"]
    if backend not in VALID_BACKENDS:
        raise ValueError(f"{manifest_path} has unsupported backend: {backend}")
    compute_class = spec.get("compute_class", _default_compute_class(backend))
    if compute_class not in VALID_COMPUTE_CLASSES:
        raise ValueError(
            f"{manifest_path} has unsupported compute_class: {compute_class}"
        )
    response_format_type = spec.get("response_format_type")
    if response_format_type and response_format_type not in VALID_OPENAI_RESPONSE_FORMATS:
        raise ValueError(
            f"{manifest_path} has unsupported response_format_type: {response_format_type}"
        )
    thinking_mode = spec.get("thinking_mode")
    if thinking_mode and thinking_mode not in VALID_THINKING_MODES:
        raise ValueError(
            f"{manifest_path} has unsupported thinking_mode: {thinking_mode}"
        )


def _default_compute_class(backend: str) -> str:
    return "local" if backend == "ollama" else "api"


def load_model_definition(manifest_path: Path):
    spec = yaml.safe_load(manifest_path.read_text())
    _validate_model_manifest(spec, manifest_path)

    backend = spec["backend"]
    model = {
        "name": spec["name"],
        "order": int(spec.get("order", 999)),
        "backend": backend,
        "display_name": spec.get("display_name", spec["name"]),
        "compute_class": spec.get("compute_class", _default_compute_class(backend)),
        "think": bool(spec.get("think", False)),
        "provider": spec.get("provider"),
        "reasoning_effort": spec.get("reasoning_effort"),
        "response_format_type": spec.get("response_format_type"),
        "thinking_mode": spec.get("thinking_mode"),
        "cost_per_call_usd": (
            float(spec["cost_per_call_usd"]) if spec.get("cost_per_call_usd") is not None else None
        ),
        "ollama_url": spec.get("ollama_url"),
        "base_url": spec.get("base_url"),
        "api_key_env": spec.get("api_key_env"),
        "api_key_file": _resolve_optional_path(spec.get("api_key_file")),
        "manifest_path": manifest_path,
    }
    return model


def load_model_definitions(model_manifest=None, models_dir=None):
    manifests = _resolve_model_manifest_paths(
        model_manifest=model_manifest,
        models_dir=models_dir,
    )
    models = [load_model_definition(path) for path in manifests]
    return sorted(models, key=lambda m: (m["order"], m["name"]))


def load_model_definitions_from_args(args):
    return load_model_definitions(
        model_manifest=getattr(args, "model_manifest", None),
        models_dir=getattr(args, "models_dir", None),
    )
