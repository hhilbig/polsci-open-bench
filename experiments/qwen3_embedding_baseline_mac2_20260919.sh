#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/Users/hannohilbig/sidecar-runs/polsci-qwen3-embedding-20260919}"
PROJECT_ROOT="${PROJECT_ROOT:-$RUN_ROOT/repo}"
MODEL_ID="Qwen/Qwen3-Embedding-8B"
MODEL_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs" "$RUN_ROOT/cache" "$RUN_ROOT/uvbin"
export HF_HOME="$RUN_ROOT/hf"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export UV_INSTALL_DIR="$RUN_ROOT/uvbin"
export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/uvpython"
export UV_CACHE_DIR="$RUN_ROOT/uvcache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

if [[ ! -x "$UV_INSTALL_DIR/uv" ]]; then
    curl -LsSf --retry 5 --retry-all-errors https://astral.sh/uv/install.sh \
        | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

if [[ ! -x "$RUN_ROOT/venv/bin/python" ]]; then
    uv venv "$RUN_ROOT/venv" --python 3.12
fi
source "$RUN_ROOT/venv/bin/activate"
uv pip install -q \
    "torch==2.8.0" \
    "transformers==4.57.1" \
    "huggingface-hub==0.36.0" \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "PyYAML==6.0.3" \
    "scipy==1.16.3" \
    "scikit-learn==1.8.0"
uv pip check

MODEL_SNAPSHOT=$(python - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, revision = sys.argv[1:]
path = Path(snapshot_download(repo_id=model_id, revision=revision)).resolve()
if path.name.lower() != revision.lower():
    raise SystemExit(f"snapshot mismatch: expected {revision}, observed {path}")
print(path)
PY
)
export QWEN3_EMBEDDING_SNAPSHOT_PATH="$MODEL_SNAPSHOT"
export QWEN3_EMBEDDING_BATCH_SIZE="${QWEN3_EMBEDDING_BATCH_SIZE:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

python "$PROJECT_ROOT/code/build_supervised_baseline.py" \
    --method qwen3 \
    --output "$RUN_ROOT/output/supervised_baseline_qwen3_8b.csv"

python - "$PROJECT_ROOT" "$RUN_ROOT" "$MODEL_SNAPSHOT" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

project, root, snapshot = map(Path, sys.argv[1:])
result = root / "output/supervised_baseline_qwen3_8b.csv"
record = {
    "model": "Qwen/Qwen3-Embedding-8B",
    "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "snapshot_path": str(snapshot),
    "method": "frozen embeddings plus the existing class-weighted logistic regression",
    "pooling": "last non-padding token",
    "normalized": True,
    "max_tokens": 8192,
    "embedding_dimension": 4096,
    "training_sizes": [50, 100, 250, 500, 1000, 2000],
    "draws_per_size": 5,
    "result_sha256": hashlib.file_digest(result.open("rb"), "sha256").hexdigest(),
    "result_rows": sum(1 for _ in result.open()) - 1,
    "source_git_head": "acf8c8ff585421a18d6e94af66a80b3a341580c8",
    "script_sha256": hashlib.file_digest((project / "code/build_supervised_baseline.py").open("rb"), "sha256").hexdigest(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "packages": {name: metadata.version(name) for name in ["torch", "transformers", "numpy", "pandas", "scikit-learn"]},
}
(root / "output/run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
