#!/usr/bin/env bash
#SBATCH --job-name=qwen3-embed-baseline
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/hive/scratch/hhilbig/polsci-qwen3-embedding-20260919/logs/%x_%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/hive/scratch/hhilbig/polsci-qwen3-embedding-20260919/repo}"
RUN_ROOT="${RUN_ROOT:-/nfs/hive/scratch/hhilbig/polsci-qwen3-embedding-20260919/output}"
MODEL_ID="Qwen/Qwen3-Embedding-8B"
MODEL_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
RUNTIME_LOCK="/nfs/hive/scratch/hhilbig/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt"
RUNTIME_LOCK_SHA256="4f46101fa493e38b312af034b8d740c612bd0001bab4cb0f624f4a9e1c49f5b3"
JOB_ROOT="/scratch/$USER/qwen3-embed-baseline-${SLURM_JOB_ID}"

mkdir -p "$JOB_ROOT/tmp" "$RUN_ROOT" "$(dirname "$RUN_ROOT")/logs"
export TMPDIR="$JOB_ROOT/tmp"
export HF_HOME="/nfs/hive/scratch/$USER/hf"
export UV_INSTALL_DIR="/nfs/hive/scratch/$USER/uvbin"
export UV_PYTHON_INSTALL_DIR="/nfs/hive/scratch/$USER/uvpython-3.12"
export UV_CACHE_DIR="/nfs/hive/scratch/$USER/uvcache-vllm-0.26.0"
export XDG_CACHE_HOME="/nfs/hive/scratch/$USER/cache"
mkdir -p "$HF_HOME" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

echo "=== host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) ==="
df -h "$HOME" /nfs/hive/scratch
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
test -f "$PROJECT_ROOT/code/build_supervised_baseline.py"
test -f "$RUNTIME_LOCK"
test "$(sha256sum "$RUNTIME_LOCK" | awk '{print $1}')" = "$RUNTIME_LOCK_SHA256"

if [[ ! -x "$UV_INSTALL_DIR/uv" ]]; then
    curl -LsSf --retry 5 --retry-all-errors https://astral.sh/uv/install.sh \
        | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

uv venv "$JOB_ROOT/venv" --python 3.12
source "$JOB_ROOT/venv/bin/activate"
if ! uv pip install -q --offline --torch-backend=auto -r "$RUNTIME_LOCK"; then
    uv pip install -q --torch-backend=auto -r "$RUNTIME_LOCK"
fi
uv pip install -q \
    "pandas==2.3.3" "PyYAML==6.0.3" "scikit-learn==1.8.0"
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
export QWEN3_EMBEDDING_BATCH_SIZE="${QWEN3_EMBEDDING_BATCH_SIZE:-16}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

python "$PROJECT_ROOT/code/build_supervised_baseline.py" \
    --method qwen3 \
    --output "$RUN_ROOT/supervised_baseline_qwen3_8b.csv"

python - "$PROJECT_ROOT" "$RUN_ROOT" "$MODEL_SNAPSHOT" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

project, output, snapshot = map(Path, sys.argv[1:])
result = output / "supervised_baseline_qwen3_8b.csv"
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
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "packages": {name: metadata.version(name) for name in ["torch", "transformers", "numpy", "pandas", "scikit-learn"]},
}
(output / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

echo "=== completed end=$(date -Is) ==="
