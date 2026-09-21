#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=polsci-model-bakeoff
#SBATCH --output=/home/hhilbig/%x_%j.log

set -euo pipefail

if [[ -z "${MODEL_KEY:-}" ]]; then
    echo "MODEL_KEY is required" >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/hhilbig/polsci-open-bench-hive-20260804}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/experiments/hive_model_bakeoff_20260804.yaml}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/output/sidecar/hive_model_bakeoff_20260804}"
SCRATCH_ROOT="/scratch/$USER/polsci_bakeoff_${SLURM_JOB_ID}"

mkdir -p "$SCRATCH_ROOT" "$RUN_ROOT/$MODEL_KEY" "$PROJECT_ROOT/hive_logs"
export HF_HOME="/nfs/hive/scratch/$USER/hf"
export PIP_CACHE_DIR="/nfs/hive/scratch/$USER/pipcache-vllm-0.26.0"
export TMPDIR="$SCRATCH_ROOT/tmp"
export UV_INSTALL_DIR="/nfs/hive/scratch/$USER/uvbin"
export UV_PYTHON_INSTALL_DIR="/nfs/hive/scratch/$USER/uvpython-3.12"
export UV_CACHE_DIR="/nfs/hive/scratch/$USER/uvcache-vllm-0.26.0"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/nfs/hive/scratch/$USER/tiktoken-encodings}"
DEFAULT_BAKEOFF_RUNTIME_LOCK="/nfs/hive/scratch/$USER/polsci-bakeoff-vllm-0.26.0-py312.freeze.txt"
export BAKEOFF_RUNTIME_LOCK="${BAKEOFF_RUNTIME_LOCK:-$DEFAULT_BAKEOFF_RUNTIME_LOCK}"
BAKEOFF_RUNTIME_BASE_LOCK="${BAKEOFF_RUNTIME_BASE_LOCK:-$DEFAULT_BAKEOFF_RUNTIME_LOCK}"
CUDA_TOOLKIT_META_VERSION="${CUDA_TOOLKIT_META_VERSION:-}"
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

if [[ "$MODEL_KEY" == gpt_oss_* ]]; then
    HARMONY_VOCAB="$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"
    HARMONY_VOCAB_SHA256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    if [[ ! -f "$HARMONY_VOCAB" ]]; then
        echo "GPT-OSS requires the pinned Harmony vocabulary at $HARMONY_VOCAB" >&2
        exit 3
    fi
    if [[ "$(sha256sum "$HARMONY_VOCAB" | awk '{print $1}')" != "$HARMONY_VOCAB_SHA256" ]]; then
        echo "GPT-OSS Harmony vocabulary hash mismatch" >&2
        exit 3
    fi
fi

echo "=== host=$(hostname) job=$SLURM_JOB_ID model=$MODEL_KEY start=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

if [[ ! -x "$UV_INSTALL_DIR/uv" ]]; then
    installed=0
    for attempt in 1 2 3 4 5; do
        if curl -LsSf --retry 5 --retry-all-errors https://astral.sh/uv/install.sh \
            | env INSTALLER_NO_MODIFY_PATH=1 sh; then
            installed=1
            break
        fi
        sleep 15
    done
    if [[ "$installed" -ne 1 ]]; then
        echo "failed to install uv after five attempts" >&2
        exit 3
    fi
fi
export PATH="$UV_INSTALL_DIR:$PATH"

uv venv "$SCRATCH_ROOT/venv" --python 3.12
source "$SCRATCH_ROOT/venv/bin/activate"
UV_PIP_FLAGS=(-q)
if [[ "${OFFLINE_INSTALL:-0}" == "1" ]]; then
    UV_PIP_FLAGS+=(--offline)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
CREATE_RUNTIME_LOCK=0
if [[ -f "$BAKEOFF_RUNTIME_LOCK" ]]; then
    uv pip install "${UV_PIP_FLAGS[@]}" --torch-backend=auto -r "$BAKEOFF_RUNTIME_LOCK"
elif [[ -f "$BAKEOFF_RUNTIME_BASE_LOCK" ]]; then
    uv pip install "${UV_PIP_FLAGS[@]}" --torch-backend=auto -r "$BAKEOFF_RUNTIME_BASE_LOCK"
    CREATE_RUNTIME_LOCK=1
else
    uv pip install "${UV_PIP_FLAGS[@]}" "vllm==0.26.0" --torch-backend=auto
    uv pip install "${UV_PIP_FLAGS[@]}" \
        "pandas==2.3.3" \
        "PyYAML==6.0.3" \
        "httpx==0.28.1" \
        "openai==2.26.0"
    CREATE_RUNTIME_LOCK=1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_NAME_LOWER=$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]')
if [[ "$GPU_NAME_LOWER" == *blackwell* || "$GPU_NAME_LOWER" == *"rtx pro 6000"* ]]; then
    if [[ -n "$CUDA_TOOLKIT_META_VERSION" ]]; then
        if [[ "$CUDA_TOOLKIT_META_VERSION" != "13.0.2" ]]; then
            echo "no audited CUDA component pins for toolkit $CUDA_TOOLKIT_META_VERSION" >&2
            exit 4
        fi
        uv pip install "${UV_PIP_FLAGS[@]}" \
            "cuda-toolkit[cccl,crt,cudart,nvcc,nvrtc,nvvm]==13.0.2" \
            "nvidia-cuda-cccl==13.0.85" \
            "nvidia-cuda-crt==13.0.88" \
            "nvidia-cuda-nvcc==13.0.88" \
            "nvidia-cuda-nvrtc==13.0.88" \
            "nvidia-cuda-runtime==13.0.96" \
            "nvidia-nvvm==13.0.88"
    else
        uv pip install "${UV_PIP_FLAGS[@]}" "nvidia-cuda-nvcc>=13,<14"
    fi
    NVCC_PATH=$(find "$SCRATCH_ROOT/venv" -name nvcc -type f | head -1)
    if [[ -z "$NVCC_PATH" ]]; then
        echo "Blackwell job could not find nvcc after installation" >&2
        exit 4
    fi
    export PATH="$(dirname "$NVCC_PATH"):$PATH"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
    CUDA_RUNTIME_LIB="$CUDA_HOME/lib"
    CUDA_RUNTIME_SONAME="$CUDA_RUNTIME_LIB/libcudart.so.13"
    CUDA_NVRTC_SONAME="$CUDA_RUNTIME_LIB/libnvrtc.so.13"
    CUDA_CUBLAS_SONAME="$CUDA_RUNTIME_LIB/libcublas.so.13"
    CUDA_CUBLASLT_SONAME="$CUDA_RUNTIME_LIB/libcublasLt.so.13"
    if [[ ! -f "$CUDA_RUNTIME_SONAME" ]]; then
        echo "Blackwell pip toolkit could not find $CUDA_RUNTIME_SONAME" >&2
        exit 4
    fi
    if [[ ! -f "$CUDA_NVRTC_SONAME" ]]; then
        echo "Blackwell pip toolkit could not find $CUDA_NVRTC_SONAME" >&2
        exit 4
    fi
    if [[ ! -f "$CUDA_CUBLAS_SONAME" || ! -f "$CUDA_CUBLASLT_SONAME" ]]; then
        echo "Blackwell pip runtime could not find cuBLAS shared libraries" >&2
        exit 4
    fi
    if [[ ! -e "$CUDA_HOME/lib64" ]]; then
        ln -s lib "$CUDA_HOME/lib64"
    fi
    if [[ ! -e "$CUDA_HOME/lib64/libcudart.so" ]]; then
        ln -s "$CUDA_RUNTIME_SONAME" "$CUDA_HOME/lib64/libcudart.so"
    fi
    if [[ ! -e "$CUDA_HOME/lib64/libnvrtc.so" ]]; then
        ln -s "$CUDA_NVRTC_SONAME" "$CUDA_HOME/lib64/libnvrtc.so"
    fi
    if [[ ! -e "$CUDA_HOME/lib64/libcublas.so" ]]; then
        ln -s "$CUDA_CUBLAS_SONAME" "$CUDA_HOME/lib64/libcublas.so"
    fi
    if [[ ! -e "$CUDA_HOME/lib64/libcublasLt.so" ]]; then
        ln -s "$CUDA_CUBLASLT_SONAME" "$CUDA_HOME/lib64/libcublasLt.so"
    fi
    export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    CUDA_LINK_TEST="$SCRATCH_ROOT/libcuda_link_preflight.so"
    c++ -shared -x c++ /dev/null -Wl,--no-as-needed \
        -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" \
        -lcudart -lcuda -lnvrtc -lcublas -lcublasLt -o "$CUDA_LINK_TEST"
    readelf -d "$CUDA_LINK_TEST" | grep -q 'libcudart.so.13'
    readelf -d "$CUDA_LINK_TEST" | grep -q 'libcuda.so.1'
    readelf -d "$CUDA_LINK_TEST" | grep -q 'libnvrtc.so.13'
    readelf -d "$CUDA_LINK_TEST" | grep -q 'libcublas.so'
    readelf -d "$CUDA_LINK_TEST" | grep -q 'libcublasLt.so'
else
    module load cuda/12.6.2
fi

uv pip check
nvcc --version

EFFECTIVE_RUNTIME_LOCK="$SCRATCH_ROOT/effective-runtime.freeze.txt"
uv pip freeze | sort > "$EFFECTIVE_RUNTIME_LOCK"
if [[ "$CREATE_RUNTIME_LOCK" -eq 1 ]]; then
    mkdir -p "$(dirname "$BAKEOFF_RUNTIME_LOCK")"
    LOCK_TMP="${BAKEOFF_RUNTIME_LOCK}.tmp-${SLURM_JOB_ID}"
    cp "$EFFECTIVE_RUNTIME_LOCK" "$LOCK_TMP"
    mv "$LOCK_TMP" "$BAKEOFF_RUNTIME_LOCK"
elif ! cmp -s "$EFFECTIVE_RUNTIME_LOCK" "$BAKEOFF_RUNTIME_LOCK"; then
    echo "installed runtime differs from frozen target lock" >&2
    echo "effective_sha256=$(shasum -a 256 "$EFFECTIVE_RUNTIME_LOCK" | awk '{print $1}')" >&2
    echo "target_sha256=$(shasum -a 256 "$BAKEOFF_RUNTIME_LOCK" | awk '{print $1}')" >&2
    exit 5
fi

# Retain the verified Blackwell-safe settings while vLLM 0.26.0 is audited.
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
echo "FlashInfer build parallelism: MAX_JOBS=$MAX_JOBS FLASHINFER_NVCC_THREADS=$FLASHINFER_NVCC_THREADS"

python - <<'PY'
import importlib.metadata
for package in [
    "vllm",
    "torch",
    "transformers",
    "pandas",
    "PyYAML",
    "cuda-toolkit",
    "nvidia-cuda-cccl",
    "nvidia-cuda-crt",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-nvvm",
]:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "MISSING"
    print(f"{package}={version}")
PY

BAKEOFF_MODEL_SNAPSHOT_PATH=$(python - "$CONFIG_PATH" "$MODEL_KEY" "${OFFLINE_INSTALL:-0}" <<'PY'
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

config_path = Path(sys.argv[1])
model_key = sys.argv[2]
local_files_only = sys.argv[3] == "1"
config = yaml.safe_load(config_path.read_text())
model = config["models"][model_key]
snapshot_path = snapshot_download(
    repo_id=model["model_id"],
    revision=model["revision"],
    local_files_only=local_files_only,
)
snapshot_path = Path(snapshot_path).resolve()
if snapshot_path.name.lower() != str(model["revision"]).lower():
    raise SystemExit(
        "immutable snapshot mismatch: "
        f"expected {model['revision']}, observed {snapshot_path}"
    )
print(snapshot_path, flush=True)
PY
)
export BAKEOFF_MODEL_SNAPSHOT_PATH
echo "resolved immutable snapshot model=$MODEL_KEY path=$BAKEOFF_MODEL_SNAPSHOT_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
    "${BAKEOFF_RUNNER_PATH:-$PROJECT_ROOT/code/hive_vllm_benchmark.py}"
    --config "$CONFIG_PATH"
    --model-key "$MODEL_KEY"
    --output-dir "$RUN_ROOT/$MODEL_KEY"
)
if [[ -n "${ONLY_TASK:-}" ]]; then
    ARGS+=(--only-task "$ONLY_TASK")
fi
if [[ -n "${LIMIT_ITEMS:-}" ]]; then
    ARGS+=(--limit-items "$LIMIT_ITEMS")
fi
if [[ -n "${PANEL_MANIFEST:-}" ]]; then
    ARGS+=(--panel-manifest "$PANEL_MANIFEST")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
    ARGS+=(--force)
fi

python "${ARGS[@]}"
echo "=== done model=$MODEL_KEY end=$(date -Is) ==="
