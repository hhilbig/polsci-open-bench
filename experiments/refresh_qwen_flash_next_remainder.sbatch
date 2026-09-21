#!/usr/bin/env bash
#SBATCH --job-name=refresh-qflash-remainder
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/hhilbig/refresh-qflash-remainder_%j.log
set -euo pipefail

# Submit only after the structural pilot validates exact output and runtime fit.
# Keep its immutable runner, image, overlay, model revision and decoding settings.
project_root=/home/hhilbig/polsci-refresh-20260914
audit_root=/nfs/hive/scratch/hhilbig/refresh-qwen-flash-runtime
hf_root=/nfs/hive/scratch/hhilbig/hf-qwen38-flash-next
image_path="$audit_root/qwen38-flash-next-0aea3024.sif"
job_root="/scratch/$USER/refresh-qflash-remainder-${SLURM_JOB_ID}"
config_path=/project/experiments/refresh_qwen_flash_next_remainder.yaml
output_path=/project/output/sidecar/refresh_hive_remainder/qwen3_8_flash_next_fp8
model_id=Qwen/Qwen3.8-Flash-Next-FP8
revision=236dfdf285828023ca3bcd3f37366c58a3469b13

if [[ ! -f "$image_path" || ! -f "$audit_root/metadata/runtime.freeze.txt" \
      || ! -d "$audit_root/overlay-pandas233-base-numpy226" ]]; then
  echo 'Pinned image, runtime lock, or compatible package overlay missing' >&2
  exit 3
fi
mkdir -p "$hf_root" "$job_root/tmp" "$job_root/cache" "$job_root/home" \
  "$project_root/output/sidecar/refresh_hive_remainder"
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$job_root/tmp"

container_options=(
  --nv --cleanenv
  # Bind an empty job-local home. Apptainer resets HOME after processing
  # --env, so HOME=/job alone does not make implicit ~/.triton writes safe.
  --home "$job_root/home:/home/hhilbig"
  --bind "$project_root:/project"
  --bind "$audit_root:/audit"
  --bind "$hf_root:/hf"
  --bind "$job_root:/job"
  --pwd /project
  --env PYTHONPATH=/audit/overlay-pandas233-base-numpy226
  --env HF_HOME=/hf
  --env TMPDIR=/job/tmp
  --env XDG_CACHE_HOME=/job/cache
  # FlashInfer ignores XDG_CACHE_HOME and otherwise writes under the read-only
  # container home during vLLM architecture inspection.
  --env FLASHINFER_WORKSPACE_BASE=/job/cache
  --env TRITON_CACHE_DIR=/job/cache/triton
  --env TORCHINDUCTOR_CACHE_DIR=/job/cache/torchinductor
  --env TORCH_EXTENSIONS_DIR=/job/cache/torch_extensions
  --env CUDA_CACHE_PATH=/job/cache/cuda
)

echo "host=$(hostname) job=$SLURM_JOB_ID gpu_count=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# A preemption or retry reuses the immutable snapshot and verified task chunks.
exec 9>"$audit_root/qwen-flash-snapshot.lock"
flock 9
snapshot_path=$(apptainer exec "${container_options[@]}" "$image_path" \
  python3 - "$model_id" "$revision" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot = Path(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])).resolve()
if snapshot.name != sys.argv[2]:
    raise SystemExit('Immutable Qwen Flash snapshot revision mismatch')
print(snapshot)
PY
)
flock -u 9

apptainer exec "${container_options[@]}" \
  --env BAKEOFF_RUNTIME_LOCK=/audit/metadata/runtime.freeze.txt \
  --env BAKEOFF_CONTAINER_IMAGE_PATH=/audit/qwen38-flash-next-0aea3024.sif \
  --env BAKEOFF_MODEL_SNAPSHOT_PATH="$snapshot_path" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  "$image_path" python3 /project/code/hive_vllm_benchmark_qflash_gitfallback.py \
    --config "$config_path" --model-key qwen3_8_flash_next_fp8 \
    --output-dir "$output_path"
