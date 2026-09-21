#!/usr/bin/env bash
#SBATCH --job-name=refresh-qflash-pilot
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/hhilbig/refresh-qflash-pilot_%j.log
set -euo pipefail

# This uses the pinned official dedicated image and an isolated copy of the
# benchmark runner, so concurrent historical backfills keep their own code.
project_root=/home/hhilbig/polsci-refresh-20260914
audit_root=/nfs/hive/scratch/hhilbig/refresh-qwen-flash-runtime
hf_root=/nfs/hive/scratch/hhilbig/hf-qwen38-flash-next
image_path="$audit_root/qwen38-flash-next-0aea3024.sif"
job_root="/scratch/$USER/refresh-qflash-pilot-${SLURM_JOB_ID}"
config_path=/project/experiments/refresh_qwen_flash_next_pilot.yaml
output_path=/project/output/sidecar/refresh_hive_pilot/qwen3_8_flash_next_fp8
model_id=Qwen/Qwen3.8-Flash-Next-FP8
revision=236dfdf285828023ca3bcd3f37366c58a3469b13

if [[ ! -f "$image_path" || ! -f "$audit_root/metadata/runtime.freeze.txt" \
      || ! -d "$audit_root/overlay-pandas233-base-numpy226" ]]; then
  echo 'Pinned image, runtime lock, or compatible package overlay missing' >&2
  exit 3
fi
mkdir -p "$hf_root" "$job_root/tmp" "$job_root/cache" "$project_root/output/sidecar/refresh_hive_pilot"
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$job_root/tmp"

container_options=(
  --nv --cleanenv --no-home
  --bind "$project_root:/project"
  --bind "$audit_root:/audit"
  --bind "$hf_root:/hf"
  --bind "$job_root:/job"
  --pwd /project
  --env PYTHONPATH=/audit/overlay-pandas233-base-numpy226
  --env HF_HOME=/hf
  --env TMPDIR=/job/tmp
  --env XDG_CACHE_HOME=/job/cache
)

echo "host=$(hostname) job=$SLURM_JOB_ID gpu_count=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Only re-creatable official weights are kept in shared scratch. Lock snapshot
# mutation in case another job for this exact checkpoint starts later.
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
