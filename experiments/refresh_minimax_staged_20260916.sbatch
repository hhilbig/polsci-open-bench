#!/usr/bin/env bash
#SBATCH --job-name=refresh-minimax-staged
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/hhilbig/refresh-minimax-staged_%j.log
set -euo pipefail
project_root=/home/hhilbig/polsci-expansion-20260916
audit_root=/nfs/hive/scratch/hhilbig/refresh-expansion-20260916
hf_root=/nfs/hive/scratch/hhilbig/hf-minimax-m3-nvfp4
image_path="$audit_root/minimax-m3-a8c89303.sif"
job_root="/scratch/$USER/refresh-minimax-${SLURM_JOB_ID}"
config_root=/project/output/sidecar/refresh_expansion_20260916/config
output_root=/project/output/sidecar/refresh_expansion_20260916/minimax_m3_nvfp4

# Preparation must finish and pin both stages before this job is submitted.
test -f "$audit_root/metadata/runtime.freeze.txt"
test -f "$project_root/output/sidecar/refresh_expansion_20260916/config/pilot.yaml"
test -f "$project_root/output/sidecar/refresh_expansion_20260916/config/remainder.yaml"
test -d "$audit_root/overlay"
mkdir -p "$hf_root" "$job_root/tmp" "$job_root/cache" "$job_root/home"
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$job_root/tmp"
container_options=(
  --nv --cleanenv --home "$job_root/home:/home/hhilbig"
  --bind "$project_root:/project" --bind "$audit_root:/audit"
  --bind "$hf_root:/hf" --bind "$job_root:/job" --pwd /project
  --env PYTHONPATH=/audit/overlay
  --env HF_HOME=/hf --env TMPDIR=/job/tmp --env XDG_CACHE_HOME=/job/cache
  --env FLASHINFER_WORKSPACE_BASE=/job/cache
  --env TRITON_CACHE_DIR=/job/cache/triton
  --env TORCHINDUCTOR_CACHE_DIR=/job/cache/torchinductor
  --env TORCH_EXTENSIONS_DIR=/job/cache/torch_extensions
  --env CUDA_CACHE_PATH=/job/cache/cuda
  --env SLURM_JOB_ID="$SLURM_JOB_ID" --env SLURM_JOB_NUM_NODES=1
)
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
# Prevent a duplicate submission from entering this model's output directory.
exec 8>"$audit_root/minimax-run.lock"
flock -n 8 || { echo 'Another MiniMax expansion job holds the run lock'; exit 3; }
exec 9>"$audit_root/minimax-snapshot.lock"
flock 9
snapshot_path=$(apptainer exec "${container_options[@]}" "$image_path" python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
revision = '901464083161bf8612a29ff7ad29914cd4ab4a85'
snapshot = Path(snapshot_download(repo_id='nvidia/MiniMax-M3-NVFP4', revision=revision)).resolve()
if snapshot.name != revision:
    raise SystemExit('Pinned snapshot revision mismatch')
print(snapshot)
PY
)
flock -u 9
apptainer exec "${container_options[@]}" \
  --env BAKEOFF_RUNTIME_LOCK=/audit/metadata/runtime.freeze.txt \
  --env BAKEOFF_CONTAINER_IMAGE_PATH=/audit/minimax-m3-a8c89303.sif \
  --env BAKEOFF_MODEL_SNAPSHOT_PATH="$snapshot_path" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_FLOAT32_MATMUL_PRECISION=high \
  "$image_path" python3 /project/code/run_refresh_expansion.py \
    --pilot-config "$config_root/pilot.yaml" \
    --remainder-config "$config_root/remainder.yaml" \
    --model-key minimax_m3_nvfp4 --output-root "$output_root"
