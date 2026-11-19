#!/usr/bin/env bash
#SBATCH --job-name=ext-qflash
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --exclude=hive-dc-7-5-58
#SBATCH --output=/nfs/hive/scratch/hhilbig/polsci-ext-20261114/logs/ext-qflash_%j.log
# Qwen3.8 Flash-Next on the 2026-11-14 task pool, in November's pinned container
# (experiments/refresh_qwen_flash_next_remainder.sbatch), with the task-pool runner
# of the other open models. Checkpoints per task, so a requeue resumes.
set -euo pipefail
# Second checkout at 03058c2 (tensor-parallel opt-in); the shared checkout
# polsci-ext-20261114 stays at 2cae5cb. Runtime and logs stay in the shared one.
shared_root=/nfs/hive/scratch/hhilbig/polsci-ext-20261114
project_root=/nfs/hive/scratch/hhilbig/polsci-ext-20261114-qflash
audit_root=$shared_root/qflash-runtime
hf_root=/nfs/hive/scratch/hhilbig/hf-qwen38-flash-next
image_path="$audit_root/qwen38-flash-next-0aea3024.sif"
job_root="/scratch/$USER/ext-qflash-${SLURM_JOB_ID}"
config_path=/project/experiments/task_ext_20261114/qflash.yaml
output_path=/project/output/sidecar/task_ext_20261114/qwen3_8_flash_next_fp8
model_id=Qwen/Qwen3.8-Flash-Next-FP8
revision=236dfdf285828023ca3bcd3f37366c58a3469b13

if [[ ! -f "$image_path" || ! -f "$audit_root/metadata/runtime.freeze.txt" \
      || ! -d "$audit_root/overlay-pandas233-base-numpy226" ]]; then
  echo 'Pinned image, runtime lock, or package overlay missing' >&2; exit 3
fi
mkdir -p "$job_root/tmp" "$job_root/cache" "$job_root/home" "$project_root/output/sidecar/task_ext_20261114"
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$job_root/tmp"
jobstatus set polsci-ext-open-qflash --project polsci-open-bench --task "Run Qwen3.8 Flash-Next on the new benchmark tasks (Hive)" \
  --unit items --total 23413 --host hive --log "$shared_root/logs/ext-qflash_${SLURM_JOB_ID}.log" \
  --state running --note "Slurm $SLURM_JOB_ID on $(hostname)" >/dev/null 2>&1 || true
trap 'rc=$?; [ $rc -eq 0 ] && st=finished || st=failed; jobstatus set polsci-ext-open-qflash --project polsci-open-bench --task "Run Qwen3.8 Flash-Next on the new benchmark tasks (Hive)" --unit items --total 23413 --host hive --state $st --note "Slurm $SLURM_JOB_ID exit $rc" >/dev/null 2>&1 || true' EXIT

container_options=(
  --nv --cleanenv
  --home "$job_root/home:/home/hhilbig"
  --bind "$project_root:/project"
  --bind "$shared_root/.git:$shared_root/.git:ro"
  --bind "$audit_root:/audit"
  --bind "$hf_root:/hf"
  --bind "$job_root:/job"
  --pwd /project
  --env PYTHONPATH=/audit/overlay-pandas233-base-numpy226
  --env HF_HOME=/hf
  --env TMPDIR=/job/tmp
  --env XDG_CACHE_HOME=/job/cache
  --env FLASHINFER_WORKSPACE_BASE=/job/cache
  --env TRITON_CACHE_DIR=/job/cache/triton
  --env TORCHINDUCTOR_CACHE_DIR=/job/cache/torchinductor
  --env TORCH_EXTENSIONS_DIR=/job/cache/torch_extensions
  --env CUDA_CACHE_PATH=/job/cache/cuda
)
echo "host=$(hostname) job=$SLURM_JOB_ID gpu_count=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
# Job 25780028 got two GPUs with only ~30 GB free each on hive-dc-7-5-58
# (excluded above); vLLM needs 0.85 x 98 GB per GPU.
if nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 < 85000 {bad=1} END {exit !bad}'; then
  echo 'An allocated GPU has less than 85 GB free; stopping before model load' >&2; exit 4
fi

snapshot_path=$(apptainer exec "${container_options[@]}" --env HF_HUB_OFFLINE=1 "$image_path" \
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

apptainer exec "${container_options[@]}" \
  --env BAKEOFF_RUNTIME_LOCK=/audit/metadata/runtime.freeze.txt \
  --env BAKEOFF_CONTAINER_IMAGE_PATH=/audit/qwen38-flash-next-0aea3024.sif \
  --env BAKEOFF_MODEL_SNAPSHOT_PATH="$snapshot_path" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  "$image_path" python3 /project/code/hive_vllm_benchmark.py \
    --config "$config_path" --model-key qwen3_8_flash_next_fp8 \
    --output-dir "$output_path"
