#!/bin/bash -l
#SBATCH --job-name=oct-weights
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hive/scratch/hhilbig/polsci-refresh-oct-20261026/logs/%x_%A_%a.log

# CPU-only: cache the pinned snapshots on shared scratch so the GPU jobs only
# verify them. Re-creatable data only; no inference.
set -euo pipefail
SPECS=(
  "XiaomiMiMo/MiMo-V2.6-Flash-MOPD b12f5573350bb1704513fcf119fca48e498b0423"
  "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
  "internlm/Intern-S2-397B-FP8 31b460c8ebb8bdda6f1d77651cfad7ab811910e3"
)
read -r REPO_ID REVISION <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
export HF_HOME=/nfs/hive/scratch/$USER/hf
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} repo=$REPO_ID rev=$REVISION start=$(date -Is)"
for attempt in 1 2 3 4 5; do
  if /nfs/hive/scratch/$USER/hfvenv/bin/python - "$REPO_ID" "$REVISION" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], max_workers=8)).resolve()
if path.name != sys.argv[2]:
    raise SystemExit(f'snapshot revision mismatch: {path}')
size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
print(f'snapshot={path} bytes={size}')
PY
  then echo "done=$(date -Is)"; exit 0; fi
  echo "attempt $attempt failed; retrying" >&2; sleep 60
done
exit 1
