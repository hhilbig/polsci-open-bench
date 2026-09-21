#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-11%1
#SBATCH --job-name=refresh-backfill-remainder
#SBATCH --output=/home/hhilbig/%x_%A_%a.log

set -euo pipefail
export PROJECT_ROOT=/home/hhilbig/polsci-refresh-20260914
cd "$PROJECT_ROOT"
export MODEL_KEY=$(python3 - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path('output/sidecar/refresh_20260910_release/backfills/manifest.json').read_text())
models = sorted(manifest['models'], key=lambda row: (row['model_key'] != 'gpt_oss_120b_mxfp4', row['model_key']))
assert len(models) == 12
entry = models[int(sys.argv[1])]
assert entry['stages']['remainder']['rows'] > 0
print(entry['model_key'])
PY
)
export CONFIG_PATH="$PROJECT_ROOT/output/sidecar/refresh_20260910_release/backfills/$MODEL_KEY/remainder.yaml"
export RUN_ROOT="$PROJECT_ROOT/output/sidecar/refresh_backfill_remainders"
export BAKEOFF_RUNTIME_LOCK=/nfs/hive/scratch/hhilbig/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt
export CUDA_TOOLKIT_META_VERSION=13.0.2
exec bash "$PROJECT_ROOT/experiments/hive_model_bakeoff_20260804.sbatch"
