#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --qos=publicgrp-low-qos
#SBATCH --nodes=1
#SBATCH --gpus=6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --job-name=refresh-gptoss-retry
#SBATCH --output=/home/hhilbig/%x_%j.log

set -euo pipefail
export PROJECT_ROOT=/home/hhilbig/polsci-refresh-20260914
cd "$PROJECT_ROOT"
export MODEL_KEY=gpt_oss_120b_mxfp4
export CONFIG_PATH="$PROJECT_ROOT/output/sidecar/refresh_20260910_release/backfills/$MODEL_KEY/remainder.yaml"
export RUN_ROOT="$PROJECT_ROOT/output/sidecar/refresh_backfill_remainders"
export BAKEOFF_RUNTIME_LOCK=/nfs/hive/scratch/hhilbig/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt
export CUDA_TOOLKIT_META_VERSION=13.0.2
export BAKEOFF_RUNNER_PATH="$PROJECT_ROOT/code/hive_vllm_benchmark_gptoss_retry_20260915.py"
exec bash "$PROJECT_ROOT/experiments/hive_model_bakeoff_gptoss_retry_20260915.sbatch"
