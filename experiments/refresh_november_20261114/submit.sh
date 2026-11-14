#!/bin/bash -l
# Submit one stage (pilot or remainder) for one November small model on Hive.
# Usage (from the project root on Hive): submit.sh STAGE MODEL_KEY [extra sbatch args]
# Single-GPU models on the September refresh runtime (vLLM 0.26.0, CUDA 13.0.2 lock).
set -euo pipefail
STAGE=$1; MODEL_KEY=$2; shift 2
P=/nfs/hive/scratch/$USER/polsci-refresh-nov-small-20261114
S=/nfs/hive/scratch/$USER
case "$MODEL_KEY" in
  qwen3_5_9b_bf16|gemma3_12b_it_fp8_dynamic) ;;
  *) echo "unknown model $MODEL_KEY" >&2; exit 2 ;;
esac
cd "$P"
sbatch --parsable --account=publicgrp --partition=high --nodes=1 --gpus=6000_blackwell:1 \
  --cpus-per-task=2 --mem=64G --time=02:00:00 \
  --job-name="nov-$STAGE-${MODEL_KEY%%_*}" --output="$P/logs/%x_%j.log" "$@" \
  --export=ALL,MODEL_KEY=$MODEL_KEY,PROJECT_ROOT=$P,CONFIG_PATH=$P/experiments/refresh_november_20261114/$MODEL_KEY/$STAGE.yaml,RUN_ROOT=$P/output/sidecar/refresh_november_20261114/$STAGE,BAKEOFF_RUNTIME_LOCK=$S/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt,CUDA_TOOLKIT_META_VERSION=13.0.2,OFFLINE_INSTALL=0,VLLM_CACHE_ROOT=$S/vllm-cache-nov,XDG_CACHE_HOME=$S/xdg-cache-nov,TRITON_CACHE_DIR=$S/triton-cache-nov \
  experiments/hive_model_bakeoff_20260804.sbatch
