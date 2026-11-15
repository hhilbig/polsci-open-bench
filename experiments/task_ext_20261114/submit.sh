#!/bin/bash -l
# Submit one model of the 2026-11-14 open-model run (47 tasks_ext + tasks_v2
# tasks, full pools). Run from anywhere on Hive: submit.sh MODEL_KEY [extra sbatch args]
# Single-GPU models: vllm026.yaml, one Blackwell on high (non-preemptible).
# MiMo: vllm032.yaml, four Blackwells on low, as in its panel run.
set -euo pipefail
MODEL_KEY=$1; shift
P=/nfs/hive/scratch/$USER/polsci-ext-20261114
S=/nfs/hive/scratch/$USER
cd "$P"
RUN_ROOT=${RUN_ROOT:-$P/output/sidecar/task_ext_20261114}
COMMON="ALL,MODEL_KEY=$MODEL_KEY,PROJECT_ROOT=$P,RUN_ROOT=$RUN_ROOT,OFFLINE_INSTALL=0"
if [[ "$MODEL_KEY" == mimo_v2_6_flash_mopd_fp8 ]]; then
  sbatch --parsable --account=publicgrp --partition=low --nodes=1 --gpus=6000_blackwell:4 \
    --cpus-per-task=8 --mem=320G --time=12:00:00 --requeue \
    --job-name="ext-${MODEL_KEY:0:20}" --output="$P/logs/%x_%j.log" "$@" \
    --export=$COMMON,CONFIG_PATH=$P/experiments/task_ext_20261114/vllm032.yaml,BAKEOFF_RUNTIME_LOCK=$S/polsci-bakeoff-vllm-0.32.0-py312-cuda1303-blackwell.freeze.txt,BAKEOFF_RUNTIME_BASE_LOCK=$S/none-0.32-base.freeze.txt,BAKEOFF_VLLM_VERSION=0.32.0,CUDA_TOOLKIT_META_VERSION=13.0.3,TORCH_BACKEND=cu130,VLLM_CACHE_ROOT=$S/vllm-cache-oct,XDG_CACHE_HOME=$S/xdg-cache-oct,TRITON_CACHE_DIR=$S/triton-cache-oct \
    experiments/hive_model_bakeoff_20260804.sbatch
else
  TIME=1-00:00:00
  [[ "$MODEL_KEY" == qwen3_30b_a3b_bf16 ]] && TIME=3-00:00:00  # 2.4 GPU-h for the 3,300-item panel
  sbatch --parsable --account=publicgrp --partition=high --nodes=1 --gpus=6000_blackwell:1 \
    --cpus-per-task=2 --mem=96G --time=$TIME \
    --job-name="ext-${MODEL_KEY:0:20}" --output="$P/logs/%x_%j.log" "$@" \
    --export=$COMMON,CONFIG_PATH=$P/experiments/task_ext_20261114/vllm026.yaml,BAKEOFF_RUNTIME_LOCK=$S/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt,CUDA_TOOLKIT_META_VERSION=13.0.2,VLLM_CACHE_ROOT=$S/vllm-cache-nov,XDG_CACHE_HOME=$S/xdg-cache-nov,TRITON_CACHE_DIR=$S/triton-cache-nov \
    experiments/hive_model_bakeoff_20260804.sbatch
fi
