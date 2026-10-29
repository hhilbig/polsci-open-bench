#!/bin/bash -l
# Submit one stage (pilot or remainder) for one October model on Hive.
# Usage (from the project root on Hive): submit.sh STAGE MODEL_KEY [extra sbatch args]
# Multi-GPU jobs go to low: the free high tier allows one GPU per job.
set -euo pipefail
STAGE=$1; MODEL_KEY=$2; shift 2
P=/nfs/hive/scratch/$USER/polsci-refresh-oct-20261026
S=/nfs/hive/scratch/$USER
case "$MODEL_KEY" in
  mimo_v2_6_flash_mopd_fp8|nemotron3_5_super_vl_bf16)
    RES=(--gpus=6000_blackwell:4 --mem=320G)
    LOCK=$S/polsci-bakeoff-vllm-0.32.0-py312-cuda1303-blackwell.freeze.txt
    EXTRA="" ;;
  intern_s2_397b_fp8)
    RES=(--gpus=a100:8 --mem=480G)
    LOCK=$S/polsci-bakeoff-vllm-0.32.0-py312-cuda1303-a100.freeze.txt
    EXTRA=",PIP_CUDA_ANY_GPU=1" ;;
  *) echo "unknown model $MODEL_KEY" >&2; exit 2 ;;
esac
cd "$P"
sbatch --parsable --partition=low --nodes=1 --cpus-per-task=8 --time=06:00:00 "${RES[@]}" \
  --job-name="oct-$STAGE-${MODEL_KEY%%_*}" --output="$P/logs/%x_%j.log" "$@" \
  --export=ALL,MODEL_KEY=$MODEL_KEY,PROJECT_ROOT=$P,CONFIG_PATH=$P/experiments/refresh_october_20261028/$MODEL_KEY/$STAGE.yaml,RUN_ROOT=$P/output/sidecar/refresh_october_20261028/$STAGE,BAKEOFF_RUNTIME_LOCK=$LOCK,BAKEOFF_RUNTIME_BASE_LOCK=$S/none-0.32-base.freeze.txt,BAKEOFF_VLLM_VERSION=0.32.0,CUDA_TOOLKIT_META_VERSION=13.0.3,TORCH_BACKEND=cu130,OFFLINE_INSTALL=0,VLLM_CACHE_ROOT=$S/vllm-cache-oct,XDG_CACHE_HOME=$S/xdg-cache-oct,TRITON_CACHE_DIR=$S/triton-cache-oct$EXTRA \
  experiments/hive_model_bakeoff_20260804.sbatch
