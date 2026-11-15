#!/bin/bash -l
# Submit the AgoraSpeech speaker-shown vs speaker-hidden run (tasks_speaker_cue,
# 2 tasks, 10,558 items) for the six open-weight models of the title experiment.
# Run on Hive. One Blackwell per model on high (non-preemptible), as in
# experiments/task_ext_20261114/submit.sh. Usage: ..._submit.sh [extra sbatch args]
# As run on 2026-11-14: submitted to high, then moved to low with scontrol
# (partition=low qos=publicgrp-low-qos) because of the public group GPU cap on high.
# Prints "model_key<TAB>job_id" per model.
set -euo pipefail
S=/nfs/hive/scratch/$USER
P=$S/polsci-speakercue-20261114
RUN_ROOT=$P/output/sidecar/task_ext_20261114_speaker_cue
RESTORE_JOB=25735372   # weight restore of the 2026-11-14 open-model run
cd "$P"
mkdir -p logs
for MODEL_KEY in qwen3_6_27b_fp8 gemma4_31b_it_qat_w4a16 qwen3_8_27b_fp8 \
                 nemotron3_5_lightning_fp8 qwen3_6_35b_a3b_fp8 qwen3_30b_a3b_instruct_2507_fp8; do
  DEP=()
  case "$MODEL_KEY" in
    qwen3_6_35b_a3b_fp8|qwen3_30b_a3b_instruct_2507_fp8) DEP=(--dependency=afterok:$RESTORE_JOB) ;;
  esac
  JOB=$(sbatch --parsable --account=publicgrp --partition=high --nodes=1 --gpus=6000_blackwell:1 \
    --cpus-per-task=2 --mem=96G --time=02:00:00 "${DEP[@]}" \
    --job-name="spk-${MODEL_KEY:0:20}" --output="$P/logs/%x_%j.log" "$@" \
    --export=ALL,MODEL_KEY=$MODEL_KEY,PROJECT_ROOT=$P,RUN_ROOT=$RUN_ROOT,OFFLINE_INSTALL=0,CONFIG_PATH=$P/experiments/task_ext_20261114_speaker_cue.yaml,BAKEOFF_RUNTIME_LOCK=$S/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt,CUDA_TOOLKIT_META_VERSION=13.0.2,VLLM_CACHE_ROOT=$S/vllm-cache-nov,XDG_CACHE_HOME=$S/xdg-cache-nov,TRITON_CACHE_DIR=$S/triton-cache-nov \
    experiments/hive_model_bakeoff_20260804.sbatch)
  printf '%s\t%s\n' "$MODEL_KEY" "$JOB"
done
