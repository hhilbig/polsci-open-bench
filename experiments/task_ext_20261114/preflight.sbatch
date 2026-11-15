#!/bin/bash -l
#SBATCH --job-name=ext-preflight
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
# CPU-only: counts every prompt of the 2026-11-14 run with each model's tokenizer.
set -euo pipefail
P=/nfs/hive/scratch/$USER/polsci-ext-20261114
W=/scratch/$USER/preflight_$SLURM_JOB_ID
mkdir -p "$W"
export UV_CACHE_DIR=$W/uvcache UV_PYTHON_INSTALL_DIR=/nfs/hive/scratch/$USER/uvpython-3.12
export PATH=/nfs/hive/scratch/$USER/uvbin:$PATH
export HF_HOME=/nfs/hive/scratch/$USER/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv venv --clear "$W/venv" --python 3.12
source "$W/venv/bin/activate"
for i in 1 2 3; do
  uv pip install -q pandas==2.3.3 PyYAML==6.0.3 packaging numpy httpx==0.28.1 openai==2.26.0 \
    "transformers==5.14.1" jinja2 tokenizers sentencepiece tiktoken protobuf && break
  sleep 20
done
cd "$P"
python code/preflight_prompt_budget.py --config experiments/task_ext_20261114/vllm026.yaml \
  --out output/sidecar/task_ext_20261114/preflight_vllm026.json
# vllm032 (MiMo) passed at 16384 with transformers 5.19 (job 25735164); not rerun here.
echo PREFLIGHT_DONE
