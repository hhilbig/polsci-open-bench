#!/bin/bash -l
#SBATCH --job-name=ext-weights
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# CPU-only: re-download the five post-mid-2025 models whose cached blobs were
# deleted on 2026-10-26, at the revisions the panel runs pinned, then verify
# that no snapshot file is a dangling link. Sequential, one model at a time.
set -euo pipefail
W=/scratch/$USER/weights_$SLURM_JOB_ID; mkdir -p "$W"
export PATH=/nfs/hive/scratch/$USER/uvbin:$PATH UV_CACHE_DIR=$W/uvcache
export UV_PYTHON_INSTALL_DIR=/nfs/hive/scratch/$USER/uvpython-3.12
export HF_HOME=/nfs/hive/scratch/$USER/hf HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_XET_HIGH_PERFORMANCE=1
uv venv --clear "$W/venv" --python 3.12 && source "$W/venv/bin/activate"
for i in 1 2 3; do uv pip install -q "huggingface_hub[hf_xet]==1.1.4" && break || sleep 20; done
python - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download
MODELS = [
    ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "5a5a776300a41aaa681dd7ff0106608ef2bc90db"),
    ("Qwen/Qwen3.5-35B-A3B-FP8", "9d1823d2dee688a6b25e77009dc727688c44936e"),
    ("Qwen/Qwen3.6-35B-A3B-FP8", "95a723d08a9490559dae23d0cff1d9466213d989"),
    ("mistralai/Mistral-Small-4-119B-2603-NVFP4", "b1a9048590131d38491bd23a7c9f6ed0962f0358"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", "c5f5f263bdd5cc134092897864e8905d8fe7b928"),
]
for repo, rev in MODELS:
    t = time.time()
    for attempt in range(5):
        try:
            path = Path(snapshot_download(repo_id=repo, revision=rev, max_workers=4))
            break
        except Exception as exc:
            print(f"{repo} attempt {attempt+1} failed: {exc}", flush=True); time.sleep(60)
    else:
        print(f"FAILED {repo}", flush=True); continue
    files = [p for p in path.iterdir()]
    dangling = [p.name for p in files if not p.exists()]
    size = sum(p.stat().st_size for p in files if p.exists())
    print(f"OK {repo}@{rev[:8]} files={len(files)} dangling={len(dangling)} "
          f"size={size/1e9:.1f}GB minutes={(time.time()-t)/60:.1f}", flush=True)
PY
df -h /nfs/hive/scratch | tail -1
echo WEIGHTS_DONE
