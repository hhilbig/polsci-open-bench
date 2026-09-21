#!/usr/bin/env bash
#SBATCH --job-name=refresh-qflash-image
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hive/scratch/hhilbig/refresh-qwen-flash-runtime/preflight-%j.log
# No GPU requested: this job inspects container packages, not inference.
set -euo pipefail
audit_root=/nfs/hive/scratch/hhilbig/refresh-qwen-flash-runtime
image_path="$audit_root/qwen38-flash-next-0aea3024.sif"
image_uri=docker://vllm/vllm-openai@sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$audit_root/tmp"
export TMPDIR="$audit_root/tmp"
mkdir -p "$audit_root/cache" "$audit_root/tmp" "$audit_root/metadata"
exec 9>"$audit_root/image.lock"
flock -n 9 || { echo 'Image preflight already active'; exit 1; }
if [[ ! -f "$image_path" ]]; then
  srun --cpu-bind=cores apptainer pull --reproducible "$image_path" "$image_uri"
fi
sha256sum "$image_path"
overlay_path="$audit_root/overlay-pandas233-base-numpy226"
if [[ ! -d "$overlay_path/pandas" ]]; then
  mkdir -p "$overlay_path"
  apptainer exec --cleanenv --no-home --bind "$audit_root:/audit" --pwd /audit \
    "$image_path" python3 -m pip install --disable-pip-version-check \
    --no-deps --target /audit/overlay-pandas233-base-numpy226 \
    pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2026.3.post1 \
    six==1.17.0 tzdata==2026.4
fi
apptainer exec --cleanenv --no-home --bind "$audit_root:/audit" --pwd /audit \
  --env PYTHONPATH=/audit/overlay-pandas233-base-numpy226 "$image_path" \
  python3 /audit/inspect_refresh_runtime.py --output /audit/metadata
