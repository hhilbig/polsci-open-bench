#!/usr/bin/env bash
#SBATCH --job-name=refresh-expansion-images
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hhilbig/refresh-expansion-images_%j.log
set -euo pipefail
audit_root=/nfs/hive/scratch/hhilbig/refresh-expansion-20260916
mkdir -p "$audit_root/cache" "$audit_root/tmp" "$audit_root/metadata"
export APPTAINER_CACHEDIR="$audit_root/cache"
export APPTAINER_TMPDIR="$audit_root/tmp"
export TMPDIR="$audit_root/tmp"
exec 9>"$audit_root/image.lock"
flock -n 9 || exit 3
image_path="$audit_root/minimax-m3-a8c89303.sif"
if [[ ! -f "$image_path" ]]; then
  apptainer pull --reproducible "$image_path" docker://vllm/vllm-openai@sha256:a8c89303e467d08ef77ab5fac1ab20910a09b47d10787474b4ee70dfb6bee148
fi
sha256sum "$image_path"
apptainer exec --cleanenv --no-home --bind "$audit_root:/audit" --pwd /audit \
  "$image_path" python3 - <<'PY'
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
packages = dict(sorted((d.metadata['Name'].lower().replace('_', '-'), d.version)
                      for d in metadata.distributions() if d.metadata['Name']))
freeze = ''.join(f'{k}=={v}\n' for k, v in packages.items()).encode()
spec = importlib.util.find_spec('vllm')
root = Path(spec.origin).parent
registry = (root / 'model_executor/models/registry.py').read_text()
import torch
report = {'packages': packages, 'runtime_lock_sha256': hashlib.sha256(freeze).hexdigest(),
          'torch_cuda': torch.version.cuda, 'compiled_architectures': torch._C._cuda_getArchFlags(),
          'minimax_registry_lines': [line.strip() for line in registry.splitlines() if 'MiniMax' in line or 'Minimax' in line],
          'gpu_tested': False, 'weights_downloaded': False}
Path('/audit/metadata/minimax-runtime.freeze.txt').write_bytes(freeze)
Path('/audit/metadata/minimax-runtime.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
for path in root.rglob('*minimax*'):
    print('SOURCE', path)
    if path.is_file() and path.suffix == '.py':
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if any(word in line.lower() for word in ['capability', 'sm120', 'sm_120', 'nvfp4', 'modelopt', 'msa', 'flashinfer']):
                print(number, line)
PY
