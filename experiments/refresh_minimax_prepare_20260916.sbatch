#!/usr/bin/env bash
#SBATCH --job-name=refresh-minimax-prepare
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/hhilbig/refresh-minimax-prepare_%j.log
set -euo pipefail
project_root=/home/hhilbig/polsci-expansion-20260916
audit_root=/nfs/hive/scratch/hhilbig/refresh-expansion-20260916
image_path="$audit_root/minimax-m3-a8c89303.sif"
job_root="/scratch/$USER/refresh-minimax-prepare-${SLURM_JOB_ID}"
mkdir -p "$job_root/home" "$job_root/tmp" "$job_root/cache" "$audit_root/overlay"
test -f "$audit_root/metadata/minimax-runtime.json"
exec 9>"$audit_root/prepare.lock"
flock -n 9 || exit 3
options=(--cleanenv --home "$job_root/home:/home/hhilbig"
  --bind "$project_root:/project" --bind "$audit_root:/audit" --bind "$job_root:/job"
  --pwd /project --env TMPDIR=/job/tmp --env XDG_CACHE_HOME=/job/cache)
# Separate overlay; never modify the image or the shared historical runtime.
apptainer exec "${options[@]}" "$image_path" python3 -m pip install \
  --disable-pip-version-check --no-deps --target /audit/overlay \
  numpy==2.2.6 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2026.3.post1 \
  six==1.17.0 tzdata==2026.4 scipy==1.16.3 scikit-learn==1.7.2 \
  joblib==1.5.2 threadpoolctl==3.6.0
apptainer exec "${options[@]}" --env PYTHONPATH=/audit/overlay:/project/code \
  --env HF_HOME=/audit/tokenizer-cache "$image_path" python3 - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import numpy
import pandas
import sklearn
from transformers import AutoTokenizer
from configure_refresh_minimax import configure
from hive_vllm_benchmark import (load_config, selected_tasks, build_conversations,
                                  preflight_prompt_lengths, resolved_chat_template_kwargs)
from run_refresh_expansion import validate_configs

audit = Path('/audit/metadata')
report = json.loads((audit / 'minimax-runtime.json').read_text())
if (numpy.__version__, pandas.__version__, sklearn.__version__) != ('2.2.6', '2.3.3', '1.7.2'):
    raise RuntimeError('Isolated analysis dependencies differ from pins')
packages = {}
for dist in metadata.distributions():
    if dist.metadata['Name']:
        packages.setdefault(dist.metadata['Name'].lower().replace('_', '-'), dist.version)
freeze = ''.join(f'{k}=={v}\n' for k, v in sorted(packages.items())).encode()
report.update(packages=packages, runtime_lock_sha256=hashlib.sha256(freeze).hexdigest())
(audit / 'runtime.freeze.txt').write_bytes(freeze)
(audit / 'runtime.json').write_text(json.dumps(report, indent=2) + '\n')
image_hash = hashlib.file_digest(open('/audit/minimax-m3-a8c89303.sif', 'rb'), 'sha256').hexdigest()
paths = configure(audit / 'runtime.json', audit / 'runtime.freeze.txt', image_hash,
                  'output/sidecar/refresh_expansion_20260916/config')
validate_configs(*paths, 'minimax_m3_nvfp4')
revision = '901464083161bf8612a29ff7ad29914cd4ab4a85'
tokenizer = AutoTokenizer.from_pretrained('nvidia/MiniMax-M3-NVFP4', revision=revision, trust_remote_code=False)
rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Return a classification.'}],
                                        tokenize=False, add_generation_prompt=True, thinking_mode='disabled')
if not rendered.endswith('</mm:think>'):
    raise RuntimeError('Official tokenizer did not close the thinking prefix')
counts = []
for path in paths:
    config = load_config(Path(path))
    model = config['models']['minimax_m3_nvfp4']
    lengths = []
    for task in selected_tasks(config, None):
        conversations = build_conversations(Path(task['prompt_path']).read_text(), task['loader']())
        lengths.extend(preflight_prompt_lengths(tokenizer, conversations,
                       max_model_len=config['runtime']['max_model_len'], max_tokens=256,
                       margin_tokens=128, chat_template_kwargs=resolved_chat_template_kwargs(model)))
    counts.append({'stage': config['refresh_sample']['stage'], 'items': len(lengths),
                   'max_prompt_tokens': max(lengths)})
if [r['items'] for r in counts] != [68, 3332]:
    raise RuntimeError('Token preflight did not cover all frozen items')
result = {'stage_token_checks': counts, 'thinking_disabled_verified': True,
          'container_image_sha256': image_hash, 'weights_downloaded': False,
          'runtime_lock_sha256': report['runtime_lock_sha256']}
(audit / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
