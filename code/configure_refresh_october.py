"""Generate frozen pilot and remainder configs for the October 2026 open-model additions.

MiMo-V2.6-Flash and Nemotron 3.5 Super VL run on one 4x Blackwell node, Intern-S2
on one 8x A100 node. Prompts, sample, decoding and scoring match the September
refresh; only the vLLM runtime (0.32.0, needed for these architectures) differs.
"""
import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from hive_vllm_benchmark import REVIEWED_REMOTE_CODE, load_config

OUTPUT = Path('experiments/refresh_october_20261026')
CUDA_COMPONENTS = {
    'nvidia-cuda-cccl': '13.0.85',
    'nvidia-cuda-crt': '13.0.88',
    'nvidia-cuda-nvcc': '13.0.88',
    'nvidia-cuda-nvrtc': '13.0.88',
    'nvidia-cuda-runtime': '13.0.96',
    'nvidia-nvvm': '13.0.88',
}
MODELS = {
    'mimo_v2_6_flash_mopd_fp8': dict(
        hardware=dict(nodes=1, gpu_count=4, gpu_type='6000_blackwell'),
        model=dict(model_id='XiaomiMiMo/MiMo-V2.6-Flash-MOPD',
                   revision='b12f5573350bb1704513fcf119fca48e498b0423',
                   display_name='MiMo-V2.6-Flash MOPD FP8', quantization='FP8 (official)')),
    'nemotron3_5_super_vl_bf16': dict(
        hardware=dict(nodes=1, gpu_count=4, gpu_type='6000_blackwell'),
        model=dict(model_id='nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16',
                   revision='fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe',
                   display_name='Nemotron 3.5 Super VL 120B-A12B BF16', quantization='BF16',
                   language_model_only=True)),
    'intern_s2_397b_fp8': dict(
        hardware=dict(nodes=1, gpu_count=8, gpu_type='a100'),
        model=dict(model_id='internlm/Intern-S2-397B-FP8',
                   revision='31b460c8ebb8bdda6f1d77651cfad7ab811910e3',
                   display_name='Intern-S2 397B FP8', quantization='FP8 (official)',
                   language_model_only=True)),
}


def configure(output=OUTPUT):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    base = yaml.safe_load(Path('experiments/refresh_hive_20260914_pilot.yaml').read_text())
    base['benchmark_commit'] = commit
    base['runtime'] = dict(vllm_version='0.32.0', python_version='3.12',
                           cuda_toolkit_meta_version='13.0.3',
                           cuda_component_versions=CUDA_COMPONENTS,
                           max_model_len=16384, gpu_memory_utilization=0.90)
    written = []
    for key, spec in MODELS.items():
        model = dict(role='refresh_candidate_no_accuracy_selection', **spec['model'])
        if (model['model_id'], model['revision']) not in REVIEWED_REMOTE_CODE:
            raise ValueError(f'{key} remote code has not been reviewed')
        model.update(reviewed_remote_code=True,
                     llm_kwargs=dict(tensor_parallel_size=spec['hardware']['gpu_count']),
                     chat_template_kwargs=dict(enable_thinking=False))
        config = deepcopy(base)
        config['hardware'] = spec['hardware']
        config['models'] = {key: model}
        config['promotion_gate']['baseline_model_key'] = key
        for stage, count in [('pilot', 68), ('remainder', 3332)]:
            staged = deepcopy(config)
            staged['run_id'] = f'refresh_october_{key}_{stage}_20261026'
            staged['refresh_sample']['stage'] = stage
            staged['task_scope']['expected_items'] = count
            path = Path(output) / key / f'{stage}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            content = yaml.safe_dump(staged, sort_keys=False)
            # Never mutate settings for an existing attempt.
            if path.exists() and path.read_text() != content:
                raise ValueError(f'{path} exists with different settings; use a fresh directory')
            path.write_text(content)
            load_config(path)
            written.append(str(path))
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    print(json.dumps(configure(**vars(parser.parse_args())), indent=2))
