"""Generate both frozen stages from an inspected, pinned MiniMax runtime."""
import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

from hive_vllm_benchmark import load_config


def configure(metadata, freeze, image_sha256, output):
    meta = json.loads(Path(metadata).read_text())
    lock = Path(freeze).read_bytes()
    if hashlib.sha256(lock).hexdigest() != meta['runtime_lock_sha256']:
        raise ValueError('Runtime freeze does not match inspected metadata')
    if 'sm_120' not in (meta.get('compiled_architectures') or ''):
        raise ValueError('Runtime lacks compiled SM120 support')
    if len(image_sha256) != 64 or any(c not in '0123456789abcdef' for c in image_sha256):
        raise ValueError('Image digest must be a SHA256')
    if not any('MiniMaxM3' in line for line in meta['minimax_registry_lines']):
        raise ValueError('MiniMax M3 architecture not registered')
    base = yaml.safe_load(Path('experiments/refresh_qwen_flash_next_pilot.yaml').read_text())
    key = 'minimax_m3_nvfp4'
    base['hardware'] = dict(nodes=1, gpu_count=4, gpu_type='6000_blackwell')
    base['runtime'] = dict(vllm_version=meta['packages']['vllm'], python_version='3.12',
                           runtime_lock_sha256=meta['runtime_lock_sha256'],
                           container_image_sha256=image_sha256,
                           required_environment={'VLLM_FLOAT32_MATMUL_PRECISION': 'high'},
                           max_model_len=16384, gpu_memory_utilization=0.90)
    base['models'] = {key: dict(
        role='refresh_candidate_no_accuracy_selection',
        model_id='nvidia/MiniMax-M3-NVFP4',
        revision='901464083161bf8612a29ff7ad29914cd4ab4a85',
        display_name='MiniMax M3 NVFP4', quantization='NVIDIA ModelOpt NVFP4',
        language_model_only=True,
        reviewed_remote_config='836c3e4aff06f88bd2891b18292424ed7a90158b508d9235d490a33d9e3cba32',
        llm_kwargs=dict(tensor_parallel_size=4, block_size=128, max_num_seqs=32),
        chat_template_kwargs={'thinking_mode': 'disabled'})}
    base['promotion_gate']['baseline_model_key'] = key
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for stage, count in [('pilot', 68), ('remainder', 3332)]:
        config = deepcopy(base)
        config['run_id'] = f'refresh_minimax_m3_{stage}_20260916'
        config['refresh_sample']['stage'] = stage
        config['task_scope']['expected_items'] = count
        path = output / f'{stage}.yaml'
        # Never mutate settings for an existing attempt directory.
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise ValueError('Existing stage configuration differs; use a fresh attempt directory')
        path.write_text(content)
        load_config(path)
    return [str(output / f'{s}.yaml') for s in ['pilot', 'remainder']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--image-sha256', required=True)
    parser.add_argument('--output', required=True)
    print(json.dumps(configure(**vars(parser.parse_args()))))
