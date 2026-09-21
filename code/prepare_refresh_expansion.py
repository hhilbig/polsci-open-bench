"""Pin metadata for the approved Hive-only expansion; never download weights."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_refresh_candidates import get_json, image_metadata, model_metadata

DEST = Path('output/sidecar/refresh_expansion_20260916')
MODELS = {
    'glm5_3_flash': ('zai-org/GLM-5.3-Flash', 'eb9eb208eb0d988989d07a6a12d0fdeb5f52574a'),
    'glm5_3': ('zai-org/GLM-5.3', 'aca966e4e02791568aa6a4ced368624b3d897f42'),
    'minimax_m3_nvfp4': ('nvidia/MiniMax-M3-NVFP4', '901464083161bf8612a29ff7ad29914cd4ab4a85'),
    'kimi_k3': ('moonshotai/Kimi-K3', 'f831ab66814297da540d832a5235f8e904f29d06'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST / 'candidates.json')
    args = parser.parse_args()
    records = {key: model_metadata(*identity) for key, identity in MODELS.items()}
    issue = get_json('https://api.github.com/repos/vllm-project/vllm/issues/55773')
    pr = get_json('https://api.github.com/repos/vllm-project/vllm/pulls/38476')
    records['glm5_3_flash'].update(
        status='not evaluated',
        reason='Supported SM120 NoPE sparse-MLA cache path not established; upstream issue remains open, including an independent September 16 failure. No supported A100 path documented.',
        reasoning={'reasoning_effort': 'low', 'thinking_disabled': False},
        runtime_source='https://recipes.vllm.ai/zai-org/GLM-5.3-Flash',
        upstream_issue={k: issue[k] for k in ['html_url', 'state', 'updated_at']},
        alternative=model_metadata('RedHatAI/GLM-5.3-Flash-NVFP4', 'c245560b6d7e62c329cd3042343b358a4279affd'))
    records['glm5_3'].update(
        status='not evaluated',
        reason='Official FP8 exceeds an eight-A100 node; NVFP4 exceeds four Blackwell GPUs. AWQ can fit eight A100s in memory but supported Ampere sparse-attention runtime is not established.',
        runtime_source='https://recipes.vllm.ai/zai-org/GLM-5.3',
        upstream_pr={'url': pr['html_url'], 'state': pr['state'], 'merged': pr['merged'], 'head': pr['head']['sha']},
        alternative=model_metadata('cyankiwi/GLM-5.3-AWQ-INT4'))
    records['kimi_k3'].update(
        status='not evaluated', reason='Native MXFP4 weight files alone exceed the largest currently listed Hive single-node GPU memory; no multi-node or paid fallback authorized.')
    records['minimax_m3_nvfp4'].update(
        status='not evaluated',
        reason=('Pinned four-GPU Hive pilot job 23508830 failed before inference. '
                'The official NVFP4 runtime has no MoE backend that preserves MiniMax M3\'s '
                'SwiGLU clamp on Hive\'s SM120 RTX PRO 6000 Blackwell GPUs; zero predictions were produced.'),
        runtime_source='https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3',
        runtime_image=image_metadata('minimax-m3'),
        reasoning={'thinking_mode': 'disabled'},
        allocation={'nodes': 1, 'gpu_count': 4, 'gpu_type': '6000_blackwell'},
        hive_attempt={'job_id': '23508830', 'state': 'FAILED', 'exit_code': '1:0',
                      'elapsed': '00:07:22', 'stage': 'pilot', 'predictions': 0,
                      'failure': 'No NvFp4 MoE backend supports the deployment configuration.',
                      'hardware': '4x RTX PRO 6000 Blackwell (SM120)'},
        original=model_metadata('MiniMaxAI/MiniMax-M3', 'f0e1c1e04d40177e4673a22097036854f536e9c0'),
        official_quantization=model_metadata('MiniMaxAI/MiniMax-M3-MXFP8', 'c5454eb03678d8710e54a4e0fc681b9f3b4a3dba'))
    if issue['state'] != 'open' or pr['merged']:
        raise RuntimeError('Upstream eligibility changed; review before exporting exclusions')
    payload = {'checked_at': datetime.now(timezone.utc).isoformat(), 'models': records,
               'panel': 'output/sidecar/refresh_20260910/panel.json',
               'policy': 'Hive only; frozen 3400 items; no accuracy gate; retain malformed answers; no deployment.',
               'memory_note': 'Distinct indexed shards in bytes. Twenty percent overhead is a screening heuristic, not measured fit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: {'revision': row['revision'], 'bytes': row['weight_bytes'], 'status': row['status']} for key, row in records.items()}, indent=2))


if __name__ == '__main__':
    main()
