"""Record official pinned candidate metadata without downloading model weights."""
import json
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

CANDIDATES = [
    ('Qwen3.8 27B', 'Qwen/Qwen3.8-27B-FP8'),
    ('Qwen3.8 Flash-Next', 'Qwen/Qwen3.8-Flash-Next-FP8'),
    ('Mistral Medium 3.5 128B', 'mistralai/Mistral-Medium-3.5-128B'),
    ('GLM-5.3', 'zai-org/GLM-5.3'),
    ('GLM-5.3-Flash', 'zai-org/GLM-5.3-Flash'),
]
DEST = Path('output/sidecar/refresh_20260910_release/candidates.json')
EXPANSION = Path('output/sidecar/refresh_expansion_20260916/candidates.json')
MINIMAX_ATTEMPT = Path('output/sidecar/refresh_expansion_20260916/remote_attempt_23508830/minimax_m3_nvfp4/status.json')
QWEN_FLASH_RUNTIME = DEST.parent / 'qwen_flash_runtime.json'
QWEN_FLASH_FREEZE = DEST.parent / 'qwen_flash_runtime.freeze.txt'
RECIPES = {
    'Qwen3.8 Flash-Next': 'https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next',
    'Mistral Medium 3.5 128B': 'https://recipes.vllm.ai/mistralai/Mistral-Medium-3.5-128B',
    'GLM-5.3': 'https://recipes.vllm.ai/zai-org/GLM-5.3',
    'GLM-5.3-Flash': 'https://recipes.vllm.ai/zai-org/GLM-5.3-Flash',
}


def get_bytes(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def model_metadata(model, revision=None):
    """Select only shards used by one index, avoiding duplicate model formats."""
    endpoint = 'https://huggingface.co/api/models/' + model
    metadata = get_json(endpoint + ('/revision/' + revision if revision else '') + '?blobs=true')
    revision = metadata['sha']
    base = f'https://huggingface.co/{model}/resolve/{revision}/'
    all_files = {f['rfilename']: f for f in metadata['siblings']}
    config_raw = get_bytes(base+'config.json')
    config = json.loads(config_raw)
    indexes = [p for p in all_files if p.endswith('.safetensors.index.json')]
    index_name = 'model.safetensors.index.json' if 'model.safetensors.index.json' in indexes else (sorted(indexes)[0] if indexes else None)
    if index_name:
        index_raw = get_bytes(base+index_name)
        selected = sorted(set(json.loads(index_raw)['weight_map'].values()))
    else:
        index_raw = None
        selected = sorted(p for p in all_files if p.endswith('.safetensors'))
    assert selected and all(all_files[p].get('size') for p in selected)
    files = [dict(path=p, bytes=all_files[p]['size'], lfs=all_files[p].get('lfs')) for p in selected]
    size = sum(f['bytes'] for f in files)
    small = {}
    for path in ['LICENSE','README.md','chat_template.jinja','tokenizer_config.json','tokenizer.json','tekken.json']:
        if path in all_files:
            raw = get_bytes(base+path)
            small[path] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), source=base+path)
    card = metadata.get('cardData') or {}
    return dict(model_id=model, revision=revision, source=f'https://huggingface.co/{model}/tree/{revision}',
                license=card.get('license'), license_name=card.get('license_name'),
                architectures=config.get('architectures'), quantization=(config.get('quantization_config') or {}).get('quant_method'),
                weight_bytes=size, weight_gib=size/2**30, preliminary_weight_plus_20_percent_bytes=int(size*1.2),
                weight_index=index_name, weight_index_sha256=hashlib.sha256(index_raw).hexdigest() if index_raw else None,
                excluded_alternate_weight_files=sorted(p for p in all_files if p.endswith('.safetensors') and p not in selected),
                weight_files=files, config_sha256=hashlib.sha256(config_raw).hexdigest(), pinned_artifacts=small)


def image_metadata(tag):
    """Resolve public registry digests/config only; never fetch image layers."""
    repo = 'vllm/vllm-openai'
    token = get_json('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repo+':pull')['token']
    headers = {'Authorization':'Bearer '+token, 'Accept':'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'}
    def request(path):
        with urllib.request.urlopen(urllib.request.Request('https://registry-1.docker.io/v2/'+repo+'/'+path,headers=headers),timeout=30) as response:
            return json.load(response), response.headers.get('Docker-Content-Digest')
    index, digest = request('manifests/'+tag)
    amd = next(x for x in index['manifests'] if x.get('platform') == {'architecture':'amd64','os':'linux'})
    manifest, _ = request('manifests/'+amd['digest'])
    config, _ = request('blobs/'+manifest['config']['digest'])
    labels = config.get('config',{}).get('Labels',{}) or {}
    return dict(image=repo+':'+tag, index_digest=digest, amd64_digest=amd['digest'],
                immutable_amd64=repo+'@'+amd['digest'], config_digest=manifest['config']['digest'],
                build_created=config.get('created'), labels=labels,
                compressed_layer_bytes=sum(x['size'] for x in manifest['layers']),
                sm120_validation='Not established by registry metadata. Must validate native kernels on Hive; GB200/GB300 recipe validation is not SM120 proof.')


def qwen_flash_cpu_preflight():
    """Accept only the compatible saved CPU inspection, not a GPU fit claim."""
    if not QWEN_FLASH_RUNTIME.exists() or not QWEN_FLASH_FREEZE.exists():
        return None
    raw = QWEN_FLASH_RUNTIME.read_bytes()
    meta = json.loads(raw)
    freeze_sha = hashlib.sha256(QWEN_FLASH_FREEZE.read_bytes()).hexdigest()
    if (meta.get('runtime_lock_sha256') != freeze_sha or
            meta.get('qwen4_registered') is not True or
            meta.get('thinking_disabled_template_checked') is not True or
            meta.get('gpu_execution_tested') is not False or
            meta.get('inference_performed') is not False or
            meta.get('missing_required_modules') or
            meta.get('dependency_compatibility') != {'numpy':'2.2.6','pandas':'2.3.3'}):
        raise ValueError('Saved Qwen Flash CPU preflight does not pass the compatibility checks')
    return dict(metadata=str(QWEN_FLASH_RUNTIME), metadata_sha256=hashlib.sha256(raw).hexdigest(),
                runtime_lock_sha256=freeze_sha, qwen4_registered=True,
                thinking_disabled_template_checked=True, gpu_execution_tested=False,
                package_compatibility=meta['dependency_compatibility'])


def qwen27_complete(validation, revision):
    pilot = validation.get('pilot', {})
    remainder = validation.get('remainder', {})
    return (validation.get('complete') is True
            and pilot.get('passed') is True and pilot.get('rows') == 68
            and remainder.get('passed') is True and remainder.get('rows') == 3332
            and pilot.get('revision') == remainder.get('revision') == revision
            and pilot.get('settings') == remainder.get('settings'))


def main():
    existing = json.loads(DEST.read_text()) if DEST.exists() else {}
    previous = {r['model']: r for r in existing.get('candidates',[])}
    records = []
    for name, model in CANDIDATES:
        old = previous.get(name,{})
        row = dict(old, **model_metadata(model, old.get('revision')))
        row['model'] = name
        row.setdefault('status','pending')
        row.setdefault('reason','Hive feasibility and structural pilot pending; memory estimate is not proof of fit.')
        if name in RECIPES:
            row['runtime_recipe'] = RECIPES[name]
            row['runtime_recipe_sha256'] = hashlib.sha256(get_bytes(RECIPES[name])).hexdigest()
        records.append(row)
    by_name = {r['model']:r for r in records}
    q27 = by_name['Qwen3.8 27B']
    q27_pilot_path = DEST.parent / 'new_models/qwen3_8_27b_fp8/validation.json'
    if q27_pilot_path.exists():
        q27_validation = json.loads(q27_pilot_path.read_text())
        q27_pilot = q27_validation.get('pilot', {})
        q27_remainder = q27_validation.get('remainder', {})
        if qwen27_complete(q27_validation, q27['revision']):
            q27['status'] = 'completed'
            q27['reason'] = 'The pinned Hive pilot and remainder passed validation on all 3,400 frozen texts with identical inference settings.'
        elif (q27_pilot.get('passed') is True and q27_pilot.get('rows') == 68
                and q27_pilot.get('revision') == q27['revision']):
            q27['reason'] = 'The 68-text Hive structural pilot passed; the remaining 3,332 predictions have not been validated.'
    q = by_name['Qwen3.8 Flash-Next']
    qwen_preflight = qwen_flash_cpu_preflight()
    if qwen_preflight:
        q['reason'] = 'The pinned CPU runtime preflight passed; GPU fit and the 68-text structural pilot remain untested.'
    q.update(thinking_settings={'enable_thinking':False}, runtime_image=image_metadata('qwen38-flash-next'),
             feasibility=dict(status='cpu_preflight_passed_gpu_pilot_pending' if qwen_preflight else 'runtime_preflight_pending',
                              cpu_preflight=qwen_preflight,
                              preferred_allocation='one node, 2x98GB Blackwell with PLE CPU offload; use 4 GPUs if pilot memory requires',
                              host_memory_gb_minimum=128, cpu_offload='VLLM_PLE_CPU_OFFLOAD=1',
                              rationale='Official FP8 weights include approximately 51GB PLE lookup embeddings; after documented CPU offload, two Blackwell GPUs are a plausible pilot allocation. Full fit remains unproven.',
                              runtime='Dedicated qwen38-flash-next image required by current recipe; PyPI installation explicitly unsupported. Qwen4Exp architecture absent from shared vLLM0.26 registry.',
                              hazards=['No stock TP8 for FP8 block shapes; recipe requires expert parallelism at TP8.', 'SM120 kernels and PLE behavior need direct validation; skip speculative decoding.']))
    m = by_name['Mistral Medium 3.5 128B']
    m['status'] = 'not evaluated'
    m['reason'] = 'License eligibility is unresolved; no Hive inference has been submitted.'
    m.update(thinking_settings={'reasoning_effort':'none'},
             feasibility=dict(status='license_review_required', preferred_allocation='one node, 2x98GB Blackwell',
                              rationale='Selected Hugging Face index totals 133.606 GB; the previous 267.212 GB count duplicated native Mistral and HF formats. Two Blackwell GPUs fit weights plus 20% planning overhead, before measured pilot validation.',
                              runtime='Mistral3ForConditionalGeneration and Ministral3ForCausalLM are registered in shared vLLM0.26. Official card recommends nightly with mistral_common>=1.11.1 and transformers>=5.4.0; exact pinned runtime still requires pilot.',
                              license_note='Pinned Modified MIT license restricts use when company or employer global consolidated monthly revenue exceeds US$20 million. No nonprofit or research exemption appears. Do not assume institutional eligibility.',
                              license_source=m['pinned_artifacts']['LICENSE']['source']))
    g = by_name['GLM-5.3']
    g['status'] = 'not evaluated'
    g['reason'] = 'No supported single-node Hive runtime was verified for the available quantized weights.'
    g.update(thinking_settings={'reasoning_effort':'low'},
             alternatives=[model_metadata('Inferact/GLM-5.3-NVFP4'),model_metadata('cyankiwi/GLM-5.3-AWQ-INT4')])
    pr = get_json('https://api.github.com/repos/vllm-project/vllm/pulls/38476')
    g['feasibility'] = dict(status='no_verified_single_node_runtime',
                           rationale='Official FP8 weights exceed an 8x80GB A100 node. Inferact NVFP4 exceeds a 4x98GB Blackwell node. Cyankiwi AWQ-INT4 totals 488.176 GB and could fit 8xA100 on memory alone, but stock sparse-MLA/indexer support is not established on Ampere.',
                           awq_candidate='cyankiwi/GLM-5.3-AWQ-INT4', awq_runtime='Requires experimental Ampere sparse-MLA/indexer fallback rather than verified shared runtime. Not selected for inference by this metadata audit.',
                           upstream_a100_support_pr=dict(url=pr['html_url'],state=pr['state'],merged=pr['merged'],head_sha=pr['head']['sha'],updated_at=pr['updated_at']),
                           reason_not_general_impossibility='A100-compatible quantized weights exist; an unmerged backend proposal is not proven reliable Hive support. Do not describe the model as inherently impossible to run.')
    f = by_name['GLM-5.3-Flash']
    flash_issue = get_json('https://api.github.com/repos/vllm-project/vllm/issues/55773')
    f['status'] = 'not evaluated'
    f['reason'] = ('The SM120 NoPE-MLA runtime issue is open; GPU pilot and fit remain untested.'
                   if flash_issue['state'] == 'open' else
                   'The reported SM120 NoPE-MLA issue changed state; the pinned image still needs GPU verification.')
    f.update(thinking_settings={'reasoning_effort':'low'}, runtime_image=image_metadata('glm53-flash'),
             alternatives=[model_metadata('RedHatAI/GLM-5.3-Flash-NVFP4')],
             feasibility=dict(status='runtime_preflight_pending', preferred_checkpoint='zai-org/GLM-5.3-Flash', preferred_allocation='one node, 4x98GB Blackwell',
                              rationale='Official FP8 weights are 328.337 GB (305.788 GiB); adding 20% gives 366.946 GiB. Check actual per-GPU memory bytes before judging four-GPU fit; do not mix decimal GB with GiB. Prefer an official FP8 pilot if headroom suffices. Documented RedHatAI NVFP4 weights total 197.844 GB as a fallback. Neither is a measured fit.',
                              runtime='Official recipe requires dedicated glm53-flash container and FlashInfer>=0.6.17; Glm5Next architecture absent from shared vLLM0.26 registry.',
                              hazards=['SM120 NoPE-MLA startup issue reported upstream; require pilot on exact image digest.', 'Thinking cannot be disabled; low effort must still share the 256 output tokens.', 'Fallback quantized scores must be labeled NVFP4, not official FP8.'],
                              sm120_issue=dict(url=flash_issue['html_url'], state=flash_issue['state'],
                                               updated_at=flash_issue['updated_at'])))
    if EXPANSION.exists():
        expansion = json.loads(EXPANSION.read_text())['models']
        minimax = dict(expansion['minimax_m3_nvfp4'])
        minimax['model'] = 'MiniMax M3 NVFP4'
        if MINIMAX_ATTEMPT.exists():
            attempt = json.loads(MINIMAX_ATTEMPT.read_text())
            if attempt.get('status') != 'infrastructure_failed' or attempt.get('stage') != 'pilot':
                raise ValueError('Unexpected MiniMax attempt state')
            minimax.update(
                status='not evaluated',
                reason=("The pinned four-GPU Hive pilot failed before inference: the official "
                        "NVFP4 runtime has no MoE backend that preserves MiniMax M3's SwiGLU "
                        "clamp on SM120 RTX PRO 6000 Blackwell GPUs. Zero predictions were produced."),
                feasibility={'status': 'unsupported_sm120_nvfp4_moe_backend',
                             'job_id': '23508830', 'stage': 'pilot', 'predictions': 0,
                             'hardware': '4x RTX PRO 6000 Blackwell (SM120)'})
        kimi = dict(expansion['kimi_k3'])
        kimi['model'] = 'Kimi K3'
        records.extend([minimax, kimi])
    registry_url='https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/registry.py'
    registry = get_bytes(registry_url)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(existing, checked_at=datetime.now(timezone.utc).isoformat(),candidates=records,
                 hardware_scope='Metadata screening against one node with at most 8xA100 80GB or 4xBlackwell 98GB; no jobs or weight downloads performed.',
                 shared_runtime_registry=dict(source=registry_url,sha256=hashlib.sha256(registry).hexdigest()),
                 memory_note='Bytes from distinct shard paths referenced by one pinned index. 20% overhead is a planning heuristic, not proof of fit. GPU memory totals do not establish kernel support.')
    DEST.write_text(json.dumps(payload,indent=2)+'\n')
    for row in records:
        print(row['model'],row['revision'],round(row['weight_bytes']/1e9,2),'GB',row['architectures'])


if __name__ == '__main__':
    main()
