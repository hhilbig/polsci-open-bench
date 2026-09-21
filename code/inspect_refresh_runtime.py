"""Inspect an isolated image's packages and template; never load model weights."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import urllib.request


def inspect(output, expected_lock=None):
    output=Path(output)
    packages=sorted({d.metadata['Name'].lower().replace('_','-'): d.version
                     for d in importlib.metadata.distributions() if d.metadata['Name']}.items())
    lock=''.join(f'{name}=={version}\n' for name,version in packages).encode()
    if expected_lock and Path(expected_lock).read_bytes()!=lock:
        raise RuntimeError('Image package environment differs from audited lock')
    required={'pandas':'pandas','pyyaml':'yaml','httpx':'httpx','openai':'openai',
              'transformers':'transformers','vllm':'vllm','torch':'torch',
              'huggingface-hub':'huggingface_hub','jinja2':'jinja2','safetensors':'safetensors'}
    available={name:importlib.util.find_spec(module) is not None for name,module in required.items()}
    missing=[name for name,found in available.items() if not found]
    compatibility={}
    if available['pandas']:
        import numpy
        import pandas
        compatibility={"numpy": numpy.__version__, "pandas": pandas.__version__}
        if compatibility != {"numpy": "2.2.6", "pandas": "2.3.3"}:
            raise RuntimeError(f'Unexpected pandas overlay compatibility: {compatibility}')
    torch_info={}
    if available['torch']:
        import torch
        torch_info=dict(version=torch.__version__,cuda_version=torch.version.cuda,
                        compiled_architectures=torch._C._cuda_getArchFlags() if hasattr(torch._C,'_cuda_getArchFlags') else None)
    vllm_spec=importlib.util.find_spec('vllm')
    registry=Path(vllm_spec.origin).parent/'model_executor/models/registry.py' if vllm_spec else None
    registry_text=registry.read_text() if registry and registry.exists() else ''
    # The pinned official template is metadata, not a model download.
    template_url='https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/236dfdf285828023ca3bcd3f37366c58a3469b13/chat_template.jinja'
    with urllib.request.urlopen(template_url,timeout=30) as response:
        template=response.read()
    rendered=None
    if available['jinja2']:
        from jinja2 import Environment
        env=Environment()
        env.globals['raise_exception']=lambda message: (_ for _ in ()).throw(ValueError(message))
        rendered=env.from_string(template.decode()).render(messages=[{'role':'user','content':'Return a classification.'}],
                                                          add_generation_prompt=True,enable_thinking=False)
        if not rendered.endswith('<think>\n\n</think>\n\n'):
            raise RuntimeError('Thinking-disabled template did not close the thinking block as expected')
    result=dict(python_version=platform.python_version(),packages=dict(packages),required_modules=available,
                missing_required_modules=missing,torch=torch_info,
                runtime_lock_sha256=hashlib.sha256(lock).hexdigest(),
                qwen4_registered='Qwen4ExpForConditionalGeneration' in registry_text,
                vllm_registry_sha256=hashlib.sha256(registry_text.encode()).hexdigest(),
                chat_template_sha256=hashlib.sha256(template).hexdigest(),
                thinking_disabled_template_checked=rendered is not None,
                dependency_compatibility=compatibility,
                gpu_execution_tested=False, inference_performed=False)
    output.mkdir(parents=True,exist_ok=True)
    (output/'runtime.freeze.txt').write_bytes(lock)
    (output/'runtime.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['packages']},sort_keys=True))
    if missing or not result['qwen4_registered']:
        raise RuntimeError('Image lacks required runner packages or Qwen4 architecture')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-lock')
    args=parser.parse_args()
    inspect(args.output,args.expected_lock)
