"""Assemble scoring inputs for the October additions; no inference, no remote calls.

Writes one full-panel predictions.csv per OpenRouter model (pilot + remainder),
an API manifest for build_refresh_release.py, and an open manifest that adds the
collected October Hive models to the September open models. The output is a
preview: publication still needs Hanno's approval.

    python3 code/build_october_inputs.py             # predictions and manifests
    python3 code/build_october_inputs.py --costs     # cost table, after the release build
    python3 code/build_refresh_release.py --active-only \
        --output output/sidecar/refresh_october_preview \
        --open-manifest output/sidecar/refresh_october_preview_inputs/open_inputs.json \
        --api-manifest output/sidecar/refresh_jev_20260918/api_manifest.json \
        --api-manifest output/sidecar/refresh_gemini_20260920/api_manifest.json \
        --api-manifest output/sidecar/refresh_october_preview_inputs/api_manifest.json
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from page_config import load_page_config
from refresh_openrouter_batch import MODELS, ROOT, PANEL, read

OUT = Path('output/sidecar/refresh_october_preview_inputs')
HIVE = Path('output/sidecar/refresh_october_20261029/collected')
DROPPED = {'mistral-large-4'}  # Hanno dropped it after 6/68 pilot truncations.
PUBLIC = ['max_tokens', 'temperature', 'reasoning_effort', 'structured_output']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def api_entry(model, panel_sha):
    spec, stages = MODELS[model], []
    for stage in ('pilot', 'remainder'):
        state = read(ROOT / model / stage / 'state.json')
        if state['status'] != 'collected':
            raise SystemExit(f'{model} {stage} is {state["status"]}, not collected')
        frame = pd.read_csv(ROOT / model / stage / 'predictions.csv', dtype={'item_id': str}, low_memory=False)
        stages.append((state, frame.assign(stage=stage)))
    frame = pd.concat([f for _, f in stages], ignore_index=True)
    if len(frame) != 3400 or frame.duplicated(['task', 'item_id']).any():
        raise SystemExit(f'{model}: expected 3,400 unique rows, got {len(frame)}')
    path = ROOT / model / 'predictions.csv'
    frame.to_csv(path, index=False)
    settings = {'model': spec['id'], 'max_tokens': spec['max_tokens'],
                'reasoning_effort': (spec['reasoning'] or {}).get('effort', 'none'),
                'structured_output': 'json_schema'}
    if 'temperature' in spec:
        settings['temperature'] = spec['temperature']
    returned = sorted(frame.returned_model.dropna().astype(str).unique())
    return dict(model=model, model_id=spec['id'], provider='openrouter:' + ','.join(
                    sorted(frame.provider.dropna().astype(str).unique())),
                predictions=str(path), panel_sha256=panel_sha, prediction_sha256=sha(path),
                revision=','.join(returned), hardware_tier='api', settings=settings,
                run_dates=sorted({s['created'][:10] for s, _ in stages}),
                input_tokens=int(frame.input_tokens.fillna(0).sum()),
                output_tokens=int(frame.output_tokens.fillna(0).sum()),
                cost_usd_upper=round(sum(s['cost_usd'] for s, _ in stages), 6),
                returned_models=returned, documented_versions=returned,
                access_mode=spec['mode'], provenance_validated=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel_sha = canonical_sha(read(PANEL))
    api = [api_entry(m, panel_sha) for m in MODELS if m not in DROPPED]
    (OUT / 'api_manifest.json').write_text(json.dumps({'models': api}, indent=2) + '\n')
    config = load_page_config(check_paths=False)
    open_models = json.loads(Path(config.inputs['open_manifest']).read_text())['models']
    for inputs in sorted(HIVE.glob('*/open_inputs.json')):
        entries = json.loads(inputs.read_text())
        entries = entries['models'] if isinstance(entries, dict) and 'models' in entries else [entries]
        for entry in entries:
            if entry['panel_sha256'] != panel_sha:
                raise SystemExit(f'{inputs}: panel hash mismatch')
            open_models.append(entry)
    (OUT / 'open_inputs.json').write_text(json.dumps({'models': open_models}, indent=2) + '\n')
    print(json.dumps({'api_models': [e['model'] for e in api],
                      'open_models': len(open_models), 'output': str(OUT)}))


def cost_table(release_dir):
    """September cost rows plus the new API models, for the preview page's cost figure.

    Batch models: the batch figure is OpenRouter's reported cost; the standard figure
    multiplies the provider token counts by OpenRouter's listed standard prices.
    Sync models were billed at standard rates, so both use the reported cost.
    """
    config = load_page_config(check_paths=False)
    old = pd.read_csv(config.inputs['cost_table'])
    models = pd.read_csv(Path(release_dir) / 'models.csv').set_index('model')
    listing = json.load(urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=60))['data']
    prices = {m['id']: m['pricing'] for m in listing}
    rows = []
    for entry in json.loads((OUT / 'api_manifest.json').read_text())['models']:
        name, spec = entry['model'], MODELS[entry['model']]
        frame = pd.read_csv(entry['predictions'], low_memory=False)
        per_1k = entry['cost_usd_upper'] / len(frame) * 1000
        if spec['mode'] == 'batch':
            price = prices[spec['id']]
            standard = (frame.input_tokens.sum() * float(price['prompt'])
                        + frame.output_tokens.sum() * float(price['completion'])) / len(frame) * 1000
            basis, batch = 'provider_tokens', per_1k
            caveat = ('standard rate estimated from provider token counts and OpenRouter list prices '
                      f'on {pd.Timestamp.now(tz="UTC").date()}; batch figure is the cost OpenRouter reported')
        else:
            standard, basis, batch = per_1k, 'provider_reported', None
            caveat = 'cost OpenRouter reported for one request per text to a fixed provider'
        rows.append(dict(model=name, mean_task_f1=models.loc[name, 'mean_task_f1'],
                         task_ci_low=models.loc[name, 'task_ci_low'], task_ci_high=models.loc[name, 'task_ci_high'],
                         cost_usd_run=entry['cost_usd_upper'], cost_per_1k_items=standard, cost_basis=basis,
                         cost_per_1k_items_batch=batch, has_batch_endpoint=spec['mode'] == 'batch',
                         cost_caveat=caveat, cost_is_invoiced=False,
                         input_tokens_per_item=frame.input_tokens.mean(),
                         output_tokens_per_item=frame.output_tokens.mean()))
    table = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if table.model.duplicated().any():
        raise SystemExit('Duplicate model in the cost table')
    table.to_csv(OUT / 'cost_performance.csv', index=False)
    print(pd.DataFrame(rows)[['model', 'cost_per_1k_items', 'cost_per_1k_items_batch', 'cost_basis']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--costs']:
        cost_table('output/sidecar/refresh_october_preview')
    else:
        main()
