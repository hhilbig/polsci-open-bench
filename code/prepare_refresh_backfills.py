"""Prepare exact, provenance-preserving Hive backfills; never submit jobs.

Uses only completed historical multi-task runs accepted by audit_refresh_reuse.
The pilot config requests only missing members of the frozen 68-item pilot;
the remainder requests only missing non-pilot rows. Existing predictions stay
unchanged and are referenced by path and checksum.
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import yaml

from audit_refresh_reuse import digest, sha

ROOT = Path('output/sidecar/refresh_20260910_release')


def split_missing(panel, reusable):
    rows = {(r['task'], str(r['item']['item_id'])): r for r in panel['rows']}
    if not set(reusable) <= rows.keys():
        raise ValueError('Reuse keys outside frozen panel')
    if len(rows) != len(panel['rows']):
        raise ValueError('Duplicate frozen keys')
    result = {'pilot': [], 'remainder': []}
    for key, row in rows.items():
        if key not in reusable:
            result['pilot' if row['pilot'] else 'remainder'].append(list(key))
    assert len(reusable) + sum(map(len, result.values())) == len(rows)
    return result


def choose_partial_groups(audit):
    sources = {s['source']: s for s in audit['sources']}
    completed_aliases = {g['model_key'] for g in audit['groups'] if g['complete']}
    completed_checkpoints = {(g['model_id'], g['revision']) for g in audit['groups'] if g['complete']}
    candidates = defaultdict(list)
    for group in audit['groups']:
        if (group['complete'] or group['model_key'] in completed_aliases or
                (group['model_id'], group['revision']) in completed_checkpoints):
            continue
        # Pilot-only evidence does not make a historical checkpoint an evaluated baseline.
        runs = [sources[r['source']] for r in group['selected_rows']]
        if not any(len(s.get('task_checks', {})) >= 18 for s in runs):
            continue
        candidates[group['model_key']].append(group)
    return [max(groups, key=lambda g: (g['eligible_rows'], g['compatibility_group']))
            for _, groups in sorted(candidates.items())]


def matching_config(metadata, experiments=Path('experiments')):
    matches = [p for p in experiments.glob('*.yaml') if sha(p) == metadata['config_sha256']]
    if not matches:
        raise ValueError('Exact historical config unavailable; do not reconstruct inference settings')
    path = sorted(matches)[0]
    return path, yaml.safe_load(path.read_text())


def build_config(original, model_key, panel_hash, keys_path, stage, keys):
    if not keys:
        raise ValueError('Empty stage must not be submitted')
    config = copy.deepcopy(original)
    config.pop('execution', None)
    config['run_id'] = 'refresh_20260910_backfill_' + model_key + '_' + stage
    config['models'] = {model_key: config['models'][model_key]}
    config['refresh_sample'] = {'panel_sha256': panel_hash, 'stage': 'backfill',
                               'keys_manifest': str(keys_path), 'keys_sha256': sha(keys_path)}
    config['task_scope'] = {'tasks_dir': 'tasks', 'expected_tasks': len({k[0] for k in keys}), 'expected_items': len(keys)}
    # Legacy runner requires this section; these fields never select by accuracy.
    config['promotion_gate'] = {'baseline_model_key': model_key, 'minimum_mean_f1_gain': .01,
        'equivalence_tolerance': .005, 'maximum_parse_error_increase': .005, 'material_throughput_ratio': 1.25}
    return config


def prepare(audit_path=ROOT / 'reuse_audit.json', panel_path=Path('output/sidecar/refresh_20260910/panel.json'),
            output=ROOT / 'backfills', model_keys=None):
    audit = json.loads(audit_path.read_text())
    panel = json.loads(panel_path.read_text())
    if sha(panel_path) != audit['panel_sha256'] or digest(panel) != audit['panel_canonical_sha256']:
        raise ValueError('Frozen panel changed since reuse audit')
    sources = {s['source']: s for s in audit['sources']}
    groups = choose_partial_groups(audit)
    if model_keys:
        groups = [g for g in groups if g['model_key'] in model_keys]
        if {g['model_key'] for g in groups} != set(model_keys):
            raise ValueError('Requested model lacks eligible completed multi-task source')
    manifest = {'panel_sha256': digest(panel), 'reuse_audit_sha256': sha(audit_path),
                'status': 'prepared_not_submitted', 'models': [], 'ineligible': []}
    for group in groups:
        alias = group['model_key']
        source_paths = sorted({r['source'] for r in group['selected_rows']})
        try:
            for path in source_paths:
                if sha(path) != sources[path]['predictions_sha256']:
                    raise ValueError('Historical predictions changed since audit')
                if sha(sources[path]['metadata']) != sources[path]['metadata_sha256']:
                    raise ValueError('Historical metadata changed since audit')
            reference = max((sources[p] for p in source_paths), key=lambda s: s.get('completed_at') or '')
            metadata = json.loads(Path(reference['metadata']).read_text())
            config_path, original = matching_config(metadata)
            if original['models'][alias]['revision'] != group['revision']:
                raise ValueError('Historical config revision differs from audit')
            if original['generation'] != metadata['generation'] or original['runtime'] != metadata['runtime']:
                raise ValueError('Historical effective settings differ from config')
            reusable = {(r['task'], r['item_id']) for r in group['selected_rows']}
            stages = split_missing(panel, reusable)
            folder = output / alias
            folder.mkdir(parents=True, exist_ok=True)
            entry = {'model_key': alias, 'model_id': group['model_id'], 'revision': group['revision'],
                'compatibility_group': group['compatibility_group'], 'reused_rows': len(reusable),
                'backfill_rows': group['backfill_rows'], 'source_config': str(config_path),
                'source_config_sha256': sha(config_path), 'reused_predictions': [
                    {'path': p, 'sha256': sources[p]['predictions_sha256']} for p in source_paths],
                'selected_reuse_rows': group['selected_rows'], 'stages': {}}
            for stage, keys in stages.items():
                if not keys:
                    entry['stages'][stage] = {'rows': 0, 'status': 'already_reused'}
                    continue
                keys_path = folder / (stage + '_keys.json')
                keys_path.write_text(json.dumps({'panel_sha256': digest(panel), 'keys': keys}, indent=2) + '\n')
                config = build_config(original, alias, digest(panel), keys_path, stage, keys)
                target = folder / (stage + '.yaml')
                target.write_text(yaml.safe_dump(config, sort_keys=False))
                entry['stages'][stage] = {'rows': len(keys), 'config': str(target), 'keys': str(keys_path),
                    'output': str(folder / stage), 'status': 'prepared_not_submitted',
                    'command': f'python3 code/hive_vllm_benchmark.py --config {target} --model-key {alias} --output-dir {folder / stage}'}
            manifest['models'].append(entry)
        except (ValueError, KeyError, OSError) as exc:
            manifest['ineligible'].append({'model_key': alias, 'reason': str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', action='append')
    args = parser.parse_args()
    result = prepare(model_keys=args.model_key)
    print(json.dumps({'models': [{k: g[k] for k in ['model_key', 'reused_rows', 'backfill_rows', 'stages']} for g in result['models']],
                      'ineligible': result['ineligible']}, indent=2))
