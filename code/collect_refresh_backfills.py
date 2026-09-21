"""Validate and combine exact historical reuse with completed Hive backfills.

Run from the repository root. This reads local artifacts only. A partial or
settings-incompatible backfill produces a validation report, never an entry in
the ranked open-model manifest.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audit_refresh_reuse import digest, sha
from build_refresh_release import validate_panel
from collect_refresh_hive import audit_stage, gpu_hours
from hive_vllm_benchmark import load_config
from prepare_refresh_backfills import split_missing
from task_registry import load_task_definitions


PANEL = Path('output/sidecar/refresh_20260910/panel.json')
MANIFEST = Path('output/sidecar/refresh_20260910_release/backfills/manifest.json')
AUDIT = Path('output/sidecar/refresh_20260910_release/reuse_audit.json')


def keys(frame):
    return list(zip(frame.task, frame.item_id))


def verify_stage_keys(entry, panel, reusable):
    """Keep the approved pilot/remainder split, with no unseen or omitted key."""
    expected = split_missing(panel, reusable)
    for stage, wanted in expected.items():
        spec = entry['stages'][stage]
        if spec['rows'] != len(wanted):
            raise ValueError(stage + ' manifest row count differs from frozen missing keys')
        if not wanted:
            if spec.get('status') != 'already_reused':
                raise ValueError(stage + ' empty stage is not labeled already reused')
            continue
        path = Path(spec['keys'])
        payload = json.loads(path.read_text())
        if payload.get('panel_sha256') != digest(panel) or payload.get('keys') != wanted:
            raise ValueError(stage + ' key manifest differs from frozen missing keys')
        config = load_config(Path(spec['config']))
        selection = config.get('refresh_sample', {})
        if (selection.get('stage') != 'backfill' or selection.get('panel_sha256') != digest(panel)
                or selection.get('keys_manifest') != str(path) or selection.get('keys_sha256') != sha(path)):
            raise ValueError(stage + ' configuration does not pin the exact key manifest')
    return expected


def reused_frame(entry, audit, panel):
    sources = {s['source']: s for s in audit['sources']}
    by_source = defaultdict(set)
    for row in entry['selected_reuse_rows']:
        by_source[row['source']].add((row['task'], str(row['item_id'])))
    if sum(map(len, by_source.values())) != entry['reused_rows']:
        raise ValueError('Duplicate selected historical reuse key')
    frozen = {(r['task'], str(r['item']['item_id'])) for r in panel['rows']}
    if not set.union(set(), *by_source.values()) <= frozen:
        raise ValueError('Selected historical key is outside frozen panel')
    if set(by_source) != {p['path'] for p in entry['reused_predictions']}:
        raise ValueError('Historical source list differs from selected row sources')
    frames, metadata = [], []
    for source in entry['reused_predictions']:
        path = source['path']
        info = sources[path]
        if source['sha256'] != info['predictions_sha256'] or sha(path) != source['sha256']:
            raise ValueError('Historical predictions changed after reuse audit: ' + path)
        if sha(info['metadata']) != info['metadata_sha256']:
            raise ValueError('Historical metadata changed after reuse audit: ' + path)
        meta = json.loads(Path(info['metadata']).read_text())
        if (meta['model_key'] != entry['model_key'] or meta['model_id'] != entry['model_id']
                or meta['revision'] != entry['revision'] or info['compatibility_group'] != entry['compatibility_group']):
            raise ValueError('Historical checkpoint identity or compatibility group changed')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        selected = frame.loc[[key in by_source[path] for key in keys(frame)]].copy()
        if set(keys(selected)) != by_source[path] or selected.duplicated(['task', 'item_id']).any():
            raise ValueError('Historical selected keys absent or duplicated: ' + path)
        frames.append(selected)
        metadata.append(meta)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != entry['reused_rows'] or combined.duplicated(['task', 'item_id']).any():
        raise ValueError('Historical reuse selection has duplicate or missing rows')
    return combined, metadata


def comparable_runtime(historical, new):
    """The immutable package lock matters; driver/node readings may differ."""
    for key in ['packages', 'runtime_lock', 'torch_cuda', 'nvcc']:
        if historical['runtime_versions'].get(key) != new.get(key):
            raise ValueError('Backfill runtime differs from historical source: ' + key)
    if historical['runtime_versions'].get('python', '')[:4] != new.get('python', '')[:4]:
        raise ValueError('Backfill Python minor version differs from historical source')


def comparable_effective_model(historical, new):
    old, current = historical['llm_kwargs'], new['llm_kwargs']
    if old != current:
        raise ValueError('Backfill effective model/decoding settings differ from historical source')


def collect_model(entry, audit, panel, stage_dirs, output, sacct=None):
    historical, source_meta = reused_frame(entry, audit, panel)
    reusable = set(keys(historical))
    expected = verify_stage_keys(entry, panel, reusable)
    original_config = Path(entry['source_config'])
    if sha(original_config) != entry['source_config_sha256']:
        raise ValueError('Exact historical configuration changed')
    original = load_config(original_config)
    model_key = entry['model_key']
    if original['models'][model_key]['revision'] != entry['revision']:
        raise ValueError('Original configuration checkpoint revision differs')
    stages, new_frames, new_meta = {}, [], []
    for stage in ['pilot', 'remainder']:
        spec = entry['stages'][stage]
        if not expected[stage]:
            continue
        config_path = Path(spec['config'])
        config = load_config(config_path)
        if (config['models'][model_key] != original['models'][model_key]
                or config['generation'] != original['generation'] or config['runtime'] != original['runtime']):
            raise ValueError(stage + ' requested inference settings differ from original run')
        directory = Path(stage_dirs[stage])
        if not (directory / 'run_metadata.json').exists():
            stages[stage] = {'passed': False, 'directory': str(directory),
                             'issues': ['run_metadata_missing'], 'failure_types': ['infrastructure']}
            return {'complete': False, 'model': model_key, 'stages': stages,
                    'reason': stage + ' stage has not produced run metadata'}, None
        report, frame = audit_stage(config_path, directory)
        stages[stage] = report
        if not report['passed'] or frame is None:
            return {'complete': False, 'model': model_key, 'stages': stages,
                    'reason': stage + ' stage is absent, incomplete or failed validation'}, None
        if set(keys(frame)) != {tuple(k) for k in expected[stage]} or len(frame) != spec['rows']:
            raise ValueError(stage + ' output covers different frozen missing keys')
        meta = json.loads((directory / 'run_metadata.json').read_text())
        for source in source_meta:
            comparable_runtime(source, meta['runtime_versions'])
            comparable_effective_model(source, meta)
        new_frames.append(frame)
        new_meta.append(meta)
    combined = pd.concat([historical] + new_frames, ignore_index=True).sort_values(['task', 'item_id'])
    if len(combined) != len(panel['rows']) or combined.duplicated(['task', 'item_id']).any():
        raise ValueError('Historical plus backfill rows do not form one exact frozen panel')
    validate_panel(combined, panel, {t['name']: t for t in load_task_definitions()})
    all_meta = source_meta + new_meta
    gpu_sets = [(m.get('gpu_after_load') or {}).get('gpus', []) for m in all_meta]
    gpu_counts = {len(g) for g in gpu_sets}
    if len(gpu_counts) != 1 or 0 in gpu_counts:
        raise ValueError('Historical and backfill GPU allocations differ or are undocumented')
    quantizations = {m.get('quantization') or 'not documented' for m in all_meta}
    if len(quantizations) != 1:
        raise ValueError('Historical and backfill quantization labels differ')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'predictions.csv'
    model = original['models'][model_key]
    settings = dict(original['generation'], **original['runtime'],
                    **model.get('chat_template_kwargs', {}), **model.get('llm_kwargs', {}))
    settings['structured_output'] = settings.pop('structured_outputs', None)
    settings['tokenizer_revision'] = entry['revision']
    remainder_report = stages.get('remainder', {})
    remainder_seconds = remainder_report.get('generation_seconds')
    remainder_unknown_usage = remainder_report.get('output_token_usage_unavailable', [])
    comparable_rate = (len(expected['remainder']) / remainder_seconds
                       if remainder_seconds and not remainder_unknown_usage else None)
    remainder_gpus = (remainder_report.get('hardware') or {}).get('gpus', [])
    result = {'model': model_key, 'model_id': entry['model_id'], 'predictions': str(target),
              'panel_sha256': digest(panel),
              'revision': entry['revision'], 'quantization': quantizations.pop(),
              'hardware': '; '.join(sorted({g.get('name', 'unknown GPU') for gs in gpu_sets for g in gs})),
              'hardware_tier': 'single-gpu' if next(iter(gpu_counts)) == 1 else 'multi-gpu',
              'settings': settings,
              'run_dates': sorted({m[k][:10] for m in all_meta for k in ['started_at', 'completed_at'] if m.get(k)}),
              'provenance_validated': True, 'compatibility_group': entry['compatibility_group'],
              'sources': sorted({r['source'] for r in entry['selected_reuse_rows']}),
              'generation_items_per_second': comparable_rate,
              'generation_tokens_per_second': (remainder_report.get('generation_tokens_per_second')
                  if comparable_rate is not None else None),
              'throughput_keyset_sha256': digest(expected['remainder']),
              'throughput_settings_sha256': digest({
                  'generation': original['generation'],
                  'runtime': original['runtime'],
                  'max_num_seqs': model.get('llm_kwargs', {}).get('max_num_seqs', 'runtime_default'),
                  'max_num_batched_tokens': model.get('llm_kwargs', {}).get(
                      'max_num_batched_tokens', 'runtime_default')}),
              'throughput_items': len(expected['remainder']),
              'throughput_hardware': '; '.join(sorted({g['name'] for g in remainder_gpus})),
              'throughput_note': 'Generation rate covers only the frozen missing-item remainder '
                  'on Hive; model load, historical reused rows and queue time are excluded. '
                  'Compare rates only for the same keyset, generation/runtime and batch '
                  'concurrency settings, and GPU hardware.'}
    if sacct:
        ledger_path = Path(sacct)
        ledger = json.loads(ledger_path.read_text())
        source_lookup = {row['source']: row for row in audit['sources']}
        source_hashes = [sha(source_lookup[row['path']]['metadata'])
                         for row in entry['reused_predictions']]
        expected_jobs = {str(attempt['slurm_job_id']) for meta in all_meta
                         for attempt in meta.get('attempts', [])}
        recorded_jobs = {str(row['job_id']) for row in ledger.get('attempts', [])}
        if (ledger.get('model') != model_key or ledger.get('model_id') != entry['model_id']
                or ledger.get('revision') != entry['revision']
                or ledger.get('source_metadata_sha256') != source_hashes
                or ledger.get('pilot_metadata_sha256') != sha(Path(stage_dirs['pilot'])/'run_metadata.json')
                or ledger.get('remainder_metadata_sha256') != sha(Path(stage_dirs['remainder'])/'run_metadata.json')
                or recorded_jobs != expected_jobs):
            raise ValueError('sacct ledger identity, metadata or attempt coverage differs')
        hours = gpu_hours(ledger, expected_jobs)
        reported = ledger.get('gpu_hours', {})
        if hours is None or not math.isclose(hours, float(ledger.get('gpu_hours_total', -1)),
                                             rel_tol=0, abs_tol=1e-9):
            raise ValueError('sacct total GPU hours differs from recorded attempts')
        if any(scope not in reported or not math.isclose(
                float(reported[scope]),
                sum(row['elapsed_seconds']*row['gpu_count']/3600
                    for row in ledger['attempts'] if row.get('scope') == scope),
                rel_tol=0, abs_tol=1e-9) for scope in ['historical','pilot','remainder']):
            raise ValueError('sacct stage GPU hours differ from recorded attempts')
        result.update(gpu_hours=hours,
                      gpu_hours_historical=float(reported['historical']),
                      gpu_hours_backfill=float(reported['pilot'])+float(reported['remainder']),
                      gpu_hours_note=ledger['note'],
                      sacct_attempts_sha256=sha(ledger_path))
    combined.to_csv(target, index=False)
    result['prediction_sha256'] = sha(target)
    return {'complete': True, 'model': model_key, 'reused_rows': len(historical),
            'backfill_rows': sum(len(f) for f in new_frames), 'rows': len(combined), 'stages': stages}, result


def collect(model_key, pilot_dir=None, remainder_dir=None, output=None, sacct=None,
            manifest_path=MANIFEST, audit_path=AUDIT, panel_path=PANEL):
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    panel = json.loads(Path(panel_path).read_text())
    if (manifest['panel_sha256'] != digest(panel) or
            manifest['reuse_audit_sha256'] != sha(audit_path) or
            audit['panel_sha256'] != sha(panel_path) or
            audit['panel_canonical_sha256'] != digest(panel)):
        raise ValueError('Frozen panel or reuse audit changed since backfill preparation')
    selected = [e for e in manifest['models'] if e['model_key'] == model_key]
    if len(selected) != 1:
        raise ValueError('Model not uniquely present in approved backfill manifest')
    entry = selected[0]
    destination = Path(output or Path(manifest_path).parent / model_key / 'collected')
    stage_dirs = {'pilot': pilot_dir or entry['stages']['pilot'].get('output'),
                  'remainder': remainder_dir or entry['stages']['remainder'].get('output')}
    if any(entry['stages'][stage]['rows'] and not stage_dirs[stage] for stage in stage_dirs):
        raise ValueError('Missing stage directory')
    report, export = collect_model(entry, audit, panel, stage_dirs, destination, sacct=sacct)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'validation.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    if export is not None:
        (destination / 'open_inputs.json').write_text(json.dumps({'models': [export]}, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--pilot-dir', type=Path)
    parser.add_argument('--remainder-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sacct', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args)), indent=2, default=str))
