"""Audit local historical predictions against the frozen September panel.

No inference or network access. An eligible row must be covered by an intact
checkpoint whose fingerprint proves the same input, prompt, schema and gold.
Run from the repository root: python3 code/audit_refresh_reuse.py
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from hive_vllm_benchmark import task_fingerprint
from task_registry import load_task_definitions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def scalar_equal(left, right):
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(right, (int, float, bool)):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def frames_equivalent(a, b, numeric_label_columns=None):
    """Allow CSV numeric serialization differences, never label/text differences."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for column in a.columns:
        if a[column].equals(b[column]):
            continue
        numeric_label = column in numeric_label_columns if numeric_label_columns is not None else column.startswith(('pred_', 'gt_'))
        if not (numeric_label or column in
                {'latency_s', 'eval_count', 'prompt_tokens'}):
            return False
        if not a[column].eq('').equals(b[column].eq('')):
            return False
        present = a[column].ne('')
        try:
            av = pd.to_numeric(a.loc[present, column], errors='raise')
            bv = pd.to_numeric(b.loc[present, column], errors='raise')
        except ValueError:
            return False
        if column == 'latency_s':
            if not np.allclose(av, bv, rtol=1e-12, atol=1e-12):
                return False
        elif not np.array_equal(av, bv):
            return False
    return True


def metadata_issues(meta, prediction_path):
    issues = []
    if meta.get('status') != 'completed':
        issues.append('run_not_completed')
    if not re.fullmatch(r'[0-9a-f]{40}', str(meta.get('revision', ''))):
        issues.append('immutable_revision_missing')
    if not meta.get('model_id') or not meta.get('model_key'):
        issues.append('checkpoint_identity_missing')
    if meta.get('predictions_sha256') != sha(prediction_path):
        issues.append('predictions_checksum_missing_or_mismatch')
    gen = meta.get('generation', {})
    if gen.get('max_tokens') != 256 or gen.get('temperature') != 0:
        issues.append('incompatible_output_limit_or_temperature')
    if not (gen.get('enable_thinking') is False or gen.get('reasoning_effort') == 'low'):
        issues.append('thinking_settings_not_compatible')
    if gen.get('structured_outputs') != 'json_schema':
        issues.append('structured_decoding_not_documented')
    if not meta.get('runtime_versions') or not meta.get('config_sha256'):
        issues.append('runtime_or_config_provenance_missing')
    return issues


def audit_source(path, tasks, current_items, frozen):
    result = {'source': str(path), 'eligible_keys': [], 'task_checks': {}, 'issues': []}
    metadata_path = path.parent / 'run_metadata.json'
    if not metadata_path.exists():
        result['issues'] = ['run_metadata_missing']
        return result
    meta = json.loads(metadata_path.read_text())
    result.update(model_key=meta.get('model_key'), model_id=meta.get('model_id'),
                  revision=meta.get('revision'), completed_at=meta.get('completed_at'),
                  started_at=meta.get('started_at'),
                  generation=meta.get('generation'), runtime=meta.get('runtime'),
                  quantization=meta.get('quantization'),
                  hardware=meta.get('gpu_after_load'), metadata=str(metadata_path),
                  metadata_sha256=sha(metadata_path), predictions_sha256=sha(path))
    # No cross-runtime or cross-settings combination of historical predictions.
    result['compatibility_group'] = digest({k: meta.get(k) for k in
        ['model_key', 'model_id', 'revision', 'generation', 'runtime', 'runtime_versions', 'llm_kwargs']})
    result['issues'] = metadata_issues(meta, path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    result['source_rows'] = len(frame)
    if not {'task', 'item_id'}.issubset(frame.columns):
        result['issues'].append('row_keys_missing')
        return result
    overlap = {(r.task, r.item_id) for r in frame[['task', 'item_id']].itertuples(index=False)} & set(frozen)
    result['overlapping_keys'] = len(overlap)
    if frame.duplicated(['task', 'item_id']).any():
        result['issues'].append('duplicate_row_keys')
    if result['issues']:
        return result
    for name, group in frame.groupby('task', sort=False):
        if name not in tasks:
            continue
        issues = []
        checkpoint = path.parent / 'tasks' / (name + '.csv')
        record = meta.get('task_results', {}).get(name, {})
        if not checkpoint.exists() or record.get('checkpoint_sha256') != sha(checkpoint):
            issues.append('task_checkpoint_missing_or_checksum_mismatch')
        else:
            saved = pd.read_csv(checkpoint, dtype=str, keep_default_na=False)
            lookup = current_items[name]
            ids = saved['item_id'].tolist()
            if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
                issues.append('task_item_identity_mismatch')
            else:
                items = [lookup[i] for i in ids]
                expected = task_fingerprint(tasks[name], items, meta['model_key'],
                    {'model_id': meta['model_id'], 'revision': meta['revision']},
                    meta['config_sha256'], meta['generation'], meta['runtime'],
                    meta['runtime_versions'], meta.get('panel_sha256'),
                    record.get('panel_task_fingerprint'))
                if expected != record.get('task_fingerprint'):
                    issues.append('input_prompt_schema_or_settings_fingerprint_mismatch')
                # The combined export must actually reproduce the intact checkpoint.
                common = sorted(set(group.columns) & set(saved.columns))
                a = group[common].sort_values('item_id').reset_index(drop=True)
                b = saved[common].sort_values('item_id').reset_index(drop=True)
                numeric_columns = [] if tasks[name]['label_kind'] == 'categorical' else [
                    c for c in common if c.startswith(('pred_', 'gt_'))]
                if not frames_equivalent(a, b, numeric_columns):
                    issues.append('combined_export_differs_from_checkpoint')
        if set(group.get('model_id', [])) != {meta['model_id']} or set(group.get('model_revision', [])) != {meta['revision']}:
            issues.append('row_checkpoint_identity_mismatch')
        eligible = []
        if not issues:
            for _, row in group.iterrows():
                key = (name, row['item_id'])
                if key not in frozen:
                    continue
                target = frozen[key]
                if not all(scalar_equal(row.get('gt_' + k), v) for k, v in target['gt'].items()):
                    issues.append('frozen_gold_mismatch')
                    break
                if current_items[name][row['item_id']] != target:
                    issues.append('frozen_item_differs_from_current_source')
                    break
                eligible.append(list(key))
        result['task_checks'][name] = {'issues': sorted(set(issues)), 'eligible': 0 if issues else len(eligible),
                                      'fingerprint': record.get('task_fingerprint')}
        if not issues:
            result['eligible_keys'].extend(eligible)
    return result


def aggregate(sources, frozen_keys):
    groups = defaultdict(list)
    for source in sources:
        if source.get('compatibility_group') and source.get('eligible_keys'):
            groups[source['compatibility_group']].append(source)
    results = []
    for group_id, runs in sorted(groups.items()):
        selected = {}
        # Prefer the newest compatible run, never accuracy-dependent selection.
        runs.sort(key=lambda x: (x.get('completed_at') or '', x['source']), reverse=True)
        for run in runs:
            for key in run['eligible_keys']:
                selected.setdefault(tuple(key), run['source'])
        missing = sorted(set(frozen_keys) - set(selected))
        results.append({'compatibility_group': group_id, 'model_key': runs[0]['model_key'],
            'model_id': runs[0]['model_id'], 'revision': runs[0]['revision'],
            'eligible_rows': len(selected), 'backfill_rows': len(missing),
            'complete': not missing, 'selected_rows': [dict(task=k[0], item_id=k[1], source=v)
                for k, v in sorted(selected.items())], 'backfill_keys': [list(k) for k in missing]})
    return results


def export_open(report, output_dir):
    """Export one newest complete compatible source group per checkpoint alias."""
    sources = {s['source']: s for s in report['sources']}
    chosen = {}
    for group in report['groups']:
        if not group['complete']:
            continue
        date = max(sources[r['source']].get('completed_at') or '' for r in group['selected_rows'])
        alias = group['model_key']
        if alias not in chosen or date > chosen[alias][0]:
            chosen[alias] = (date, group)
    exports = []
    for alias, (_, group) in sorted(chosen.items()):
        by_source = defaultdict(set)
        for row in group['selected_rows']:
            by_source[row['source']].add((row['task'], row['item_id']))
        frames = []
        for path, keys in sorted(by_source.items()):
            assert sha(path) == sources[path]['predictions_sha256'], 'Source changed after audit'
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame = frame.loc[[key in keys for key in zip(frame.task, frame.item_id)]].copy()
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True).sort_values(['task', 'item_id'])
        assert len(combined) == 3400 and not combined.duplicated(['task', 'item_id']).any()
        path = output_dir / 'open_inputs' / alias / 'predictions.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
        source = sources[next(iter(by_source))]
        model_id = source['model_id']
        quantization = source.get('quantization')
        if not quantization:
            # Only identify formats explicitly named by the immutable checkpoint.
            quantization = next((q for q in ['NVFP4', 'FP8', 'W4A16', 'AWQ'] if q.lower() in model_id.lower()), 'not documented')
        gpus = (source.get('hardware') or {}).get('gpus', [])
        hardware = '; '.join(sorted(set(g.get('name', 'unknown GPU') for g in gpus))) or 'not documented'
        gpu_count = len(gpus)
        settings = dict(source['generation'], **source['runtime'])
        settings['structured_output'] = settings.pop('structured_outputs', None)
        exports.append({'model': alias, 'model_id': model_id, 'predictions': str(path),
            'panel_sha256': report['panel_canonical_sha256'], 'prediction_sha256': sha(path),
            'revision': group['revision'], 'quantization': quantization,
            'hardware': hardware, 'hardware_tier': 'single-gpu' if gpu_count == 1 else 'multi-gpu' if gpu_count > 1 else 'unknown',
            'settings': settings, 'run_dates': sorted(set(
                sources[p][k][:10] for p in by_source for k in ['started_at', 'completed_at'] if sources[p].get(k))),
            'provenance_validated': True, 'compatibility_group': group['compatibility_group'],
            'sources': sorted(by_source), 'throughput_note': 'Historical generation timings cover a different batch/sample size; no matched-sample speed estimate is exported.'})
    manifest = output_dir / 'open_inputs.json'
    manifest.write_text(json.dumps({'models': exports, 'audit': str(output_dir / 'reuse_audit.json')}, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('output/sidecar/refresh_20260910/panel.json'))
    parser.add_argument('--search-root', type=Path, default=Path('output'))
    parser.add_argument('--output', type=Path, default=Path('output/sidecar/refresh_20260910_release/reuse_audit.json'))
    parser.add_argument('--export-open', action='store_true', help='Export proven complete matched open-model inputs')
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    frozen = {(r['task'], str(r['item']['item_id'])): r['item'] for r in panel['rows']}
    assert panel['seed'] == 20260910 and len(frozen) == 3400
    tasks = {t['name']: t for t in load_task_definitions()}
    current = {n: {str(i['item_id']): i for i in t['loader']()} for n, t in tasks.items()}
    assert len(tasks) == 34 and all(sum(k[0] == n for k in frozen) == 100 for n in tasks)
    for name, task in tasks.items():
        assert digest([list(current[name].values()), Path(task['prompt_path']).read_text(), task['json_schema']]) == panel['sources'][name], name
    paths = sorted(p for p in args.search_root.rglob('predictions*.csv')
                   if 'refresh_20260910' not in p.parts and 'refresh_20260910_release' not in p.parts)
    sources = []
    for path in paths:
        try:
            sources.append(audit_source(path, tasks, current, frozen))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            sources.append({'source': str(path), 'issues': ['audit_error:' + str(exc)], 'eligible_keys': []})
    report = {'schema_version': 1, 'panel': str(args.panel), 'panel_sha256': sha(args.panel),
              'panel_canonical_sha256': digest(panel),
              'seed': panel['seed'], 'expected_rows_per_model': 3400,
              'policy': 'Exact task fingerprints and file checksums; no cross-settings/runtime combination; newest compatible prediction, never quality-selected.',
              'sources': sources, 'groups': aggregate(sources, frozen)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if args.export_open:
        export_open(report, args.output.parent)
    print(json.dumps({'sources_audited': len(sources), 'groups': [
        {k: v for k, v in g.items() if k not in ['selected_rows', 'backfill_keys']}
        for g in report['groups']]}, indent=2))


if __name__ == '__main__':
    main()
