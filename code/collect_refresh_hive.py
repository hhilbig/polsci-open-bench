"""Validate collected Hive refresh artifacts and export completed open models.

No inference, remote calls, or retries. Legacy previews >=200 characters are
unverifiable; they are never treated as complete responses. Optional sacct JSON:
{"complete":true,"source":"sacct","attempts":[{"job_id":"...","state":"COMPLETED",
"elapsed_seconds":123,"gpu_count":1}]}. Supply all attempts across both stages.
"""
import argparse
import json
import math
from collections import Counter
from io import StringIO
from pathlib import Path

import pandas as pd

from audit_refresh_reuse import sha, digest, scalar_equal, frames_equivalent
from hive_vllm_benchmark import (load_config, selected_tasks, task_fingerprint,
                                 assert_runtime_versions, resolved_chat_template_kwargs)


def strict_response(row, task):
    if 'raw_content' in row and pd.notna(row['raw_content']):
        text = str(row['raw_content'])
        provenance = 'full_raw_content'
    else:
        text = str(row.get('raw_content_preview', ''))
        provenance = 'complete_legacy_preview'
        if len(text) >= 200:
            return None, 'unverifiable_preview', 'legacy_preview_at_limit'
    try:
        obj = json.loads(text)
        expected = task['labels'] if task['label_kind'] == 'multi_binary' else [task['label_key']]
        if not isinstance(obj, dict) or set(obj) != set(expected):
            raise ValueError('Missing or unexpected fields')
        if task['label_kind'] == 'categorical':
            if obj[task['label_key']] not in task['labels']:
                raise ValueError('Invalid categorical label')
        elif any(type(v) is not int or v not in (0, 1) for v in obj.values()):
            raise ValueError('Binary fields must be integer 0 or 1')
    except (ValueError, TypeError) as exc:
        return None, 'schema_invalid: ' + str(exc), provenance
    return obj, None, provenance


def output_token_usage_status(row):
    """Flag generated responses for which the endpoint returned no usage."""
    status = str(row.get('token_usage_status', 'reported') or 'reported')
    if status not in {'reported', 'unavailable'}:
        raise ValueError('Invalid output token usage status')
    if status == 'unavailable':
        if (str(row.get('finish_reason')) != 'failed'
                or not str(row.get('parse_error', '')).startswith('harmony_')
                or str(row.get('eval_count')) not in {'0', '0.0'}):
            raise ValueError('Unreported output usage must be a retained Harmony failure')
    return status


def gpu_hours(ledger, job_ids):
    if not ledger or ledger.get('complete') is not True or ledger.get('source') != 'sacct':
        return None
    if not job_ids:
        return None
    attempts = ledger.get('attempts', [])
    observed = [str(a.get('job_id')) for a in attempts]
    if len(observed) != len(set(observed)) or not set(job_ids) <= set(observed):
        raise ValueError('sacct ledger must cover all recorded attempt job IDs without duplicates')
    total = 0.
    for a in attempts:
        seconds, count = float(a['elapsed_seconds']), float(a['gpu_count'])
        if not math.isfinite(seconds) or seconds < 0 or not count.is_integer() or count < 1 or a['state'] not in {
            'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED', 'OUT_OF_MEMORY', 'NODE_FAIL'}:
            raise ValueError('Invalid or nonterminal sacct attempt')
        total += seconds * count / 3600
    return total


def checked_sacct_hours(path, job_ids, failed_job_ids=()):
    """Attribute GPU time to audited stages and named pre-inference failures."""
    ledger = json.loads(Path(path).read_text())
    observed = {str(row.get('job_id')) for row in ledger.get('attempts', [])}
    completed = {str(job_id) for job_id in job_ids}
    failed = {str(job_id) for job_id in failed_job_ids}
    if completed & failed:
        raise ValueError('Pre-inference failure is also a completed stage attempt')
    expected = completed | failed
    if observed != expected:
        raise ValueError('sacct ledger differs from exact refresh attempt IDs')
    if any(row.get('state') not in {'FAILED', 'CANCELLED', 'TIMEOUT',
                                   'OUT_OF_MEMORY', 'NODE_FAIL'}
           for row in ledger['attempts'] if str(row['job_id']) in failed):
        raise ValueError('Named pre-inference attempt did not fail')
    hours = gpu_hours(ledger, expected)
    if hours is None:
        raise ValueError('sacct ledger is incomplete')
    return hours


def hardware_label(gpus):
    """Retain GPU count as well as GPU model for multi-GPU candidates."""
    names = Counter(str(gpu['name']) for gpu in gpus)
    if not names:
        raise ValueError('GPU hardware identity is unavailable')
    if len(gpus) == 1:
        return next(iter(names))
    return '; '.join(f'{count} × {name}' for name, count in sorted(names.items()))


def generation_rates(items, seconds, output_tokens, usage_unavailable=False):
    """Keep item throughput when a retained failure has no output-token usage."""
    if items < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Generation item count or timing is invalid')
    return (items / seconds,
            None if usage_unavailable else output_tokens / seconds)


def audit_stage(config_path, directory):
    config_path, directory = Path(config_path), Path(directory)
    config = load_config(config_path)
    meta = json.loads((directory / 'run_metadata.json').read_text())
    report = {'directory': str(directory), 'config': str(config_path), 'status': meta.get('status'),
              'issues': [], 'schema_failures': [], 'truncations': [], 'unverifiable_responses': [],
              'output_token_usage_unavailable': [],
              'raw_provenance_counts': {}, 'failure_types': [], 'passed': False}
    if meta.get('status') != 'completed':
        report['issues'].append('runtime_not_completed: ' + str(meta.get('error', meta.get('status'))))
        report['failure_types'] = ['runtime']
        return report, None
    try:
        model_key = meta['model_key']
        model = config['models'][model_key]
        for field, expected in [('config_sha256', sha(config_path)), ('revision', model['revision']),
            ('model_id', model['model_id']), ('generation', config['generation']), ('runtime', config['runtime'])]:
            if meta.get(field) != expected:
                raise ValueError('Metadata mismatch: ' + field)
        assert_runtime_versions(config['runtime'], meta['runtime_versions'])
        runtime = config['runtime']
        if runtime.get('required_environment') or runtime.get('container_image_sha256'):
            observed = meta.get('refresh_runtime_requirements', {})
            if (observed.get('required_environment') != runtime.get('required_environment', {})
                    or observed.get('container_image_sha256') != runtime.get('container_image_sha256')):
                raise ValueError('Pinned container image or offload settings were not verified')
        if meta.get('chat_template_kwargs') != resolved_chat_template_kwargs(model):
            raise ValueError('Chat template settings mismatch')
        effective = meta.get('llm_kwargs', {})
        if model.get('inference_interface') == 'vllm_responses_harmony':
            if (effective.get('served_model_name') != model['model_id'] or
                    Path(str(effective.get('model', ''))).name != model['revision'] or
                    meta.get('provenance_status') != 'qualified'):
                raise ValueError('Harmony effective snapshot identity not verified')
        else:
            if (effective.get('model') != model['model_id'] or effective.get('revision') != model['revision']
                    or effective.get('tokenizer_revision') != model['revision']
                    or effective.get('seed') != config['generation']['seed']):
                raise ValueError('Effective model/tokenizer/seed differs from pinned configuration')
            for key, value in model.get('llm_kwargs', {}).items():
                if effective.get(key) != value:
                    raise ValueError('Effective inference setting mismatch: ' + key)
        gpus = meta.get('gpu_after_load', {}).get('gpus', [])
        if len(gpus) != int(model.get('llm_kwargs', {}).get('tensor_parallel_size', 1)):
            raise ValueError('Allocated GPU count differs from configured tensor parallelism')
        peak = meta.get('observed_peak_gpu_memory_used_mib')
        if peak is None or not gpus or float(peak) > sum(float(g['memory_total_mib']) for g in gpus):
            raise ValueError('Peak memory is missing or exceeds allocated capacity')
        if sha(directory / 'predictions.csv') != meta.get('predictions_sha256'):
            raise ValueError('Combined predictions checksum mismatch')
        tasks = selected_tasks(config)
        task_map = {t['name']: t for t in tasks}
        expected = {(t['name'], str(i['item_id'])): i for t in tasks for i in t['loader']()}
        frame = pd.read_csv(directory / 'predictions.csv', dtype=str, keep_default_na=False)
        actual = list(zip(frame.task, frame.item_id))
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError('Exact frozen stage coverage failed')
        for t in tasks:
            name = t['name']
            done = json.loads((directory / 'tasks' / (name + '.done.json')).read_text())
            record = meta['task_results'][name]
            if done != record:
                raise ValueError('Task done record differs from metadata: ' + name)
            checkpoint = directory / 'tasks' / (name + '.csv')
            if sha(checkpoint) != record.get('checkpoint_sha256'):
                raise ValueError('Task checkpoint checksum mismatch: ' + name)
            fingerprint = task_fingerprint(t, t['loader'](), model_key, model, sha(config_path),
                config['generation'], config['runtime'], meta['runtime_versions'])
            if fingerprint != record.get('task_fingerprint'):
                raise ValueError('Input/prompt/schema/settings fingerprint mismatch: ' + name)
            saved = pd.read_csv(checkpoint, dtype=str, keep_default_na=False)
            group = frame.loc[frame.task == name]
            columns = sorted(set(saved.columns) & set(group.columns))
            numeric = [] if t['label_kind'] == 'categorical' else [c for c in columns if c.startswith(('gt_', 'pred_'))]
            if not frames_equivalent(group[columns].sort_values('item_id').reset_index(drop=True),
                saved[columns].sort_values('item_id').reset_index(drop=True), numeric):
                raise ValueError('Combined export/checkpoint mismatch: ' + name)
        for index, row in frame.iterrows():
            key = (row['task'], row['item_id'])
            if row['model_id'] != model['model_id'] or row['model_revision'] != model['revision'] or row['model'] != model_key:
                raise ValueError('Per-row model identity mismatch')
            for label, gold in expected[key]['gt'].items():
                if not scalar_equal(row.get('gt_' + label), gold):
                    raise ValueError('Frozen gold mismatch')
            obj, error, raw_provenance = strict_response(row, task_map[key[0]])
            report['raw_provenance_counts'][raw_provenance] = report['raw_provenance_counts'].get(raw_provenance, 0) + 1
            if error == 'unverifiable_preview':
                report['unverifiable_responses'].append(list(key))
            elif error:
                report['schema_failures'].append({'key': list(key), 'error': error})
                # Derived export marks strict failures; original checkpoints remain intact.
                frame.loc[index, 'parse_error'] = error
            elif any(not scalar_equal(row.get('pred_' + label), value) for label, value in obj.items()):
                raise ValueError('Parsed prediction differs from complete raw response')
            if row.get('finish_reason') in {'length', 'max_tokens'}:
                report['truncations'].append(list(key))
            if output_token_usage_status(row) == 'unavailable':
                report['output_token_usage_unavailable'].append(list(key))
            for column in ['eval_count', 'prompt_tokens']:
                value = float(row[column])
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError('Invalid token usage')
            if float(row['eval_count']) > config['generation']['max_tokens']:
                raise ValueError('Output tokens exceed recorded allowance')
        seconds = sum(float(r['generation_seconds']) for r in meta['task_results'].values())
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Generation timing is missing or nonpositive')
        out_tokens = int(pd.to_numeric(frame.eval_count).sum())
        jobs = sorted({str(a['slurm_job_id']) for a in meta.get('attempts', []) if a.get('slurm_job_id')})
        report.update(model=model_key, model_id=model['model_id'], revision=model['revision'],
            expected_rows=len(expected), rows=len(frame), tasks=len(tasks),
            input_tokens=int(pd.to_numeric(frame.prompt_tokens).sum()), output_tokens=out_tokens,
            generation_seconds=seconds,
            generation_tokens_per_second=(out_tokens / seconds if seconds > 0
                and not report['output_token_usage_unavailable'] else None),
            peak_gpu_memory_mib=meta.get('observed_peak_gpu_memory_used_mib'),
            hardware=meta.get('gpu_after_load'), job_ids=jobs,
            settings={'generation': config['generation'], 'runtime': config['runtime'], 'model': model},
            started_at=meta.get('started_at'), completed_at=meta.get('completed_at'),
            schema_valid=not report['schema_failures'])
        report['output_token_usage_note'] = ('Reported output tokens exclude retained Harmony '
            'failures whose endpoint returned no usage; token throughput is unavailable for '
            'this stage.' if report['output_token_usage_unavailable'] else
            'All output token counts are available.')
        report['passed'] = not report['truncations'] and not report['unverifiable_responses']
        report['failure_types'] = [name for name, present in [('schema', report['schema_failures']),
            ('truncation', report['truncations']), ('unverifiable_raw', report['unverifiable_responses'])] if present]
        report['malformed_handling'] = 'Retain and score strict schema failures incorrect, without retries.'
        return report, frame
    except (KeyError, ValueError, TypeError, OSError) as exc:
        report['issues'].append(str(exc))
        report['failure_types'].append('provenance_or_artifact_integrity')
        return report, None


def collect(pilot_config, pilot_dir, output, remainder_config=None, remainder_dir=None,
            sacct=None, failed_attempt_id=None, pilot_attempt_id=None,
            remainder_attempt_id=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pilot, pf = audit_stage(pilot_config, pilot_dir)
    explicit_stage_ids = {
        'pilot': [str(job_id) for job_id in (pilot_attempt_id or [])],
        'remainder': [str(job_id) for job_id in (remainder_attempt_id or [])],
    }
    for scope, stage in [('pilot', pilot)]:
        if explicit_stage_ids[scope]:
            if stage.get('job_ids'):
                raise ValueError(scope + ' explicit attempt duplicates recorded metadata')
            if any(not job_id.isdigit() for job_id in explicit_stage_ids[scope]):
                raise ValueError(scope + ' explicit attempt lacks a numeric Slurm job ID')
            stage['job_ids'] = explicit_stage_ids[scope]
            stage['job_id_provenance'] = 'explicit_registry_handle'
    report = {'pilot': pilot, 'complete': False}
    if remainder_config and remainder_dir:
        remainder, rf = audit_stage(remainder_config, remainder_dir)
        if explicit_stage_ids['remainder']:
            if remainder.get('job_ids'):
                raise ValueError('remainder explicit attempt duplicates recorded metadata')
            if any(not job_id.isdigit() for job_id in explicit_stage_ids['remainder']):
                raise ValueError('remainder explicit attempt lacks a numeric Slurm job ID')
            remainder['job_ids'] = explicit_stage_ids['remainder']
            remainder['job_id_provenance'] = 'explicit_registry_handle'
        report['remainder'] = remainder
        if pilot['passed'] and remainder['passed']:
            if pilot['settings'] != remainder['settings'] or pilot['revision'] != remainder['revision']:
                raise ValueError('Pilot/remainder settings differ; do not combine')
            pilot_gpus = (pilot.get('hardware') or {}).get('gpus', [])
            remainder_gpus = (remainder.get('hardware') or {}).get('gpus', [])
            if sorted(g['name'] for g in pilot_gpus) != sorted(g['name'] for g in remainder_gpus):
                raise ValueError('Pilot/remainder GPU hardware differs; do not combine')
            combined = pd.concat([pf, rf], ignore_index=True)
            from build_refresh_release import validate_panel
            from task_registry import load_task_definitions
            panel = json.loads(Path('output/sidecar/refresh_20260910/panel.json').read_text())
            # Validate numeric CSV labels through the same release reader.
            destination = output / 'predictions.csv'
            serialized = combined.to_csv(index=False)
            validate_panel(pd.read_csv(StringIO(serialized), dtype={'item_id': str}), panel,
                {t['name']: t for t in load_task_definitions()})
            destination.write_text(serialized)
            model = pilot['settings']['model']
            gpus = pilot_gpus
            item_rate, token_rate = generation_rates(
                len(combined), pilot['generation_seconds'] + remainder['generation_seconds'],
                pilot['output_tokens'] + remainder['output_tokens'],
                bool(pilot['output_token_usage_unavailable']
                     or remainder['output_token_usage_unavailable']))
            quantization = model.get('quantization') or next((q for q in ['NVFP4', 'FP8', 'W4A16', 'AWQ']
                if q.lower() in model['model_id'].lower()), 'not documented')
            entry = {'model': pilot['model'], 'model_id': pilot['model_id'], 'predictions': str(destination), 'panel_sha256': digest(panel),
                'prediction_sha256': sha(destination), 'revision': pilot['revision'],
                'quantization': quantization,
                'hardware': hardware_label(gpus),
                'hardware_tier': 'single-gpu' if len(gpus) == 1 else 'multi-gpu',
                'settings': dict(pilot['settings']['generation'], **pilot['settings']['runtime'],
                    **model.get('chat_template_kwargs', {}),
                    structured_output=pilot['settings']['generation']['structured_outputs'],
                    tokenizer_revision=model['revision'], **model.get('llm_kwargs', {})),
                'run_dates': sorted({s[k][:10] for s in [pilot, remainder] for k in ['started_at', 'completed_at'] if s.get(k)}),
                'provenance_validated': True,
                'generation_items_per_second': item_rate,
                'generation_tokens_per_second': token_rate,
                'throughput_keyset_sha256': digest(sorted(zip(combined.task, combined.item_id))),
                'throughput_settings_sha256': digest({
                    'generation': pilot['settings']['generation'],
                    'runtime': pilot['settings']['runtime'],
                    'max_num_seqs': model.get('llm_kwargs', {}).get('max_num_seqs', 'runtime_default'),
                    'max_num_batched_tokens': model.get('llm_kwargs', {}).get(
                        'max_num_batched_tokens', 'runtime_default')}),
                'throughput_items': len(combined),
                'throughput_hardware': hardware_label(gpus),
                'throughput_note': 'Generation rate covers the full frozen 3,400-item panel on '
                    'Hive; model load and queue time are excluded. Compare only with the same '
                    'keyset, generation/runtime and batch concurrency settings, and GPU hardware.',
                'output_token_usage_unavailable_count': (len(pilot['output_token_usage_unavailable'])
                    + len(remainder['output_token_usage_unavailable']))}
            if sacct:
                ledger = json.loads(Path(sacct).read_text())
                stage_hashes = ('pilot_metadata_sha256', 'remainder_metadata_sha256')
                if pilot['settings']['runtime'].get('container_image_sha256') and any(
                        field not in ledger for field in stage_hashes):
                    raise ValueError('Dedicated-image sacct ledger lacks stage metadata hashes')
                if any(field in ledger for field in stage_hashes):
                    if (ledger.get('model') != pilot['model']
                            or ledger.get('model_id') != pilot['model_id']
                            or ledger.get('revision') != pilot['revision']
                            or ledger.get('pilot_metadata_sha256') != sha(Path(pilot_dir)/'run_metadata.json')
                            or ledger.get('remainder_metadata_sha256') != sha(Path(remainder_dir)/'run_metadata.json')
                            or set(ledger.get('failed_preinference_ids', [])) != set(failed_attempt_id or ())):
                        raise ValueError('sacct ledger checkpoint or stage metadata differs')
                entry['gpu_hours'] = checked_sacct_hours(
                    sacct, set(pilot['job_ids']) | set(remainder['job_ids']),
                    failed_attempt_id or ())
                if 'gpu_hours_total' in ledger and not math.isclose(
                        entry['gpu_hours'], float(ledger['gpu_hours_total']),
                        rel_tol=0, abs_tol=1e-9):
                    raise ValueError('sacct total GPU hours differs from recorded attempts')
                entry['preinference_failed_attempt_ids'] = sorted(failed_attempt_id or [])
                entry['gpu_hours_note'] = ledger.get('note') or ('Allocated Hive GPU time includes every recorded '
                    'pilot and full-panel attempt, including pre-inference failures. It is '
                    'separate from generation throughput and is not an economic cost estimate.')
                if 'gpu_hours' in ledger:
                    for field, scope in [('gpu_hours_preinference','preinference_failure'),
                                         ('gpu_hours_pilot','pilot'),
                                         ('gpu_hours_remainder','remainder')]:
                        entry[field] = float(ledger['gpu_hours'][scope])
                entry['sacct_attempts_sha256'] = sha(sacct)
                entry['attempt_id_provenance'] = {
                    'pilot': pilot.get('job_id_provenance', 'stage_metadata'),
                    'remainder': remainder.get('job_id_provenance', 'stage_metadata')}
            (output / 'open_inputs.json').write_text(json.dumps({'models': [entry]}, indent=2) + '\n')
            report['complete'] = True
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-config', required=True, type=Path)
    parser.add_argument('--pilot-dir', required=True, type=Path)
    parser.add_argument('--remainder-config', type=Path)
    parser.add_argument('--remainder-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sacct', type=Path)
    parser.add_argument('--failed-attempt-id', action='append', default=[],
                        help='Registry-recorded pre-inference failure absent from stage metadata')
    parser.add_argument('--pilot-attempt-id', action='append', default=[],
                        help='Completed pilot Slurm ID when absent inside container metadata')
    parser.add_argument('--remainder-attempt-id', action='append', default=[],
                        help='Completed remainder Slurm ID when absent inside container metadata')
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args)), indent=2))
