"""Record allocated Hive GPU time for audited historical reuse and backfills.

This measures all recorded source and backfill attempts, including failed
attempts. Historical source jobs processed more than the frozen 3,400-item
panel, so their GPU time is provenance, not a matched-sample speed or cost.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from audit_refresh_reuse import sha
from collect_refresh_backfills import MANIFEST, AUDIT, PANEL, reused_frame


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED',
            'OUT_OF_MEMORY', 'NODE_FAIL'}


def attempt_ids(metadata, scope):
    if metadata.get('status') != 'completed':
        raise ValueError(scope + ' metadata is not completed')
    ids = []
    for attempt in metadata.get('attempts', []):
        job_id = str(attempt.get('slurm_job_id') or '')
        if not re.fullmatch(r'[0-9]+', job_id):
            raise ValueError(scope + ' attempt lacks a numeric Slurm job ID')
        ids.append(job_id)
    if not ids:
        raise ValueError(scope + ' metadata has no recorded Slurm attempts')
    return ids


def refresh_attempt_ids(metadata, scope, explicit_ids=()):
    """Use explicit stage IDs only when the container recorded none.

    Dedicated Apptainer images may not expose Slurm environment variables to
    the runner.  In that case the immutable stage metadata retains a null job
    ID and the caller must supply the independently recorded Slurm handle.
    """
    explicit = [str(job_id) for job_id in explicit_ids]
    if any(not re.fullmatch(r'[0-9]+', job_id) for job_id in explicit):
        raise ValueError(scope + ' explicit attempt lacks a numeric Slurm job ID')
    recorded = metadata.get('attempts', [])
    recorded_ids = [str(attempt.get('slurm_job_id') or '') for attempt in recorded]
    if any(recorded_ids):
        if explicit:
            raise ValueError(scope + ' explicit attempt duplicates recorded metadata')
        return attempt_ids(metadata, scope), 'stage_metadata'
    if metadata.get('status') != 'completed':
        raise ValueError(scope + ' metadata is not completed')
    if not recorded or not explicit:
        raise ValueError(scope + ' metadata has no recorded Slurm attempt; explicit ID required')
    if len(explicit) != len(set(explicit)):
        raise ValueError(scope + ' explicit attempts contain duplicates')
    return explicit, 'explicit_registry_handle'


def parse_sacct(output, wanted):
    """Ignore batch/extern steps and require every top-level attempt."""
    found = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) < 4 or fields[0] not in wanted:
            continue
        job_id, state, elapsed, tres = fields[:4]
        if job_id in found:
            raise ValueError('Duplicate top-level sacct attempt: ' + job_id)
        state = state.split()[0]
        gpu = next((value.split('=', 1)[1] for value in tres.split(',')
                    if value.startswith('gres/gpu=')), None)
        if state not in TERMINAL or not elapsed.isdigit() or not gpu or not gpu.isdigit():
            raise ValueError('Incomplete or invalid sacct GPU allocation: ' + job_id)
        seconds, count = int(elapsed), int(gpu)
        if count < 1:
            raise ValueError('Attempt has no allocated GPUs: ' + job_id)
        found[job_id] = dict(job_id=job_id, state=state,
                             elapsed_seconds=seconds, gpu_count=count)
    if set(found) != set(wanted):
        raise ValueError('sacct did not return every recorded attempt: ' +
                         ','.join(sorted(set(wanted) - set(found))))
    return found


def query_sacct(job_ids):
    ordered = sorted(set(job_ids), key=int)
    command = ('sacct -j ' + ','.join(ordered) +
               ' --format=JobIDRaw,State,ElapsedRaw,AllocTRES -P -n')
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
         'hhilbig@hive.hpc.ucdavis.edu', 'bash -lc "' + command + '"'],
        capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError('Hive sacct query failed: ' + result.stderr[:300])
    return parse_sacct(result.stdout, ordered)


def build(model_key, pilot_dir, remainder_dir, output):
    manifest = json.loads(MANIFEST.read_text())
    audit = json.loads(AUDIT.read_text())
    panel = json.loads(PANEL.read_text())
    entries = [row for row in manifest['models'] if row['model_key'] == model_key]
    if len(entries) != 1:
        raise ValueError('Backfill model is not uniquely in the frozen manifest')
    entry = entries[0]
    _, source_meta = reused_frame(entry, audit, panel)
    source_info = {row['source']: row for row in audit['sources']}
    paths = {'pilot': Path(pilot_dir) / 'run_metadata.json',
             'remainder': Path(remainder_dir) / 'run_metadata.json'}
    metadata = {'historical': source_meta,
                'pilot': [json.loads(paths['pilot'].read_text())],
                'remainder': [json.loads(paths['remainder'].read_text())]}
    scopes = {}
    for scope, records in metadata.items():
        for record in records:
            for job_id in attempt_ids(record, scope):
                if job_id in scopes:
                    raise ValueError('Slurm attempt appears in more than one source/stage')
                scopes[job_id] = scope
    observed = query_sacct(scopes)
    attempts = [dict(record, scope=scopes[job_id])
                for job_id, record in sorted(observed.items(), key=lambda item: int(item[0]))]
    hours = {scope: sum(row['elapsed_seconds'] * row['gpu_count'] / 3600
                        for row in attempts if row['scope'] == scope)
             for scope in metadata}
    payload = {'model': model_key, 'model_id': entry['model_id'],
               'revision': entry['revision'], 'source': 'sacct',
               'complete': True, 'attempts': attempts,
               'gpu_hours': hours, 'gpu_hours_total': sum(hours.values()),
               'source_metadata_sha256': [sha(source_info[row['path']]['metadata'])
                                          for row in entry['reused_predictions']],
               'pilot_metadata_sha256': sha(paths['pilot']),
               'remainder_metadata_sha256': sha(paths['remainder']),
               'note': 'Allocated GPU time includes full historical source jobs and all recorded '
                       'backfill attempts, including failed attempts. Historical jobs processed '
                       'more than the frozen comparison panel; GPU time is not a matched speed '
                       'or an economic cost estimate.'}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    return {'model': model_key, 'attempts': len(attempts),
            'gpu_hours_total': payload['gpu_hours_total'],
            'output': str(target), 'sha256': sha(target)}


def build_refresh(model_key, pilot_dir, remainder_dir, output,
                  failed_attempt_id=(), pilot_attempt_id=(),
                  remainder_attempt_id=()):
    """Account for both new-model stages and named pre-inference failures."""
    paths = {'pilot': Path(pilot_dir) / 'run_metadata.json',
             'remainder': Path(remainder_dir) / 'run_metadata.json'}
    metadata = {scope: json.loads(path.read_text()) for scope, path in paths.items()}
    pilot, remainder = metadata['pilot'], metadata['remainder']
    if (pilot.get('model_key') != model_key or remainder.get('model_key') != model_key
            or pilot.get('model_id') != remainder.get('model_id')
            or pilot.get('revision') != remainder.get('revision')
            or pilot.get('generation') != remainder.get('generation')
            or pilot.get('runtime') != remainder.get('runtime')):
        raise ValueError('Pilot/remainder model identity or inference settings differ')
    scopes = {}
    explicit = {'pilot': pilot_attempt_id, 'remainder': remainder_attempt_id}
    attempt_id_provenance = {}
    for scope, record in metadata.items():
        stage_ids, provenance = refresh_attempt_ids(record, scope, explicit[scope])
        attempt_id_provenance[scope] = provenance
        for job_id in stage_ids:
            if job_id in scopes:
                raise ValueError('Refresh Slurm attempt appears in both stages')
            scopes[job_id] = scope
    for job_id in failed_attempt_id:
        job_id = str(job_id)
        if not re.fullmatch(r'[0-9]+', job_id) or job_id in scopes:
            raise ValueError('Named pre-inference attempt is invalid or duplicated')
        scopes[job_id] = 'preinference_failure'
    observed = query_sacct(scopes)
    attempts = [dict(record, scope=scopes[job_id])
                for job_id, record in sorted(observed.items(), key=lambda item: int(item[0]))]
    if any(row['state'] not in {'FAILED', 'CANCELLED', 'TIMEOUT',
                                'OUT_OF_MEMORY', 'NODE_FAIL'}
           for row in attempts if row['scope'] == 'preinference_failure'):
        raise ValueError('Named pre-inference attempt did not fail')
    gpu_counts = {len((record.get('gpu_after_load') or {}).get('gpus', []))
                  for record in metadata.values()}
    if len(gpu_counts) != 1 or 0 in gpu_counts or any(
            row['gpu_count'] != next(iter(gpu_counts)) for row in attempts):
        raise ValueError('Slurm GPU allocation differs from completed refresh hardware')
    hours = {scope: sum(row['elapsed_seconds'] * row['gpu_count'] / 3600
                        for row in attempts if row['scope'] == scope)
             for scope in ['preinference_failure', 'pilot', 'remainder']}
    payload = {'model': model_key, 'model_id': pilot['model_id'],
               'revision': pilot['revision'], 'source': 'sacct', 'complete': True,
               'attempts': attempts, 'gpu_hours': hours,
               'gpu_hours_total': sum(hours.values()),
               'pilot_metadata_sha256': sha(paths['pilot']),
               'remainder_metadata_sha256': sha(paths['remainder']),
               'failed_preinference_ids': sorted(
                   str(row) for row in failed_attempt_id),
               'attempt_id_provenance': attempt_id_provenance,
               'note': 'Allocated Hive GPU time includes all recorded pilot and remainder '
                       'attempts plus named pre-inference failures. This is separate from '
                       'generation throughput and is not an economic cost estimate.'}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    return {'model': model_key, 'attempts': len(attempts),
            'gpu_hours_total': payload['gpu_hours_total'],
            'output': str(target), 'sha256': sha(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--remainder-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true',
                        help='New refresh model; do not use the historical reuse manifest')
    parser.add_argument('--failed-attempt-id', action='append', default=[],
                        help='Registry-recorded Slurm pre-inference failure')
    parser.add_argument('--pilot-attempt-id', action='append', default=[],
                        help='Completed pilot Slurm ID when absent inside container metadata')
    parser.add_argument('--remainder-attempt-id', action='append', default=[],
                        help='Completed remainder Slurm ID when absent inside container metadata')
    args = parser.parse_args()
    if args.refresh:
        print(json.dumps(build_refresh(args.model_key, args.pilot_dir,
                                       args.remainder_dir, args.output,
                                       args.failed_attempt_id,
                                       args.pilot_attempt_id,
                                       args.remainder_attempt_id)))
    else:
        if args.failed_attempt_id or args.pilot_attempt_id or args.remainder_attempt_id:
            parser.error('attempt ID overrides are only valid with --refresh')
        print(json.dumps(build(args.model_key, args.pilot_dir,
                               args.remainder_dir, args.output)))
