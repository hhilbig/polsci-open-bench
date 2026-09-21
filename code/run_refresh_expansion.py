"""Run one frozen Hive candidate through pilot, remainder, and collection.

No answer retries or quality selection. Existing task checkpoints provide restart
safety; completed stages are always re-audited before reuse.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_refresh_reuse import digest
from collect_refresh_hive import collect
from hive_vllm_benchmark import load_config

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / 'output/sidecar/refresh_20260910/panel.json'


def validate_configs(pilot_path, remainder_path, model_key):
    configs = [load_config(Path(p)) for p in (pilot_path, remainder_path)]
    panel_hash = digest(json.loads(PANEL.read_text()))
    normalized = []
    for config, stage, count in zip(configs, ('pilot', 'remainder'), (68, 3332)):
        if set(config['models']) != {model_key}:
            raise ValueError('Each configuration must contain exactly the requested model')
        if config.get('refresh_sample') != {'stage': stage, 'panel_sha256': panel_hash}:
            raise ValueError('Frozen panel hash or stage differs')
        if (config['task_scope'].get('expected_items') != count
                or config['task_scope'].get('expected_tasks') != 34):
            raise ValueError('Expected frozen stage coverage differs')
        comparable = deepcopy(config)
        comparable.pop('run_id')
        comparable['refresh_sample'].pop('stage')
        comparable['task_scope'].pop('expected_items')
        normalized.append(comparable)
    if normalized[0] != normalized[1]:
        raise ValueError('Pilot/remainder settings differ')


def run(pilot_config, remainder_config, model_key, output_root):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    configs = {'pilot': Path(pilot_config).resolve(),
               'remainder': Path(remainder_config).resolve()}
    dirs = {stage: root / stage for stage in ('pilot', 'remainder')}
    dirs['collected'] = root / 'collected'
    dirs.update({stage + '_audit': root / (stage + '_audit')
                 for stage in ('pilot', 'remainder')})
    status_path = root / 'status.json'
    events = []
    if status_path.exists():
        events = json.loads(status_path.read_text()).get('events', [])

    def status(state, stage, reason):
        events.append({'status': state, 'stage': stage, 'reason': reason,
                       'at': datetime.now(timezone.utc).isoformat()})
        record = dict(events[-1], model_key=model_key,
                      directories={k: str(v) for k, v in dirs.items()},
                      configs={k: str(v) for k, v in configs.items()}, events=events)
        temporary = root / f'.status.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)
        return record

    stage = 'preflight'
    try:
        status('running', stage, 'Checking frozen configurations')
        validate_configs(configs['pilot'], configs['remainder'], model_key)
        for stage in ('pilot', 'remainder'):
            directory = dirs[stage]
            metadata_path = directory / 'run_metadata.json'
            completed = (metadata_path.exists()
                         and json.loads(metadata_path.read_text()).get('status') == 'completed')
            if not completed:
                status('running', stage, 'Running or resuming checkpointed inference')
                subprocess.run([sys.executable, str(REPO / 'code/hive_vllm_benchmark.py'),
                                '--config', str(configs[stage]), '--model-key', model_key,
                                '--output-dir', str(directory)], cwd=REPO, check=True)
            status('validating', stage, 'Auditing all saved outputs before reuse')
            report = collect(configs[stage], directory, dirs[stage + '_audit'])['pilot']
            if report.get('truncations') or (stage == 'pilot' and report.get('schema_failures')):
                return status(f'{stage}_configuration_review', stage,
                              'Retained truncation or pilot schema failure; no answer retry')
            if (not report.get('passed') or report.get('output_token_usage_unavailable')
                    or report.get('rows') != (68 if stage == 'pilot' else 3332)
                    or report.get('tasks') != 34):
                return status('validation_failed', stage,
                              'Coverage, provenance, raw response, or token accounting audit failed')
        stage = 'collection'
        status('validating', stage, 'Validating combined frozen panel')
        report = collect(configs['pilot'], dirs['pilot'], dirs['collected'],
                         remainder_config=configs['remainder'], remainder_dir=dirs['remainder'])
        if not report.get('complete'):
            return status('validation_failed', stage, 'Combined collector rejected outputs')
        return status('completed', stage, 'Validated all 3,400 predictions; GPU accounting pending sacct')
    except Exception as exc:
        return status('infrastructure_failed' if isinstance(exc, subprocess.CalledProcessError)
                      else 'validation_failed', stage, f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-config', type=Path, required=True)
    parser.add_argument('--remainder-config', type=Path, required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
