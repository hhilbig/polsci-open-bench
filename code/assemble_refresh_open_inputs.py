"""Combine audited open-model exports for the matched release.

Pass the seven-model historical manifest first, then only completed collector
exports. This does not run inference or publish a release. Every source must
already cover the same frozen panel and carry verified provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_refresh_reuse import digest, sha
from build_refresh_release import validate_panel
from task_registry import load_task_definitions


PANEL = Path('output/sidecar/refresh_20260910/panel.json')
BASELINE = Path('output/sidecar/refresh_20260910_release/open_inputs.json')
OUTPUT = Path('output/sidecar/refresh_20260910_release/assembled_open_inputs.json')


def assemble(sources, output=OUTPUT, panel_path=PANEL):
    panel = json.loads(Path(panel_path).read_text())
    panel_hash = digest(panel)
    task_defs = {task['name']: task for task in load_task_definitions()}
    entries, names = [], set()
    for index, source in enumerate(sources):
        source = Path(source)
        payload = json.loads(source.read_text())
        if not isinstance(payload.get('models'), list) or not payload['models']:
            raise ValueError('Empty or invalid open-model source: ' + str(source))
        if index and len(payload['models']) != 1:
            raise ValueError('Completed collector export must contain exactly one model')
        if index:
            validation_path = source.with_name('validation.json')
            validation = json.loads(validation_path.read_text())
            if validation.get('complete') is not True:
                raise ValueError('Collector export lacks completed validation: ' + str(source))
            stages = validation.get('stages', validation)
            for stage in ['pilot', 'remainder']:
                stage_report = stages.get(stage)
                if stage_report is not None and stage_report.get('passed') is not True:
                    raise ValueError('Collector stage failed validation: ' + str(source))
            if validation.get('model') and validation['model'] != payload['models'][0]['model']:
                raise ValueError('Collector validation model differs from export')
        for entry in payload['models']:
            name = entry.get('model')
            if not name or name in names:
                raise ValueError('Duplicate or missing open-model name: ' + str(name))
            if (entry.get('panel_sha256') != panel_hash or
                    entry.get('provenance_validated') is not True):
                raise ValueError('Frozen panel or provenance mismatch: ' + name)
            prediction_path = Path(entry['predictions'])
            if sha(prediction_path) != entry.get('prediction_sha256'):
                raise ValueError('Prediction bytes changed after audit: ' + name)
            frame = pd.read_csv(prediction_path, dtype={'item_id': str}, low_memory=False)
            if set(frame.model) != {name}:
                raise ValueError('Prediction model identity mismatch: ' + name)
            validate_panel(frame, panel, task_defs)
            entries.append(entry)
            names.add(name)
    result = {'models': sorted(entries, key=lambda entry: entry['model'])}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
    return {'models': len(entries), 'predictions': len(entries) * len(panel['rows']),
            'panel_sha256': panel_hash, 'output': str(target), 'sha256': sha(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--collector', type=Path, action='append', default=[],
                        help='Completed collector open_inputs.json; repeat for each model')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(assemble([args.baseline, *args.collector], args.output), indent=2))
