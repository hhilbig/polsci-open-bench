import json
import sys
import unittest
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import audit_refresh_reuse as audit


def fixture_source(tmp_path):
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('Classify this text.')
    task = {'name': 't', 'prompt_path': str(prompt), 'json_schema': {'type': 'object'}, 'label_kind': 'binary'}
    item = {'item_id': '001', 'user_content': 'Frozen input', 'gt': {'label': 1}}
    generation = {'max_tokens': 256, 'temperature': 0, 'enable_thinking': False,
                  'structured_outputs': 'json_schema'}
    meta = {'status': 'completed', 'model_id': 'org/model', 'model_key': 'model',
            'revision': 'a' * 40, 'config_sha256': 'b' * 64, 'generation': generation,
            'runtime': {}, 'runtime_versions': {'python': '3.12'}}
    frame = pd.DataFrame([{'task': 't', 'item_id': '001', 'model_id': 'org/model',
        'model_revision': 'a' * 40, 'gt_label': 1, 'pred_label': 0, 'parse_error': ''}])
    root = tmp_path / 'run'
    (root / 'tasks').mkdir(parents=True)
    checkpoint = root / 'tasks/t.csv'
    frame.to_csv(checkpoint, index=False)
    path = root / 'predictions.csv'
    frame.to_csv(path, index=False)
    meta['predictions_sha256'] = audit.sha(path)
    meta['task_results'] = {'t': {'checkpoint_sha256': audit.sha(checkpoint),
        'task_fingerprint': audit.task_fingerprint(task, [item], 'model', meta,
            meta['config_sha256'], generation, {}, meta['runtime_versions'])}}
    (root / 'run_metadata.json').write_text(json.dumps(meta))
    return path, {'t': task}, {'t': {'001': item}}, {('t', '001'): item}


def test_exact_provenance_accepts_wrong_prediction(tmp_path):
    args = fixture_source(tmp_path)
    result = audit.audit_source(*args)
    assert result['eligible_keys'] == [['t', '001']]
    assert not result['issues']


def test_changed_prompt_rejected(tmp_path):
    args = fixture_source(tmp_path)
    Path(args[1]['t']['prompt_path']).write_text('A changed prompt')
    result = audit.audit_source(*args)
    assert not result['eligible_keys']
    assert 'input_prompt_schema_or_settings_fingerprint_mismatch' in result['task_checks']['t']['issues']


def test_changed_input_rejected(tmp_path):
    args = fixture_source(tmp_path)
    args[2]['t']['001']['user_content'] = 'Changed input'
    assert not audit.audit_source(*args)['eligible_keys']


def test_corrupted_prediction_rejected(tmp_path):
    args = fixture_source(tmp_path)
    args[0].write_text(args[0].read_text().replace(',0,', ',1,'))
    assert 'predictions_checksum_missing_or_mismatch' in audit.audit_source(*args)['issues']


def test_generation_mismatch_rejected(tmp_path):
    args = fixture_source(tmp_path)
    path = args[0].parent / 'run_metadata.json'
    meta = json.loads(path.read_text())
    meta['generation']['max_tokens'] = 1024
    path.write_text(json.dumps(meta))
    assert 'incompatible_output_limit_or_temperature' in audit.audit_source(*args)['issues']


def test_missing_checkpoint_rejected(tmp_path):
    args = fixture_source(tmp_path)
    (args[0].parent / 'tasks/t.csv').unlink()
    result = audit.audit_source(*args)
    assert not result['eligible_keys']
    assert 'task_checkpoint_missing_or_checksum_mismatch' in result['task_checks']['t']['issues']


def test_frozen_gold_mismatch_rejected(tmp_path):
    args = fixture_source(tmp_path)
    args[3][('t', '001')] = dict(args[3][('t', '001')], gt={'label': 0})
    result = audit.audit_source(*args)
    assert not result['eligible_keys']
    assert 'frozen_gold_mismatch' in result['task_checks']['t']['issues']


def test_aggregate_never_combines_settings():
    common = {'model_key': 'x', 'model_id': 'org/x', 'revision': 'a' * 40}
    sources = [dict(common, compatibility_group='a', source='old', completed_at='2026-01-01', eligible_keys=[['t', '1']]),
               dict(common, compatibility_group='b', source='other', completed_at='2026-02-01', eligible_keys=[['t', '2']]),
               dict(common, compatibility_group='a', source='new', completed_at='2026-03-01', eligible_keys=[['t', '1']])]
    groups = audit.aggregate(sources, [('t', '1'), ('t', '2')])
    assert all(g['backfill_rows'] == 1 for g in groups)
    assert groups[0]['selected_rows'][0]['source'] == 'new'


def test_numeric_gold_and_missing():
    assert audit.scalar_equal('1.0', 1)
    assert not audit.scalar_equal('', 1)
    assert not audit.scalar_equal(None, 1)


def test_numeric_csv_serialization():
    a = pd.DataFrame({'gt_label': ['1.0'], 'item_id': ['001']})
    b = pd.DataFrame({'gt_label': ['1'], 'item_id': ['001']})
    assert audit.frames_equivalent(a, b)
    assert not audit.frames_equivalent(a, b, [])
    b['item_id'] = ['1']
    assert not audit.frames_equivalent(a, b)


class ReuseAuditTests(unittest.TestCase):
    pass


def as_unittest(function):
    def run(self):
        if function.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))
        else:
            function()
    return run


for name, function in list(globals().items()):
    if name.startswith('test_') and callable(function):
        setattr(ReuseAuditTests, name, as_unittest(function))


if __name__ == '__main__':
    unittest.main()
