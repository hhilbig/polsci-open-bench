import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from build_refresh_release import (headline_replicates, class_metrics, validate_panel,
                                   compute, clean_json, public_settings,
                                   load_historical_exclusions)
from build_frontier_2026 import _score_malformed_as_incorrect


def test_fast_score_matches_existing_and_retains_failures(kind):
    categorical = kind == 'categorical'
    task = dict(name='t', label_kind=kind, label_key='a', labels=['x','y','z'] if categorical else ['a'])
    g = pd.DataFrame(dict(gt_a=['x','y','x','y'] if categorical else [1,0,1,0],
                          pred_a=['x','y','y','x'] if categorical else [1,0,0,1],
                          parse_error=['invalid',None,None,None]))
    weights = np.array([[1,1,1,1], [2,0,1,1], [0,1,0,3]], dtype=float)
    scores = headline_replicates(g, task, weights)
    for i, row in enumerate(weights):
        expanded = g.iloc[np.repeat(np.arange(4), row.astype(int))]
        assert np.isclose(scores[i], _score_malformed_as_incorrect(expanded, task)['headline_f1'])
    assert g.pred_a.iloc[0] == ('x' if categorical else 1)


def test_undefined_mcc_absent_class_f1():
    task = dict(label_kind='categorical', label_key='a', labels=['x','y'])
    rows = class_metrics(pd.DataFrame(dict(gt_a=['x','x'],pred_a=['x','x'],parse_error=[None,None])),task)
    assert rows[1]['support'] == 0 and rows[1]['f1'] == 0
    assert rows[1]['mcc'] is None
    assert clean_json({'a':float('nan')}) == {'a':None}


def test_unscorable_historical_attempt_cannot_be_ranked_as_clean():
    entry=dict(model='GPT-OSS 120B',model_id='openai/gpt-oss-120b',
               revision='pinned',status='excluded',reason='Item ID unavailable')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'historical_exclusions.json'
        path.write_text(json.dumps({'exclusions':[entry]}))
        assert load_historical_exclusions(path,{})==[entry]
        with unittest.TestCase().assertRaisesRegex(ValueError,'ranked'):
            load_historical_exclusions(path,{'gpt':{'model_id':entry['model_id'],
                                                    'revision':entry['revision']}})


def test_frozen_keys_and_gold_required():
    panel = dict(rows=[dict(task='t',item=dict(item_id='1',gt={'a':1}))])
    task = dict(name='t',label_kind='binary',label_key='a',labels=['a'])
    g = pd.DataFrame(dict(task=['t'],item_id=['1'],gt_a=[1],pred_a=[1],parse_error=[None]))
    validate_panel(g,panel,{'t':task})
    with unittest.TestCase().assertRaisesRegex(ValueError,'Duplicate'):
        validate_panel(pd.concat([g,g]),panel,{'t':task})
    with unittest.TestCase().assertRaisesRegex(ValueError,'Incomplete'):
        validate_panel(g.iloc[:0],panel,{'t':task})
    with unittest.TestCase().assertRaisesRegex(ValueError,'Gold'):
        validate_panel(g.assign(gt_a=0),panel,{'t':task})
    with unittest.TestCase().assertRaisesRegex(ValueError,'Unmarked'):
        validate_panel(g.assign(pred_a=7),panel,{'t':task})


def test_paired_draws_and_determinism():
    task = dict(name='t',label_kind='binary',label_key='a',labels=['a'])
    g = pd.DataFrame(dict(task=['t']*100,item_id=[str(i) for i in range(100)],gt_a=[0,1]*50,
                          pred_a=[0,1,1,0]*25,parse_error=[None]*100,model=['m1']*100))
    frame = pd.concat([g,g.assign(model='m2')],ignore_index=True)
    one=compute(frame,{'t':task},{'t':'category'},20)
    two=compute(frame.sample(frac=1,random_state=3),{'t':task},{'t':'category'},20)
    assert one == two
    assert one['pairs'][0]['difference']==0
    assert one['pairs'][0]['item_ci_low']==one['pairs'][0]['item_ci_high']==0


class ReleaseTests(unittest.TestCase):
    def test_existing_scorer(self):
        for kind in ['binary','categorical','multi_binary']:
            test_fast_score_matches_existing_and_retains_failures(kind)

    def test_undefined(self):
        test_undefined_mcc_absent_class_f1()

    def test_validation(self):
        test_frozen_keys_and_gold_required()

    def test_pairing(self):
        test_paired_draws_and_determinism()

    def test_public_runtime_provenance_keeps_controls_but_not_paths(self):
        settings = {'container_image_sha256': 'abc',
                    'required_environment': {'VLLM_PLE_CPU_OFFLOAD': '1'},
                    'kernel_config': {'enable_flashinfer_autotune': False, 'moe_backend': 'triton'},
                    'runtime_lock_path': '/private/runtime.freeze.txt',
                    'model_snapshot_path': '/private/model'}
        published = public_settings(settings)
        self.assertEqual(published['container_image_sha256'], 'abc')
        self.assertEqual(published['required_environment'], {'VLLM_PLE_CPU_OFFLOAD': '1'})
        self.assertNotIn('runtime_lock_path', published)
        self.assertNotIn('model_snapshot_path', published)


if __name__ == '__main__':
    unittest.main()
