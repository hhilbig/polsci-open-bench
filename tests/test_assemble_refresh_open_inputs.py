"""Guard the final open-model manifest assembly against partial exports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from assemble_refresh_open_inputs import assemble
from audit_refresh_reuse import digest, sha


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.panel = {'rows': [{'task': 't', 'item': {'item_id': '1', 'gt': {'a': 1}}}]}
        self.panel_path = self.root / 'panel.json'
        self.panel_path.write_text(json.dumps(self.panel))
        self.task = {'name': 't', 'label_kind': 'binary', 'label_key': 'a', 'labels': ['a']}
        self.mock_tasks = patch('assemble_refresh_open_inputs.load_task_definitions', return_value=[self.task])
        self.mock_tasks.start()
        self.addCleanup(self.mock_tasks.stop)

    def source(self, model, collector=False, complete=True):
        directory = self.root / model
        directory.mkdir()
        predictions = directory / 'predictions.csv'
        pd.DataFrame([{'task': 't', 'item_id': '1', 'gt_a': 1, 'pred_a': 1,
                       'parse_error': None, 'model': model}]).to_csv(predictions, index=False)
        entry = {'model': model, 'predictions': str(predictions),
                 'panel_sha256': digest(self.panel), 'prediction_sha256': sha(predictions),
                 'revision': 'pin', 'quantization': 'FP8', 'hardware': 'GPU',
                 'hardware_tier': 'single-gpu', 'settings': {}, 'run_dates': ['2026-09-14'],
                 'provenance_validated': True}
        manifest = directory / 'open_inputs.json'
        manifest.write_text(json.dumps({'models': [entry]}))
        if collector:
            (directory / 'validation.json').write_text(json.dumps({
                'complete': complete, 'model': model,
                'stages': {'pilot': {'passed': complete}, 'remainder': {'passed': complete}}}))
        return manifest

    def test_completed_collector_combines_deterministically(self):
        baseline = self.source('baseline')
        completed = self.source('candidate', collector=True)
        first = assemble([baseline, completed], self.root / 'one.json', self.panel_path)
        second = assemble([baseline, completed], self.root / 'two.json', self.panel_path)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(first['predictions'], 2)

    def test_incomplete_collector_is_rejected(self):
        baseline = self.source('baseline')
        incomplete = self.source('candidate', collector=True, complete=False)
        with self.assertRaisesRegex(ValueError, 'completed validation'):
            assemble([baseline, incomplete], self.root / 'combined.json', self.panel_path)

    def test_duplicate_or_changed_prediction_is_rejected(self):
        baseline = self.source('baseline')
        duplicate = json.loads(baseline.read_text())
        duplicate['models'].append(dict(duplicate['models'][0]))
        baseline.write_text(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            assemble([baseline], self.root / 'combined.json', self.panel_path)
        duplicate['models'].pop()
        baseline.write_text(json.dumps(duplicate))
        completed = self.source('candidate', collector=True)
        prediction_path = completed.parent / 'predictions.csv'
        prediction_path.write_text(prediction_path.read_text().replace(',1,1,', ',1,0,'))
        with self.assertRaisesRegex(ValueError, 'changed after audit'):
            assemble([baseline, completed], self.root / 'combined.json', self.panel_path)

    def test_wrong_frozen_gold_is_rejected(self):
        baseline = self.source('baseline')
        manifest = json.loads(baseline.read_text())
        prediction_path = Path(manifest['models'][0]['predictions'])
        frame = pd.read_csv(prediction_path)
        frame['gt_a'] = 0
        frame.to_csv(prediction_path, index=False)
        manifest['models'][0]['prediction_sha256'] = sha(prediction_path)
        baseline.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Gold label mismatch'):
            assemble([baseline], self.root / 'combined.json', self.panel_path)


if __name__ == '__main__':
    unittest.main()
