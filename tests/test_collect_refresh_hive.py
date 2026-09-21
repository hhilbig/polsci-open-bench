import sys
import unittest
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from collect_refresh_hive import (strict_response, output_token_usage_status, gpu_hours,
                                  checked_sacct_hours, hardware_label, generation_rates,
                                  audit_stage)


class HiveCollectionTests(unittest.TestCase):
    def setUp(self):
        self.task = {'label_kind': 'multi_binary', 'labels': ['a', 'b']}

    def test_complete_legacy_preview(self):
        obj, error, provenance = strict_response({'raw_content_preview': '{"a":0,"b":1}'}, self.task)
        self.assertIsNone(error)
        self.assertEqual(obj, {'a': 0, 'b': 1})
        self.assertEqual(provenance, 'complete_legacy_preview')

    def test_at_limit_preview_is_unverifiable_even_if_valid_json(self):
        text = '{"a":0,"b":1}' + ' ' * 187
        self.assertEqual(len(text), 200)
        self.assertEqual(strict_response({'raw_content_preview': text}, self.task)[1], 'unverifiable_preview')
        self.assertIsNone(strict_response({'raw_content': text}, self.task)[1])

    def test_missing_field_not_silently_zero(self):
        self.assertIn('Missing', strict_response({'raw_content': '{"a":0}'}, self.task)[1])
        self.assertIn('integer', strict_response({'raw_content': '{"a":false,"b":1}'}, self.task)[1])

    def test_retained_harmony_error_has_malformed_schema_and_unavailable_usage(self):
        row = {'raw_content': '', 'raw_content_preview': '', 'finish_reason': 'failed',
               'parse_error': 'harmony_unknown_generated_channel', 'eval_count': '0',
               'token_usage_status': 'unavailable'}
        obj, error, _ = strict_response(row, self.task)
        self.assertIsNone(obj)
        self.assertTrue(error.startswith('schema_invalid: '))
        self.assertEqual(output_token_usage_status(row), 'unavailable')
        with self.assertRaises(ValueError):
            output_token_usage_status(dict(row, finish_reason='length'))
        with self.assertRaises(ValueError):
            output_token_usage_status(dict(row, parse_error='missing_final_output'))

    def test_sacct_unknown_omitted_and_attempts_counted(self):
        self.assertIsNone(gpu_hours(None, ['1']))
        ledger = {'source': 'sacct', 'complete': True, 'attempts': [
            {'job_id': '1', 'state': 'COMPLETED', 'elapsed_seconds': 3600, 'gpu_count': 2},
            {'job_id': '2', 'state': 'FAILED', 'elapsed_seconds': 1800, 'gpu_count': 2}]}
        self.assertEqual(gpu_hours(ledger, ['1']), 3)
        self.assertIsNone(gpu_hours(ledger, []))
        with self.assertRaises(ValueError):
            gpu_hours(ledger, ['3'])
        ledger['attempts'][0]['state'] = 'RUNNING'
        with self.assertRaises(ValueError):
            gpu_hours(ledger, ['1'])

    def test_refresh_sacct_requires_exact_attempt_ids(self):
        ledger = {'source': 'sacct', 'complete': True, 'attempts': [
            {'job_id': '1', 'state': 'COMPLETED', 'elapsed_seconds': 3600, 'gpu_count': 1},
            {'job_id': '2', 'state': 'FAILED', 'elapsed_seconds': 1800, 'gpu_count': 1}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sacct.json'
            path.write_text(json.dumps(ledger))
            self.assertEqual(checked_sacct_hours(path, {'1','2'}), 1.5)
            with self.assertRaisesRegex(ValueError, 'exact refresh attempt'):
                checked_sacct_hours(path, {'1'})
            self.assertEqual(checked_sacct_hours(path, {'1'}, {'2'}), 1.5)
            ledger['attempts'][1]['state'] = 'COMPLETED'
            path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, 'did not fail'):
                checked_sacct_hours(path, {'1'}, {'2'})

    def test_multi_gpu_hardware_label_keeps_allocation_count(self):
        self.assertEqual(hardware_label([{'name':'Blackwell'}]), 'Blackwell')
        self.assertEqual(hardware_label([{'name':'Blackwell'}, {'name':'Blackwell'}]),
                         '2 × Blackwell')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            hardware_label([])

    def test_item_throughput_remains_defined_when_output_usage_is_unavailable(self):
        self.assertEqual(generation_rates(68, 4, 120, True), (17, None))
        self.assertEqual(generation_rates(68, 4, 120, False), (17, 30))
        with self.assertRaisesRegex(ValueError, 'Generation item count'):
            generation_rates(68, 0, 120, True)

    def test_actual_collected_qwen_pilot(self):
        path = Path('output/sidecar/refresh_hive_pilot/qwen3_8_27b_fp8')
        if not (path / 'run_metadata.json').exists():
            self.skipTest('Collected Qwen pilot not available')
        report, frame = audit_stage('experiments/refresh_hive_20260914_pilot.yaml', path)
        self.assertTrue(report['passed'], report['issues'])
        self.assertEqual((report['rows'], report['tasks']), (68, 34))
        self.assertEqual(report['raw_provenance_counts'], {'complete_legacy_preview': 68})
        self.assertFalse(report['schema_failures'])
        self.assertFalse(report['truncations'])
        self.assertEqual(len(frame), 68)

    def test_runtime_failure_not_counted_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'run_metadata.json').write_text(json.dumps({'status': 'failed', 'error': 'CUDA out of memory'}))
            report, frame = audit_stage('experiments/refresh_hive_20260914_pilot.yaml', path)
            self.assertFalse(report['passed'])
            self.assertEqual(report['failure_types'], ['runtime'])
            self.assertIsNone(frame)

    def test_metadata_settings_mismatch_rejected(self):
        source = Path('output/sidecar/refresh_hive_pilot/qwen3_8_27b_fp8/run_metadata.json')
        if not source.exists():
            self.skipTest('Collected pilot unavailable')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            meta = json.loads(source.read_text())
            meta['generation']['max_tokens'] = 512
            (path / 'run_metadata.json').write_text(json.dumps(meta))
            report, frame = audit_stage('experiments/refresh_hive_20260914_pilot.yaml', path)
            self.assertFalse(report['passed'])
            self.assertIn('Metadata mismatch: generation', report['issues'])
            self.assertIsNone(frame)


if __name__ == '__main__':
    unittest.main()
