import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_refresh_expansion as driver
import hive_vllm_benchmark as benchmark
import configure_refresh_minimax as minimax_config


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def report(self, count=68, **changes):
        report = dict(passed=True, rows=count, tasks=34, schema_failures=[],
                      truncations=[], output_token_usage_unavailable=[])
        report.update(changes)
        return {'pilot': report}

    def execute(self, reports, error=None):
        with patch.object(driver, 'validate_configs'), patch.object(
                driver, 'collect', side_effect=reports) as collector, patch.object(
                driver.subprocess, 'run', side_effect=error) as runner:
            result = driver.run('pilot.yaml', 'rest.yaml', 'candidate', self.root)
        self.assertEqual(json.loads((self.root / 'status.json').read_text()), result)
        return result, runner, collector

    def test_success_ignores_accuracy_and_retains_full_stage_schema_failures(self):
        result, runner, collector = self.execute([
            self.report(accuracy=0, mean_f1=0),
            self.report(3332, schema_failures=[{'key': ['task', 'id']}]),
            {'complete': True}])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(collector.call_count, 3)
        self.assertIn('--model-key', runner.call_args.args[0])

    def test_pilot_truncation_stops_without_remainder(self):
        result, runner, _ = self.execute([self.report(truncations=[['task', 'id']])])
        self.assertEqual(result['status'], 'pilot_configuration_review')
        self.assertEqual(runner.call_count, 1)

    def test_pilot_schema_failure_not_retried(self):
        result, runner, _ = self.execute([self.report(schema_failures=[{'key': ['t', 'i']}])])
        self.assertEqual(result['status'], 'pilot_configuration_review')
        self.assertEqual(runner.call_count, 1)

    def test_invalid_coverage_or_usage_stops(self):
        for changes in ({'passed': False}, {'rows': 67}, {'tasks': 33},
                        {'output_token_usage_unavailable': [['t', 'i']]}):
            result, runner, _ = self.execute([self.report(**changes)])
            self.assertEqual(result['status'], 'validation_failed')
            self.assertEqual(runner.call_count, 1)

    def test_exception_retains_stage_and_paths(self):
        result, runner, _ = self.execute([], subprocess.CalledProcessError(1, ['runner']))
        self.assertEqual(result['status'], 'infrastructure_failed')
        self.assertEqual(result['stage'], 'pilot')
        self.assertEqual(set(result['directories']),
                         {'pilot', 'remainder', 'collected', 'pilot_audit', 'remainder_audit'})
        self.assertIn('CalledProcessError', result['reason'])

    def test_restart_reaudits_completed_stages_without_inference(self):
        for stage in ('pilot', 'remainder'):
            (self.root / stage).mkdir()
            (self.root / stage / 'run_metadata.json').write_text('{"status":"completed"}')
        result, runner, collector = self.execute([
            self.report(), self.report(3332), {'complete': True}])
        self.assertEqual(result['status'], 'completed')
        runner.assert_not_called()
        self.assertEqual(collector.call_count, 3)

    def test_invalid_completed_pilot_never_regenerated(self):
        (self.root / 'pilot').mkdir()
        (self.root / 'pilot/run_metadata.json').write_text('{"status":"completed"}')
        result, runner, _ = self.execute([self.report(passed=False)])
        self.assertEqual(result['status'], 'validation_failed')
        runner.assert_not_called()

    def test_config_settings_must_match_and_panel_must_be_frozen(self):
        configs = []
        panel_hash = driver.digest(json.loads(driver.PANEL.read_text()))
        for stage, count in [('pilot', 68), ('remainder', 3332)]:
            configs.append(dict(run_id=stage, models={'candidate': {'revision': 'a'}},
                                refresh_sample={'stage': stage, 'panel_sha256': panel_hash},
                                task_scope={'expected_tasks': 34, 'expected_items': count},
                                generation={'max_tokens': 256}, runtime={}))
        with patch.object(driver, 'load_config', side_effect=configs):
            driver.validate_configs('p', 'r', 'candidate')
        configs[1]['generation']['max_tokens'] = 512
        with patch.object(driver, 'load_config', side_effect=configs):
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                driver.validate_configs('p', 'r', 'candidate')
        configs[1]['refresh_sample']['panel_sha256'] = 'other'
        with patch.object(driver, 'load_config', side_effect=configs):
            with self.assertRaisesRegex(ValueError, 'panel hash'):
                driver.validate_configs('p', 'r', 'candidate')


class ExpansionRuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            'model_id': 'nvidia/MiniMax-M3-NVFP4',
            'revision': '901464083161bf8612a29ff7ad29914cd4ab4a85',
            'reviewed_remote_config': '836c3e4aff06f88bd2891b18292424ed7a90158b508d9235d490a33d9e3cba32',
            'llm_kwargs': {'block_size': 64},
        }
        self.runtime = {'max_model_len': 16384, 'gpu_memory_utilization': .9}
        self.generation = {'seed': 20260910}
        self.llm = MagicMock()
        modules = patch.dict(sys.modules, {'vllm': SimpleNamespace(LLM=self.llm)})
        modules.start()
        self.addCleanup(modules.stop)

    def test_positive_block_size_accepted_and_nonpositive_rejected(self):
        path = driver.REPO / 'experiments/refresh_hive_20260914_pilot.yaml'
        original = benchmark.load_config(path)
        for value in (16, 64, 0, -1):
            config = deepcopy(original)
            next(iter(config['models'].values()))['llm_kwargs']['block_size'] = value
            with self.subTest(value=value), patch.object(benchmark.yaml, 'safe_load', return_value=config):
                if value > 0:
                    self.assertEqual(benchmark.load_config(path), config)
                else:
                    with self.assertRaisesRegex(benchmark.BakeoffError, 'block_size must be positive'):
                        benchmark.load_config(path)

    def test_unreviewed_id_revision_or_hash_rejected_before_snapshot_access(self):
        for field in ('model_id', 'revision', 'reviewed_remote_config'):
            model = dict(self.model, **{field: 'unreviewed'})
            with self.subTest(field=field), patch.object(
                    benchmark, 'assert_compact_offline_provenance') as snapshot:
                with self.assertRaisesRegex(benchmark.BakeoffError, 'has not been reviewed'):
                    benchmark.create_llm(model, self.runtime, self.generation)
                snapshot.assert_not_called()
        self.llm.assert_not_called()

    def test_reviewed_module_checksum_mismatch_rejected(self):
        with patch.object(benchmark, 'assert_compact_offline_provenance', return_value=Path('/snapshot')), \
                patch.object(benchmark, 'file_sha256', return_value='changed'):
            with self.assertRaisesRegex(benchmark.BakeoffError, 'checksum mismatch'):
                benchmark.create_llm(self.model, self.runtime, self.generation)
        self.llm.assert_not_called()

    def test_valid_reviewed_module_requires_offline_identity_and_pins_code_revision(self):
        with patch.object(benchmark, 'assert_compact_offline_provenance',
                          return_value=Path('/snapshot')) as provenance, \
                patch.object(benchmark, 'file_sha256',
                             return_value=self.model['reviewed_remote_config']) as checksum:
            result, kwargs = benchmark.create_llm(self.model, self.runtime, self.generation)
        provenance.assert_called_once_with(
            {'require_exact_snapshot_path': True, 'require_offline_inference': True}, self.model)
        checksum.assert_called_once_with(Path('/snapshot/configuration_minimax_m3_vl.py'))
        self.assertIs(result, self.llm.return_value)
        self.assertTrue(kwargs['trust_remote_code'])
        self.assertEqual(kwargs['code_revision'], self.model['revision'])
        self.assertEqual(kwargs['block_size'], 64)
        self.llm.assert_called_once_with(**kwargs)

    def test_existing_model_does_not_enable_remote_code(self):
        model = {'model_id': 'Qwen/existing', 'revision': 'a' * 40}
        with patch.object(benchmark, 'assert_compact_offline_provenance') as provenance:
            _, kwargs = benchmark.create_llm(model, self.runtime, self.generation)
        self.assertFalse(kwargs['trust_remote_code'])
        self.assertNotIn('code_revision', kwargs)
        provenance.assert_not_called()


class MiniMaxConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.freeze = self.root / 'runtime.freeze'
        self.freeze.write_text('vllm==0.99.0\n')
        self.metadata = self.root / 'metadata.json'
        self.meta = {
            'runtime_lock_sha256': hashlib.sha256(self.freeze.read_bytes()).hexdigest(),
            'compiled_architectures': 'sm_80 sm_90 sm_120',
            'minimax_registry_lines': ['MiniMaxM3ForConditionalGeneration'],
            'packages': {'vllm': '0.99.0'},
        }
        self.output = self.root / 'configs'

    def configure(self, image_hash='a' * 64):
        self.metadata.write_text(json.dumps(self.meta))
        return minimax_config.configure(self.metadata, self.freeze, image_hash, self.output)

    def test_generated_stages_preserve_frozen_protocol(self):
        paths = self.configure()
        for path, stage, count in zip(paths, ('pilot', 'remainder'), (68, 3332)):
            config = benchmark.load_config(Path(path))
            self.assertEqual(config['refresh_sample'], {
                'stage': stage,
                'panel_sha256': driver.digest(json.loads(driver.PANEL.read_text()))})
            self.assertEqual(config['task_scope']['expected_items'], count)
            self.assertEqual(config['task_scope']['expected_tasks'], 34)
            self.assertEqual(config['generation']['max_tokens'], 256)
            self.assertEqual(config['generation']['seed'], 20260910)
            self.assertFalse(config['generation']['enable_thinking'])
            self.assertEqual(config['hardware'],
                             {'nodes': 1, 'gpu_count': 4, 'gpu_type': '6000_blackwell'})
            model = config['models']['minimax_m3_nvfp4']
            self.assertEqual(model['chat_template_kwargs'], {'thinking_mode': 'disabled'})
            self.assertEqual(model['llm_kwargs']['tensor_parallel_size'], 4)
            self.assertNotIn('truncate_prompt_tokens', config['generation'])
            self.assertEqual(config['runtime']['max_model_len'], 16384)
            self.assertEqual(config['runtime']['runtime_lock_sha256'], self.meta['runtime_lock_sha256'])
            self.assertEqual(config['runtime']['container_image_sha256'], 'a' * 64)
        driver.validate_configs(*paths, 'minimax_m3_nvfp4')
        self.assertEqual(self.configure(), paths)  # Identical restart is safe.

    def test_runtime_lock_mismatch_rejected(self):
        self.meta['runtime_lock_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'freeze does not match'):
            self.configure()
        self.assertFalse(self.output.exists())

    def test_missing_sm120_rejected(self):
        self.meta['compiled_architectures'] = 'sm_80 sm_90'
        with self.assertRaisesRegex(ValueError, 'SM120'):
            self.configure()

    def test_unknown_architecture_rejected(self):
        self.meta['minimax_registry_lines'] = ['UnknownForConditionalGeneration']
        with self.assertRaisesRegex(ValueError, 'not registered'):
            self.configure()

    def test_invalid_image_hash_rejected(self):
        for image_hash in ('short', 'z' * 64):
            with self.subTest(image_hash=image_hash), self.assertRaisesRegex(ValueError, 'Image digest'):
                self.configure(image_hash)

    def test_existing_settings_cannot_be_overwritten(self):
        paths = self.configure()
        original = [Path(path).read_bytes() for path in paths]
        with self.assertRaisesRegex(ValueError, 'fresh attempt directory'):
            self.configure('b' * 64)
        self.assertEqual([Path(path).read_bytes() for path in paths], original)


if __name__ == '__main__':
    unittest.main()
