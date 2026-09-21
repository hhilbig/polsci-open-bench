import json
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from audit_refresh_reuse import sha
from prepare_refresh_backfills import split_missing, build_config
from hive_vllm_benchmark import selected_tasks, load_config, validate_scope, BakeoffError
import hive_vllm_benchmark as hive


class RefreshBackfillTests(unittest.TestCase):
    def test_split_excludes_reused_and_preserves_pilot(self):
        panel = {'rows': [{'task': 't', 'item': {'item_id': str(i)}, 'pilot': i < 2} for i in range(4)]}
        actual = split_missing(panel, {('t', '0'), ('t', '2')})
        self.assertEqual(actual, {'pilot': [['t', '1']], 'remainder': [['t', '3']]})
        with self.assertRaises(ValueError):
            split_missing(panel, {('t', 'x')})

    def test_empty_stage_disallowed(self):
        with self.assertRaises(ValueError):
            build_config({}, 'm', 'p', Path('unused'), 'pilot', [])

    def test_prepared_gpt_oss_config(self):
        path = Path('output/sidecar/refresh_20260910_release/backfills/gpt_oss_120b_mxfp4/pilot.yaml')
        if not path.exists():
            self.skipTest('Run prepare_refresh_backfills.py to create local audit products')
        config = load_config(path)
        self.assertEqual(config['generation']['seed'], 20260820)
        self.assertEqual(config['generation']['max_tokens'], 256)
        model = config['models']['gpt_oss_120b_mxfp4']
        self.assertEqual(model['inference_interface'], 'vllm_responses_harmony')
        self.assertEqual(model['chat_template_kwargs']['reasoning_effort'], 'low')
        tasks = selected_tasks(config)
        scope = validate_scope(config, tasks)
        self.assertEqual(scope['total_items'], 45)

    def test_runner_subset_validation(self):
        panel = json.loads(Path('output/sidecar/refresh_20260910/panel.json').read_text())
        row = panel['rows'][0]
        key = [row['task'], str(row['item']['item_id'])]
        panel_hash = '20294ce2229b45dd3d0aa97f5e52f1ebc2292c6bae186475d998e00e8140bf5a'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'keys.json'
            def config(keys, bad_hash=False):
                path.write_text(json.dumps({'keys': keys}))
                return {'task_scope': {'tasks_dir': 'tasks'}, 'refresh_sample': {
                    'stage': 'backfill', 'panel_sha256': panel_hash,
                    'keys_manifest': str(path), 'keys_sha256': 'bad' if bad_hash else sha(path)}}
            tasks = selected_tasks(config([key]))
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]['loader'](), [row['item']])
            for keys, bad_hash in [([key, key], False), ([[key[0], 'outside-panel']], False), ([key], True)]:
                with self.assertRaises(BakeoffError):
                    selected_tasks(config(keys, bad_hash))

    def test_actual_refresh_harmony_run_and_resume(self):
        config_path = Path('output/sidecar/refresh_20260910_release/backfills/gpt_oss_120b_mxfp4/pilot.yaml')
        if not config_path.exists():
            self.skipTest('Prepare local backfill products first')
        config = load_config(config_path)
        model_key = 'gpt_oss_120b_mxfp4'
        model = config['models'][model_key]
        keys = json.loads(Path(config['refresh_sample']['keys_manifest']).read_text())['keys']
        generated_keys = []
        def generate(**kwargs):
            rows = []
            for item in kwargs['items']:
                generated_keys.append((kwargs['task']['name'], str(item['item_id'])))
                row = {'task': kwargs['task']['name'], 'model': model_key,
                       'model_id': model['model_id'], 'model_revision': model['revision'],
                       'item_id': str(item['item_id']), 'latency_s': .01,
                       'eval_count': 1, 'prompt_tokens': 1, 'parse_error': None,
                       'finish_reason': 'stop', 'structured_output_backend': 'fake',
                       'raw_content_preview': '{}'}
                for k, v in item['gt'].items():
                    row['gt_' + k] = v
                    row['pred_' + k] = v
                if len(generated_keys) == 1:
                    row['parse_error'] = 'malformed retained, not retried'
                rows.append(row)
            return pd.DataFrame(rows), {'generation_seconds': .1, 'items_per_second': len(rows) / .1,
                'max_prompt_tokens': 1, 'mean_output_tokens': 1., 'parse_ok': sum(r['parse_error'] is None for r in rows)}
        gpu = {'gpus': [{'name': 'RTX PRO 6000 Blackwell', 'memory_total_mib': 97887,
                         'memory_used_mib': 50000}], 'total_used_mib': 50000}
        sidecar = Path('output/sidecar')
        with tempfile.TemporaryDirectory(dir=sidecar) as directory:
            root = Path(directory)
            snapshot = root / model['revision']
            snapshot.mkdir()
            backend = SimpleNamespace(tokenizer=object(), stop=MagicMock())
            args = SimpleNamespace(config=config_path, model_key=model_key, output_dir=root/'result',
                panel_manifest=None, only_task=None, limit_items=None, force=False, plan_only=False)
            monitor = MagicMock(max_total_used_mib=50000)
            with patch.dict('os.environ', {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'BAKEOFF_MODEL_SNAPSHOT_PATH': str(snapshot.resolve()), 'SLURM_JOB_NUM_NODES': '1'}), \
                patch.object(hive, 'assert_runtime_versions'), patch.object(hive, 'runtime_versions', return_value={'test': 'mocked'}), \
                patch.object(hive, 'query_gpu', return_value=gpu), patch.object(hive, 'GpuMemoryMonitor', return_value=monitor), \
                patch.object(hive, 'start_vllm_responses_backend', return_value=(backend, {'inference_interface': 'vllm_responses_harmony'})) as start, \
                patch.object(hive, 'create_llm') as ordinary, \
                patch.object(hive, 'generate_task_batch', side_effect=generate), redirect_stdout(io.StringIO()):
                hive.run(args)
                first_count = len(generated_keys)
                hive.run(args)
            self.assertEqual(first_count, 45)
            self.assertEqual(len(generated_keys), first_count)
            self.assertEqual(set(generated_keys), {tuple(k) for k in keys})
            ordinary.assert_not_called()
            self.assertEqual(start.call_count, 2)
            metadata = json.loads((root/'result/run_metadata.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['rows_written'], 45)
            self.assertEqual(metadata['parse_ok'], 44)
            self.assertEqual(metadata['provenance_status'], 'qualified')
            self.assertEqual(metadata['model_snapshot_path'], str(snapshot.resolve()))


if __name__ == '__main__':
    unittest.main()
