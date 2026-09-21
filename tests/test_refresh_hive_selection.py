import sys
import unittest
import tempfile
import hashlib
import yaml
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import hive_vllm_benchmark as h
import refresh_pilot as p


class RefreshSelectionTests(unittest.TestCase):
    def test_refresh_schema_rejects_legacy_coercions(self):
        task = {'name': 'test', 'label_kind': 'multi_binary', 'labels': ['a', 'b'],
                '_refresh_strict_schema': True}
        for text in ['{"a": 1}', '{"a": 0.9, "b": 1}', '{"a": true, "b": 0}',
                     '{"a": 1, "b": 0, "extra": 1}']:
            self.assertIn('schema_invalid', h.parse_refresh_content(text, task)[1])
        self.assertIsNone(h.parse_refresh_content('{"a": 1, "b": 0}', task)[1])

    def test_exact_pilot_and_remainder_partition(self):
        config = h.load_config(Path('experiments/refresh_hive_20260914_pilot.yaml'))
        frozen = p.read(p.ROOT/'panel.json')
        observed = []
        for stage, count in [('pilot', 68), ('remainder', 3332)]:
            config['refresh_sample']['stage'] = stage
            tasks = h.selected_tasks(config)
            keys = [(t['name'], str(i['item_id'])) for t in tasks for i in t['loader']()]
            expected = [(r['task'], str(r['item']['item_id'])) for r in frozen['rows']
                        if bool(r['pilot']) == (stage == 'pilot')]
            self.assertEqual(set(keys), set(expected))
            self.assertEqual(len(keys), count)
            observed.extend(keys)
        self.assertEqual(len(set(observed)), 3400)

    def test_hash_drift_rejected(self):
        config = h.load_config(Path('experiments/refresh_hive_20260914_pilot.yaml'))
        config['refresh_sample']['panel_sha256'] = 'wrong'
        with self.assertRaisesRegex(h.BakeoffError, 'fingerprint'):
            h.selected_tasks(config)

    def test_multi_gpu_requires_explicit_single_node_refresh_tier(self):
        config = h.load_config(Path('experiments/refresh_hive_20260914_pilot.yaml'))
        config['models']['qwen3_8_27b_fp8']['llm_kwargs']['tensor_parallel_size'] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(h.BakeoffError):
                h.load_config(path)
            config['hardware'] = {'nodes': 1, 'gpu_count': 2}
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(h.load_config(path)['hardware']['gpu_count'], 2)
            config.pop('refresh_sample')
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(h.BakeoffError):
                h.load_config(path)

    def test_qwen_flash_pilot_pins_offload_image_and_exact_68_keys(self):
        config = h.load_config(Path('experiments/refresh_qwen_flash_next_pilot.yaml'))
        model = config['models']['qwen3_8_flash_next_fp8']
        self.assertEqual(config['hardware']['gpu_count'], 2)
        self.assertEqual(model['llm_kwargs']['tensor_parallel_size'], 2)
        self.assertEqual(model['llm_kwargs']['kernel_config'],
                         {'enable_flashinfer_autotune': False, 'moe_backend': 'triton'})
        self.assertEqual(config['runtime']['required_environment'], {'VLLM_PLE_CPU_OFFLOAD': '1'})
        self.assertEqual(h.validate_scope(config, h.selected_tasks(config))['total_items'], 68)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            model['llm_kwargs']['kernel_config']['enable_flashinfer_autotune'] = 'false'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(h.BakeoffError, 'kernel_config'):
                h.load_config(path)

    def test_qwen_flash_remainder_preserves_pilot_settings_and_exact_partition(self):
        pilot = h.load_config(Path('experiments/refresh_qwen_flash_next_pilot.yaml'))
        remainder = h.load_config(Path('experiments/refresh_qwen_flash_next_remainder.yaml'))
        for field in ['hardware','runtime','generation','models']:
            self.assertEqual(pilot[field], remainder[field], field)
        self.assertEqual(remainder['refresh_sample']['stage'], 'remainder')
        self.assertEqual(pilot['refresh_sample']['panel_sha256'],
                         remainder['refresh_sample']['panel_sha256'])
        selected = []
        for config, count in [(pilot,68),(remainder,3332)]:
            tasks = h.selected_tasks(config)
            self.assertEqual(h.validate_scope(config, tasks)['total_items'], count)
            selected.extend((task['name'], str(item['item_id']))
                            for task in tasks for item in task['loader']())
        self.assertEqual(len(selected), 3400)
        self.assertEqual(len(set(selected)), 3400)

    def test_qwen_flash_remainder_uses_writable_flashinfer_cache(self):
        wrapper = Path('experiments/refresh_qwen_flash_next_remainder.sbatch').read_text()
        self.assertIn('--home "$job_root/home:/home/hhilbig"', wrapper)
        self.assertIn('--env FLASHINFER_WORKSPACE_BASE=/job/cache', wrapper)
        self.assertIn('--env TRITON_CACHE_DIR=/job/cache/triton', wrapper)
        self.assertIn('--env TORCHINDUCTOR_CACHE_DIR=/job/cache/torchinductor', wrapper)
        self.assertIn('--env TORCH_EXTENSIONS_DIR=/job/cache/torch_extensions', wrapper)
        self.assertIn('--env CUDA_CACHE_PATH=/job/cache/cuda', wrapper)
        self.assertIn('--bind "$job_root:/job"', wrapper)

    def test_refresh_container_and_offload_checked_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.sif'
            path.write_bytes(b'pinned image fixture')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            runtime = {'required_environment': {'VLLM_PLE_CPU_OFFLOAD': '1'},
                       'container_image_sha256': expected}
            with patch.dict('os.environ', {'VLLM_PLE_CPU_OFFLOAD': '1',
                                           'BAKEOFF_CONTAINER_IMAGE_PATH': str(path)}):
                record = h.assert_refresh_runtime_requirements(runtime)
                self.assertEqual(record['container_image_sha256'], expected)
                self.assertEqual(record['required_environment']['VLLM_PLE_CPU_OFFLOAD'], '1')
                with patch.dict('os.environ', {'VLLM_PLE_CPU_OFFLOAD': '0'}):
                    with self.assertRaisesRegex(h.BakeoffError, 'environment differs'):
                        h.assert_refresh_runtime_requirements(runtime)
                path.write_bytes(b'changed image')
                with self.assertRaisesRegex(h.BakeoffError, 'SHA-256 mismatch'):
                    h.assert_refresh_runtime_requirements(runtime)

    def test_commit_verification_without_git_binary_still_checks_frozen_head(self):
        frozen = '3c7ad0756d447b1d57ed4daf26bbb85f5f296042'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_dir = root / '.git'
            git_dir.mkdir()
            (git_dir / 'HEAD').write_text('ref: refs/heads/main\n')
            branch = git_dir / 'refs' / 'heads' / 'main'
            branch.parent.mkdir(parents=True)
            branch.write_text(frozen + '\n')
            with patch.object(h, 'REPO', root), patch.object(
                    h.subprocess, 'check_output', side_effect=FileNotFoundError):
                self.assertEqual(h.assert_benchmark_commit({'benchmark_commit':frozen}), frozen)
                with self.assertRaisesRegex(h.BakeoffError, 'frozen config requires'):
                    h.assert_benchmark_commit({'benchmark_commit':'0'*40})
                branch.unlink()
                (git_dir / 'packed-refs').write_text(frozen + ' refs/heads/main\n')
                self.assertEqual(h.assert_benchmark_commit({'benchmark_commit':frozen}), frozen)
                (git_dir / 'packed-refs').write_text('')
                with self.assertRaisesRegex(h.BakeoffError, 'no readable Git HEAD'):
                    h.assert_benchmark_commit({'benchmark_commit':frozen})


if __name__ == '__main__':
    unittest.main()
