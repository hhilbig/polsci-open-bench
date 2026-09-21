import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import inspect_refresh_candidates as candidates


class CandidatePreflightTests(unittest.TestCase):
    def test_qwen27_is_completed_only_for_matching_full_validation(self):
        validation = {
            'complete': True,
            'pilot': {'passed': True, 'rows': 68, 'revision': 'pinned', 'settings': {'max_tokens': 256}},
            'remainder': {'passed': True, 'rows': 3332, 'revision': 'pinned', 'settings': {'max_tokens': 256}},
        }
        self.assertTrue(candidates.qwen27_complete(validation, 'pinned'))
        self.assertFalse(candidates.qwen27_complete(validation, 'different'))
        validation['remainder']['settings']['max_tokens'] = 512
        self.assertFalse(candidates.qwen27_complete(validation, 'pinned'))
        validation['remainder']['settings']['max_tokens'] = 256
        validation['complete'] = False
        self.assertFalse(candidates.qwen27_complete(validation, 'pinned'))

    def test_saved_qwen_flash_cpu_preflight_is_not_gpu_fit(self):
        report = candidates.qwen_flash_cpu_preflight()
        if report is None:
            self.skipTest('Qwen Flash runtime artifacts have not been collected')
        self.assertTrue(report['qwen4_registered'])
        self.assertTrue(report['thinking_disabled_template_checked'])
        self.assertFalse(report['gpu_execution_tested'])
        self.assertEqual(report['runtime_lock_sha256'],
                         json.loads(candidates.QWEN_FLASH_RUNTIME.read_text())['runtime_lock_sha256'])

    def test_changed_or_incompatible_preflight_is_rejected(self):
        original = json.loads(candidates.QWEN_FLASH_RUNTIME.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'runtime.json'
            freeze = root / 'runtime.freeze.txt'
            freeze.write_bytes(candidates.QWEN_FLASH_FREEZE.read_bytes())
            metadata.write_text(json.dumps(original))
            with patch.object(candidates, 'QWEN_FLASH_RUNTIME', metadata), \
                    patch.object(candidates, 'QWEN_FLASH_FREEZE', freeze):
                self.assertIsNotNone(candidates.qwen_flash_cpu_preflight())
                original['dependency_compatibility']['numpy'] = '2.5.3'
                metadata.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, 'compatibility'):
                    candidates.qwen_flash_cpu_preflight()


if __name__ == '__main__':
    unittest.main()
