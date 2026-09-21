import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from collect_refresh_backfills import (MANIFEST, AUDIT, PANEL, collect, collect_model, comparable_effective_model,
                                       comparable_runtime, reused_frame, verify_stage_keys)


class BackfillCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in [MANIFEST, AUDIT, PANEL]:
            if not path.exists():
                raise unittest.SkipTest('Prepared reuse and backfill manifests are required')
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.audit = json.loads(AUDIT.read_text())
        cls.panel = json.loads(PANEL.read_text())
        cls.entry = next(e for e in cls.manifest['models'] if e['model_key'] == 'gpt_oss_120b_mxfp4')

    def test_historical_selection_and_exact_missing_stage_keys(self):
        historical, metadata = reused_frame(self.entry, self.audit, self.panel)
        self.assertEqual(len(historical), self.entry['reused_rows'])
        self.assertTrue(metadata)
        expected = verify_stage_keys(self.entry, self.panel,
                                     set(zip(historical.task, historical.item_id)))
        self.assertEqual(len(expected['pilot']), self.entry['stages']['pilot']['rows'])
        self.assertEqual(len(expected['remainder']), self.entry['stages']['remainder']['rows'])
        self.assertEqual(len(historical) + len(expected['pilot']) + len(expected['remainder']), 3400)

    def test_changed_stage_key_manifest_is_rejected(self):
        historical, _ = reused_frame(self.entry, self.audit, self.panel)
        altered = json.loads(json.dumps(self.entry))
        altered['stages']['pilot']['rows'] += 1
        with self.assertRaisesRegex(ValueError, 'manifest row count'):
            verify_stage_keys(altered, self.panel, set(zip(historical.task, historical.item_id)))

    def test_runtime_and_effective_settings_mismatch_is_rejected(self):
        _, metadata = reused_frame(self.entry, self.audit, self.panel)
        source = metadata[0]
        runtime = json.loads(json.dumps(source['runtime_versions']))
        runtime['nvidia_driver'] = 'different-node-driver'
        comparable_runtime(source, runtime)
        runtime['packages']['vllm'] = 'different'
        with self.assertRaisesRegex(ValueError, 'packages'):
            comparable_runtime(source, runtime)
        effective = json.loads(json.dumps(source))
        comparable_effective_model(source, effective)
        effective['llm_kwargs']['max_num_seqs'] += 1
        with self.assertRaisesRegex(ValueError, 'effective model'):
            comparable_effective_model(source, effective)

    def test_missing_stage_writes_incomplete_report_without_ranked_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = collect(self.entry['model_key'], pilot_dir=root / 'missing-pilot',
                             remainder_dir=root / 'missing-remainder', output=root / 'export')
            self.assertFalse(report['complete'])
            self.assertIn('not produced run metadata', report['reason'])
            self.assertTrue((root / 'export' / 'validation.json').exists())
            self.assertFalse((root / 'export' / 'open_inputs.json').exists())
            self.assertFalse((root / 'export' / 'predictions.csv').exists())

    def test_full_union_requires_and_exports_exact_3400_keys(self):
        historical, metadata = reused_frame(self.entry, self.audit, self.panel)
        missing = verify_stage_keys(self.entry, self.panel, set(zip(historical.task, historical.item_id)))
        frozen = {(r['task'], str(r['item']['item_id'])): r['item'] for r in self.panel['rows']}

        def synthetic(stage):
            rows = []
            for task, item_id in missing[stage]:
                item = frozen[(task, item_id)]
                row = {column: '' for column in historical.columns}
                row.update(task=task, item_id=item_id, model=self.entry['model_key'],
                           model_id=self.entry['model_id'], model_revision=self.entry['revision'],
                           parse_error='', eval_count='1', prompt_tokens='1')
                for label, gold in item['gt'].items():
                    row['gt_' + label] = str(gold)
                    row['pred_' + label] = str(gold)
                rows.append(row)
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = {}
            paths = {}
            for stage in ['pilot', 'remainder']:
                frames[stage] = synthetic(stage)
                paths[stage] = root / stage
                paths[stage].mkdir()
                meta = json.loads(json.dumps(metadata[0]))
                meta['started_at'] = '2026-09-15T00:00:00Z'
                meta['completed_at'] = '2026-09-15T00:01:00Z'
                (paths[stage] / 'run_metadata.json').write_text(json.dumps(meta))

            def checked_stage(config_path, directory):
                stage = Path(config_path).stem
                report = {'passed': True, 'rows': len(frames[stage])}
                if stage == 'remainder':
                    report.update(generation_seconds=100., generation_tokens_per_second=50.,
                                  output_token_usage_unavailable=[],
                                  hardware=metadata[0]['gpu_after_load'])
                return report, frames[stage]

            with patch('collect_refresh_backfills.audit_stage', side_effect=checked_stage):
                report, export = collect_model(self.entry, self.audit, self.panel,
                                               paths, root / 'export')
            self.assertTrue(report['complete'])
            self.assertEqual(report['rows'], 3400)
            self.assertEqual(export['quantization'], 'MXFP4')
            self.assertTrue(export['provenance_validated'])
            self.assertEqual(export['throughput_items'], len(missing['remainder']))
            self.assertAlmostEqual(export['generation_items_per_second'],
                                   len(missing['remainder']) / 100.)
            self.assertEqual(export['generation_tokens_per_second'], 50.)
            actual = pd.read_csv(export['predictions'], dtype=str, keep_default_na=False)
            self.assertEqual(len(actual), 3400)
            self.assertEqual(len(set(zip(actual.task, actual.item_id))), 3400)

    def test_sacct_ledger_requires_all_attempts_and_correct_totals(self):
        name = 'deepseek_r1_distill_qwen_32b_bf16'
        root = MANIFEST.parent.parent
        pilot = root / 'imported_hive_20260915' / 'backfill_pilots' / name
        remainder = root / 'imported_hive_20260915' / 'backfill_remainders' / name
        source = root / 'backfills' / name / 'collected' / 'sacct_attempts.json'
        if not all(path.exists() for path in [pilot, remainder, source]):
            self.skipTest('Imported completed stages and sacct ledger are required')
        original = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for label, changed, expected in [
                ('missing_attempt', dict(original, attempts=original['attempts'][:-1]), 'attempt coverage'),
                ('wrong_total', dict(original, gpu_hours_total=original['gpu_hours_total'] + 1),
                 'total GPU hours')]:
                ledger = temporary / (label + '.json')
                ledger.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, expected):
                    collect(name, pilot_dir=pilot, remainder_dir=remainder,
                            output=temporary / label, sacct=ledger)
                self.assertFalse((temporary / label / 'open_inputs.json').exists())


if __name__ == '__main__':
    unittest.main()
