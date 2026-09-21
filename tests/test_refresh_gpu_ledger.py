import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from build_refresh_gpu_ledger import (attempt_ids, refresh_attempt_ids,
                                      parse_sacct, build_refresh)


class RefreshGpuLedgerTests(unittest.TestCase):
    def test_all_recorded_attempts_include_failed_gpu_time(self):
        source = ('123|COMPLETED|120|billing=2,cpu=2,gres/gpu=1,mem=96G,node=1\n'
                  '123.batch|COMPLETED|120|billing=2,cpu=2,gres/gpu=1,mem=96G,node=1\n'
                  '124|FAILED|180|billing=2,cpu=2,gres/gpu=2,mem=192G,node=1\n')
        found = parse_sacct(source, ['123', '124'])
        self.assertEqual(found['123']['elapsed_seconds'], 120)
        self.assertEqual(found['124']['gpu_count'], 2)
        self.assertEqual(sum(a['elapsed_seconds'] * a['gpu_count'] for a in found.values()), 480)

    def test_missing_or_running_attempt_is_not_complete(self):
        with self.assertRaisesRegex(ValueError, 'every recorded'):
            parse_sacct('123|COMPLETED|120|gres/gpu=1\n', ['123', '124'])
        with self.assertRaisesRegex(ValueError, 'invalid'):
            parse_sacct('123|RUNNING|120|gres/gpu=1\n', ['123'])

    def test_metadata_requires_completed_numeric_attempts(self):
        self.assertEqual(attempt_ids({'status':'completed','attempts':[{'slurm_job_id':'123'}]},
                                     'pilot'), ['123'])
        with self.assertRaisesRegex(ValueError, 'not completed'):
            attempt_ids({'status':'failed','attempts':[{'slurm_job_id':'123'}]}, 'pilot')
        with self.assertRaisesRegex(ValueError, 'numeric'):
            attempt_ids({'status':'completed','attempts':[{'slurm_job_id':None}]}, 'pilot')

    def test_refresh_allows_explicit_id_only_when_container_recorded_none(self):
        missing = {'status':'completed','attempts':[{'slurm_job_id':None}]}
        self.assertEqual(refresh_attempt_ids(missing, 'pilot', ['123']),
                         (['123'], 'explicit_registry_handle'))
        with self.assertRaisesRegex(ValueError, 'explicit ID required'):
            refresh_attempt_ids(missing, 'pilot')
        recorded = {'status':'completed','attempts':[{'slurm_job_id':'123'}]}
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            refresh_attempt_ids(recorded, 'pilot', ['123'])

    def test_new_refresh_ledger_counts_failed_and_completed_attempts(self):
        base = {'status':'completed','model_key':'flash','model_id':'Qwen/Flash',
                'revision':'pinned','generation':{'max_tokens':256},
                'runtime':{'image':'pinned'},
                'gpu_after_load':{'gpus':[{'name':'Blackwell'}, {'name':'Blackwell'}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for scope, job_id in [('pilot','10'),('remainder','11')]:
                stage=root/scope
                stage.mkdir()
                (stage/'run_metadata.json').write_text(json.dumps(
                    dict(base, attempts=[{'slurm_job_id':job_id}])))
            attempts={job_id:dict(job_id=job_id,state=state,
                                  elapsed_seconds=seconds,gpu_count=2)
                      for job_id,state,seconds in [('8','FAILED',30),
                                                    ('9','FAILED',60),
                                                    ('10','COMPLETED',120),
                                                    ('11','COMPLETED',180)]}
            target=root/'ledger.json'
            with patch('build_refresh_gpu_ledger.query_sacct', return_value=attempts):
                report=build_refresh('flash',root/'pilot',root/'remainder',target,['8','9'])
            ledger=json.loads(target.read_text())
            self.assertEqual(report['attempts'],4)
            self.assertAlmostEqual(report['gpu_hours_total'],(30+60+120+180)*2/3600)
            self.assertEqual(ledger['failed_preinference_ids'],['8','9'])
            self.assertTrue(ledger['pilot_metadata_sha256'])
            attempts['9']['state']='COMPLETED'
            with patch('build_refresh_gpu_ledger.query_sacct', return_value=attempts):
                with self.assertRaisesRegex(ValueError, 'did not fail'):
                    build_refresh('flash',root/'pilot',root/'remainder',target,['8','9'])

    def test_new_refresh_ledger_records_explicit_completed_ids(self):
        base = {'status':'completed','model_key':'flash','model_id':'Qwen/Flash',
                'revision':'pinned','generation':{'max_tokens':256},
                'runtime':{'image':'pinned'},
                'gpu_after_load':{'gpus':[{'name':'Blackwell'}, {'name':'Blackwell'}]},
                'attempts':[{'slurm_job_id':None}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for scope in ['pilot','remainder']:
                stage=root/scope
                stage.mkdir()
                (stage/'run_metadata.json').write_text(json.dumps(base))
            attempts={job_id:dict(job_id=job_id,state='COMPLETED',
                                  elapsed_seconds=120,gpu_count=2)
                      for job_id in ['10','11']}
            target=root/'ledger.json'
            with patch('build_refresh_gpu_ledger.query_sacct', return_value=attempts):
                build_refresh('flash',root/'pilot',root/'remainder',target,[],['10'],['11'])
            ledger=json.loads(target.read_text())
            self.assertEqual(ledger['attempt_id_provenance'],
                             {'pilot':'explicit_registry_handle',
                              'remainder':'explicit_registry_handle'})


if __name__ == '__main__':
    unittest.main()
