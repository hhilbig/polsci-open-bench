import sys
import unittest
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import refresh_pilot as p


class RefreshPilotTests(unittest.TestCase):
    def test_pro_approved_failure_is_exact(self):
        row=dict(model='deepseek-v4-pro', task='erlich_ati_topics',item_id='3480',
                 parse_error='schema_invalid: Missing or unexpected output fields',truncated=False,
                 raw_content='{"Activities": 0, "Budget": 0, "Evaluation": 0, "Institutional Structure": 0, "Other": 0, "Regulatory": 1}')
        self.assertTrue(p.approved_pro_failure(row))
        for change in [dict(item_id='3481'),dict(model='deepseek-v4-flash'),dict(truncated=True),dict(raw_content='{}')]:
            self.assertFalse(p.approved_pro_failure(dict(row,**change)))

    def test_server_timeout_payload_is_retryable_but_other_errors_are_not(self):
        saved={'status':'completed','response':{'choices':None,'usage':None,'error':{
            'message':'We were unable to start processing your request within the 900-second timeout limit. Please try again later.'}}}
        self.assertTrue(p.is_timeout_attempt(saved))
        saved['response']['error']['message']='Invalid request'
        self.assertFalse(p.is_timeout_attempt(saved))
        self.assertFalse(p.is_timeout_attempt({'status':'submitting'}))

    def test_shared_timeout_limit_preserves_attempts_and_cost(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p, 'ROOT', Path(folder)):
            p.save(p.ROOT/'timeout_retries.json', {'approved_limit':10,'per_retry_cap':.03,'retries':[]})
            request={'custom_id':'r0001','cost_upper':.01}
            for i in range(10):
                model='deepseek-v4-flash' if i<6 else 'deepseek-v4-pro'
                dest=p.ROOT/model/'remainder'
                p.save(dest/'r0001.json', {'status':'error','error_type':'APITimeoutError'})
                p.reserve_timeout_retry(model,request,dest)
            self.assertAlmostEqual(p.uncertain_attempt_cost('deepseek-v4-flash','remainder'),.06)
            self.assertAlmostEqual(p.uncertain_attempt_cost('deepseek-v4-pro','remainder'),.04)
            self.assertEqual(len(p.read(p.ROOT/'timeout_retries.json')['retries']),10)
            with self.assertRaisesRegex(AssertionError,'exhausted'):
                p.reserve_timeout_retry(model,request,dest)
            self.assertTrue((dest/'r0001_timeout_retry_10.json').exists())

    def test_non_timeout_and_ambiguous_attempts_cannot_retry(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p, 'ROOT', Path(folder)):
            p.save(p.ROOT/'timeout_retries.json', {'approved_limit':10,'per_retry_cap':.03,'retries':[]})
            dest=p.ROOT/'deepseek-v4-flash/remainder'
            for state in [{'status':'submitting'}, {'status':'error','error_type':'BadRequestError'}]:
                p.save(dest/'r0001.json',state)
                with self.assertRaisesRegex(AssertionError,'Only approved'):
                    p.reserve_timeout_retry('deepseek-v4-flash',{'custom_id':'r0001','cost_upper':.01},dest)
            self.assertEqual(p.read(p.ROOT/'timeout_retries.json')['retries'],[])

    def test_interrupted_approval_is_exact_and_consumed_once(self):
        fingerprint='9d62ae82883e0174bf878d6e61c0525faf4b85424827da096b8d334ba3bb34c0'
        with tempfile.TemporaryDirectory() as folder, patch.object(p,'ROOT',Path(folder)), patch.object(p,'digest',return_value=fingerprint):
            p.save(p.ROOT/'timeout_retries.json',{'approved_limit':10,'per_retry_cap':.03,'retries':[{}]*8})
            request={'custom_id':'r2200','cost_upper':.01}
            saved={'status':'submitting','request_sha256':fingerprint,'submitted_at':'2026-09-14T19:19:18.269062+00:00'}
            self.assertTrue(p.approved_interrupted_retry('deepseek-v4-flash',request,saved))
            self.assertFalse(p.approved_interrupted_retry('deepseek-v4-pro',request,saved))
            self.assertFalse(p.approved_interrupted_retry('deepseek-v4-flash',request,dict(saved,submitted_at='later')))
            dest=p.ROOT/'deepseek-v4-flash/remainder'
            p.save(dest/'r2200.json',saved)
            p.reserve_timeout_retry('deepseek-v4-flash',request,dest)
            self.assertFalse(p.approved_interrupted_retry('deepseek-v4-flash',request,saved))
            self.assertEqual(p.read(dest/'r2200_timeout_retry_9.json')['previous_attempt'],saved)

    def test_recovery_preserves_timeout_and_submits_only_planned_requests(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as folder, patch.object(p, 'ROOT', Path(folder)), \
             patch.object(p, 'validate_sources'), patch.object(p, 'register'), \
             patch.object(p, 'now', return_value='2026-09-12T16:00:00'), \
             patch.object(p, 'load_task_definitions', return_value=[{'name':'t'}]), \
             patch.object(p, 'decode', return_value={'parse_error':None,'truncated':False,'usage_present':True}):
            dest=p.ROOT/'deepseek-v4-flash/remainder'
            original={'status':'error','error_type':'APITimeoutError'}
            p.save(dest/'r0256.json', original)
            p.save(dest/'r0000.json', {'status':'completed','response':{'unchanged':True}})
            rows=[{'custom_id':i,'task':'t','params':{'thinking':{'type':'disabled'}}}
                  for i in ['r0256','r0257']]
            c=MagicMock()
            c.chat.completions.create.return_value.model_dump.return_value={'model':'deepseek-flash'}
            with patch.object(p,'recovery_plan',return_value=({'status':'needs_attention'},rows,.002,56.17)), \
                 patch.object(p,'client',return_value=c):
                p.recover_flash()
            self.assertEqual(c.chat.completions.create.call_count,2)
            self.assertEqual(p.read(dest/'r0256_attempt1_timeout.json'),original)
            self.assertEqual(p.read(dest/'r0000.json')['response'],{'unchanged':True})
            self.assertEqual(p.read(dest/'submission.json')['status'],'completed')
            self.assertEqual(p.uncertain_attempt_cost('deepseek-v4-flash','remainder'),.002)

    def test_recovery_cannot_replay_and_cost_is_retained(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p, 'ROOT', Path(folder)):
            self.assertEqual(p.uncertain_attempt_cost('deepseek-v4-flash', 'remainder'), 0)
            p.save(p.ROOT/'deepseek-v4-flash/remainder/recovery.json',
                   {'uncertain_attempt_cost_upper': .00220528})
            self.assertEqual(p.uncertain_attempt_cost('deepseek-v4-flash', 'remainder'), .00220528)
            with self.assertRaisesRegex(AssertionError, 'already attempted'):
                p.recovery_plan()
            p.save(p.ROOT/'deepseek-v4-flash/remainder/recovery2.json',
                   {'uncertain_attempt_cost_upper': .00221672})
            self.assertAlmostEqual(p.uncertain_attempt_cost('deepseek-v4-flash', 'remainder'), .004422)
            with self.assertRaisesRegex(AssertionError, 'already attempted'):
                p.recovery_plan(2)
            with self.assertRaisesRegex(AssertionError, 'Only two'):
                p.recovery_plan(3)

    def test_finalize_requires_all_stages_to_pass(self):
        with patch.object(p, 'validate_sources'), \
             patch.object(p, 'read', return_value={'all_passed': False}), \
             patch.object(p, 'combined_predictions') as combine:
            with self.assertRaisesRegex(AssertionError, 'Unfinished'):
                p.finalize()
            combine.assert_not_called()

    def test_approved_malformed_is_scored_wrong_even_if_prediction_matches(self):
        import pandas as pd
        from build_frontier_2026 import _score_malformed_as_incorrect
        task = dict(name='t', label_kind='categorical', label_key='y', labels=['a', 'b'])
        df = pd.DataFrame([dict(gt_y='a', pred_y='a', parse_error='schema_invalid'),
                           dict(gt_y='b', pred_y='b', parse_error=None)])
        metrics = _score_malformed_as_incorrect(df, task)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['headline_f1'], .5)
        self.assertEqual(metrics['parse_ok'], 1)
        self.assertEqual(df.loc[0, 'pred_y'], 'a')

    def test_approved_failure_is_exact_and_stage_specific(self):
        import pandas as pd
        import json
        row = dict(task='agoraspeech_criticism_agenda',
                   item_id='agora_Velopoulos_2023_06_18_Volos_p9',
                   parse_error='schema_invalid: Unknown categorical label',
                   raw_content=json.dumps({'criticism_or_agenda':json.dumps({'criticism_or_agenda':'criticism'})}),
                   malformed=True, truncated=False)
        df=pd.DataFrame([row])
        self.assertTrue(p.approved_failure_mask('claude-sonnet-5','remainder',df).all())
        for model,stage in [('claude-opus-5','remainder'),('claude-sonnet-5','pilot')]:
            self.assertFalse(p.approved_failure_mask(model,stage,df).any())
        df.loc[0,'truncated']=True
        self.assertFalse(p.approved_failure_mask('claude-sonnet-5','remainder',df).any())
        df.loc[0,'truncated']=False
        df.loc[0,'raw_content']='{}'
        self.assertFalse(p.approved_failure_mask('claude-sonnet-5','remainder',df).any())

    def test_remainder_budget_is_bound_to_requests(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p,'ROOT',Path(folder)), \
             patch.object(p,'MODELS',{'m':('openai',1,1,'none')}), \
             patch.object(p,'requests_for',return_value=[{'id':1}]):
            p.save(p.ROOT/'panel.json',{})
            p.save(p.ROOT/'pilot_report.json',{'actual_cost':1})
            audit={'panel_sha256':p.digest({}),'models':[{'model':'m','remaining_cost_at_256_output_tokens':80}],
                   'requests_sha256':{'m':p.digest([{'id':1}])}}
            p.save(p.ROOT/'budget_audit.json',audit)
            self.assertEqual(p.remainder_budget('m'),(91,80))
            with patch.object(p,'requests_for',return_value=[{'id':2}]):
                with self.assertRaisesRegex(RuntimeError,'fingerprint'): p.remainder_budget('m')
            audit['models'][0]['remaining_cost_at_256_output_tokens']=90
            p.save(p.ROOT/'budget_audit.json',audit)
            with self.assertRaisesRegex(RuntimeError,'budget'): p.remainder_budget('m')
            p.save(p.ROOT/'m/remainder/submission.json',{'status':'submitted'})
            with self.assertRaisesRegex(RuntimeError,'previous stage'): p.remainder_budget('m')

    def test_flash_version_tracks_observed_alias_transition(self):
        self.assertEqual(p.DOCUMENTED_VERSIONS['deepseek-v4-flash'],'DeepSeek-V4.1-Flash')

    def test_pro_alias_change_blocks_remainder(self):
        with patch.object(p,'validate_sources'), patch.object(p,'now',return_value='2026-09-14T04:00:00+00:00'):
            with self.assertRaisesRegex(RuntimeError,'version review'):
                p.submit('remainder')

    def test_over_budget_pilot_cannot_reach_provider(self):
        with patch.object(p,'validate_sources'), patch.object(p,'count_pilot'), \
             patch.object(p,'requests_for',return_value=[{}]), \
             patch.object(p,'upper_cost',return_value=1), patch.object(p,'client') as client:
            with self.assertRaisesRegex(RuntimeError,'exceeds'):
                p.submit('pilot')
            client.assert_not_called()

    def test_failed_pilot_blocks_remainder(self):
        with patch.object(p,'validate_sources'), patch.object(p,'now',return_value='2026-09-12T00:00:00'), patch.object(p,'read',return_value={'all_passed':False}), \
             patch.object(p,'client') as client:
            with self.assertRaisesRegex(RuntimeError,'diagnosis'):
                p.submit('remainder')
            client.assert_not_called()
    def test_provider_timestamp_serializes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'submission.json'
            p.save(path,{'created_at':datetime.now(timezone.utc)})
            self.assertIsInstance(p.read(path)['created_at'],str)
    def test_selection_is_order_independent_and_has_two_distinct_pilots(self):
        items=[{'item_id':str(i),'user_content':'x'*i,'gt':{'y':0}} for i in range(150)]
        a, pilots=p.select(items,'task')
        b, other=p.select(list(reversed(items)),'task')
        self.assertEqual(a,b)
        self.assertEqual(pilots,other)
        self.assertEqual(len(a),100)
        self.assertEqual(len(pilots),2)
        lengths=sorted(a,key=lambda x:(len(x['user_content']),str(x['item_id'])))
        self.assertEqual(pilots,{lengths[49]['item_id'],lengths[-1]['item_id']})

    def test_missing_binary_field_is_invalid_not_silently_zero(self):
        task={'name':'t','label_kind':'binary','label_key':'y','labels':['y'],
              'json_schema':{'type':'object','properties':{'y':{'type':'integer','enum':[0,1]}},'required':['y']}}
        result={'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':4},'model':'gpt-5.6-sol'}
        row=p.decode('gpt-5.6-sol',{'item':{'item_id':'a','gt':{'y':1}}},result,task)
        self.assertTrue(row['parse_error'].startswith('schema_invalid'))

    def test_reasoning_length_stop_is_flagged(self):
        task={'name':'t','label_kind':'binary','label_key':'y','labels':['y'],
              'json_schema':{'type':'object','properties':{'y':{'type':'integer'}},'required':['y']}}
        result={'choices':[{'message':{'content':'{"y":1}'},'finish_reason':'length'}],
                'usage':{'prompt_tokens':100,'completion_tokens':256},'model':'gpt-6-astra'}
        row=p.decode('gpt-6-astra',{'item':{'item_id':'a','gt':{'y':1}}},result,task)
        self.assertTrue(row['truncated'])
        self.assertAlmostEqual(row['cost_usd_upper'],(100*5*1.25+256*25)/1e6)

if __name__=='__main__': unittest.main()
