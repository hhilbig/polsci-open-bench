"""Budgeted September 2026 API comparison; all artifacts stay in the sidecar.

Commands: prepare, submit, collect, remainder. No automatic paid retries.
"""
import argparse
import hashlib
import json
import math
import fcntl
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tiktoken
from openai import OpenAI
import anthropic

import benchmark
from anthropic_batch_submit import build_message_params, anthropic_key_map
from anthropic_token_count import token_count_params
from openai_batch_submit import build_chat_body
from task_registry import load_task_definitions
from run_registry import append_event, render_markdown, read_events, DEFAULT_STATUS_MD

ROOT = Path('output/sidecar/refresh_20260910')
SEED = 20260910
# Batch prices for OpenAI/Anthropic; peak uncached direct prices for DeepSeek.
MODELS = {
    'gpt-6-astra': ('openai', 5., 25., 'low'),
    'gpt-5.6-sol': ('openai', 2., 10., 'none'),
    'gpt-5.6-terra': ('openai', 1., 6., 'none'),
    'gpt-5.6-luna': ('openai', .1, .6, 'none'),
    'claude-sonnet-5': ('anthropic', 1., 5., 'disabled'),
    'claude-opus-5': ('anthropic', 2.5, 12.5, 'disabled'),
    'deepseek-v4-flash': ('deepseek', .44, 1.32, 'disabled'),
    'deepseek-v4-pro': ('deepseek', 1.32, 3.96, 'disabled'),
}


DOCUMENTED_VERSIONS = {'deepseek-v4-flash': 'DeepSeek-V4.1-Flash',
                       'deepseek-v4-pro': 'DeepSeek-V4-Pro-0813'}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def client(provider):
    key = benchmark._load_api_key({}, provider.upper() + '_API_KEY',
                                  Path.home() / ('.' + provider + '_api_key'))
    if not key:
        raise RuntimeError('Missing credentials: ' + provider)
    if provider == 'anthropic':
        return anthropic.Anthropic(api_key=key, max_retries=0, timeout=120)
    return OpenAI(api_key=key, max_retries=0, timeout=120,
                  base_url='https://api.deepseek.com' if provider == 'deepseek' else None)


def select(items, task):
    assert len({str(x['item_id']) for x in items}) == len(items), task
    assert len(items) >= 100, task
    ranked = sorted(items, key=lambda x: (digest([SEED, task, str(x['item_id'])]), str(x['item_id'])))[:100]
    ordered = sorted(ranked, key=lambda x: (len(x['user_content']), str(x['item_id'])))
    # Lower median for an even sample; length is Unicode character count.
    pilots = {str(ordered[49]['item_id']), str(ordered[-1]['item_id'])}
    return ranked, pilots


def body(model, task, item):
    provider, _, _, effort = MODELS[model]
    config = {'name': model, 'max_output_tokens': 256}
    prompt = Path(task['prompt_path']).read_text()
    if provider == 'anthropic':
        config['thinking_mode'] = effort
        return build_message_params(config, task, prompt, item['user_content'])
    config.update(provider=provider, reasoning_effort=effort if provider == 'openai' else None)
    if provider == 'deepseek':
        config.update(thinking_mode='disabled', response_format_type='json_object')
    result = build_chat_body(config, task, prompt, item['user_content'])
    if provider == 'openai':
        result.pop('max_tokens', None)
        result.pop('temperature', None)
        result['max_completion_tokens'] = 256
    return result


def register(model, stage, status, note=''):
    append_event(run_id=f'refresh-20260910-{model}-{stage}', event='update', status=status,
                 host='local', runner='refresh_pilot.py', model_scope=model,
                 task_scope=f'34_tasks:{68 if stage == "pilot" else 3332}_items',
                 batch_sizes='individual requests', output=str(ROOT/model/stage),
                 log=str(ROOT/model/stage/'submission.json'), cost_cap_usd=5 if stage=='pilot' else 100,
                 note=note)
    render_markdown(status_path=DEFAULT_STATUS_MD)


def prepare():
    tasks = load_task_definitions()
    assert len(tasks) == 34
    rows = []
    sources = {}
    for task in tasks:
        items = task['loader']()
        chosen, pilots = select(items, task['name'])
        sources[task['name']] = digest([items, Path(task['prompt_path']).read_text(), task['json_schema']])
        for item in chosen:
            rows.append({'task': task['name'], 'item': item, 'pilot': str(item['item_id']) in pilots})
    panel = {'seed': SEED, 'rows': rows, 'sources': sources}
    path = ROOT/'panel.json'
    if path.exists():
        assert read(path) == panel, 'Frozen panel changed'
    else:
        save(path, panel)
    by_name = {t['name']: t for t in tasks}
    enc = tiktoken.get_encoding('o200k_base')
    costs = []
    for model, (provider, inp, out, effort) in MODELS.items():
        requests = []
        for i, row in enumerate(rows):
            params = body(model, by_name[row['task']], row['item'])
            # Conservative input estimates, including protocol overhead. Anthropic
            # pilots are replaced by official counts before paid submission.
            serialized = json.dumps(params, ensure_ascii=False)
            tokens = (len(serialized.encode()) + 1024 if provider != 'openai'
                      else math.ceil(len(enc.encode(serialized)) * 1.25) + 256)
            # Budget all OpenAI input as cache writes, with no hit discount.
            upper = (tokens * inp * (1.25 if provider == 'openai' else 1) + 256*out)/1e6
            requests.append({'custom_id': f'r{i:04d}', 'task': row['task'],
                             'item': row['item'], 'pilot': row['pilot'], 'params': params,
                             'input_upper': tokens, 'cost_upper': upper})
        request_path = ROOT/model/'requests.json'
        payload = {'panel_sha256': digest(panel), 'requests': requests}
        if request_path.exists():
            assert read(request_path) == payload, 'Prepared requests changed'
        else:
            save(request_path, payload)
        costs.append({'model': model, 'pilot_upper': sum(r['cost_upper'] for r in requests if r['pilot']),
                      'full_upper': sum(r['cost_upper'] for r in requests)})
    save(ROOT/'preflight.json', {'created': now(), 'models': costs,
         'pilot_upper': sum(x['pilot_upper'] for x in costs),
         'full_upper': sum(x['full_upper'] for x in costs),
         'note': 'Conservative estimates; official Anthropic pilot counts required before submit.'})
    print(json.dumps(read(ROOT/'preflight.json'), indent=2))


def requests_for(model, stage):
    payload = read(ROOT/model/'requests.json')
    assert payload['panel_sha256'] == digest(read(ROOT/'panel.json'))
    return [r for r in payload['requests'] if r['pilot'] == (stage == 'pilot')]


def validate_sources():
    panel=read(ROOT/'panel.json')
    for task in load_task_definitions():
        current=digest([task['loader'](),Path(task['prompt_path']).read_text(),task['json_schema']])
        assert current==panel['sources'][task['name']], 'Source changed: '+task['name']


def count_pilot():
    from api_batch_common import canonical_sha256
    cache_path = ROOT/'anthropic_counts.json'
    cache = read(cache_path) if cache_path.exists() else {}
    # Reuse free historical counts only for identical input-bearing payloads.
    historical={}
    for path in Path('output/sidecar/frontier_2026').rglob('token_counts.json'):
        for record in read(path).get('counts',[]):
            if record.get('counted_payload_sha256') and 'input_tokens' in record:
                historical[record['counted_payload_sha256']]=record
    for model,(provider,*_) in MODELS.items():
        if provider!='anthropic':
            continue
        for r in read(ROOT/model/'requests.json')['requests']:
            params=token_count_params(r['params'])
            old=historical.get(canonical_sha256(params))
            if old and digest(params) not in cache:
                cache[digest(params)]={'input_tokens':old['input_tokens'],
                    'source':'previous official count with identical payload',
                    'counted_at':old.get('counted_at')}
    save(cache_path,cache)
    c = client('anthropic')
    for model, (provider, *_rest) in MODELS.items():
        if provider != 'anthropic':
            continue
        for r in requests_for(model, 'pilot'):
            params = token_count_params(r['params'])
            key = digest(params)
            if key not in cache:
                cache[key] = c.messages.count_tokens(**params).model_dump()
                save(cache_path, cache)
        print(model, 'official pilot token counts complete', flush=True)


@lru_cache(maxsize=2)
def count_cache(path, mtime_ns):
    return read(Path(path))


def upper_cost(model, r):
    provider, inp, out, _ = MODELS[model]
    if provider == 'anthropic' and (ROOT/'anthropic_counts.json').exists():
        path=ROOT/'anthropic_counts.json'
        counts = count_cache(str(path),path.stat().st_mtime_ns)
        key = digest(token_count_params(r['params']))
        if key in counts:
            return (counts[key]['input_tokens']*inp + 256*out)/1e6
    return r['cost_upper']


def remainder_budget(selected_model):
    audit = read(ROOT/'budget_audit.json')
    if audit['panel_sha256'] != digest(read(ROOT/'panel.json')):
        raise RuntimeError('Budget panel fingerprint changed')
    estimates = {x['model']: x['remaining_cost_at_256_output_tokens'] for x in audit['models']}
    if set(estimates) != set(MODELS):
        raise RuntimeError('Budget model coverage changed')
    previous = read(ROOT/'remainder_report.json') if (ROOT/'remainder_report.json').exists() else {'models':[]}
    completed = {x['model']:x for x in previous['models']}
    total = read(ROOT/'pilot_report.json')['actual_cost'] + 10
    first_unsubmitted = None
    for model in MODELS:
        if audit.get('requests_sha256',{}).get(model) != digest(requests_for(model,'remainder')):
            raise RuntimeError('Budget request fingerprint changed: ' + model)
        state_path = ROOT/model/'remainder/submission.json'
        if state_path.exists():
            result = completed.get(model,{})
            if not result.get('passed'):
                raise RuntimeError('Collect and validate previous stage before continuing: ' + model)
            total += result['cost_usd_upper']
        else:
            first_unsubmitted = first_unsubmitted or model
            total += estimates[model]
    if selected_model != first_unsubmitted:
        raise RuntimeError('Next model must be ' + str(first_unsubmitted))
    if not math.isfinite(total) or total >= 100:
        raise RuntimeError(f'Conservative total with reserve ${total:.2f} exceeds budget')
    return total, estimates[selected_model]


def submit(stage, selected_model=None):
    validate_sources()
    if stage == 'remainder':
        if now() >= '2026-09-14T04:00:00':
            raise RuntimeError('DeepSeek Pro alias changed; version review required')
        report = read(ROOT/'pilot_report.json')
        if not report['all_passed']:
            raise RuntimeError('Pilot requires diagnosis before continuation')
        total, stage_reservation = remainder_budget(selected_model)
    else:
        count_pilot()
        total = sum(upper_cost(m,r) for m in MODELS for r in requests_for(m,stage))
        if not math.isfinite(total) or total > 5:
            raise RuntimeError(f'Pilot upper cost ${total:.2f} exceeds $5')
    print(f'{stage} maximum authorized reservation: ${total:.4f}', flush=True)
    for model, (provider, *_rest) in MODELS.items():
        if stage == 'remainder' and model != selected_model:
            continue
        dest = ROOT/model/stage
        state_path = dest/'submission.json'
        if state_path.exists():
            print(model, 'already has submission record; will not resubmit', flush=True)
            continue
        rows = requests_for(model, stage)
        c = client(provider)
        register(model,stage,'running',f'User-approved budget; stage total reservation ${total:.4f}')
        state = {'status':'submitting', 'created':now(), 'requests_sha256':digest(rows),
                 'reserved_cost':stage_reservation if stage=='remainder' else sum(upper_cost(model,r) for r in rows)}
        save(state_path,state)  # A crash or timeout requires reconciliation, never blind retry.
        try:
            if provider == 'anthropic':
                result = c.messages.batches.create(requests=[{'custom_id':r['custom_id'],'params':r['params']} for r in rows])
                state.update(status='submitted',batch_id=result.id,response=result.model_dump())
            elif provider == 'openai':
                raw = '\n'.join(json.dumps({'custom_id':r['custom_id'],'method':'POST',
                     'url':'/v1/chat/completions','body':r['params']}, ensure_ascii=False) for r in rows)+'\n'
                upload=c.files.create(file=('requests.jsonl',raw.encode()),purpose='batch')
                state['file_id']=upload.id
                save(state_path,state)
                result=c.batches.create(input_file_id=upload.id,endpoint='/v1/chat/completions',completion_window='24h')
                state.update(status='submitted',batch_id=result.id,response=result.model_dump())
            else:
                for r in rows:
                    # Persist intent before each billable call. SDK retries are disabled.
                    target=dest/(r['custom_id']+'.json')
                    save(target,{'status':'submitting','request_sha256':digest(r)})
                    params=dict(r['params'])
                    params['extra_body']={'thinking':params.pop('thinking')}
                    try:
                        result=c.chat.completions.create(**params)
                        save(target,{'status':'completed','response':result.model_dump(),'completed':now()})
                    except Exception as exc:
                        save(target,{'status':'error','error':str(exc),'error_type':type(exc).__name__})
                        raise
                state['status']='completed'
            save(state_path,state)
            print(model,state['status'],flush=True)
        except Exception as exc:
            state.update(status='needs_attention',error=str(exc),error_type=type(exc).__name__)
            save(state_path,state)
            register(model,stage,'failed',type(exc).__name__ + ': inspect submission.json')
            print(model,type(exc).__name__,str(exc)[:200],flush=True)


def uncertain_attempt_cost(model, stage):
    paths = [ROOT/model/stage/name for name in ['recovery.json', 'recovery2.json']]
    cost = sum(read(path)['uncertain_attempt_cost_upper'] for path in paths if path.exists())
    ledger = ROOT/'timeout_retries.json'
    if stage == 'remainder' and ledger.exists():
        cost += sum(x['cost_upper'] for x in read(ledger)['retries'] if x['model'] == model)
    return cost


def is_timeout_attempt(saved):
    if saved.get('status') == 'error':
        return saved.get('error_type') == 'APITimeoutError'
    response = saved.get('response') or {}
    return (saved.get('status') == 'completed' and not response.get('choices') and
            (response.get('error') or {}).get('message') ==
            'We were unable to start processing your request within the 900-second timeout limit. Please try again later.')


def approved_interrupted_retry(model, request, saved):
    """Exact unknown-outcome attempt approved September 14; consumes shared slot nine."""
    return (model == 'deepseek-v4-flash' and request['custom_id'] == 'r2200' and
            digest(request) == '9d62ae82883e0174bf878d6e61c0525faf4b85424827da096b8d334ba3bb34c0' and
            saved == {'status': 'submitting',
                      'request_sha256': digest(request),
                      'submitted_at': '2026-09-14T19:19:18.269062+00:00'} and
            len(read(ROOT/'timeout_retries.json')['retries']) == 8)


def reserve_timeout_retry(model, request, dest):
    """Persist authorization use and the uncertain prior attempt before retrying."""
    path = ROOT/'timeout_retries.json'
    ledger = read(path)
    assert ledger['approved_limit'] == 10 and len(ledger['retries']) < 10, 'Timeout retry limit exhausted'
    previous = read(dest/(request['custom_id']+'.json'))
    assert is_timeout_attempt(previous) or approved_interrupted_retry(model, request, previous), 'Only approved retries allowed'
    assert request['cost_upper'] <= ledger['per_retry_cap'], 'Retry cost exceeds allowance'
    entry = dict(model=model, custom_id=request['custom_id'], request_sha256=digest(request),
                 cost_upper=request['cost_upper'], reserved_at=now(), previous_attempt=previous)
    ledger['retries'].append(entry)
    save(path, ledger)
    save(dest/(request['custom_id']+f"_timeout_retry_{len(ledger['retries'])}.json"), entry)


def bounded_deepseek(model):
    """Use the September 14 authorization: at most ten additional timeout retries."""
    assert model in ['deepseek-v4-flash', 'deepseek-v4-pro']
    validate_sources()
    ledger = read(ROOT/'timeout_retries.json')
    assert ledger['approved_limit'] == 10 and len(ledger['retries']) <= 10
    review = read(ROOT/'deepseek_version_review.json')
    assert review['versions'] == DOCUMENTED_VERSIONS, 'Version review mismatch'
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(review['checked_at'])).total_seconds()
    assert 0 <= age < 86400, 'Refresh official version and pricing review before submission'
    assert review['peak_rates'] == {'deepseek-v4-flash':[.30,1.20], 'deepseek-v4-pro':[1.32,3.96]}
    audit = read(ROOT/'budget_audit.json')
    assert audit['panel_sha256'] == digest(read(ROOT/'panel.json'))
    reports = {x['model']:x for x in read(ROOT/'remainder_report.json')['models']}
    assert {x['model'] for x in audit['models']} == set(MODELS), 'Budget model coverage changed'
    assert read(ROOT/'pilot_report.json')['all_passed'], 'Pilot requires diagnosis'
    # Retain both full original DeepSeek reservations plus all ten retry caps.
    total = read(ROOT/'pilot_report.json')['actual_cost'] + 10 + .004422 + 10*ledger['per_retry_cap']
    for e in audit['models']:
        m = e['model']
        assert audit['requests_sha256'][m] == digest(requests_for(m, 'remainder'))
        if m.startswith('deepseek'):
            total += e['remaining_cost_at_256_output_tokens']
        else:
            assert reports[m]['passed'], 'Previous model has not passed'
            total += reports[m]['cost_usd_upper']
    assert total < 56.39 and total < 100, 'Approved budget exceeded'
    if model == 'deepseek-v4-pro':
        assert reports['deepseek-v4-flash']['passed'], 'Flash must pass before Pro'
    dest = ROOT/model/'remainder'
    rows = requests_for(model, 'remainder')
    state_path = dest/'submission.json'
    if state_path.exists():
        state = read(state_path)
        server_timeout_crash = (state.get('error_type') == 'TypeError' and
                                (dest/'r2200.json').exists() and is_timeout_attempt(read(dest/'r2200.json')))
        interrupted = (state['status'] == 'submitting' and any(
            r['custom_id'] == 'r2200' and approved_interrupted_retry(model, r, read(dest/'r2200.json'))
            for r in rows) and (dest/'r2200.json').exists())
        approved_output_stop = (model == 'deepseek-v4-pro' and state.get('error_type') == 'AssertionError'
                                and state.get('error') == 'Response requires diagnosis')
        assert interrupted or (state['status'] == 'needs_attention' and (state['error_type'] == 'APITimeoutError' or server_timeout_crash or approved_output_stop)), 'Do not replay active or completed stage'
        assert state['requests_sha256'] == digest(rows)
    else:
        assert model == 'deepseek-v4-pro', 'Unexpected missing stage'
        state = dict(created=now(), requests_sha256=digest(rows), reserved_cost=next(
            e['remaining_cost_at_256_output_tokens'] for e in audit['models'] if e['model']==model))
    tasks = {t['name']:t for t in load_task_definitions()}
    pending = []
    for r in rows:
        path = dest/(r['custom_id']+'.json')
        if path.exists():
            saved = read(path)
            if saved['status'] == 'completed' and not is_timeout_attempt(saved):
                row = decode(model, r, saved['response'], tasks[r['task']])
                approved = approved_pro_failure(row)
                assert (not row['parse_error'] or approved) and not row['truncated'] and row['usage_present'], 'Saved response needs diagnosis'
                continue
            assert is_timeout_attempt(saved) or approved_interrupted_retry(model, r, saved), 'Ambiguous or unapproved failure'
        pending.append(r)
    save(dest/f"submission_before_bounded_retries_{len(ledger['retries'])}.json", state)
    state.update(status='submitting', bounded_retry_started_at=now())
    save(state_path, state)
    register(model, 'remainder', 'running', f'Approved shared ten-timeout retry limit; total reservation ${total:.6f}')
    print(f'{model}: {len(pending)} requests remain; total reservation ${total:.6f}', flush=True)
    c = client('deepseek')
    try:
        for r in pending:
            path = dest/(r['custom_id']+'.json')
            while True:
                if path.exists() and (is_timeout_attempt(read(path)) or approved_interrupted_retry(model, r, read(path))):
                    reserve_timeout_retry(model, r, dest)
                save(path, dict(status='submitting', request_sha256=digest(r), submitted_at=now()))
                params = dict(r['params'])
                params['extra_body'] = {'thinking':params.pop('thinking')}
                try:
                    response = c.chat.completions.create(**params).model_dump()
                except Exception as exc:
                    save(path, dict(status='error', error_type=type(exc).__name__, error=str(exc), failed_at=now()))
                    if type(exc).__name__ == 'APITimeoutError':
                        continue
                    raise
                save(path, dict(status='completed', response=response, completed=now()))
                if is_timeout_attempt(read(path)):
                    continue
                decoded = decode(model, r, response, tasks[r['task']])
                assert not decoded['parse_error'] and not decoded['truncated'] and decoded['usage_present'], 'Response requires diagnosis'
                expected = 'deepseek-flash' if model.endswith('flash') else 'deepseek-v4-pro'
                assert decoded['returned_model']==expected, 'Returned model changed'
                break
        state['status']='completed'
    except Exception as exc:
        state.update(status='needs_attention', error_type=type(exc).__name__, error=str(exc))
        register(model, 'remainder', 'failed', 'Bounded continuation stopped: '+type(exc).__name__)
        print(type(exc).__name__, str(exc)[:200], flush=True)
    save(state_path, state)
    print(model, state['status'], flush=True)


def recovery_plan(attempt=1):
    """Only the single Flash timeout approved by Hanno on September 12."""
    model = 'deepseek-v4-flash'
    dest = ROOT/model/'remainder'
    assert attempt in (1, 2), 'Only two specific recoveries approved'
    record = 'recovery.json' if attempt == 1 else 'recovery2.json'
    assert not (dest/record).exists(), 'Recovery already attempted; do not replay'
    if attempt == 2:
        assert read(dest/'recovery.json')['status'] == 'needs_attention'
    expected_done, retry_id, task_name, item_id = (
        (250, 'r0256', 'gilardi_stance', '1343672969885872130') if attempt == 1 else
        (1054, 'r1076', 'osnabruegge_cross_domain_topic', 'osnabruegge_cross_domain_topic_278'))
    state = read(dest/'submission.json')
    assert state['status'] == 'needs_attention' and state['error_type'] == 'APITimeoutError'
    rows = requests_for(model, 'remainder')
    assert state['requests_sha256'] == digest(rows)
    done, todo, failed = [], [], []
    for r in rows:
        f = dest/(r['custom_id']+'.json')
        if not f.exists():
            todo.append(r)
        elif read(f)['status'] == 'completed':
            done.append(r)
        else:
            failed.append(r)
    assert len(done) == expected_done and len(failed) == 1, 'Unexpected recovery scope'
    retry = failed[0]
    assert retry['custom_id'] == retry_id and retry['task'] == task_name
    assert str(retry['item']['item_id']) == item_id
    assert read(dest/(retry_id+'.json'))['error_type'] == 'APITimeoutError'
    audit = read(ROOT/'budget_audit.json')
    assert audit['panel_sha256'] == digest(read(ROOT/'panel.json'))
    reports = {x['model']: x for x in read(ROOT/'remainder_report.json')['models']}
    total = read(ROOT/'pilot_report.json')['actual_cost'] + 10 + retry['cost_upper'] + uncertain_attempt_cost(model, 'remainder')
    for entry in audit['models']:
        m = entry['model']
        assert audit['requests_sha256'][m] == digest(requests_for(m, 'remainder'))
        if m in ['deepseek-v4-flash', 'deepseek-v4-pro']:
            total += entry['remaining_cost_at_256_output_tokens']
        else:
            assert reports[m]['passed'], 'Previous stage has not passed'
            total += reports[m]['cost_usd_upper']
    assert math.isfinite(total) and total < 100, 'Recovery exceeds budget'
    return state, [retry] + todo, retry['cost_upper'], total


def recover_flash(attempt=1):
    validate_sources()
    assert now() < '2026-09-14T04:00:00', 'Version review required'
    state, pending, uncertain, total = recovery_plan(attempt)
    model = 'deepseek-v4-flash'
    dest = ROOT/model/'remainder'
    # Preserve the original records before replacing the active response slot.
    record = 'recovery.json' if attempt == 1 else 'recovery2.json'
    snapshot = 'original_submission_before_recovery.json' if attempt == 1 else 'original_submission_before_recovery2.json'
    retry_id = pending[0]['custom_id']
    save(dest/snapshot, state)
    save(dest/(retry_id+'_attempt1_timeout.json'), read(dest/(retry_id+'.json')))
    recovery = dict(approved_at=now(), started_at=now(),
                    approved_retry=retry_id, status='submitting',
                    uncertain_attempt_cost_upper=uncertain, total_reservation=total)
    save(dest/record, recovery)
    state.update(status='submitting', recovery_started_at=recovery['started_at'])
    save(dest/'submission.json', state)
    register(model, 'remainder', 'running', f'Approved one-time timeout recovery; total reservation ${total:.6f}')
    print(f'Recovery reservation ${total:.6f}; {len(pending)} requests, {3332-len(pending)} saved responses reused', flush=True)
    c = client('deepseek')
    tasks = {t['name']: t for t in load_task_definitions()}
    try:
        for r in pending:
            target = dest/(r['custom_id']+'.json')
            save(target, dict(status='submitting', request_sha256=digest(r)))
            params = dict(r['params'])
            params['extra_body'] = {'thinking': params.pop('thinking')}
            try:
                result = c.chat.completions.create(**params).model_dump()
            except Exception as exc:
                save(target, dict(status='error', error=str(exc), error_type=type(exc).__name__))
                raise
            save(target, dict(status='completed', response=result, completed=now()))
            decoded = decode(model, r, result, tasks[r['task']])
            assert not decoded['parse_error'] and not decoded['truncated'] and decoded['usage_present'], 'Response requires diagnosis'
        state['status'] = recovery['status'] = 'completed'
    except Exception as exc:
        state.update(status='needs_attention', error=str(exc), error_type=type(exc).__name__)
        recovery['status'] = 'needs_attention'
        register(model, 'remainder', 'failed', 'Recovery stopped: '+type(exc).__name__)
        print(type(exc).__name__, str(exc)[:200], flush=True)
    save(dest/'submission.json', state)
    save(dest/record, recovery)
    print(model, state['status'], flush=True)


def decode(model, request, result, task):
    provider, inp, out, _ = MODELS[model]
    if provider == 'anthropic':
        usage=result.get('usage',{})
        text=''
        for block in result.get('content',[]):
            if block['type']=='tool_use':
                reverse={v:k for k,v in anthropic_key_map(task).items()}
                text=json.dumps({reverse.get(k,k):v for k,v in block['input'].items()})
        stop=result.get('stop_reason')
        input_tokens=usage.get('input_tokens',0)
        output_tokens=usage.get('output_tokens',0)
        cost=(input_tokens*inp+usage.get('cache_creation_input_tokens',0)*inp*1.25+
              usage.get('cache_read_input_tokens',0)*inp*.1+output_tokens*out)/1e6
    else:
        choice=result.get('choices',[{}])[0]
        text=choice.get('message',{}).get('content') or ''
        stop=choice.get('finish_reason')
        usage=result.get('usage',{})
        input_tokens=usage.get('prompt_tokens',0)
        output_tokens=usage.get('completion_tokens',0)
        # Conservative billed-cost accounting where cache-write breakdown is unavailable.
        cost=(input_tokens*inp*(1.25 if provider=='openai' else 1)+output_tokens*out)/1e6
    predictions,error=benchmark.parse_content(text,task)
    try:
        obj=json.loads(text)
        expected=task['labels'] if task['label_kind']=='multi_binary' else [task['label_key']]
        if not isinstance(obj,dict) or set(obj)!=set(expected):
            raise ValueError('Missing or unexpected output fields')
        if task['label_kind']=='categorical':
            if obj[task['label_key']] not in task['labels']:
                raise ValueError('Unknown categorical label')
        elif any(type(v) is not int or v not in (0,1) for v in obj.values()):
            raise ValueError('Binary fields must be integers 0 or 1')
    except ValueError as exc:
        error='schema_invalid: ' + str(exc)[:160]
    row={'task':task['name'],'model':model,'item_id':request['item']['item_id'],
         'parse_error':error,'truncated':stop in ['length','max_tokens'],
         'stop_reason':stop,'returned_model':result.get('model'),'input_tokens':input_tokens,
         'output_tokens':output_tokens,'cost_usd_upper':cost,'raw_content':text,
         'usage_present':bool(usage),
         'documented_version':DOCUMENTED_VERSIONS.get(model,model),
         'version_documentation_checked':'2026-09-10',
         'provider_created':result.get('created')}
    row.update({'gt_'+k:v for k,v in request['item']['gt'].items()})
    row.update({'pred_'+k:v for k,v in predictions.items()})
    return row


def approved_pro_failure(row):
    """Exact missing-field response approved September 14, retained and scored wrong."""
    return (row.get('model') == 'deepseek-v4-pro' and row.get('task') == 'erlich_ati_topics'
            and str(row.get('item_id')) == '3480'
            and row.get('parse_error') == 'schema_invalid: Missing or unexpected output fields'
            and row.get('raw_content') == '{"Activities": 0, "Budget": 0, "Evaluation": 0, "Institutional Structure": 0, "Other": 0, "Regulatory": 1}'
            and not row.get('truncated'))


def approved_failure_mask(model, stage, df):
    """Hanno approved this exact Sonnet failure on 2026-09-12, scored wrong."""
    mask = pd.Series(False, index=df.index)
    if model == 'claude-sonnet-5' and stage == 'remainder' and len(df):
        mask = (df.task.eq('agoraspeech_criticism_agenda') &
                df.item_id.eq('agora_Velopoulos_2023_06_18_Volos_p9') &
                df.parse_error.eq('schema_invalid: Unknown categorical label') &
                df.raw_content.eq(json.dumps({'criticism_or_agenda':
                    json.dumps({'criticism_or_agenda': 'criticism'})})) &
                df.malformed & ~df.truncated)
    if model == 'deepseek-v4-pro' and stage == 'remainder' and len(df):
        mask = df.apply(approved_pro_failure, axis=1) & df.malformed & ~df.truncated
    return mask


def collect(stage):
    validate_sources()
    from api_prediction_validation import prediction_malformed_masks
    tasks=load_task_definitions()
    by_name={t['name']:t for t in tasks}
    reports=[]
    for model,(provider,*_) in MODELS.items():
        dest=ROOT/model/stage
        path=dest/'submission.json'
        if not path.exists():
            reports.append({'model':model,'passed':False,'status':'not_submitted'})
            continue
        state=read(path)
        rows=requests_for(model,stage)
        assert digest(rows)==state['requests_sha256']
        if state['status']=='submitting':
            reports.append({'model':model,'passed':False,'status':'submitting_or_requires_reconciliation',
                            'reserved_cost':state['reserved_cost']})
            continue
        responses={}
        if 'batch_id' in state:
            c=client(provider)
            b=c.messages.batches.retrieve(state['batch_id']) if provider=='anthropic' else c.batches.retrieve(state['batch_id'])
            save(dest/'batch_status.json',b.model_dump())
            done=b.processing_status=='ended' if provider=='anthropic' else b.status in ['completed','failed','expired','cancelled']
            if not done:
                reports.append({'model':model,'passed':False,'status':'pending',
                                'reserved_cost':state['reserved_cost']})
                continue
            raw_path=dest/'batch_results.json'
            if raw_path.exists():
                raw=read(raw_path)
            elif provider=='anthropic':
                raw=[x.model_dump() for x in c.messages.batches.results(b.id)]
                save(raw_path,raw)
            else:
                raw=[]
                for fid in [b.output_file_id,b.error_file_id]:
                    if fid:
                        raw.extend(json.loads(line) for line in c.files.content(fid).text.splitlines() if line)
                save(raw_path,raw)
            for x in raw:
                if x['custom_id'] in responses or x['custom_id'] not in {r['custom_id'] for r in rows}:
                    raise RuntimeError('Duplicate or unknown provider result identifier')
                if provider=='anthropic':
                    responses[x['custom_id']]=x.get('result',{}).get('message')
                else:
                    responses[x['custom_id']]=(x.get('response') or {}).get('body')
        else:
            for r in rows:
                f=dest/(r['custom_id']+'.json')
                if f.exists():
                    responses[r['custom_id']]=read(f).get('response')
        decoded=[decode(model,r,responses[r['custom_id']],by_name[r['task']]) for r in rows
                 if responses.get(r['custom_id']) and 'error' not in responses[r['custom_id']]]
        df=pd.DataFrame(decoded)
        if len(df):
            df['run_submitted_at']=state['created']
            malformed,_=prediction_malformed_masks(df,tasks)
            df['malformed']=malformed
            df.to_csv(dest/'predictions.csv',index=False)
            bad=int(malformed.sum()); truncated=int(df.truncated.sum())
            cost=float(df.cost_usd_upper.sum())
        else:
            bad=truncated=0; cost=0
        approved = approved_failure_mask(model, stage, df)
        approved_count = int(approved.sum())
        passed=len(df)==len(rows) and bad==approved_count and truncated==0 and bool(df.usage_present.all())
        projected=None
        projected_capped=None
        if stage=='pilot' and len(df)==68:
            _,input_rate,output_rate,_=MODELS[model]
            input_factor=1.25 if provider=='openai' else 1
            # Descriptive extrapolation, never the authoritative spending gate:
            # equal task weights; each pilot deliberately includes a long text.
            projected=float(df.groupby('task').cost_usd_upper.mean().sum()*100)
            projected_input=float(df.groupby('task').input_tokens.mean().sum()*100)
            projected_capped=(projected_input*input_rate*input_factor+3400*256*output_rate)/1e6
        reports.append({'model':model,'status':'passed' if passed else 'needs_attention','passed':passed,
                        'expected':len(rows),'observed':len(df),'malformed':bad,'truncated':truncated,
                        'approved_failures_scored_incorrect':approved_count,
                        'cost_usd_upper':cost + uncertain_attempt_cost(model, stage),
                        'uncertain_attempt_cost_upper':uncertain_attempt_cost(model, stage),
                        'reserved_cost':state['reserved_cost'],
                        'input_tokens':int(df.input_tokens.sum()) if len(df) else 0,
                        'output_tokens':int(df.output_tokens.sum()) if len(df) else 0,
                        'returned_models':sorted(df.returned_model.dropna().unique().tolist()) if len(df) else [],
                        'projected_full_cost_from_pilot':projected,
                        'projected_full_cost_at_output_cap':projected_capped,
                        'full_panel_preflight_upper':sum(upper_cost(model,r) for r in read(ROOT/model/'requests.json')['requests'])})
        register(model,stage,'completed' if passed else 'failed',f'{len(df)}/{len(rows)} rows; {bad} malformed; {truncated} truncated; cost upper ${cost:.4f}')
    report={'models':reports,'all_passed':all(x['passed'] for x in reports),
            'actual_cost':sum(x.get('cost_usd_upper',0) for x in reports),
            'pending_reserved_cost':sum(x.get('reserved_cost',0) for x in reports if 'cost_usd_upper' not in x),
            'cost_note':'Usage-based upper estimate; no assumed cache discounts; not an invoice.'}
    save(ROOT/(stage+'_report.json'),report)
    print(json.dumps(report,indent=2))


def combined_predictions(model):
    """Validate saved stages against the frozen requests before scoring."""
    from api_prediction_validation import prediction_malformed_masks
    tasks = load_task_definitions()
    frames = []
    for stage in ['pilot', 'remainder']:
        rows = requests_for(model, stage)
        dest = ROOT/model/stage
        assert read(dest/'submission.json')['requests_sha256'] == digest(rows)
        df = pd.read_csv(dest/'predictions.csv', dtype={'item_id': str})
        expected = {(r['task'], str(r['item']['item_id'])): r for r in rows}
        keys = list(zip(df.task, df.item_id))
        assert len(keys) == len(set(keys)) and set(keys) == set(expected), 'Coverage mismatch'
        assert df.model.eq(model).all(), 'Model mismatch'
        assert df.usage_present.eq(True).all() and df.returned_model.notna().all()
        assert df.truncated.eq(False).all(), 'Truncation'
        assert df.output_tokens.between(1, 256).all(), 'Output token allowance'
        assert df.input_tokens.gt(0).all() and df.cost_usd_upper.ge(0).all()
        for record in df.to_dict('records'):
            gold = expected[(record['task'], record['item_id'])]['item']['gt']
            for key, value in gold.items():
                assert record['gt_'+key] == value, 'Gold label mismatch'
        malformed, _ = prediction_malformed_masks(df, tasks)
        assert malformed.equals(df.malformed), 'Malformed flags changed'
        assert (malformed == approved_failure_mask(model, stage, df)).all(), 'Unapproved failure'
        df['stage'] = stage
        if model.startswith('deepseek'):
            completed = []
            for key in keys:
                request = expected[key]
                saved = read(dest/(request['custom_id']+'.json'))
                assert saved['status'] == 'completed' and saved.get('completed'), 'Missing observed completion date'
                datetime.fromisoformat(saved['completed'])
                completed.append(saved['completed'])
            df['observed_completed_at'] = completed
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == 3400 and not combined.duplicated(['task', 'item_id']).any()
    counts = combined.groupby('task').size()
    assert len(counts) == 34 and counts.eq(100).all()
    return combined


def finalize():
    """Write the comparison only after every model passes the final audit."""
    from build_frontier_2026 import _score_malformed_as_incorrect
    validate_sources()
    for stage in ['pilot', 'remainder']:
        report = read(ROOT/(stage+'_report.json'))
        assert report['all_passed'], 'Unfinished or failed stage'
        assert {r['model'] for r in report['models']} == set(MODELS)
    tasks = {t['name']: t for t in load_task_definitions()}
    predictions, scores, totals = [], [], []
    for model in MODELS:
        df = combined_predictions(model)
        predictions.append(df)
        model_scores = []
        for name, group in df.groupby('task', sort=True):
            metrics = _score_malformed_as_incorrect(group, tasks[name])
            # Latency was not measured by this API run.
            metrics.pop('mean_latency_s', None)
            metrics.pop('median_latency_s', None)
            scores.append(dict(model=model, task=name, **metrics))
            model_scores.append(metrics['headline_f1'])
        totals.append(dict(model=model, n=len(df), tasks=df.task.nunique(),
                           mean_task_f1=sum(model_scores)/len(model_scores),
                           malformed=int(df.malformed.sum()), truncated=int(df.truncated.sum()),
                           input_tokens=int(df.input_tokens.sum()), output_tokens=int(df.output_tokens.sum()),
                           cost_usd_upper=float(df.cost_usd_upper.sum()) + uncertain_attempt_cost(model, 'remainder'),
                           uncertain_attempt_cost_upper=uncertain_attempt_cost(model, 'remainder'),
                           returned_models=sorted(df.returned_model.unique().tolist()),
                           documented_versions=sorted(df.documented_version.unique().tolist()),
                           run_submitted_dates=sorted(df.run_submitted_at.str[:10].unique().tolist()),
                           observed_completion_dates=(sorted(df.observed_completed_at.str[:10].unique().tolist())
                                                      if 'observed_completed_at' in df else [])))
    cost = sum(r['cost_usd_upper'] for r in totals)
    assert cost + 10 < 100, 'Final budget exceeded'
    # Publish only after all eight models have passed the preceding checks.
    pd.concat(predictions, ignore_index=True).to_csv(ROOT/'comparison_predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(ROOT/'comparison_task_scores.csv', index=False)
    save(ROOT/'comparison_report.json', dict(
        completed_at=now(), panel_sha256=digest(read(ROOT/'panel.json')), models=totals,
        cost_usd_upper=cost, reserve_usd=10, total_with_reserve=cost+10,
        scoring='Existing task headline F1, averaged equally across 34 tasks. Approved malformed response scored incorrect.',
        cost_note='Usage-based conservative estimate, not an invoice. Raw provider results and failed attempts retained.'))
    print(json.dumps(read(ROOT/'comparison_report.json'), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','submit','collect','remainder','collect-remainder','finalize','recover-flash','recover-flash-second','bounded-deepseek'])
    p.add_argument('--model',choices=list(MODELS),help='Required for one-model-at-a-time remainder submission')
    args=p.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='finalize': finalize()
    elif args.command in ['submit','remainder','recover-flash','recover-flash-second','bounded-deepseek']:
        ROOT.mkdir(parents=True,exist_ok=True)
        with (ROOT/'submit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == 'bounded-deepseek': bounded_deepseek(args.model)
            elif args.command.startswith('recover-flash'): recover_flash(2 if args.command.endswith('-second') else 1)
            else: submit('pilot' if args.command=='submit' else 'remainder',args.model)
    else: collect('remainder' if args.command=='collect-remainder' else 'pilot')
