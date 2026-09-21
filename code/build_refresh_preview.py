"""Build a static, aggregate-only homepage preview. Never deploys the page."""
from __future__ import annotations

import argparse
import html
import hashlib
import json
import csv
from datetime import date
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / 'output/sidecar/refresh_20260910_release_33'
# Standard-rate API costs with their evidence label, shared with the README.
COST_TABLE = ROOT / 'output/sidecar/jev_sidecar/cost_performance.csv'
# Panel size is read from the release (see set_panel), not assumed: the
# 34-task panel became 33 when halterman_ccc_protest was held out.
N_TASKS = 34
N_ITEMS = 3400
FEATURED_MODELS = frozenset([
    'gpt-6-astra','gpt-5.6-sol','gpt-5.6-terra','gpt-5.6-luna',
    'claude-sonnet-5','claude-opus-5','deepseek-v4-flash','deepseek-v4-pro',
    'jev-1.13.0','gemini-3.8-flash','gemini-3.1-flash-lite',
    'qwen3_8_27b_fp8','qwen3_8_flash_next_fp8','gemma4_31b_it_qat_w4a16',
    'mistral_small_4_119b_nvfp4','qwen3_6_27b_fp8',
    'llama3_1_70b_instruct_fp8_dynamic_full34','llama3_3_70b_instruct_fp8_dynamic'])
def homepage_link_proposal():
    return ('<li><b><a href="llm-benchmark/">Political Science LLM Benchmark</a></b> '
            f'2026. Matched evaluation of language models on {N_TASKS} political-science text-coding tasks.</li>\n')
SITEMAP_ENTRY_PROPOSAL = ('<url>\n  <loc>https://www.hannohilbig.com/llm-benchmark/</loc>\n'
                          '  <priority>0.8</priority>\n</url>\n')
LABELS={'gpt-6-astra':'GPT-6 Astra','gpt-5.6-sol':'GPT-5.6 Sol','gpt-5.6-terra':'GPT-5.6 Terra','gpt-5.6-luna':'GPT-5.6 Luna','claude-sonnet-5':'Claude Sonnet 5','claude-opus-5':'Claude Opus 5','deepseek-v4-flash':'DeepSeek V4 Flash','deepseek-v4-pro':'DeepSeek V4 Pro','jev-1.13.0':'Jev 1.13','gemini-3.8-flash':'Gemini 3.8 Flash','gemini-3.1-flash-lite':'Gemini 3.1 Flash-Lite'}
LABELS.update({'gemma4_31b_it_qat_w4a16':'Gemma 4 31B (W4A16)','glm4_7_flash':'GLM-4.7-Flash','llama3_1_70b_instruct_fp8_dynamic_full34':'Llama 3.1 70B (FP8)','mistral_small_4_119b_nvfp4':'Mistral Small 4 (NVFP4)','qwen3_30b_a3b_instruct_2507_fp8':'Qwen3 30B-A3B 2507 (FP8)','qwen3_6_27b_fp8':'Qwen3.6 27B (FP8)','qwen3_6_35b_a3b_fp8':'Qwen3.6 35B-A3B (FP8)'})
LABELS['qwen3_8_27b_fp8']='Qwen3.8 27B (FP8)'
LABELS['qwen3_8_flash_next_fp8']='Qwen3.8 Flash-Next (FP8)'
LABELS.update({
    'deepseek_r1_distill_qwen_32b_bf16':'DeepSeek-R1 Distill Qwen 32B (BF16)',
    'gemma3_27b_it_fp8_dynamic':'Gemma 3 27B (dynamic FP8)',
    'llama3_3_70b_instruct_fp8_dynamic':'Llama 3.3 70B (dynamic FP8)',
    'mistral_small_3_1_24b_bf16':'Mistral Small 3.1 24B (BF16)',
    'qwen1_5_32b_chat':'Qwen 1.5 32B Chat',
    'qwen2_5_32b_instruct_bf16':'Qwen 2.5 32B Instruct (BF16)',
    'qwen2_5_72b_instruct_fp8_dynamic':'Qwen 2.5 72B Instruct (dynamic FP8)',
    'qwen3_30b_a3b_bf16':'Qwen3 30B-A3B (BF16)',
    'qwen3_32b_bf16':'Qwen3 32B (BF16)',
    'qwen3_5_35b_a3b_fp8':'Qwen3.5 35B-A3B (FP8)',
    'qwen3_next_80b_a3b_fp8':'Qwen3-Next 80B-A3B (FP8)',
})

CSS = '''
:root{--text:#222;--muted:#666;--link:#176ca4;--accent:#3498db;--border:#d8d8d8}
*{box-sizing:border-box}body{margin:0;padding:36px 20px 56px;background:#fff;color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:16px;line-height:1.55}.wrapper{max-width:760px;margin:0 auto}a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,button:focus-visible,select:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--link);outline-offset:3px}h1,h2,h3{color:#111;font-weight:650;line-height:1.2}h1{margin:0 0 .45rem;font-size:2rem}h2,h3{margin:2.2rem 0 .8rem;padding-top:1.1rem;border-top:1px solid var(--border);font-size:1.25rem}p,ul,table{margin:0 0 1.1rem}header{margin-bottom:2rem}.title,small,footer,.note{color:var(--muted)}nav{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}.notice{border-left:3px solid var(--accent);padding:.4rem 1rem}.controls{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}label{display:flex;flex-direction:column;font-size:.9rem}select{font:inherit;color:inherit;background:white;border:1px solid #888;padding:.4rem;max-width:100%}.table-scroll{overflow-x:auto;margin-bottom:1rem}table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{padding:.6rem .65rem .6rem 0;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}.numeric{font-variant-numeric:tabular-nums;white-space:nowrap}th button{font:inherit;font-weight:650;border:0;background:none;color:var(--link);padding:0;cursor:pointer}caption{text-align:left;font-weight:650;margin:.5rem 0}.model-label{min-width:145px}td small{display:block;font-size:.8rem}.bar{display:inline-block;height:5px;background:var(--accent);max-width:90px;vertical-align:middle;margin-right:6px}footer{margin-top:2.6rem;font-size:.9rem}.skip-link{position:absolute;left:-10000px}.skip-link:focus{left:20px;top:5px;background:white}.js-only{display:none}.js .js-only{display:flex}[hidden]{display:none!important}.tradeoff{display:grid;grid-template-columns:1fr;gap:.3rem}ul{padding-left:1.35rem}li{margin-bottom:.6rem}details{margin:.8rem 0}summary{cursor:pointer;color:var(--link)}@media(max-width:560px){body{padding:24px 16px 42px;font-size:15px}h1{font-size:1.8rem}.controls{display:block}label{margin-bottom:.6rem}.table-scroll table{min-width:620px}}@media print{body{padding:.4in}.controls{display:none!important}a{color:#111}}
.sort-button{display:none}.js .sort-button{display:inline}.js .static-heading{display:none}
.class-support-details{display:none}.js .class-support-details{display:block}
'''

JS = '''
'use strict';
document.documentElement.classList.add('js');
const data=JSON.parse(document.getElementById('benchmark-data').textContent);
const byId=id=>document.getElementById(id);
const num=x=>x===null||x===undefined?'Unavailable':Number(x).toFixed(3);
const makeCell=(row,text)=>{const c=document.createElement('td');c.textContent=text;row.appendChild(c);return c;};
const labels=Object.fromEntries(data.models.map(m=>[m.model,m.label||m.model]));
let order={key:'mean_task_f1',ascending:false};
function filterModels(){
 const kind=byId('kind').value,hardware=byId('hardware').value;
 const rows=[...byId('model-rows').children];
 rows.forEach(r=>{r.hidden=!!((kind&&r.dataset.kind!==kind)||(hardware&&r.dataset.hardware!==hardware));});
 const visible=rows.filter(r=>!r.hidden).length;
 byId('model-count').textContent=visible?`${visible} models shown.`:'No models match these filters. Select All to restore results.';
}
byId('kind').addEventListener('change',filterModels);byId('hardware').addEventListener('change',filterModels);
document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{
 const key=button.dataset.sort;order={key,ascending:order.key===key?!order.ascending:key==='label'};
 const values=Object.fromEntries(data.models.map(m=>[m.model,m]));
 [...byId('model-rows').children].sort((a,b)=>{const x=values[a.dataset.model][key],y=values[b.dataset.model][key];if(x==null)return y==null?0:1;if(y==null)return -1;return (typeof x==='string'?x.localeCompare(y):x-y)*(order.ascending?1:-1);}).forEach(r=>byId('model-rows').appendChild(r));
 document.querySelectorAll('[aria-sort]').forEach(th=>th.setAttribute('aria-sort','none'));
 button.parentNode.setAttribute('aria-sort',order.ascending?'ascending':'descending');
}));
function taskOptions(){
 const category=byId('category').value;
 const tasks=[...new Set(data.task_scores.filter(r=>!category||r.category===category).map(r=>r.task))].sort();
 byId('task').replaceChildren(...tasks.map(t=>{const o=document.createElement('option');o.value=t;o.textContent=t.replaceAll('_',' ');return o;}));
 showTask();
}
function showTask(){
 const task=byId('task').value;byId('task-rows').replaceChildren();
 const selected=r=>!byId('task-all')||byId('task-all').checked||featuredModels.has(r.model);
 const rows=data.task_scores.filter(r=>r.task===task&&selected(r)).sort((a,b)=>(b.f1??-1)-(a.f1??-1));
 rows.forEach(r=>{const tr=document.createElement('tr');[labels[r.model]||r.model,num(r.f1),num(r.accuracy),num(r.mcc),String(r.n)].forEach(v=>makeCell(tr,v));byId('task-rows').appendChild(tr);});
 byId('class-rows').replaceChildren();
 data.class_scores.filter(r=>r.task===task&&selected(r)).forEach(r=>{const tr=document.createElement('tr');[labels[r.model]||r.model,r.label,String(r.support),num(r.f1)].forEach(v=>makeCell(tr,v));byId('class-rows').appendChild(tr);});
 byId('task-description').textContent=rows.length?`${task.replaceAll('_',' ')}: ${rows[0].n} texts per model. Class results with small support are descriptive.`:'No completed results available for this selection.';
}
byId('category').addEventListener('change',taskOptions);byId('task').addEventListener('change',showTask);taskOptions();filterModels();
byId('task-all')?.addEventListener('change',showTask);
'''

def number_word(n):
    return ['no','one','two','three','four','five','six','seven','eight','nine'][n] if 0<=n<10 else str(n)

def lead_sentences(models):
    """State the headline result from the ranked models rather than hard-coding it."""
    if not models:
        return 'No model has completed the panel yet.'
    name=lambda m: esc(m.get('label') or m['model'])
    text=f"{name(models[0])} has the highest mean F1 ({models[0]['mean_task_f1']:.3f})."
    opened=[m for m in models if m['kind']=='open']
    api=[m for m in models if m['kind']=='API']
    if opened and api:
        best=opened[0];below=sum(m['mean_task_f1']<best['mean_task_f1'] for m in api)
        text+=(f" The best open-weight model, {name(best)}, reaches {best['mean_task_f1']:.3f}"
               f" and scores above {number_word(below)} of the {len(api)} API models.")
    return text

def esc(value):
    return html.escape(str(value), quote=True)

def fmt(value, digits=3):
    return 'Unavailable' if value is None else f'{float(value):.{digits}f}'

def interval(model, kind):
    low, high = model.get(f'ci_{kind}_low'), model.get(f'ci_{kind}_high')
    return 'Unavailable' if low is None or high is None else f'{fmt(low)}–{fmt(high)}'

def hardware_label(tier):
    return {'api':'API service','single-gpu':'Single GPU',
            'multi-gpu':'Multiple GPUs'}.get(tier,tier)

def safe_payload(data):
    """Whitelist aggregate fields; never embed arbitrary release/source objects."""
    fields = {
        'models': ('model','label','kind','hardware_tier','hardware','n','tasks','mean_task_f1','ci_item_low','ci_item_high','ci_task_low','ci_task_high','malformed','cost_usd_upper','cost_per_1k_items','cost_basis','throughput_tokens_per_second','throughput_items_per_second','throughput_items','throughput_keyset_sha256','throughput_settings_sha256','throughput_hardware','documented_versions','observed_run_dates'),
        'task_scores': ('model','task','category','n','f1','accuracy','mcc'),
        'class_scores': ('model','task','label','support','f1'),
        'candidates': ('model','status','reason'),
        'categories': ('model','category','tasks','mean_task_f1','item_ci_low','item_ci_high','task_ci_low','task_ci_high'),
        'pairs': ('model','reference','difference','item_ci_low','item_ci_high','task_ci_low','task_ci_high'),
    }
    result={key:[{f:r.get(f) for f in keep} for r in data.get(key,[])] for key,keep in fields.items()}
    result.update({k:data.get(k) for k in ('release_id','status','updated_at')})
    for m in result['models']:
        # Every ranked model must cover the same complete panel of 100 texts per task.
        if (m['n'], m['tasks']) != (result['models'][0]['n'], result['models'][0]['tasks']) or m['n'] != 100*m['tasks']:
            raise ValueError(f"Incomplete model cannot enter ranking: {m['model']}")
        if not m['kind'] or not m['hardware_tier']:
            raise ValueError('Model kind and hardware tier must be explicit')
    return result

COST_BASIS_LABELS = {'provider_ledger':'Provider billing record','provider_reported':'Provider-reported cost',
                     'provider_tokens':'Estimate: provider token counts times published price',
                     'token_estimate':'Estimate: our token counts times published price'}

def standard_costs():
    if not COST_TABLE.exists():
        return {}
    with COST_TABLE.open(newline='') as handle:
        return {r['model']:(float(r['cost_per_1k_items']),r['cost_basis']) for r in csv.DictReader(handle)
                if r.get('cost_per_1k_items')}

def set_panel(release):
    """Set the panel size used in page text from the release being built."""
    global N_TASKS, N_ITEMS
    tasks={r['task'] for r in release.get('tasks',[])}
    N_TASKS=len(tasks) or max((m.get('tasks') or 0 for m in release.get('models',[])),default=N_TASKS)
    N_ITEMS=N_TASKS*100

def normalize_release(release):
    """Adapt the metrics release to the small browser-facing schema."""
    set_panel(release)
    costs=standard_costs()
    models=[]
    for source in release['models']:
        p=source['provenance']
        observed_run_dates=None
        if source['model'] in ('deepseek-v4-flash','deepseek-v4-pro'):
            recorded=p.get('observed_completion_dates') or []
            if not recorded or not p.get('documented_versions'):
                raise ValueError('DeepSeek API version and observed run dates are required')
            for value in recorded:
                if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
                    raise ValueError('Invalid DeepSeek observed run date')
            observed_run_dates=sorted(set(recorded))
        models.append(dict(source,label=LABELS.get(source['model'],source['model'].replace('_',' ')),kind='API' if p['access']=='api' else 'open',
                           hardware_tier=p['hardware_tier'],hardware=p.get('hardware'),
                           ci_item_low=source.get('item_ci_low'),
                           ci_item_high=source.get('item_ci_high'),ci_task_low=source.get('task_ci_low'),
                           ci_task_high=source.get('task_ci_high'),cost_usd_upper=p.get('cost_usd_upper'),
                           throughput_tokens_per_second=p.get('generation_tokens_per_second'),
                           throughput_items_per_second=p.get('generation_items_per_second'),
                           throughput_items=p.get('throughput_items'),
                           throughput_keyset_sha256=p.get('throughput_keyset_sha256'),
                           throughput_settings_sha256=p.get('throughput_settings_sha256'),
                           throughput_hardware=p.get('throughput_hardware'),
                           documented_versions=p.get('documented_versions') or [p.get('revision','Unavailable')],
                           observed_run_dates=observed_run_dates,
                           cost_per_1k_items=costs.get(source['model'],(None,None))[0],
                           cost_basis=costs.get(source['model'],(None,None))[1]))
    manifest=release['manifest']
    ranked_identities={(m['provenance'].get('model_id'),m['provenance'].get('revision'))
                       for m in release['models']}
    candidates=[c for c in release.get('candidates',[])
                if (c.get('model_id'),c.get('revision')) not in ranked_identities]
    status='Open-weight refresh pending' if any(c.get('status')=='pending' for c in candidates) else manifest['status'].replace('_',' ')
    run_dates=[date for m in release['models'] for date in m['provenance'].get('run_dates',[]) if date]
    updated_at=max([manifest['api_completed_at'][:10],*run_dates])
    return dict(models=models,categories=release.get('categories',[]),pairs=release.get('pairs',[]),task_scores=[dict(r,f1=r['headline_f1']) for r in release['tasks']],
                class_scores=[dict(r,label=f"{r['field']}: {r['label']}" if r.get('field') else r['label']) for r in release['classes']],
                release_id=manifest['release'],status=status,updated_at=updated_at,
                candidates=candidates)

def validate_public(value):
    """Reject row-level content or local paths before creating public assets."""
    forbidden={'text','raw_text','raw_response','response_text','messages','input','system','item_id','predictions','api_key','token','password','authorization'}
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower() in forbidden:
                raise ValueError(f'Non-public field: {key}')
            validate_public(item)
    elif isinstance(value,list):
        for item in value: validate_public(item)
    elif isinstance(value,str) and any(s in value for s in ['/Users/','/home/','/nfs/hive/',
            'output/sidecar/','/private/tmp/','sk-proj-','sk-ant-']):
        raise ValueError('Local path or credential marker in public data')


def public_candidate(candidate):
    """Keep preflight evidence but omit its private sidecar file location."""
    cleaned = deepcopy(candidate)
    preflight = (cleaned.get('feasibility') or {}).get('cpu_preflight')
    if isinstance(preflight, dict) and str(preflight.get('metadata', '')).startswith('output/sidecar/'):
        preflight.pop('metadata')
    return cleaned

def methodology(release):
    """Explain the release using its validated manifest, without row-level inputs."""
    manifest=release['manifest']
    excluded_note=''.join(f"{entry['model']}: {entry['reason']}\n\n"
                          for entry in release.get('historical_exclusions',[]))
    return (f"# Political Science LLM Benchmark methodology\n\n"
            f"This release compares language models with human coders on the same 100 texts from each of "
            f"{N_TASKS} political science coding tasks.\n\n"
            f"## Sample and prompts\n\n"
            f"I draw the texts by hashing task and item identifiers with seed {manifest['seed']}. "
            f"Every model receives the same texts, gold labels and prompts. The panel SHA-256 is "
            f"`{manifest['panel_sha256']}`. Texts and item-level data are not included.\n\n"
            f"## Scoring and uncertainty\n\n"
            f"{manifest['scoring']}\n\n"
            f"{manifest['uncertainty']} Both use {manifest['bootstrap_replicates']} draws and seed "
            f"{manifest['seed']}. A class that appears in neither the gold labels nor the predictions "
            f"gets F1 of 0 by convention; that zero says nothing about the model. Other undefined "
            f"metrics are left blank. Scores for classes with few texts are unstable.\n\n"
            f"## Model settings and failures\n\n"
            f"The release records each model's returned identifier, revision, quantization, decoding "
            f"and reasoning settings, hardware, run dates, token usage and API cost. Reasoning is off "
            f"where a model allows it and otherwise at its lowest level. Unusable answers count as "
            f"incorrect and are not retried. Infrastructure failures are recorded separately.\n\n"
            f"{excluded_note}"
            f"## Interpretation\n\n"
            f"Each task receives equal weight in the mean. These results describe {N_TASKS} tasks, and "
            f"researchers should validate candidate models on labeled examples from their own task. The open-weight "
            f"models are a selection rather than a complete list, and a missing model says nothing about "
            f"its quality.\n")

def csv_row(row):
    """Keep nested aggregate fields parseable instead of writing Python reprs."""
    return {key:json.dumps(value,ensure_ascii=False,sort_keys=True) if isinstance(value,(dict,list)) else value
            for key,value in row.items()}

def preview_release(release_dir):
    release=json.loads((release_dir/'release.json').read_text())
    candidates=list(release.get('candidates',[]))
    if (release_dir/'candidates.json').exists():
        candidates=json.loads((release_dir/'candidates.json').read_text())['candidates']
    candidates.extend(release.get('historical_exclusions',[]))
    if candidates:
        release['candidates']=[public_candidate(candidate) for candidate in candidates]
    validate_public(release)
    set_panel(release)
    return release

def verify(release_dir):
    """Check embedded and default-table metrics against the release that built them."""
    release_dir=Path(release_dir)
    release=preview_release(release_dir)
    expected=safe_payload(normalize_release(release))
    preview=release_dir/'preview/llm-benchmark'
    page=(preview/'index.html').read_text()
    script=re.search(r'<script id="benchmark-data" type="application/json">(.*?)</script>',page,re.S)
    if script is None or json.loads(script.group(1)) != expected:
        raise ValueError('Embedded preview metrics do not reproduce the release')
    static_rows=re.search(r'<tbody id="model-rows">(.*?)</tbody>',page,re.S)
    rendered_rows=re.search(r'<tbody id="model-rows">(.*?)</tbody>',render(expected),re.S)
    if (static_rows is None or rendered_rows is None or
            static_rows.group(1) != rendered_rows.group(1)):
        raise ValueError('Static model table does not reproduce the release')
    if json.loads((preview/'downloads/release.json').read_text()) != release:
        raise ValueError('Public release download differs from the validated release')
    if json.loads((preview/'downloads/manifest.json').read_text()) != release['manifest']:
        raise ValueError('Public manifest download differs from the validated release')
    for key in ('models','tasks','classes','categories','pairs'):
        source=[{field:value for field,value in row.items() if field!='provenance'}
                for row in release[key]]
        if not source:
            continue
        with (preview/f'downloads/{key}.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        target=[{field:'' if value is None else str(value)
                 for field,value in csv_row(row).items()} for row in source]
        if rows != target:
            raise ValueError(f'Public {key} CSV differs from the validated release')
    definitions=release.get('task_definitions',[])
    if definitions:
        with (preview/'downloads/task_definitions.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        target=[{key:'' if value is None else str(value) for key,value in csv_row(row).items()}
                for row in definitions]
        if rows != target:
            raise ValueError('Public task definitions differ from the validated release')
    if (preview/'downloads/methodology.md').read_text() != methodology(release):
        raise ValueError('Public methodology differs from the validated release')
    if ('<meta name="robots" content="noindex">' not in page or
            '<link rel="canonical" href="https://www.hannohilbig.com/llm-benchmark/">' not in page):
        raise ValueError('Preview metadata is missing the local noindex or canonical link')
    if (release_dir/'preview/homepage-link-proposal.html').read_text() != homepage_link_proposal():
        raise ValueError('Homepage link proposal differs from the approved preview path')
    if (release_dir/'preview/sitemap-entry-proposal.xml').read_text() != SITEMAP_ENTRY_PROPOSAL:
        raise ValueError('Sitemap proposal differs from the approved preview path')
    expected_downloads={'release.json','manifest.json','methodology.md'}
    expected_downloads.update(f'{key}.csv' for key in ('models','tasks','classes','categories','pairs')
                              if release[key])
    if definitions:
        expected_downloads.add('task_definitions.csv')
    if any(f'href="downloads/{name}"' not in page for name in expected_downloads):
        raise ValueError('Preview is missing a validated aggregate download link')
    return dict(models=len(expected['models']),tasks=len({r['task'] for r in release['tasks']}),
                task_definitions=len(definitions))

def comparable_speed_rows(models):
    """Show generation rates only inside identical subset/settings/GPU groups."""
    groups={}
    for model in models:
        if model['kind'] != 'open' or model.get('throughput_tokens_per_second') is None:
            continue
        key=(model.get('throughput_keyset_sha256'), model.get('throughput_settings_sha256'),
             model.get('throughput_hardware'), model.get('throughput_items'))
        if any(value in (None, '') for value in key):
            continue
        groups.setdefault(key, []).append(model)
    rows=[]
    for (_, _, gpu, items), group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) < 2:
            continue
        rows.append(f'<tr><td colspan="4"><strong>Comparable generation subset:</strong> '
                    f'{int(items):,} identical frozen texts on {esc(gpu)}. '
                    'Generation, runtime and batch-concurrency settings match. '
                    f'The F1 scores use all {N_ITEMS:,} texts; generation rates exclude model load '
                    'and queue time.</td></tr>')
        for model in group:
            rows.append(f"<tr><td>{esc(model.get('label') or model['model'])}</td>"
                        f"<td>{fmt(model['mean_task_f1'])}</td>"
                        f"<td>{fmt(model['throughput_tokens_per_second'],1)}</td>"
                        f"<td>{esc(model['hardware_tier'])}</td></tr>")
    return ''.join(rows)


def render(data, downloads=()):
    data=safe_payload(data)
    models=sorted(data['models'],key=lambda m:m['mean_task_f1'],reverse=True)
    rows=[]
    for m in models:
        version=', '.join(m.get('documented_versions') or []) or 'Revision unavailable'
        run_dates=m.get('observed_run_dates') or []
        if m['model'] in ('deepseek-v4-flash','deepseek-v4-pro') and not run_dates:
            raise ValueError('DeepSeek model row is missing observed run dates')
        version_details=f'<small>{esc(version)}</small>'
        if run_dates:
            version_details+=f'<small>Observed response dates: {esc(", ".join(run_dates))}</small>'
        detail=(f'${fmt(m["cost_per_1k_items"],3)} per 1,000 texts'
                if m['kind']=='API' and m.get('cost_per_1k_items') is not None
                else m.get('hardware') or 'Hardware name unavailable')
        rows.append(f'''<tr data-model="{esc(m['model'])}" data-kind="{esc(m['kind'])}" data-hardware="{esc(m['hardware_tier'])}"><td class="model-label">{esc(m.get('label') or m['model'])}{version_details}</td><td class="numeric">{fmt(m['mean_task_f1'])}</td><td class="numeric">{interval(m,'item')}<small>Items</small>{interval(m,'task')}<small>Tasks</small></td><td class="numeric">{fmt(100*m['malformed']/m['n'],2)}%</td><td>{esc(m['kind'])}<small>{esc(hardware_label(m['hardware_tier']))}</small><small>{esc(detail)}</small></td></tr>''')
    hardware=''.join(f'<option value="{esc(t)}">{esc(hardware_label(t))}</option>' for t in sorted({m['hardware_tier'] for m in models}))
    categories=''.join(f'<option value="{esc(c)}">{esc(c)}</option>' for c in sorted({r['category'] for r in data['task_scores'] if r['category']}))
    pending=''.join(f"<li><strong>{esc(r['model'])}</strong>: {esc(r['status'])}. {esc(r['reason'])}</li>" for r in data['candidates'])
    costs=''.join(f"<tr><td>{esc(m.get('label') or m['model'])}</td><td>{fmt(m['mean_task_f1'])}</td><td>${fmt(m['cost_per_1k_items'],3)}</td><td>{esc(COST_BASIS_LABELS.get(m['cost_basis'],m['cost_basis']))}</td></tr>" for m in models if m['kind']=='API' and m.get('cost_per_1k_items') is not None)
    speeds=comparable_speed_rows(models)
    links=''.join(f'<li><a href="downloads/{esc(name)}" download>{esc(name)}</a></li>' for name in downloads)
    takeaway=lead_sentences(models)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Political Science LLM Benchmark | Hanno Hilbig</title><meta name="author" content="Hanno Hilbig"><meta name="description" content="Matched evaluation of language models on {N_TASKS} political science text-coding tasks."><meta name="robots" content="noindex"><link rel="canonical" href="https://www.hannohilbig.com/llm-benchmark/"><link rel="stylesheet" href="styles.css"></head><body><a class="skip-link" href="#main">Skip to content</a><div class="wrapper"><header><h1>Political Science LLM Benchmark</h1><p class="title">{N_TASKS} text-coding tasks · 100 texts per task<br><a href="https://www.hannohilbig.com/">Hanno Hilbig</a>, University of California, Davis</p><nav aria-label="Page sections"><a href="#comparison">Models</a><a href="#tasks">Tasks</a><a href="#methods">Methods</a><a href="#downloads">Downloads</a></nav></header><main id="main" tabindex="-1"><p class="notice">Local preview, not published. Release {esc(data['release_id'])}; status: {esc(data['status'])}. Updated {esc(data['updated_at'])}.</p><p>Can researchers code political science texts with open-weight language models instead of commercial APIs? This page compares {len(models)} models on the same {N_ITEMS:,} texts, 100 from each of {N_TASKS} coding tasks drawn from political science papers and public datasets. Every model receives the same texts, prompts and gold labels.</p><p>The main result is that the differences between models are small. {takeaway} Differences of a few hundredths in the mean often reverse on individual tasks, so researchers should check the tasks closest to their own before choosing a model.</p><h2 id="comparison">Model comparison</h2><p>The table reports mean F1 for every model that coded all {N_ITEMS:,} texts. F1 is the harmonic mean of precision and recall, and each task receives equal weight.</p><div class="controls js-only"><label>Model access<select id="kind"><option value="">All</option><option value="API">API</option><option value="open">Open weights</option></select></label><label>Hardware tier<select id="hardware"><option value="">All</option>{hardware}</select></label></div><p id="model-count" role="status" aria-live="polite">{len(models)} models shown.</p><div class="table-scroll" tabindex="0" role="region" aria-label="Model comparison, horizontally scrollable"><table><caption>Equal-task mean F1 and 95% uncertainty intervals</caption><thead><tr><th aria-sort="none"><button data-sort="label">Model ↕</button></th><th aria-sort="descending"><button data-sort="mean_task_f1">F1 ↕</button></th><th>95% intervals</th><th aria-sort="none"><button data-sort="malformed">Malformed ↕</button></th><th>Access / hardware</th></tr></thead><tbody id="model-rows">{''.join(rows)}</tbody></table></div><p class="note">I report two 95% intervals. The item interval resamples texts within each task and reflects uncertainty about these tasks. The task interval resamples tasks and reflects how a model would perform on a different set of tasks, which is why it is wider. Both use 2,000 bootstrap draws.</p><noscript><p>Task and class results are in the downloads.</p></noscript><h2 id="tasks">Choose by task</h2><div class="controls js-only"><label>Category<select id="category"><option value="">All categories</option>{categories}</select></label><label>Task<select id="task"></select></label></div><p id="task-description">Task scores are in tasks.csv under Downloads.</p><div class="table-scroll js-only"><table><caption>Selected task results</caption><thead><tr><th>Model</th><th>F1</th><th>Accuracy</th><th>MCC</th><th>Texts</th></tr></thead><tbody id="task-rows"></tbody></table></div><details><summary>Class support and class F1</summary><p>MCC is the Matthews correlation coefficient. Blank cells mark undefined metrics, not zeros. Scores for classes with few texts are unstable.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Class</th><th>Reference support</th><th>F1</th></tr></thead><tbody id="class-rows"></tbody></table></div></details><h2>Practical tradeoffs</h2><h3>API quality and cost</h3><p>Costs are per 1,000 texts at standard prices in September 2026. Where provider bills are available, I use them; otherwise, I multiply token counts by list prices. OpenAI, Anthropic and Google charge half for batch requests.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Mean F1</th><th>Cost per 1,000 texts</th><th>Source</th></tr></thead><tbody>{costs}</tbody></table></div><h3>Open-model quality and throughput</h3><p>Speeds are only comparable across runs on the same hardware with the same settings, which the table groups together. University GPU time is free to the researcher, but not free to provide.</p>{'<div class="table-scroll"><table><thead><tr><th>Model</th><th>Mean F1</th><th>Output tokens / second</th><th>Hardware</th></tr></thead><tbody>'+speeds+'</tbody></table></div>' if speeds else '<p>No comparable speed measurements are available.</p>'}<h2>Models outside the ranking</h2><p>The models below are not ranked. Not evaluated means that I could not run the model, for the reason listed. Excluded means that a run finished but violated the benchmark's rules. Pending means that the run is not finished. None of these categories implies a score of zero.</p><p>The open-weight models are a selection rather than a complete list. Whether a model runs depends on its weight files, quantization, software support and GPU memory, not only on its size. A missing model therefore says nothing about its quality.</p><ul>{pending or '<li>No additional candidate status has been recorded.</li>'}</ul><h2 id="methods">Methods and limitations</h2><p>Sample. For each task, I draw 100 texts by hashing task and item identifiers with seed 20260910. Every model receives the same texts, gold labels and prompts. I reuse earlier predictions only when their inputs and settings match exactly.</p><p>Scoring. For binary tasks, I use the F1 score for the positive class. For tasks with several binary labels, I average the per-label F1 scores. For single-label categorical tasks, I use macro F1, which gives each class equal weight. A class that appears in neither the gold labels nor the predictions receives an F1 of 0 by convention; that zero says nothing about the model. Unusable answers count as incorrect, and I record infrastructure failures separately.</p><p>Settings. Reasoning is disabled where a model allows it and otherwise set to its lowest level. API models may return at most 256 tokens. The downloads report model versions, quantization and runtime settings, which can affect performance.</p><p>Limits. These results describe {N_TASKS} tasks. Researchers should validate candidate models on labeled examples from their own task before using them at scale.</p><h2 id="downloads">Downloads and reproducibility</h2><p>The downloads contain aggregate scores and run records. I do not include the texts or item-level predictions while redistribution rights are checked.</p><ul>{links}<li><a href="https://github.com/hhilbig/polsci-open-bench">Benchmark code and task definitions</a></li></ul></main><footer>This is a local preview. Publication requires approval.</footer></div><script id="benchmark-data" type="application/json">{json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')}</script><script src="benchmark.js" defer></script></body></html>'''

def paper_figures(preview, release):
    manifest_path=preview/'figures/figure-data.json'
    if not manifest_path.exists():
        return ''
    manifest=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(json.dumps(release,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if manifest.get('release_sha256')!=digest:
        raise ValueError('Paper figures are stale; rerun build_refresh_figures.py')
    titles=['Overall performance','API and single-GPU open models by task',
            'Performance by annotation type','Coding complexity','Label structure',
            'Quality and generation time','Generation time per 1,000 items']
    if len(manifest['figures'])!=len(titles):
        raise ValueError('Expected seven paper figures')
    def figure_html(title,figure):
        stem=figure['file']
        if not re.fullmatch(r'fig-[a-z0-9-]+',stem):
            raise ValueError('Invalid figure filename')
        for ext in ['svg','pdf']:
            if not (preview/'figures'/f'{stem}.{ext}').is_file():
                raise ValueError(f'Missing figure {stem}.{ext}')
        wide=' wide' if stem=='fig-family' else ''
        return f'<h3>{esc(title)}</h3><figure class="paper-figure"><div class="figure-scroll" tabindex="0" role="region" aria-label="{esc(title)}, horizontally scrollable"><img class="paper-plot{wide}" src="figures/{stem}.svg" alt="{esc(title)}. {esc(figure["caption"])}" loading="lazy"></div><figcaption>{esc(figure["caption"])} <a href="figures/{stem}.svg">Full-size SVG</a> · <a href="figures/{stem}.pdf">PDF</a></figcaption></figure>'
    blocks=[f'<h2 id="figures">Recent models and reference baselines</h2><p>The figures below focus on {len(FEATURED_MODELS)} models: all API models, the most recent open-weight models and two Llama 70B models as reference points. The full comparison further down includes all models.</p>']
    featured=manifest.get('featured_figures',[])
    if featured:
        if len(featured)!=4 or set(manifest.get('featured_models',[]))!=FEATURED_MODELS:
            raise ValueError('Featured figure selection differs from page selection')
        blocks.extend(figure_html(title,fig) for title,fig in zip(['Overall performance','Performance by annotation type','API and open models by task','Coding complexity'],featured))
        blocks.append('<details class="disclosure" id="more-analyses"><summary>More analyses and all-model figures</summary><p>These figures include all models. Comparisons between API and open models use only open models that run on a single GPU.</p>')
    blocks.extend(figure_html(title,figure) for title,figure in zip(titles,manifest['figures']))
    blocks.append('<p><a href="figures/figure-data.json">Download figure values and task groupings (JSON)</a></p>')
    if featured: blocks.append('</details>')
    return ''.join(blocks)


def compact_page(page,data):
    """Keep the full static comparison accessible inside native disclosures."""
    models=sorted((m for m in data['models'] if m['model'] in FEATURED_MODELS),key=lambda m:-m['mean_task_f1'])
    if not models: return page
    rows=''.join(f'<tr><td>{esc(m["label"])}</td><td>{fmt(m["mean_task_f1"])}</td><td>{interval(m,"item")}</td><td>{interval(m,"task")}</td><td>{esc(hardware_label(m["hardware_tier"]))}</td></tr>' for m in models)
    compact=f'<div class="table-scroll" tabindex="0" role="region" aria-label="Recent models and reference baselines"><table><caption>Recent models and reference baselines: equal-task mean F1</caption><thead><tr><th>Model</th><th>F1</th><th>95% item interval</th><th>95% task interval</th><th>Access / hardware</th></tr></thead><tbody id="featured-rows">{rows}</tbody></table></div><p class="note">The item interval resamples texts within each task, and the task interval resamples tasks. The full table below adds model versions, failure rates and filters.</p>'
    start=page.index('<div class="controls js-only">',page.index('<h2 id="comparison">'))
    end=page.index('<h2 id="tasks">',start)
    page=page[:start]+compact+f'<details class="disclosure" id="all-models"><summary>Show all {len(data["models"])} models</summary>'+page[start:end]+'</details>'+page[end:]
    page=page.replace('<h2 id="tasks">Choose by task</h2>','<h2 id="tasks">Choose by task</h2><label class="js-only task-scope"><input type="checkbox" id="task-all"> Show all evaluated models (default: recent models and reference baselines)</label>',1)
    # Each section retains its content and stable navigation anchor.
    for start_marker,end_marker,label,anchor in [
        ('<h2>Practical tradeoffs</h2>','<h2>Models outside the ranking</h2>','API costs and open-model runtime','tradeoffs'),
        ('<h2>Models outside the ranking</h2>','<h2>Paired comparisons</h2>','Not evaluated and excluded models','excluded'),
        ('<h2>Paired comparisons</h2>','<h2 id="methods">','Paired comparisons and category summaries','paired-details'),
        ('<h2 id="methods">Methods and limitations</h2>','<h2 id="downloads">','Methods and limitations','methods')]:
        if start_marker not in page or end_marker not in page: continue
        start=page.index(start_marker);end=page.index(end_marker,start)
        content=page[start+len(start_marker):end]
        page=page[:start]+f'<details class="disclosure" id="{anchor}"><summary>{label}</summary>'+content+'</details>'+page[end:]
    return page


def build(release_dir=DEFAULT_RELEASE):
    release_dir=Path(release_dir)
    release=preview_release(release_dir)
    data=normalize_release(release)
    preview=release_dir/'preview/llm-benchmark'
    preview.mkdir(parents=True,exist_ok=True)
    downloads=preview/'downloads';downloads.mkdir(exist_ok=True)
    # Re-export a whitelisted aggregate payload rather than copying arbitrary files.
    safe=safe_payload(data)
    (downloads/'release.json').write_text(json.dumps(release,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    names=['release.json']
    for key in ['models','tasks','classes','categories','pairs']:
        rows=[{k:v for k,v in row.items() if k!='provenance'} for row in release[key]]
        if not rows:
            continue
        name=key+'.csv'
        with (downloads/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(csv_row(row) for row in rows)
        names.append(name)
    definitions=release.get('task_definitions',[])
    if definitions:
        with (downloads/'task_definitions.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(definitions[0]));writer.writeheader();writer.writerows(csv_row(row) for row in definitions)
        names.append('task_definitions.csv')
    (downloads/'manifest.json').write_text(json.dumps(release['manifest'],indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    names.append('manifest.json')
    methods=methodology(release)
    validate_public(methods)
    (downloads/'methodology.md').write_text(methods)
    names.append('methodology.md')
    page=render(safe,names)
    for label,key in (('Model','label'),('F1','mean_task_f1'),('Malformed','malformed')):
        original=f'<button data-sort="{key}">{label} ↕</button>'
        accessible=(f'<span class="static-heading">{label}</span>'
                    f'<button class="sort-button" data-sort="{key}">{label} ↕</button>')
        if original not in page:
            raise ValueError(f'Missing sortable heading: {label}')
        page=page.replace(original,accessible,1)
    class_details='<details><summary>Class support and class F1</summary>'
    if class_details not in page:
        raise ValueError('Missing class-support details')
    page=page.replace(class_details,
                      '<details class="class-support-details"><summary>Class support and class F1</summary>',1)
    extra='''<h2>Paired comparisons</h2><p>Both models are scored on the same texts. Positive differences favor the first model. The 95% intervals apply to one comparison at a time and are not adjusted for multiple comparisons.</p><div class="controls js-only"><label>Model<select id="pair-model"></select></label><label>Reference<select id="pair-reference"></select></label></div><p id="pair-result" role="status" aria-live="polite">Paired differences are in pairs.csv.</p><h2>Category summaries</h2><div class="controls js-only"><label>Category summary<select id="summary-category"></select></label></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Mean F1</th><th>Tasks</th><th>95% item interval</th><th>95% task interval</th></tr></thead><tbody id="category-rows"></tbody></table></div><noscript><p>Download categories.csv for category scores and uncertainty.</p></noscript>'''
    page=page.replace('<h2 id="methods">',extra+'<h2 id="methods">')
    page=page.replace('<meta name="robots"','<link rel="icon" href="data:,"><meta name="robots"')
    figures=paper_figures(preview,release)
    if figures:
        page=page.replace('<h2 id="comparison">',figures+'<h2 id="comparison">',1)
        page=page.replace('<a href="#comparison">Models</a>','<a href="#figures">Figures</a> <a href="#comparison">Models</a>',1)
    page=compact_page(page,data)
    (preview/'index.html').write_text(page)
    (preview/'styles.css').write_text(CSS+'\n.paper-figure{margin:1.4rem 0}.figure-scroll{overflow-x:auto;border:1px solid var(--border,#ddd);border-radius:6px}.paper-plot{display:block;width:100%;min-width:720px;height:auto}.paper-plot.wide{min-width:1100px}.paper-figure figcaption{font-size:.88em;color:#555;margin-top:.4rem;line-height:1.5}\n')
    extra_js='''
function addOptions(id,values){byId(id).replaceChildren(...values.map(([value,label])=>{const o=document.createElement('option');o.value=value;o.textContent=label;return o;}));}
addOptions('pair-model',data.models.map(m=>[m.model,m.label]));addOptions('pair-reference',data.models.map(m=>[m.model,m.label]));
byId('pair-reference').value=data.models.some(m=>m.model==='qwen3_6_27b_fp8')?'qwen3_6_27b_fp8':data.models[1]?.model||data.models[0]?.model;
function showPair(){const model=byId('pair-model').value,reference=byId('pair-reference').value;const r=data.pairs.find(r=>r.model===model&&r.reference===reference);byId('pair-result').textContent=model===reference?'Select two different models.':r?`${labels[model]} minus ${labels[reference]}: ${num(r.difference)} F1. 95% item interval ${num(r.item_ci_low)} to ${num(r.item_ci_high)}; task interval ${num(r.task_ci_low)} to ${num(r.task_ci_high)}.`:'Paired comparison unavailable.';}
byId('pair-model').addEventListener('change',showPair);byId('pair-reference').addEventListener('change',showPair);showPair();
addOptions('summary-category',[...new Set(data.categories.map(r=>r.category))].sort().map(c=>[c,c]));
function showCategory(){byId('category-rows').replaceChildren();data.categories.filter(r=>r.category===byId('summary-category').value).sort((a,b)=>b.mean_task_f1-a.mean_task_f1).forEach(r=>{const tr=document.createElement('tr');[labels[r.model],num(r.mean_task_f1),String(r.tasks),`${num(r.item_ci_low)} to ${num(r.item_ci_high)}`,`${num(r.task_ci_low)} to ${num(r.task_ci_high)}`].forEach(t=>makeCell(tr,t));byId('category-rows').appendChild(tr);});}
byId('summary-category').addEventListener('change',showCategory);showCategory();
'''
    (preview/'styles.css').write_text((preview/'styles.css').read_text()+'\n.disclosure{margin:1.4rem 0;border-top:1px solid #ddd;padding-top:1rem}.disclosure>summary{cursor:pointer;font-weight:600;color:#0069b4}.disclosure[open]>summary{margin-bottom:1rem}.task-scope{margin-bottom:1rem}.task-scope input{width:auto;margin-right:.4rem}\n')
    (preview/'benchmark.js').write_text('const featuredModels=new Set('+json.dumps(sorted(FEATURED_MODELS))+');\n'+JS+extra_js+'\ndocument.querySelectorAll(\'a[href="#methods"]\').forEach(a=>a.addEventListener("click",()=>{byId("methods").open=true;}));\n')
    (release_dir/'preview/homepage-link-proposal.html').write_text(homepage_link_proposal())
    (release_dir/'preview/sitemap-entry-proposal.xml').write_text(SITEMAP_ENTRY_PROPOSAL)
    verify(release_dir)
    return preview

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,default=DEFAULT_RELEASE)
    parser.add_argument('--verify-only',action='store_true',help='Check an existing preview against its release')
    args=parser.parse_args()
    print(verify(args.release_dir) if args.verify_only else build(args.release_dir))
