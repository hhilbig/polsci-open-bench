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
const byId=id=>document.getElementById(id);
document.querySelectorAll('a[href="#methods"]').forEach(a=>a.addEventListener('click',()=>{byId('methods').open=true;}));
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
        'models': ('model','label','kind','hardware_tier','hardware','n','tasks','mean_task_f1','ci_item_low','ci_item_high','ci_task_low','ci_task_high','malformed','cost_usd_upper','cost_per_1k_items','cost_per_1k_items_batch','cost_basis','throughput_tokens_per_second','throughput_items_per_second','throughput_items','throughput_keyset_sha256','throughput_settings_sha256','throughput_hardware','documented_versions','observed_run_dates'),
        'task_scores': ('model','task','category','n','f1','accuracy','mcc'),
        'class_scores': ('model','task','label','support','f1'),
        'candidates': ('model','status','reason'),
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
        # Batch price is blank for providers without a batch option (DeepSeek, Jev).
        return {r['model']:(float(r['cost_per_1k_items']),r['cost_basis'],
                            float(r['cost_per_1k_items_batch']) if r.get('cost_per_1k_items_batch') else None)
                for r in csv.DictReader(handle) if r.get('cost_per_1k_items')}

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
                           cost_per_1k_items=costs.get(source['model'],(None,None,None))[0],
                           cost_basis=costs.get(source['model'],(None,None,None))[1],
                           cost_per_1k_items_batch=costs.get(source['model'],(None,None,None))[2]))
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
    manifest=figure_manifest(preview,release)
    if manifest is not None:
        # The overview figures carry the ranking now, so their plotted means must match the release.
        means={m['model']:m['mean_task_f1'] for m in release['models']}
        for figure in manifest['featured_figures']+manifest['figures']:
            if figure['file'] in ('fig-mean-f1','fig-recent-mean-f1'):
                for row in figure['data']:
                    if abs(row['mean_task_f1']-means[row['model']])>1e-9:
                        raise ValueError('Overview figure does not reproduce the release means')
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
    return dict(models=len(expected['models']),tasks=len({r['task'] for r in release['tasks']}),
                task_definitions=len(definitions))

def figure_manifest(preview, release):
    """Load the rendered figures' manifest and refuse figures built from another release."""
    manifest_path=preview/'figures/figure-data.json'
    if not manifest_path.exists():
        return None
    manifest=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(json.dumps(release,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if manifest.get('release_sha256')!=digest:
        raise ValueError('Paper figures are stale; rerun build_refresh_figures.py')
    if set(manifest.get('featured_models',[]))!=FEATURED_MODELS:
        raise ValueError('Featured figure selection differs from page selection')
    return manifest


def paper_figures(preview, release):
    manifest=figure_manifest(preview, release)
    if manifest is None:
        return ''
    def figure_html(figure):
        stem,title=figure['file'],figure['title']
        if not re.fullmatch(r'fig-[a-z0-9-]+',stem):
            raise ValueError('Invalid figure filename')
        for ext in ['svg','pdf']:
            if not (preview/'figures'/f'{stem}.{ext}').is_file():
                raise ValueError(f'Missing figure {stem}.{ext}')
        return (f'<h3>{esc(title)}</h3><figure class="paper-figure"><img class="paper-plot" src="figures/{stem}.svg" '
                f'alt="{esc(title)}. {esc(figure["caption"])}" loading="lazy"><figcaption>{esc(figure["caption"])}</figcaption></figure>')
    blocks=['<div id="figures"></div>']
    blocks.extend(figure_html(fig) for fig in manifest['featured_figures'])
    blocks.append(f'<details class="disclosure" id="more-analyses"><summary>All {len(release.get("models",[]))} models and open-model speed</summary>'
                  f'<p>The first figure repeats the overview for all {len(release.get("models",[]))} models. The second compares the speed of the '
                  f'{sum(len(g["models"]) for g in manifest.get("throughput_groups",[]))} open-weight models for which I recorded generation time.</p>')
    blocks.extend(figure_html(fig) for fig in manifest['figures'])
    blocks.append('</details>')
    return ''.join(blocks)


def render(data, downloads=(), figures=''):
    data=safe_payload(data)
    models=sorted(data['models'],key=lambda m:m['mean_task_f1'],reverse=True)
    for m in models:
        if m['model'] in ('deepseek-v4-flash','deepseek-v4-pro') and not m.get('observed_run_dates'):
            raise ValueError('DeepSeek model row is missing observed run dates')
    pending=''.join(f"<li><strong>{esc(r['model'])}</strong>: {esc(r['status'])}. {esc(r['reason'])}</li>" for r in data['candidates'])
    takeaway=lead_sentences(models)
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Political Science LLM Benchmark | Hanno Hilbig</title><meta name="author" content="Hanno Hilbig"><meta name="description" content="Matched evaluation of language models on {N_TASKS} political science text-coding tasks."><link rel="icon" href="data:,"><meta name="robots" content="noindex"><link rel="canonical" href="https://www.hannohilbig.com/llm-benchmark/"><link rel="stylesheet" href="styles.css"></head><body><a class="skip-link" href="#main">Skip to content</a><div class="wrapper"><header><h1>Political Science LLM Benchmark</h1><p class="title"><a href="https://www.hannohilbig.com/">Hanno Hilbig</a>, University of California, Davis</p><nav aria-label="Page sections"><a href="#figures">Figures</a><a href="#hardware">Hardware</a><a href="#methods">Methods</a></nav></header><main id="main" tabindex="-1"><p class="notice">Local preview, not published. Release {esc(data['release_id'])}; status: {esc(data['status'])}. Updated {esc(data['updated_at'])}.</p><p>Below, I compare the performance and cost of {len(models)} language models for text classification, {sum(m['kind']=='open' for m in models)} with open weights and {sum(m['kind']=='API' for m in models)} commercial. I use {N_TASKS} coding tasks from political science papers, replication archives and public datasets, covering relevance, stance, tone, events, claims and topics. For each task, every model codes the same 100 texts, and I compare its labels with those of human coders.</p><p>The main result is that the differences between models are small. {takeaway} Differences of a few hundredths in the mean often reverse on individual tasks, so researchers should check the tasks closest to their own before choosing a model.</p>{figures}<details class="disclosure" id="excluded"><summary>Not evaluated and excluded models</summary><p>The models below are not ranked. Not evaluated means that I could not run the model, for the reason listed. Excluded means that a run finished but violated the benchmark's rules.</p><p>The open-weight models are a selection rather than a complete list. Whether a model runs depends on its weight files, quantization, software support and GPU memory, not only on its size. A missing model therefore says nothing about its quality.</p><ul>{pending or '<li>No additional candidate status has been recorded.</li>'}</ul></details><h2 id="hardware">Where the models ran</h2><ul><li>API models: called through each provider's API between 10 and 20 September 2026. Requests to OpenAI and Anthropic went through their batch endpoints; the other providers received one request per text.</li><li>Open-weight models: each ran on a single NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB of memory on UC Davis's Hive computing cluster, using vLLM 0.26 at temperature 0. Qwen3.8 Flash-Next needed two of these GPUs and a development version of vLLM.</li><li>Settings: reasoning is disabled where a model allows it and otherwise set to its lowest level. Every model may return at most 256 tokens, except Jev 1.13, which returns a choice among the labels rather than free text.</li><li>Not tested: the API models with extended reasoning, which raises cost and response time, and the largest open-weight models (Kimi K3, GLM-5.3 and MiniMax M3), which do not run on this hardware. The top scores on this page may therefore understate what the strongest configurations of these models reach.</li></ul><details class="disclosure" id="methods"><summary>Methods and limitations</summary><p>Sample. For each task, I draw 100 texts by hashing task and item identifiers with seed 20260910. Every model receives the same texts, gold labels and prompts. I reuse earlier predictions only when their inputs and settings match exactly.</p><p>Scoring. For binary tasks, I use the F1 score for the positive class. For tasks with several binary labels, I average the per-label F1 scores. For single-label categorical tasks, I use macro F1, which gives each class equal weight. A class that appears in neither the gold labels nor the predictions receives an F1 of 0 by convention; that zero says nothing about the model. Unusable answers count as incorrect, and I record infrastructure failures separately.</p><p>Limits. These results describe {N_TASKS} tasks. Researchers should validate candidate models on labeled examples from their own task before using them at scale.</p></details></main><footer>Benchmark code and task definitions: <a href="https://github.com/hhilbig/polsci-open-bench">github.com/hhilbig/polsci-open-bench</a>. This is a local preview. Publication requires approval.</footer></div><script id="benchmark-data" type="application/json">{payload}</script><script src="benchmark.js" defer></script></body></html>'''


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
    page=render(safe,names,paper_figures(preview,release))
    (preview/'index.html').write_text(page)
    (preview/'styles.css').write_text(CSS+'\n.paper-figure{margin:1.2rem 0 2rem}.paper-plot{display:block;width:100%;height:auto}.paper-figure figcaption{font-size:.88em;color:#555;margin-top:.5rem;line-height:1.5}'
        '.disclosure{margin:1.4rem 0;border-top:1px solid #ddd;padding-top:1rem}.disclosure>summary{cursor:pointer;font-weight:600;color:#0069b4}.disclosure[open]>summary{margin-bottom:1rem}\n')
    (preview/'benchmark.js').write_text(JS)
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
