"""Prepare the benchmark page's figure data and render the figures in R.

Python assembles what the figures need from the frozen release (task families,
coding complexity, labels in practice, model labels, the comparable throughput
group) and writes it to figure-data.json. render_refresh_paper_figures.R draws
every figure from that file in the CLARA house style and records each figure's
title and caption back into it, which the page builder reads.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from build_refresh_preview import normalize_release, preview_release, FEATURED_MODELS
from page_config import load_page_config
from build_task_length_audit import _effective_label_count, _coding_complexity, _word_count
from task_registry import load_task_definitions

ROOT=Path(__file__).resolve().parents[1]
CONFIG=load_page_config()
COST_TABLE=CONFIG.resolve('cost_table')


def speed_groups(release, tasks):
    """Generation time per text for each configured group of runs, over the active tasks.

    Every run must be the checkpoint the release scores, on the same GPU model and
    over the same texts as the rest of its group, or the group is not comparable
    and the build stops.
    """
    provenance={m['model']:m['provenance'] for m in release['models']}
    groups=[]
    for group in CONFIG.inputs['speed_runs']:
        rows,gpus,texts=[],set(),set()
        for model,folder in group['models'].items():
            meta=json.loads((ROOT/folder/model/'run_metadata.json').read_text())
            found=set(re.findall(r'"revision": "([0-9a-f]{40})"',json.dumps(meta)))
            if provenance[model]['revision'] not in found:
                raise ValueError(f'{group["label"]}: {model} is not the checkpoint in the release')
            gpus.update(re.findall(r'"name": "(NVIDIA [^"]+)"',json.dumps(meta)))
            results={k:v for k,v in meta['task_results'].items() if k in tasks}
            if set(results)!=set(tasks):
                raise ValueError(f'{group["label"]}: {model} does not cover every active task')
            n=sum(v['rows'] for v in results.values());texts.add(n)
            rows.append(dict(model=model,seconds_per_item=sum(v['generation_seconds'] for v in results.values())/n))
        if len(gpus)!=1 or len(texts)!=1:
            raise ValueError(f'{group["label"]}: runs differ in GPU model or number of texts')
        groups.append(dict(label=group['label'],items=texts.pop(),hardware=gpus.pop(),
                           concurrency=group['concurrency'],models=rows))
    return groups


def short_label(label):
    """'Qwen3.6 27B (FP8)' -> 'Qwen3.6 27B'; quantization details stay in the downloads."""
    return re.sub(r'\s*\([^)]*\)','',label).strip()


def build(release_dir):
    release_dir=Path(release_dir)
    release=preview_release(release_dir);data=normalize_release(release)
    panel=json.loads(CONFIG.resolve('panel').read_text())
    digest=hashlib.sha256(json.dumps(panel,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if digest!=release['manifest']['panel_sha256']:raise ValueError('Figure sample differs from release')
    source=(ROOT/'code/build_report_assets.R').read_text()
    block=source.split('family_map <-')[1].split('long_codebook_tasks <-')[0]
    families=dict(re.findall(r'"([a-z0-9_]+)"\s*,\s*"([^"]+)"',block))
    tasks={r['task']:dict(r) for r in release['task_definitions']}
    if not set(tasks)<=set(families):raise ValueError('Paper taxonomy must cover every release task')
    items=defaultdict(list)
    for row in panel['rows']:items[row['task']].append(row['item'])
    definitions={t['name']:t for t in load_task_definitions()}
    for key,t in tasks.items():
        effective=_effective_label_count(t,items[key])
        words=_word_count(Path(definitions[key]['prompt_path']).read_text())
        t.update(paper_family=families[key],effective_labels=effective,
                 complexity=_coding_complexity(t,effective,words),prompt_words=words)
    models=sorted(data['models'],key=lambda m:-m['mean_task_f1'])
    scores={(r['model'],r['task']):r['headline_f1'] for r in data['task_scores']}
    if len(scores)!=len(models)*len(tasks):raise ValueError('Incomplete model-task coverage')
    if not FEATURED_MODELS<={m['model'] for m in models}:
        raise ValueError('Featured models must all have complete release metrics')
    # Speeds are comparable only inside one group of identical texts, settings and GPU.
    comparable=defaultdict(list)
    for m in models:
        key=tuple(m.get(k) for k in ['throughput_keyset_sha256','throughput_settings_sha256','throughput_hardware','throughput_items'])
        if m['kind']=='open' and all(x is not None for x in key) and m.get('throughput_items_per_second'):comparable[key].append(m)
    key,group=max(comparable.items(),key=lambda kv:len(kv[1]))
    if len(group)<2:raise ValueError('No comparable throughput group')
    out=release_dir/'preview/llm-benchmark/figures';out.mkdir(parents=True,exist_ok=True)
    manifest={
        'sample_sha256':digest,
        'release_sha256':hashlib.sha256(json.dumps(release,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'task_metadata':list(tasks.values()),
        'featured_models':sorted(FEATURED_MODELS),
        'models':[dict(model=m['model'],label=m['label'],short_label=short_label(m['label']),kind=m['kind'],
                       hardware_tier=m['hardware_tier'],mean_task_f1=m['mean_task_f1'],
                       task_ci_low=m['ci_task_low'],task_ci_high=m['ci_task_high']) for m in models],
        'throughput_groups':speed_groups(release,tasks)+[
            dict(label='September runs',items=key[3],hardware=key[2],concurrency='16 texts at a time',
                 models=[dict(model=m['model'],seconds_per_item=1/m['throughput_items_per_second']) for m in group])],
        'figures':[],'featured_figures':[],
    }
    (out/'figure-data.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    subprocess.run(['Rscript',str(ROOT/'code/render_refresh_paper_figures.R'),str(release_dir.resolve()),str(COST_TABLE)],check=True)
    result=json.loads((out/'figure-data.json').read_text())
    if not result['featured_figures'] or not result['figures']:
        raise ValueError('The R renderer did not record any figures')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,default=CONFIG.release_dir)
    result=build(p.parse_args().release_dir)
    print(len(result['featured_figures']),'featured and',len(result['figures']),'further figures generated')
