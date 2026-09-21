"""Recreate paper figures from the frozen refresh without editing manuscript files."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_refresh_preview import normalize_release, preview_release, FEATURED_MODELS
from build_task_length_audit import _effective_label_count, _coding_complexity, _word_count
from task_registry import load_task_definitions

ROOT=Path(__file__).resolve().parents[1]


def build(release_dir):
    release_dir=Path(release_dir)
    release=preview_release(release_dir);data=normalize_release(release)
    panel=json.loads((ROOT/'output/sidecar/refresh_20260910/panel.json').read_text())
    digest=hashlib.sha256(json.dumps(panel,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if digest!=release['manifest']['panel_sha256']:raise ValueError('Figure sample differs from release')
    source=(ROOT/'code/build_report_assets.R').read_text()
    block=source.split('family_map <-')[1].split('long_codebook_tasks <-')[0]
    families=dict(re.findall(r'"([a-z0-9_]+)"\s*,\s*"([^"]+)"',block))
    tasks={r['task']:dict(r) for r in release['task_definitions']}
    n_tasks=len(tasks)
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
    if len(scores)!=len(models)*n_tasks:raise ValueError('Incomplete model-task coverage')
    out=release_dir/'preview/llm-benchmark/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
        'font.size':10,'axes.edgecolor':'#555555','axes.linewidth':.7,'axes.grid':False,
        'xtick.color':'#555555','ytick.color':'#555555','svg.fonttype':'none','svg.hashsalt':'refresh20260910'})
    palette={'qwen':'#7570b3','gemma':'#0072B2','llama':'#009E73','mistral':'#e7298a','glm':'#d95f02','deepseek':'#e6ab02'}
    colors={m['model']:('#555555' if m['kind']=='API' else next((c for f,c in palette.items() if m['model'].startswith(f)),'#0072B2')) for m in models}
    names={m['model']:m['label']+(' [2 GPUs]' if m['hardware_tier']=='multi-gpu' else '') for m in models}
    manifest={'sample_sha256':digest,'release_sha256':hashlib.sha256(json.dumps(release,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'figures':[],'task_metadata':list(tasks.values())}
    def save(fig,stem,caption,rows):
        for ext in ['svg','pdf','png']:
            meta={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
            fig.savefig(out/(stem+'.'+ext),bbox_inches='tight',dpi=150,metadata=meta)
        caption=caption.replace('{N_TASKS}',str(n_tasks)).replace('{N_TEXTS}',f'{n_tasks*100:,}')
        plt.close(fig);manifest['figures'].append(dict(file=stem,caption=caption,data=rows))
    single=[m for m in models if m['hardware_tier']=='single-gpu']
    panels=[('API models',[m for m in models if m['kind']=='API']),('Open weights · single GPU (1 of 2)',single[:10]),
            ('Open weights · single GPU (2 of 2)',single[10:]),('Open weights · multiple GPUs',[m for m in models if m['hardware_tier']=='multi-gpu'])]
    panels=[('API models',[m for m in models if m['kind']=='API']),('Open weights, one GPU',single),
            ('Open weights, two GPUs',[m for m in models if m['hardware_tier']=='multi-gpu'])]
    panels=[(t,g) for t,g in panels if g]
    fig,axes=plt.subplots(len(panels),1,figsize=(7,.26*len(models)+1.6),sharex=True,
                          gridspec_kw={'height_ratios':[len(g)+1 for _,g in panels]})
    rng=np.random.default_rng(20260910)
    for ax,(title,group) in zip(np.atleast_1d(axes),panels):
        for i,m in enumerate(group):
            xs=[scores[m['model'],t] for t in tasks]
            ax.scatter(xs,i+rng.uniform(-.2,.2,len(xs)),s=8,color='#cccccc',alpha=.45)
            ax.scatter(m['mean_task_f1'],i,s=55,color=colors[m['model']],edgecolor='#333333',zorder=3)
            ax.text(1.08,i,f"{m['mean_task_f1']:.3f}",ha='right',va='center',fontsize=8,fontweight='bold')
        ax.set_yticks(range(len(group)),[names[m['model']] for m in group],fontsize=8)
        ax.set_ylim(len(group)-.4,-.6);ax.set_xlim(0,1.08)
        ax.set_title(title,loc='left',fontsize=10,fontweight='bold')
    np.atleast_1d(axes)[-1].set_xlabel('F1');fig.tight_layout(h_pad=1)
    save(fig,'fig-mean-f1','Same as the overview above, for all models, grouped by access and hardware.',[dict(model=m['model'],mean_f1=m['mean_task_f1']) for m in models])
    groups={'API':[m for m in models if m['kind']=='API'],'Open (single GPU)':single}
    best={t:{g:max(scores[m['model'],t] for m in ms) for g,ms in groups.items()} for t in tasks}
    gaps={t:v['API']-v['Open (single GPU)'] for t,v in best.items()}
    order=sorted(tasks,key=lambda t:gaps[t],reverse=True)
    fig,ax=plt.subplots(figsize=(9,9))
    for i,t in enumerate(order):
        c='#009E73' if gaps[t]<0 else '#333333'
        ax.plot([0,gaps[t]],[i,i],color=c,lw=1);ax.scatter(gaps[t],i,color=c,s=28)
    ax.axvline(0,color='#777777',ls='--',lw=.8);ax.set_yticks(range(n_tasks),[t.replace('_',' ') for t in order],fontsize=8)
    ax.invert_yaxis();ax.set_xlabel('Best API minus best single-GPU open F1');fig.tight_layout()
    save(fig,'fig-best-local-api-gap','Same as the task-gap figure above, for every API model and every open model that runs on a single GPU.',[dict(task=t,api_minus_open=gaps[t]) for t in order])
    family_order=['Relevance & Harm','Position & Tone','Events & Actions','Claims & Relations','Issues & Topics']
    fig,axes=plt.subplots(3,2,figsize=(13,19));rows=[]
    for ax,family in zip(axes.flat,family_order):
        subset=[t for t in tasks if tasks[t]['paper_family']==family]
        values=[float(np.mean([scores[m['model'],t] for t in subset])) for m in models]
        ax.barh(range(len(models)),values,color=[colors[m['model']] for m in models],height=.72)
        for i,v in enumerate(values):ax.text(v+.015,i,f'{v:.2f}',va='center',fontsize=7)
        ax.set_yticks(range(len(models)),[names[m['model']] for m in models],fontsize=7)
        ax.invert_yaxis();ax.set_xlim(0,1.08);ax.set_xlabel('Mean F1 within type')
        ax.set_title(f'{family} ({len(subset)} tasks)',fontsize=11,fontweight='bold',backgroundcolor='#eeeeee')
        rows.extend(dict(family=family,model=m['model'],mean_f1=v) for m,v in zip(models,values))
    axes.flat[-1].axis('off');fig.tight_layout(h_pad=2,w_pad=2)
    save(fig,'fig-family','Same as the annotation-type figure above, for all models.',rows)
    fig,axes=plt.subplots(1,2,figsize=(9,4));rows=[];levels=['Low','Medium','High']
    for ax,use_best,title in zip(axes,[False,True],['All model-task results','Best model per task in each class']):
        for group,c in [('API','#333333'),('Open (single GPU)','#009E73')]:
            means=[]
            for level in levels:
                ts=[t for t in tasks if tasks[t]['complexity']==level]
                vals=[best[t][group] for t in ts] if use_best else [scores[m['model'],t] for t in ts for m in groups[group]]
                means.append(float(np.mean(vals)));rows.append(dict(panel=title,group=group,complexity=level,mean_f1=means[-1],tasks=len(ts)))
            xs=np.arange(3)+(-.07 if group=='API' else .07)
            ax.plot(xs,means,'o-',color=c,label=group,markersize=7)
            for x,y in zip(xs,means):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,10 if group=='API' else -16),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(3),levels);ax.set_ylim(.15,1.02);ax.set_title(title,fontsize=10);ax.set_xlabel('Coding complexity');ax.set_ylabel('F1')
    axes[1].legend(loc='lower left',frameon=False,fontsize=8);fig.tight_layout()
    save(fig,'fig-complexity','Same as the complexity figure above, for every API model and every open model that runs on a single GPU.',rows)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for title,marker in [('Binary / 2-class','o'),('3-class','^'),('Many-class / multi-label','s')]:
        selected=[]
        for t,d in tasks.items():
            kind='Many-class / multi-label' if d['label_kind']=='multi_binary' else 'Binary / 2-class' if d['label_kind']=='binary' or len(d['labels'])<=2 else '3-class' if len(d['labels'])==3 else 'Many-class / multi-label'
            if kind==title:selected.append(t)
        ax.scatter([tasks[t]['effective_labels'] for t in selected],[gaps[t] for t in selected],marker=marker,color='#333333',label=title,s=35)
    xs=np.array([tasks[t]['effective_labels'] for t in tasks]);ys=np.array([gaps[t] for t in tasks])
    # ggplot transforms x before the lm layer when scale_x_log10 is used.
    fit=np.polyfit(np.log10(xs),ys,1);grid=np.geomspace(xs.min(),xs.max(),100)
    ax.plot(grid,np.polyval(fit,np.log10(grid)),color='#555555',lw=1);ax.axhline(0,color='#777777',ls='--',lw=.8)
    ax.set_xscale('log');ax.set_xticks([1,2,3,5,10,20],['1','2','3','5','10','20']);ax.minorticks_off()
    ax.set_xlabel('Effective number of labels (log scale)');ax.set_ylabel('Best API minus best single-GPU open F1')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False,fontsize=8);fig.tight_layout()
    save(fig,'fig-label-structure-gap','Each point shows one task. The vertical axis shows the best API score minus the best single-GPU open score; the horizontal axis shows the number of labels in practice. The dashed line marks equal performance, and the solid line is a linear fit on the log scale.',[dict(task=t,effective_labels=tasks[t]['effective_labels'],gap=gaps[t]) for t in tasks])
    comparable=defaultdict(list)
    for m in models:
        key=tuple(m.get(k) for k in ['throughput_keyset_sha256','throughput_settings_sha256','throughput_hardware','throughput_items'])
        if m['kind']=='open' and all(x is not None for x in key) and m.get('throughput_items_per_second'):comparable[key].append(m)
    key,group=max(comparable.items(),key=lambda kv:len(kv[1]))
    if len(group)<2:raise ValueError('No comparable throughput group')
    fig,ax=plt.subplots(figsize=(9,5))
    for i,m in enumerate(group):
        x=1/m['throughput_items_per_second'];y=m['mean_task_f1']
        ax.scatter(x,y,color=colors[m['model']],s=65)
        offset=(-90,45) if m['model'].startswith('qwen2_5_32') else (8,12 if i%2==0 else -18)
        ax.annotate(names[m['model']],(x,y),xytext=offset,textcoords='offset points',fontsize=8,
                    arrowprops={'arrowstyle':'-','color':'#999999','lw':.5} if offset==(-90,45) else None)
    ax.set_xscale('log');ax.margins(x=.5,y=.35);ax.set_xlabel('Generation seconds per item (log scale)');ax.set_ylabel(f'Mean F1 across {n_tasks} tasks');fig.tight_layout()
    note=f'Mean F1 against generation time per text. All models coded the same {key[3]:,} texts on one RTX PRO 6000 GPU with identical settings. Load and queue times are excluded.'
    save(fig,'fig-speed',note,[dict(model=m['model'],seconds_per_item=1/m['throughput_items_per_second'],mean_f1=m['mean_task_f1']) for m in group])
    group=sorted(group,key=lambda m:-m['throughput_items_per_second']);fig,ax=plt.subplots(figsize=(9,4.5))
    minutes=[1000/m['throughput_items_per_second']/60 for m in group]
    ax.barh(range(len(group)),minutes,color='#888888',height=.68)
    for i,v in enumerate(minutes):ax.text(v+.01,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_yticks(range(len(group)),[names[m['model']] for m in group],fontsize=9);ax.invert_yaxis()
    ax.set_xlim(0,max(minutes)*1.2);ax.set_xlabel('Generation minutes per 1,000 items');fig.tight_layout()
    save(fig,'fig-local-runtime-per-1000','Generation minutes per 1,000 texts, from the same runs.',[dict(model=m['model'],minutes_per_1000=v) for m,v in zip(group,minutes)])
    featured=[m for m in models if m['model'] in FEATURED_MODELS]
    if {m['model'] for m in featured}!=FEATURED_MODELS:
        raise ValueError('Featured models must all have complete release metrics')
    fig,axes=plt.subplots(2,1,figsize=(9,7));rng=np.random.default_rng(20260910)
    for ax,(title,group) in zip(axes,[('API models',[m for m in featured if m['kind']=='API']),
                                    ('Open weights',[m for m in featured if m['kind']=='open'])]):
        for i,m in enumerate(group):
            ax.scatter(i+rng.uniform(-.18,.18,n_tasks),[scores[m['model'],t] for t in tasks],s=10,color='#cccccc',alpha=.5)
            ax.scatter(i,m['mean_task_f1'],s=85,color=colors[m['model']],edgecolor='#333333',zorder=3)
            ax.text(i,m['mean_task_f1']+.065,f"{m['mean_task_f1']:.3f}",ha='center',fontsize=9)
        ax.set_xticks(range(len(group)),[names[m['model']] for m in group],rotation=24,ha='right',fontsize=9)
        ax.set_ylim(0,1.04);ax.set_ylabel('F1');ax.set_title(title,loc='left',fontsize=11)
    fig.tight_layout(h_pad=2)
    save(fig,'fig-recent-mean-f1','Recent models and reference baselines, including the strong completed Llama checkpoints. Large circles show equal-task mean F1; gray dots show all {N_TASKS} task scores. Each model uses {N_TEXTS} texts. Hardware and uncertainty intervals appear in the comparison table.',[dict(model=m['model'],mean_f1=m['mean_task_f1']) for m in featured])
    # Five compact panels preserve the paper's annotation-type comparison.
    fig,axes=plt.subplots(5,1,figsize=(9,14));rows=[]
    for ax,family in zip(axes.flat,family_order):
        subset=[t for t in tasks if tasks[t]['paper_family']==family]
        values=[float(np.mean([scores[m['model'],t] for t in subset])) for m in featured]
        ax.barh(range(len(featured)),values,color=[colors[m['model']] for m in featured],height=.7)
        for i,v in enumerate(values):ax.text(v+.015,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set_yticks(range(len(featured)),[names[m['model']] for m in featured],fontsize=8)
        ax.invert_yaxis();ax.set_xlim(0,1.08);ax.set_xlabel('Mean F1')
        ax.set_title(f'{family} ({len(subset)} tasks)',fontsize=10,backgroundcolor='#eeeeee')
        rows.extend(dict(family=family,model=m['model'],mean_f1=v) for m,v in zip(featured,values))
    fig.tight_layout(h_pad=2,w_pad=2)
    save(fig,'fig-recent-family','Mean F1 within the five annotation types used in the paper, with each task weighted equally. These types are not the categories in the task selector.',rows)
    manifest['featured_figures']=manifest['figures'][-2:]
    manifest['figures']=manifest['figures'][:-2]
    manifest['featured_models']=sorted(FEATURED_MODELS)
    (out/'figure-data.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    subprocess.run(['Rscript',str(ROOT/'code/render_refresh_paper_figures.R'),str(release_dir.resolve())],check=True)
    return json.loads((out/'figure-data.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,default=ROOT/'output/sidecar/refresh_20260910_release_33')
    print(len(build(p.parse_args().release_dir)['figures']),'paper figures generated')
