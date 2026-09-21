import importlib.util
import json
from pathlib import Path

import unittest
import tempfile
import hashlib


def test_paper_figures_require_matching_release_and_assets():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        assert preview.paper_figures(root,{})==''
        folder=root/'figures';folder.mkdir()
        manifest={'release_sha256':'stale','figures':[],'featured_figures':[]}
        (folder/'figure-data.json').write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError,'stale'):
            preview.paper_figures(root,{})
        manifest['release_sha256']=hashlib.sha256(b'{}').hexdigest()
        manifest['featured_models']=sorted(preview.FEATURED_MODELS)
        manifest['featured_figures']=[{'file':f'fig-top-{i}','title':f'Top {i}','caption':'Matched results.'} for i in range(5)]
        manifest['figures']=[{'file':f'fig-more-{i}','title':f'More {i}','caption':'Matched results.'} for i in range(7)]
        (folder/'figure-data.json').write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Missing figure'):
            preview.paper_figures(root,{})
        for row in manifest['featured_figures']+manifest['figures']:
            for ext in ['svg','pdf']:
                (folder/(row['file']+'.'+ext)).write_text('test')
        result=preview.paper_figures(root,{})
        default,expanded=result.split('<details class="disclosure" id="more-analyses">')
        assert default.count('<figure ')==5 and '<h3>Top 0</h3>' in default
        assert expanded.count('<figure ')==7 and 'figure-data.json' in expanded
        manifest['featured_models']=['other-model']
        (folder/'figure-data.json').write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Featured figure selection'):
            preview.paper_figures(root,{})

spec=importlib.util.spec_from_file_location('preview',Path(__file__).parents[1]/'code/build_refresh_preview.py')
preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(preview)

def data():
    return dict(release_id='test',status='pending',updated_at='2026-09-14',models=[dict(model='example',label='Example',n=3400,tasks=34,kind='API',hardware_tier='api',mean_task_f1=.6,malformed=1,cost_usd_upper=2,cost_per_1k_items=.5,cost_basis='provider_ledger')],task_scores=[],class_scores=[],candidates=[dict(model='Open candidate',status='pending',reason='Pilot queued')])


def test_page_has_no_ranking_tables_and_a_short_task_table():
    source=data()
    source['models'].append(dict(source['models'][0],model='older-model',label='Older model',mean_task_f1=.99))
    page=preview.render(source)
    assert 'model-rows' not in page and 'featured-rows' not in page
    assert '<caption>Five best models on the selected task</caption>' in page
    assert '<option value="older-model">Older model</option>' in page
    assert page.index('value="older-model"')<page.index('value="example"')
    assert 'rows.slice(0,5)' in preview.JS
    assert '<details class="disclosure" id="methods">' in page and '<h2 id="downloads">' in page

def test_static_page_and_pending():
    page=preview.render(data())
    assert 'Pilot queued' in page and 'Local preview, not published' in page
    assert 'The open-weight models are a selection rather than a complete list.' in page
    assert '<noscript>' in page and 'name="robots" content="noindex"' in page

def test_deepseek_observed_run_dates_are_required():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14'),
                 models=[dict(model='deepseek-v4-flash',n=3400,tasks=34,mean_task_f1=.6,
                              malformed=0,provenance=dict(access='api',hardware_tier='api',
                                  cost_usd_upper=1,documented_versions=['DeepSeek-V4.1-Flash'],
                                  observed_completion_dates=['2026-09-12','2026-09-10','2026-09-12']))],
                 tasks=[],classes=[],categories=[],pairs=[])
    source=preview.normalize_release(release)
    assert source['models'][0]['observed_run_dates']==['2026-09-10','2026-09-12']
    assert '2026-09-10' in preview.render(source)
    release['models'][0]['provenance']['observed_completion_dates']=[]
    with unittest.TestCase().assertRaisesRegex(ValueError,'observed run dates'):
        preview.normalize_release(release)

def test_multi_gpu_model_keeps_its_hardware_tier():
    release=dict(manifest=dict(release='test',status='complete',api_completed_at='2026-09-14'),
                 models=[dict(model='larger',n=3400,tasks=34,mean_task_f1=.6,malformed=0,
                              provenance=dict(access='open',hardware_tier='multi-gpu',
                                              hardware='2 NVIDIA RTX PRO 6000 Blackwell',
                                              revision='pinned',run_dates=['2026-09-15']))],
                 tasks=[],classes=[],categories=[],pairs=[])
    normalized=preview.normalize_release(release)
    assert normalized['models'][0]['hardware_tier']=='multi-gpu'
    assert '2 NVIDIA RTX PRO 6000 Blackwell' in preview.render(normalized)

def test_partial_model_cannot_rank():
    source=data();source['models'][0]['n']=3399
    with unittest.TestCase().assertRaisesRegex(ValueError,'Incomplete'): preview.render(source)

def test_script_and_html_escape():
    source=data();source['models'][0]['label']='</script><script>alert(1)</script>'
    page=preview.render(source)
    assert '</script><script>alert(1)</script>' not in page
    assert '\\u003c/script>' in page

def test_payload_excludes_row_content():
    source=data();source['models'][0]['raw_response']='secret';source['texts']=['private']
    result=preview.safe_payload(source)
    assert 'secret' not in json.dumps(result) and 'private' not in json.dumps(result)

def test_public_guard():
    for source in [{'item_id':'1'},{'provenance':{'messages':[]}},{'file':'/Users/name/private'},
                   {'file':'output/sidecar/private/metadata.json'},{'key':'sk-proj-secret'}]:
        with unittest.TestCase().assertRaises(ValueError): preview.validate_public(source)

def test_public_candidate_keeps_preflight_hash_without_private_metadata_path():
    candidate={'model':'Qwen Flash','feasibility':{'cpu_preflight':{
        'metadata':'output/sidecar/refresh_20260910_release/qwen_flash_runtime.json',
        'metadata_sha256':'abc123','gpu_execution_tested':False}}}
    cleaned=preview.public_candidate(candidate)
    assert 'metadata' not in cleaned['feasibility']['cpu_preflight']
    assert cleaned['feasibility']['cpu_preflight']['metadata_sha256']=='abc123'
    assert candidate['feasibility']['cpu_preflight']['metadata'].startswith('output/sidecar/')
    preview.validate_public(cleaned)

def test_null_and_zero_distinct():
    assert preview.fmt(None)=='Unavailable'
    assert preview.fmt(0)=='0.000'

def test_controls_and_no_external_scripts():
    page=preview.render(data())
    for identifier in ['category','task','compare','task-rank','pair-model','pair-reference']:
        assert f'id="{identifier}"' in page
    assert 'src="https://' not in page
    assert "document.createElement('td')" in preview.JS
    assert 'ranks ${rank+1} of ${rows.length}' in preview.JS

def test_homepage_proposals_follow_current_data_and_sitemap_markup():
    assert '<div>' not in preview.homepage_link_proposal()
    assert '2026. Matched evaluation' in preview.homepage_link_proposal()
    assert '<priority>0.8</priority>' in preview.SITEMAP_ENTRY_PROPOSAL

def test_select_border_uses_contrasting_white_background_color():
    assert 'select{font:inherit;color:inherit;background:white;border:1px solid #888' in preview.CSS
    assert 'border:1px solid #999' not in preview.CSS

def test_focus_outline_uses_higher_contrast_link_blue():
    assert '[tabindex]:focus-visible{outline:2px solid var(--link)' in preview.CSS
    assert 'outline:2px solid var(--accent)' not in preview.CSS

def test_skip_link_target_can_receive_keyboard_focus():
    page=preview.render(data())
    assert '<a class="skip-link" href="#main">Skip to content</a>' in page
    assert '<main id="main" tabindex="-1">' in page

def test_no_javascript_fallbacks():
    assert '.class-support-details{display:none}.js .class-support-details{display:block}' in preview.CSS
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14',
                               seed=20260910,bootstrap_replicates=2000,panel_sha256='hash',
                               scoring='Equal-task F1.',uncertainty='Paired intervals.'),
                 models=[dict(model='example',n=3400,tasks=34,mean_task_f1=.6,malformed=0,
                              provenance=dict(access='api',hardware_tier='api',cost_usd_upper=1))],
                 tasks=[],classes=[],categories=[],pairs=[])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'release.json').write_text(json.dumps(release))
        page=(preview.build(root)/'index.html').read_text()
        assert '<details class="class-support-details"><summary>Class support and class F1 for the compared model</summary>' in page
        assert 'Task and class results are in the downloads.' in page
        assert 'that zero says nothing about the model' in page
        assert 'F1 of 0 by convention' in (root/'preview/llm-benchmark/downloads/methodology.md').read_text()

def test_build_preserves_aggregate_download_and_values():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14',seed=20260910,
                               panel_sha256='panel-hash',scoring='Equal-task F1.',uncertainty='Paired percentile intervals.',
                               bootstrap_replicates=2000),
                 models=[dict(model='example',n=3400,tasks=34,mean_task_f1=.6,malformed=0,provenance=dict(access='api',hardware_tier='api',cost_usd_upper=1))],
                 tasks=[],classes=[],categories=[],pairs=[],task_definitions=[dict(task='example',source='Example et al.',labels=['a','b'])])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'release.json').write_text(json.dumps(release))
        output=preview.build(root)
        assert json.loads((output/'downloads/release.json').read_text())==release
        assert json.loads((output/'downloads/manifest.json').read_text())==release['manifest']
        assert 'Example et al.' in (output/'downloads/task_definitions.csv').read_text()
        import csv
        with (output/'downloads/task_definitions.csv').open(newline='') as handle:
            assert json.loads(next(csv.DictReader(handle))['labels'])==['a','b']
        assert '2000 draws' in (output/'downloads/methodology.md').read_text()
        assert 'models are a selection rather than a complete list' in (output/'downloads/methodology.md').read_text()
        for name in ['task_definitions.csv','manifest.json','methodology.md']:
            assert f'downloads/{name}' in (output/'index.html').read_text()
        assert '0.600' in (output/'index.html').read_text()
        first={p.name:p.read_bytes() for p in (output/'downloads').iterdir()};first['index.html']=(output/'index.html').read_bytes();preview.build(root)
        assert first=={**{p.name:p.read_bytes() for p in (output/'downloads').iterdir()},'index.html':(output/'index.html').read_bytes()}
        assert 'pair-model' in (output/'index.html').read_text()
        assert preview.verify(root)['models']==1
        homepage=root/'preview/homepage-link-proposal.html'
        homepage_original=homepage.read_text()
        homepage.write_text(homepage_original.replace('llm-benchmark/','old-benchmark/',1))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Homepage link proposal'):
            preview.verify(root)
        homepage.write_text(homepage_original)
        sitemap=root/'preview/sitemap-entry-proposal.xml'
        sitemap_original=sitemap.read_text()
        sitemap.write_text(sitemap_original.replace('llm-benchmark/','old-benchmark/',1))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Sitemap proposal'):
            preview.verify(root)
        sitemap.write_text(sitemap_original)
        page_file=output/'index.html'
        page_original=page_file.read_text()
        page_file.write_text(page_original.replace('href="downloads/release.json"',
                                              'href="downloads/old-release.json"',1))
        with unittest.TestCase().assertRaisesRegex(ValueError,'aggregate download link'):
            preview.verify(root)
        page_file.write_text(page_original.replace('href="https://www.hannohilbig.com/llm-benchmark/"',
                                                   'href="https://www.hannohilbig.com/old-benchmark/"',1))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Preview metadata'):
            preview.verify(root)
        page_file.write_text(page_original)
        model_csv=output/'downloads/models.csv'
        original_csv=model_csv.read_text()
        changed_csv=original_csv.replace('0.6','0.7',1)
        assert changed_csv!=original_csv
        model_csv.write_text(changed_csv)
        with unittest.TestCase().assertRaisesRegex(ValueError,'Public models CSV'):
            preview.verify(root)
        model_csv.write_text(original_csv)
        method_file=output/'downloads/methodology.md'
        original_methods=method_file.read_text()
        method_file.write_text(original_methods.replace('2000 draws','1000 draws',1))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Public methodology'):
            preview.verify(root)

def test_preview_verifier_rejects_changed_embedded_metrics():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14',seed=20260910,
                               panel_sha256='panel-hash',scoring='Equal-task F1.',uncertainty='Paired intervals.',
                               bootstrap_replicates=2000),
                 models=[dict(model='example',n=3400,tasks=34,mean_task_f1=.6,malformed=0,
                              provenance=dict(access='api',hardware_tier='api',cost_usd_upper=1))],
                 tasks=[],classes=[],categories=[],pairs=[])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'release.json').write_text(json.dumps(release))
        page=preview.build(root)/'index.html'
        original=page.read_text()
        changed=original.replace('"mean_task_f1": 0.6','"mean_task_f1": 0.7',1)
        assert changed!=original
        page.write_text(changed)
        with unittest.TestCase().assertRaisesRegex(ValueError,'Embedded preview metrics'):
            preview.verify(root)

def test_verifier_rejects_overview_figure_that_differs_from_release():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14',seed=20260910,
                               panel_sha256='panel-hash',scoring='Equal-task F1.',uncertainty='Paired intervals.',
                               bootstrap_replicates=2000),
                 models=[dict(model='example',n=3400,tasks=34,mean_task_f1=.6,malformed=0,
                              provenance=dict(access='api',hardware_tier='api',cost_usd_upper=1))],
                 tasks=[],classes=[],categories=[],pairs=[])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'release.json').write_text(json.dumps(release))
        preview.build(root)
        loaded=preview.preview_release(root)
        figures=root/'preview/llm-benchmark/figures';figures.mkdir()
        manifest=dict(release_sha256=hashlib.sha256(json.dumps(loaded,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                      featured_models=sorted(preview.FEATURED_MODELS),featured_figures=[],
                      figures=[dict(file='fig-mean-f1',title='Overall',caption='x',data=[dict(model='example',mean_task_f1=.6)])])
        (figures/'figure-data.json').write_text(json.dumps(manifest))
        preview.verify(root)
        manifest['figures'][0]['data'][0]['mean_task_f1']=.61
        (figures/'figure-data.json').write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError,'Overview figure'):
            preview.verify(root)

def test_ranked_candidate_not_repeated_outside_ranking_and_update_date_is_current():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-10'),
                 models=[dict(model='qwen3_8_27b_fp8',n=3400,tasks=34,mean_task_f1=.7,
                              malformed=0,provenance=dict(access='open',hardware_tier='single-gpu',
                              model_id='Qwen/Qwen3.8-27B-FP8',revision='pinned',run_dates=['2026-09-15']))],
                 tasks=[],classes=[],categories=[],pairs=[],candidates=[
                     dict(model='Qwen3.8 27B',model_id='Qwen/Qwen3.8-27B-FP8',revision='pinned',
                          status='pending',reason='Outdated pilot note'),
                     dict(model='Qwen3.8 Flash-Next',model_id='Qwen/Flash',revision='other',
                          status='pending',reason='Pilot not run')])
    result=preview.normalize_release(release)
    assert result['updated_at']=='2026-09-15'
    assert [c['model'] for c in result['candidates']]==['Qwen3.8 Flash-Next']
    assert result['models'][0]['label']=='Qwen3.8 27B (FP8)'

def test_new_historical_labels_show_model_and_weight_format():
    assert preview.LABELS['qwen3_30b_a3b_bf16']=='Qwen3 30B-A3B (BF16)'
    assert preview.LABELS['gemma3_27b_it_fp8_dynamic']=='Gemma 3 27B (dynamic FP8)'
    assert preview.LABELS['qwen3_8_flash_next_fp8']=='Qwen3.8 Flash-Next (FP8)'

def test_historical_exclusion_stays_outside_ranked_models():
    release=dict(manifest=dict(release='test',status='pending',api_completed_at='2026-09-14',
                               seed=20260910,bootstrap_replicates=2000,panel_sha256='hash',
                               scoring='Equal-task F1.',uncertainty='Paired intervals.'),
                 models=[],tasks=[],classes=[],categories=[],pairs=[],
                 historical_exclusions=[dict(model='GPT-OSS 120B',model_id='openai/gpt-oss-120b',
                                             revision='pinned',status='excluded',
                                             reason='Malformed item ID unavailable')])
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        (root/'release.json').write_text(json.dumps(release))
        (root/'candidates.json').write_text(json.dumps({'candidates':[
            dict(model='Qwen Flash',model_id='Qwen/Flash',revision='other',
                 status='pending',reason='Pilot queued')]}))
        data=preview.normalize_release(preview.preview_release(root))
        assert len(data['models'])==0
        assert [c['status'] for c in data['candidates']]==['pending','excluded']
        assert 'GPT-OSS' in preview.methodology(preview.preview_release(root))

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items() if name.startswith('test_'))

if __name__=='__main__':
    unittest.main()
