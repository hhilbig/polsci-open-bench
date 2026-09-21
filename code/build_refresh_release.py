"""Build aggregate-only matched benchmark release; never submit inference requests.

Optional --open-manifest JSON: {"models": [{"model": ..., "predictions": ...,
"panel_sha256": ..., "prediction_sha256": ..., "revision": ..., "quantization": ...,
"hardware": ..., "hardware_tier": ..., "settings": {...}, "run_dates": [...],
"provenance_validated": true, "generation_tokens_per_second": ..., "gpu_hours": ...}]}.
Prediction files must use the API comparison's task/item_id/gt_*/pred_*/parse_error
columns, cover the entire frozen panel, and be separately provenance-audited.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from build_frontier_2026 import _score_malformed_as_incorrect
from task_registry import load_task_definitions

SEED = 20260910
SOURCE = Path('output/sidecar/refresh_20260910')
OUTPUT = Path('output/sidecar/refresh_20260910_release')
TAXONOMY = Path('output/sidecar/frontier_2026/twitter_figures/full34_task_categories.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_historical_exclusions(path, ranked_provenance):
    """Keep an unscorable historical attempt visible, never ranked as clean."""
    path = Path(path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    entries = payload.get('exclusions')
    if not isinstance(entries, list):
        raise ValueError('Historical exclusion file lacks an exclusions list')
    ranked = {(p.get('model_id'), p.get('revision')) for p in ranked_provenance.values()}
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or any(not entry.get(key) for key in
                ['model', 'model_id', 'revision', 'reason']) or entry.get('status') != 'excluded':
            raise ValueError('Historical exclusion requires identity, status and reason')
        identity = (entry['model_id'], entry['revision'])
        if identity in seen or identity in ranked:
            raise ValueError('Excluded historical checkpoint is duplicate or ranked')
        seen.add(identity)
    return entries


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def malformed_mask(group):
    return group.parse_error.fillna('').astype(str).str.strip().ne('').to_numpy()


def decisions(group, task, support_only=False):
    """Return gold/prediction arrays with precisely the existing failure rule.

    `support_only` restricts the scored classes of a categorical task to those
    that occur in its gold column. A class the panel never contains can only
    score zero, so including it penalises a model in proportion to unused slack
    in the manifest's label list. Off by default so the original release stays
    reproducible.
    """
    bad = malformed_mask(group)
    keys = task['labels'] if task['label_kind'] == 'multi_binary' else [task['label_key']]
    categorical = task['label_kind'] == 'categorical'
    result = []
    for key in keys:
        gold = group['gt_' + key].to_numpy().copy()
        pred = group['pred_' + key].to_numpy().copy()
        if categorical:
            # All-malformed CSV columns are often inferred as float (all NaN).
            pred = pred.astype(object)
            pred[bad] = '__MALFORMED_INCORRECT__'
            classes = task['labels']
            if support_only:
                present = set(pd.Series(gold).dropna().astype(str))
                classes = [c for c in classes if str(c) in present]
        else:
            gold = gold.astype(float)
            if not np.isin(gold, [0, 1]).all():
                raise ValueError('Invalid binary gold label')
            pred[bad] = 1 - gold[bad]
            pred = pred.astype(float)
            if not np.isin(pred, [0, 1]).all():
                raise ValueError('Unmarked invalid binary prediction')
            gold, pred = gold.astype(int), pred.astype(int)
            classes = [0, 1]
        result.append((key, gold, pred, classes))
    return result


def headline_replicates(group, task, weights, support_only=False):
    """Weighted confusion counts reproduce scorer F1 without slow sklearn loops."""
    scores = []
    for _, gold, pred, classes in decisions(group, task, support_only):
        for label in classes if task['label_kind'] == 'categorical' else [1]:
            tp = weights @ ((gold == label) & (pred == label)).astype(float)
            den = weights @ ((gold == label).astype(float) + (pred == label).astype(float))
            scores.append(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0))
    return np.mean(scores, axis=0)


def validate_panel(group, panel, task_defs):
    if group.duplicated(['task', 'item_id']).any():
        raise ValueError('Duplicate task/item key')
    expected = {(r['task'], str(r['item']['item_id'])): r['item']['gt'] for r in panel['rows']}
    actual = set(zip(group.task, group.item_id.astype(str)))
    if actual != set(expected):
        raise ValueError('Incomplete or mismatched frozen panel')
    for row in group.to_dict('records'):
        task = task_defs[row['task']]
        for key, value in expected[(row['task'], str(row['item_id']))].items():
            got = row['gt_' + key]
            if task['label_kind'] != 'categorical':
                got = float(got)
            if got != value:
                raise ValueError('Gold label mismatch')
    for name, rows in group.groupby('task'):
        task = task_defs[name]
        for _, gold, pred, classes in decisions(rows, task):
            if not np.isin(gold, classes).all():
                raise ValueError('Invalid gold label')
            good = ~malformed_mask(rows)
            if not np.isin(pred[good], classes).all():
                raise ValueError('Unmarked invalid prediction')


def public_settings(settings):
    """Publish decoding controls, never messages or arbitrary provenance payloads."""
    allowed = {'model','max_tokens','max_completion_tokens','temperature','top_p','top_k',
               'seed','reasoning_effort','thinking','enable_thinking','repetition_penalty',
               'min_p','dtype','quantization','tensor_parallel_size','max_model_len',
               'structured_output','chat_template_sha256','tokenizer_revision',
               'python_version','vllm_version','runtime_lock_sha256','cuda_component_versions',
               'cuda_toolkit_meta_version','gpu_memory_utilization','prompt_margin_tokens',
               'container_image_sha256','required_environment','kernel_config','max_num_seqs',
               'native_interface','prompt_adapter_sha256'}
    return {k:v for k,v in settings.items() if k in allowed}


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def class_metrics(group, task):
    result = []
    for key, gold, pred, classes in decisions(group, task):
        for label in classes:
            yes, predicted = gold == label, pred == label
            tp = int((yes & predicted).sum())
            support, calls = int(yes.sum()), int(predicted.sum())
            denom = support + calls
            # Undefined MCC is not the sklearn zero convention.
            mcc = float(matthews_corrcoef(yes, predicted)) if len(set(yes)) > 1 and len(set(predicted)) > 1 else None
            result.append(dict(field=key, label=str(label), support=support,
                               predicted_support=calls, f1=2*tp/denom if denom else 0.,
                               accuracy=float((yes == predicted).mean()), mcc=mcc,
                               precision=tp/calls if calls else None,
                               recall=tp/support if support else None))
    return result


def compute(predictions, task_defs, categories, iterations=2000, active_only=False):
    """Pair item draws within tasks and task draws across every model."""
    models, tasks = sorted(predictions.model.unique()), sorted(predictions.task.unique())
    rng = np.random.default_rng(SEED)
    weights = {name: rng.multinomial(100, np.full(100, .01), size=iterations).astype(float) for name in tasks}
    task_draws = rng.integers(len(tasks), size=(iterations, len(tasks)))
    per_task, per_class, aggregates, category_rows = [], [], [], []
    item_samples, task_samples, points = {}, {}, {}
    for model in models:
        model_data = predictions[predictions.model == model]
        matrix, estimates = [], []
        for name in tasks:
            g = model_data[model_data.task == name].sort_values('item_id').reset_index(drop=True)
            task = task_defs[name]
            metrics = _score_malformed_as_incorrect(g, task, support_only=active_only)
            draws = headline_replicates(g, task, weights[name], active_only)
            estimate = float(headline_replicates(g, task, np.ones((1, len(g))), active_only)[0])
            if not np.isclose(estimate, metrics['headline_f1'], atol=1e-12):
                raise ValueError('Fast F1 differs from authoritative scorer')
            matrix.append(draws)
            estimates.append(estimate)
            arrays = decisions(g, task, active_only)
            gold_all, pred_all = np.concatenate([r[1] for r in arrays]), np.concatenate([r[2] for r in arrays])
            acc = float((gold_all == pred_all).mean())
            mcc = float(matthews_corrcoef(gold_all, pred_all)) if len(set(gold_all)) > 1 and len(set(pred_all)) > 1 else None
            per_task.append(dict(model=model, task=name, category=categories[name], n=len(g),
                                 headline_f1=estimate, accuracy=acc, mcc=mcc,
                                 accuracy_unit='label_decision' if task['label_kind']=='multi_binary' else 'item',
                                 malformed=int(malformed_mask(g).sum()), malformed_rate=float(malformed_mask(g).mean()),
                                 item_ci_low=interval(draws)[0], item_ci_high=interval(draws)[1]))
            per_class.extend(dict(model=model, task=name, **r) for r in class_metrics(g, task))
        matrix, estimates = np.asarray(matrix), np.asarray(estimates)
        item_samples[model] = matrix.mean(axis=0)
        task_samples[model] = estimates[task_draws].mean(axis=1)
        points[model] = float(estimates.mean())
        aggregates.append(dict(model=model, n=len(model_data), tasks=len(tasks), mean_task_f1=points[model],
                               item_ci_low=interval(item_samples[model])[0], item_ci_high=interval(item_samples[model])[1],
                               task_ci_low=interval(task_samples[model])[0], task_ci_high=interval(task_samples[model])[1],
                               malformed=int(malformed_mask(model_data).sum()), malformed_rate=float(malformed_mask(model_data).mean())))
        for category in sorted(set(categories.values())):
            idx = [i for i, t in enumerate(tasks) if categories[t] == category]
            if idx:
                category_rng = np.random.default_rng(SEED)
                boot = estimates[idx][category_rng.integers(len(idx), size=(iterations, len(idx)))].mean(axis=1)
                ci = interval(matrix[idx].mean(axis=0))
                category_rows.append(dict(model=model, category=category, tasks=len(idx), mean_task_f1=float(estimates[idx].mean()),
                                          item_ci_low=ci[0], item_ci_high=ci[1], task_ci_low=interval(boot)[0], task_ci_high=interval(boot)[1]))
    pairs = []
    for model, reference in itertools.permutations(models, 2):
        pairs.append(dict(model=model, reference=reference, difference=points[model]-points[reference],
                          item_ci_low=interval(item_samples[model]-item_samples[reference])[0],
                          item_ci_high=interval(item_samples[model]-item_samples[reference])[1],
                          task_ci_low=interval(task_samples[model]-task_samples[reference])[0],
                          task_ci_high=interval(task_samples[model]-task_samples[reference])[1]))
    return dict(models=aggregates, tasks=per_task, classes=per_class, categories=category_rows, pairs=pairs)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def build(source=SOURCE, output=OUTPUT, open_manifest=None, iterations=2000,
          api_manifest=None, active_only=False):
    source, output = Path(source), Path(output)
    panel = json.loads((source/'panel.json').read_text())
    report = json.loads((source/'comparison_report.json').read_text())
    if report['panel_sha256'] != canonical_sha(panel):
        raise ValueError('Panel hash mismatch')
    counts = pd.Series([r['task'] for r in panel['rows']]).value_counts()
    if panel['seed'] != SEED or len(counts) != 34 or not counts.eq(100).all():
        raise ValueError('Expected approved 34-task 100-item panel')
    if iterations < 2:
        raise ValueError('At least two bootstrap replicates required')
    task_defs = {t['name']: t for t in load_task_definitions(active_only=active_only)}
    if active_only:
        # Drop held-out tasks from the frozen panel as well, or validate_panel
        # compares a 34-task expectation against 33 tasks of predictions. The
        # hash check above still runs against the original panel, so provenance
        # is verified before the subset is taken.
        dropped = sorted({r['task'] for r in panel['rows']} - set(task_defs))
        panel = dict(panel, rows=[r for r in panel['rows'] if r['task'] in task_defs])
        print(json.dumps({'active_only': True, 'dropped_tasks': dropped,
                          'tasks': len(task_defs)}))
    categories_df = pd.read_csv(TAXONOMY)
    if categories_df.task.duplicated().any():
        raise ValueError('Duplicate taxonomy task')
    categories = dict(zip(categories_df.task, categories_df.category))
    frames = [pd.read_csv(source/'comparison_predictions.csv', dtype={'item_id':str}, low_memory=False)]
    if active_only:
        frames[0] = frames[0][frames[0].task.isin(task_defs)].reset_index(drop=True)
    if set(frames[0].model) != {r['model'] for r in report['models']} or len(report['models']) != 8:
        raise ValueError('API model coverage mismatch')
    provenance = {}
    for entry in report['models']:
        name = entry['model']
        request = json.loads((source/name/'requests.json').read_text())['requests'][0]
        provenance[name] = dict(entry, access='api', hardware_tier='api', settings=request['body'] if 'body' in request else request.get('params'),
                                cost_note=report['cost_note'])
        # Do not export prompts or item-specific request fields.
        provenance[name]['settings'] = public_settings(provenance[name]['settings'] or {})
        provenance[name]['settings']['structured_output'] = 'tool_use' if 'tools' in request['params'] else request['params'].get('response_format', {}).get('type', 'json_prompt')
    api_addition_cost = 0.0
    # One manifest per API run (Jev, Gemini); a single path is still accepted.
    api_manifests = ([] if not api_manifest else
                     [api_manifest] if isinstance(api_manifest, (str, Path)) else list(api_manifest))
    for manifest_path in api_manifests:
        additions = json.loads(Path(manifest_path).read_text()).get('models')
        if not isinstance(additions, list) or not additions:
            raise ValueError('API addition manifest requires a nonempty models list')
        for entry in additions:
            required = ['model','model_id','provider','predictions','panel_sha256',
                        'prediction_sha256','revision','hardware_tier','settings',
                        'run_dates','input_tokens','output_tokens','cost_usd_upper',
                        'returned_models','documented_versions']
            if any(key not in entry for key in required) or entry.get('provenance_validated') is not True:
                raise ValueError('API predictions require complete audited provenance')
            if entry['hardware_tier'] != 'api':
                raise ValueError('API addition must use the API hardware tier')
            if entry['panel_sha256'] != report['panel_sha256'] or sha(entry['predictions']) != entry['prediction_sha256']:
                raise ValueError('API prediction provenance hash mismatch')
            name = entry['model']
            if name in provenance:
                raise ValueError('Duplicate model source')
            frame = pd.read_csv(entry['predictions'], dtype={'item_id':str}, low_memory=False)
            if set(frame.model) != {name}:
                raise ValueError('API addition model name mismatch')
            if active_only:
                frame = frame[frame.task.isin(task_defs)].reset_index(drop=True)
            frames.append(frame)
            provenance[name] = {key:entry[key] for key in [
                'model','model_id','provider','revision','hardware_tier','run_dates',
                'input_tokens','output_tokens','cost_usd_upper','returned_models',
                'documented_versions','prediction_sha256']}
            provenance[name]['settings'] = public_settings(entry['settings'])
            provenance[name]['access'] = 'api'
            api_addition_cost += float(entry['cost_usd_upper'])
    if open_manifest:
        for entry in json.loads(Path(open_manifest).read_text())['models']:
            required = ['model','predictions','panel_sha256','prediction_sha256','revision','quantization','hardware','hardware_tier','settings','run_dates']
            if any(key not in entry for key in required) or entry.get('provenance_validated') is not True:
                raise ValueError('Open predictions require complete audited provenance')
            if entry['panel_sha256'] != report['panel_sha256'] or sha(entry['predictions']) != entry['prediction_sha256']:
                raise ValueError('Open prediction provenance hash mismatch')
            name = entry['model']
            if name in provenance:
                raise ValueError('Duplicate model source')
            frame = pd.read_csv(entry['predictions'], dtype={'item_id':str}, low_memory=False)
            if set(frame.model) != {name}:
                raise ValueError('Open model name mismatch')
            if active_only:
                frame = frame[frame.task.isin(task_defs)].reset_index(drop=True)
            frames.append(frame)
            provenance[name] = {k:entry[k] for k in ['model','revision','quantization','hardware','hardware_tier','settings','run_dates']}
            for field in ['model_id','compatibility_group','throughput_note','prediction_sha256',
                          'throughput_keyset_sha256','throughput_settings_sha256',
                          'throughput_items','throughput_hardware',
                          'gpu_hours_historical','gpu_hours_backfill','gpu_hours_note',
                          'gpu_hours_preinference','gpu_hours_pilot','gpu_hours_remainder',
                          'sacct_attempts_sha256','preinference_failed_attempt_ids']:
                if field in entry:
                    provenance[name][field] = entry[field]
            provenance[name]['settings'] = public_settings(entry['settings'])
            provenance[name].update(access='open',
                generation_tokens_per_second=entry.get('generation_tokens_per_second'),
                generation_items_per_second=entry.get('generation_items_per_second'),
                gpu_hours=entry.get('gpu_hours'))
    predictions = pd.concat(frames, ignore_index=True)
    for name, group in predictions.groupby('model'):
        validate_panel(group, panel, task_defs)
    result = compute(predictions, task_defs, categories, iterations, active_only)
    for row in result['models']:
        row['provenance'] = provenance[row['model']]
        if row['model'] in set(frames[0].model):
            original = provenance[row['model']]
            if active_only:
                # The validated report is a 34-task legacy-metric figure, so its
                # mean_task_f1 cannot match a rescore by construction. Check the
                # row and malformed counts against the filtered predictions
                # instead, which still catches a dropped or duplicated task.
                subset = frames[0][frames[0].model == row['model']]
                if row['n'] != len(subset) or row['malformed'] != int(malformed_mask(subset).sum()):
                    raise ValueError('Rescored release disagrees with the filtered panel')
            elif row['n'] != original['n'] or row['malformed'] != original['malformed'] or not np.isclose(row['mean_task_f1'], original['mean_task_f1'], atol=1e-12):
                raise ValueError('API release disagrees with validated final report')
    task_names = sorted(predictions.task.unique())
    result['task_definitions'] = [dict(task=name, category=categories[name], source=task_defs[name].get('source'),
                                       family=task_defs[name].get('family'), label_kind=task_defs[name]['label_kind'],
                                       labels=task_defs[name]['labels'], label_key=task_defs[name].get('label_key')) for name in task_names]
    exclusions_path = output/'historical_exclusions.json'
    historical_exclusions = load_historical_exclusions(exclusions_path, provenance)
    if historical_exclusions:
        result['historical_exclusions'] = historical_exclusions
    result['manifest'] = dict(release='refresh_20260910', status='api_complete_open_refresh_pending' if not open_manifest else 'validated_local_preview_publication_approval_required',
                              seed=SEED, bootstrap_replicates=iterations, panel_sha256=report['panel_sha256'],
                              prediction_sha256=sha(source/'comparison_predictions.csv'), taxonomy_sha256=sha(TAXONOMY),
                              generator_sha256=sha(__file__),
                              open_manifest_sha256=sha(open_manifest) if open_manifest else None,
                              api_manifest_sha256=(None if not api_manifests else sha(api_manifests[0])
                                  if len(api_manifests) == 1 else [sha(p) for p in api_manifests]),
                              historical_exclusions_sha256=(sha(exclusions_path)
                                  if historical_exclusions else None),
                              scorer_sha256=sha(Path(__file__).with_name('build_summary.py')),
                              malformed_scorer_sha256=sha(Path(__file__).with_name('build_frontier_2026.py')),
                              numpy_version=np.__version__, pandas_version=pd.__version__,
                              api_completed_at=report['completed_at'],
                              api_cost_usd_upper=report['cost_usd_upper'] + api_addition_cost,
                              reserve_usd=report['reserve_usd'], baseline='Qwen3.6 27B',
                              uncertainty='95% percentile intervals; paired item resampling within each fixed task versus resampling tasks. Not simultaneous intervals.',
                              scoring='Equal-task mean F1. Binary positive-class F1; multi-binary mean positive-class F1; categorical macro F1 over declared classes. Malformed decisions incorrect. Absent-class F1 remains zero per existing scorer. Undefined MCC is null.',
                              public_data='Aggregate metrics only; no raw texts, responses, or row-level predictions.')
    result = clean_json(result)
    output.mkdir(parents=True, exist_ok=True)
    (output/'release.json').write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for key in ['models','tasks','classes','categories','pairs']:
        rows = [{k:v for k,v in row.items() if k != 'provenance'} for row in result[key]]
        if key == 'models':
            for row, original in zip(rows, result[key]):
                for field, value in original['provenance'].items():
                    if field not in row:
                        row[field] = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        pd.DataFrame(rows).to_csv(output/(key+'.csv'), index=False, float_format='%.12g')
    print(json.dumps(dict(models=len(result['models']), predictions=len(predictions), output=str(output))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--open-manifest', type=Path)
    parser.add_argument('--api-manifest', type=Path, action='append',
                        help='Audited API addition manifest; repeat for several runs.')
    parser.add_argument('--active-only', action='store_true',
                        help='Score the panel on the active task set and the '
                             'support-only metric, matching the current benchmark.')
    parser.add_argument('--iterations', type=int, default=2000)
    args = parser.parse_args()
    build(args.source, args.output, args.open_manifest, args.iterations,
          args.api_manifest, args.active_only)
