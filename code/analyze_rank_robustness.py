"""How much do the model rankings depend on single tasks? Reads a scored release; no inference.

Writes three things next to the release, under robustness/:
- rank_robustness.csv: one row per model with
  * tier: models the paired task-resampling 95% interval cannot separate. Walking down
    the ranking, a model joins the current tier unless its difference from the tier's
    top model has an interval that excludes zero; then it starts the next tier.
  * loo_rank_best / loo_rank_worst: the model's best and worst rank when any one task
    is dropped (33 reruns of the equal-task mean on 32 tasks).
  * alternative aggregates, for comparison only (the headline stays the equal-task mean):
    trimmed: each model's mean after dropping its 2 best and 2 worst tasks;
    reliability_weighted: tasks weighted by 1 / (mean item-bootstrap variance of task F1
    across models), so tasks whose F1 is noisy at 100 items count for less.
- task_weights.csv: the reliability weight of each task.
- rank_robustness.pdf/.png: the figure.

    python3 code/analyze_rank_robustness.py --release-dir output/sidecar/refresh_20261106_release_33
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRIM = 2


def tiers(order, pairs):
    """Greedy tiers from the paired task-resampling interval against each tier's top model."""
    ci = pairs.set_index(['model', 'reference'])
    result, tier, top = {}, 1, order[0]
    for model in order:
        if model != top:
            row = ci.loc[(model, top)]
            if row.task_ci_high < 0:  # clearly below the tier's top model
                tier, top = tier + 1, model
        result[model] = tier
    return result


def leave_one_out(wide):
    ranks = [wide.drop(task).mean().rank(ascending=False, method='min') for task in wide.index]
    ranks = pd.concat(ranks, axis=1)
    return ranks.min(axis=1).astype(int), ranks.max(axis=1).astype(int)


def trimmed(wide, k=TRIM):
    return wide.apply(lambda s: s.sort_values().iloc[k:len(s) - k].mean())


def reliability_weights(tasks):
    var = ((tasks.item_ci_high - tasks.item_ci_low) / (2 * 1.96)) ** 2
    per_task = var.groupby(tasks.task).mean()
    weights = 1 / per_task
    return weights / weights.sum()


def build(release_dir):
    release_dir = Path(release_dir)
    tasks = pd.read_csv(release_dir / 'tasks.csv')
    pairs = pd.read_csv(release_dir / 'pairs.csv')
    models = pd.read_csv(release_dir / 'models.csv')
    wide = tasks.pivot(index='task', columns='model', values='headline_f1')
    mean = wide.mean()
    if not np.allclose(mean[models.model].to_numpy(), models.mean_task_f1.to_numpy(), atol=1e-9):
        raise ValueError('Task scores do not reproduce the release means')
    order = list(mean.sort_values(ascending=False).index)
    weights = reliability_weights(tasks)
    weighted = (wide.mul(weights, axis=0)).sum()
    trim = trimmed(wide)
    best, worst = leave_one_out(wide)
    out = pd.DataFrame({'model': order})
    out['rank'] = range(1, len(order) + 1)
    out['mean_task_f1'] = mean[order].to_numpy()
    out = out.merge(models[['model', 'task_ci_low', 'task_ci_high']], on='model')
    out['tier'] = out.model.map(tiers(order, pairs))
    out['loo_rank_best'] = out.model.map(best)
    out['loo_rank_worst'] = out.model.map(worst)
    out['trimmed_f1'] = out.model.map(trim)
    out['trimmed_rank'] = out.model.map(trim.rank(ascending=False, method='min')).astype(int)
    out['reliability_weighted_f1'] = out.model.map(weighted)
    out['reliability_weighted_rank'] = out.model.map(weighted.rank(ascending=False, method='min')).astype(int)
    target = release_dir / 'robustness'
    target.mkdir(exist_ok=True)
    out.to_csv(target / 'rank_robustness.csv', index=False, float_format='%.6f')
    weights.rename('weight').sort_values().to_csv(target / 'task_weights.csv', float_format='%.6f')
    figure(out, target)
    return out, weights


def figure(out, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(out)
    y = np.arange(n)[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.24 * n + 1.6), sharey=True,
                               gridspec_kw={'width_ratios': [3, 2]})
    shades = ['#2b5c8a', '#8aa9c4']
    for tier, rows in out.groupby('tier'):
        color = shades[(tier - 1) % 2]
        yy = y[rows.index]
        a.errorbar(rows.mean_task_f1, yy, xerr=[rows.mean_task_f1 - rows.task_ci_low,
                                                 rows.task_ci_high - rows.mean_task_f1],
                   fmt='o', color=color, ms=4, lw=1, capsize=0)
        a.text(out.task_ci_high.max() + 0.004, yy.mean(), f'tier {tier}', va='center',
               fontsize=7, color=color)
    a.set_yticks(y)
    a.set_yticklabels(out.model, fontsize=7)
    a.set_xlabel('Mean task F1, 95% interval resampling tasks')
    a.set_title('Score and tier', fontsize=9, loc='left')
    b.hlines(y, out.loo_rank_best, out.loo_rank_worst, color='#bbbbbb', lw=4,
             label='rank range, any one task dropped')
    b.plot(out['rank'], y, 'o', color='#2b5c8a', ms=4, label='current (equal-task mean)')
    b.plot(out.trimmed_rank, y, 'x', color='#c0504d', ms=4, label=f'trimmed (drop best/worst {TRIM})')
    b.plot(out.reliability_weighted_rank, y, '^', color='#6a8f3c', ms=3.5, label='reliability-weighted')
    b.invert_xaxis()
    b.set_xlabel('Rank (1 = best)')
    b.set_title('Rank under each check', fontsize=9, loc='left')
    b.legend(fontsize=6.5, loc='lower left', frameon=False)
    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(target / 'rank_robustness.pdf')
    fig.savefig(target / 'rank_robustness.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release-dir', required=True)
    out, weights = build(parser.parse_args().release_dir)
    print(out.drop(columns=['task_ci_low', 'task_ci_high']).round(3).to_string(index=False))
    print(weights.sort_values().round(4).to_string())
