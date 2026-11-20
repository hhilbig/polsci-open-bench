# Training-data exposure: timing check

Checked 2026-11-14 on release `refresh_20261106` (33 tasks). Script:
[`code/analyze_contamination_timing.py`](../code/analyze_contamination_timing.py).
Results:
[`contamination_timing.csv`](../output/sidecar/refresh_20261106_release_33/robustness/contamination_timing.csv),
[`contamination_timing_cells.csv`](../output/sidecar/refresh_20261106_release_33/robustness/contamination_timing_cells.csv),
[figure](../output/sidecar/refresh_20261106_release_33/robustness/contamination_timing.pdf).

**Verdict.** The open-weight models do not score better on tasks that were
published before they were released. The estimate goes the other way, about 3
F1 points lower, and is not distinguishable from zero. A pre-release advantage
larger than about 1 to 1.5 F1 points is outside the 95% intervals. This
argues against exposure to the published task sources as an important driver of
the scores. It does not rule out exposure to the underlying texts.

## Why

Newer models may have been trained on the papers, replication archives and
datasets the tasks come from, including their gold labels. If so, their scores
would partly reflect memory rather than coding ability, and they would be
inflated most on tasks that were public before the model was trained. The
existing check (in [`frontier_broad_benchmark.md`](frontier_broad_benchmark.md))
compared only two models on 18 tasks. This extends it to every model in the
release with a verifiable date.

## What is compared

For each of 21 open-weight models and each of the 33 release tasks, the task's
headline F1 when the task source was published in a calendar year before the
model's release year, compared with the same model's F1 on tasks published
after its release year, after removing each model's average level and each
task's average difficulty.

Mechanically: an ordinary least squares regression of task F1 on an indicator
for "source published before the model's release year", an indicator for "same
year", model fixed effects and task fixed effects. Model fixed effects absorb
that newer models are better overall; task fixed effects absorb that some tasks
are harder. The coefficient therefore comes from comparing models released at
different times on the same task, relative to how those models compare on other
tasks. For example, on a task published in 2025, a 2026 model has potentially
seen it and a 2024 model has not.

Assumptions and choices:

1. **Model date** is the creation date of the model's Hugging Face repository,
   or of the base model's repository for third-party quantizations (RedHat FP8,
   QAT). Dates are in
   [`experiments/model_release_dates.csv`](../experiments/model_release_dates.csv).
   Training cutoffs come before release, so some tasks coded "before" were in
   fact published after the cutoff. This pushes the estimate toward zero.
2. **Task date** is the first year in the task's `source` citation. This is the
   publication year of the source paper or dataset, not the age of the texts.
3. **Granularity is the calendar year.** Tasks published in a model's release
   year are coded "same year" and estimated separately, because their order is
   unknown.
4. **The 22 API models are excluded**, because no release dates for them are
   recorded in the repository. They can be added once dated sources are
   collected.
5. **Inference** clusters standard errors by task (33 clusters). A task-level
   bootstrap (2,000 resamples of tasks) gives a second interval.

Of the 33 tasks, 21 come from 2024 to 2026 sources, where "before" or "after"
differs across models. The 12 older tasks are "before" for every open model and
contribute only to the event-time bins.

## Result

| Sample | Before vs. after release (F1 points) | 95% CI, clustered | 95% CI, bootstrap |
|---|---|---|---|
| All 33 tasks | −2.9 | −7.2 to +1.5 | −7.4 to +0.9 |
| Without 6 long-public-label tasks | −3.2 | −8.6 to +2.2 | −9.0 to +1.1 |

The "same year" estimate is −1.0 points (−3.5 to +1.6). Splitting "before" into
one year and two or more years gives −2.2 and −1.2 points, both with intervals
covering zero. Nothing in the pattern shows scores rising the longer a task has
been public.

The second row drops six tasks whose gold labels come straight from long-running
public datasets that predate the cited paper: GTD attack types, SemEval-2016
stance, Manifesto Project codes, the two Policy Agendas Project tasks, and CAMEO
gold examples. For these tasks the citation year understates how long the labels
have been public. Dropping them changes nothing.

The raw cell means show where the negative sign comes from. On the 13 tasks
from 2025 sources, the 2024 models average 0.675 F1 and the 2026 models 0.652,
although the 2026 models are not worse on other tasks. That is the opposite of
what exposure would produce.

**Implication.** The plateau and rankings in the November release are unlikely
to be driven by models having memorized the published benchmark sources. The
report's current wording, that exposure cannot be ruled out, remains accurate
for the texts. This check supports adding that no timing pattern consistent
with label exposure is visible.

## Caveats

- **Text exposure is untested.** Many inputs (tweets, manifestos, court
  opinions, news) were public years before the source papers, so every model
  may have seen them. A completion or token-probability probe on the open
  models would test this directly.
- **Capability changes over time can mask exposure.** If newer models improved
  less on the kinds of tasks published recently, that would push the estimate
  down and could hide a small exposure effect. The design cannot separate the
  two.
- **Power is limited.** Only 21 tasks carry the main contrast, and the
  intervals are about ±4 points wide. The check rules out large effects, not
  small ones.
- **Repository creation date** can come before public release if a repository
  was made private first. I did not check these dates against release
  announcements. Year-level coding makes small errors unimportant except for
  models created near a year boundary.

## Rerun on the 46-task release (2026-11-19)

Release `tasks46_20261119`, results in
[`contamination_timing.csv`](../output/sidecar/release_20261119_tasks46/robustness/contamination_timing.csv)
and
[`contamination_timing_cells.csv`](../output/sidecar/release_20261119_tasks46/robustness/contamination_timing_cells.csv).
Command:
`python3 code/analyze_contamination_timing.py --release-dir output/sidecar/release_20261119_tasks46 --tasks-dir tasks_ext_v2_20261114`.

**Verdict.** As in November, the open-weight models do not score better on
tasks published before their release: the estimate is +0.4 F1 points. The
check is much weaker this time. Its 95% interval runs from about −6 to +6
points, so it rules out only a large advantage.

What is compared is the same as above: each of the 13 open-weight models on
each of the 46 tasks, before versus after the model's release year, with model
and task fixed effects. Two release dates were added to
`experiments/model_release_dates.csv` (Gemma 3 12B, 2025-03-01; Qwen3.5 9B,
2026-02-27), both from the Hugging Face repository creation date. Three tasks of
the new pool follow the long-public-label rule and were added to that list: the
UK Policy Agendas Project PMQs topics, the Policy Agendas Project executive
order topics, and the Supreme Court Database issue areas (eight such tasks in
this pool).

| Sample | Before vs. after release (F1 points) | 95% CI, clustered | 95% CI, bootstrap |
|---|---|---|---|
| All 46 tasks | +0.4 | −5.7 to +6.4 | −5.8 to +6.1 |
| Without 8 long-public-label tasks | +2.9 | −3.9 to +9.7 | −4.3 to +9.1 |

The "same year" estimate is +2.0 points (−2.7 to +6.7). The interval is wider
than in November because the contrast rests on fewer early models: nine of the
13 open models were released in 2026, three in 2025 and one (Llama 3.3 70B) in
2024, against 21 models spread over 2024 to 2026 in November. For 33 of the 46
tasks the before/after status differs across models.

**Implication.** The page can say that this check finds no advantage on older
tasks, but not that it rules out moderate exposure effects. Text exposure
remains untested, as above.

## Glossary

- **Headline F1**: the release's per-task score (positive-class F1 for binary
  tasks, macro-F1 over configured labels for categorical tasks).
- **Fixed effects**: a separate intercept for each model and each task, so the
  comparison uses only differences within model and within task.
- **Task-level bootstrap**: redraw the 33 tasks with replacement, refit, and
  take the middle 95% of the estimates.
- **Long-public-label task**: a task whose gold labels are copied directly
  from a dataset that was public before the cited source paper.
