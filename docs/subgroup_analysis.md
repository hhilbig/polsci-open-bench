# Do the rankings change across subgroups? (release tasks46_20261119)

Run 2026-11-20. Script:
[`experiments/task_ext_20261114/subgroup_analysis.py`](../experiments/task_ext_20261114/subgroup_analysis.py).
Coding: [`task_subgroups.csv`](../experiments/task_ext_20261114/task_subgroups.csv).
Results: [`robustness/subgroups/`](../output/sidecar/release_20261119_tasks46/robustness/subgroups/).

**Verdict.** The overall ranking holds for English tasks, for each label
structure and for long versus short texts. It changes on the 10 tasks that are
not English only. There, Gemma 4 31B ranks first and Claude Haiku 5.5 second,
while GPT-6.1 Sol falls from 3rd to 20th and Claude Opus 5.5 from 2nd to 11th.
Part of this shift is a label-structure pattern: 6 of the 10 tasks are yes/no
tasks, where the reasoning models do relatively poorly anyway. Net of label
structure, the language interaction is weaker and no longer clearly separated
from chance (p = 0.12). Text length matters for every model by about the same
amount: long texts are about 1 F1 point harder for all of them.

## Why

Readers ask whether the best model overall is also the best model for their
kind of text, most often for non-English text. The main page answers this only
indirectly, through the per-task figure. This analysis checks whether the
ranking changes for a few groups of tasks or texts that I fixed before looking
at the results.

## What is compared, and why only relative performance

Within each group, every model codes the same texts against the same gold
labels, so the comparison of two models within a group holds the tasks fixed.
Comparing the levels of F1 across groups would not be credible: the
non-English tasks differ from the English ones in their codebooks, number of
classes and gold-label quality, so a lower mean F1 there says nothing about
language. The analysis therefore asks only whether a model's advantage over the
other models changes from one group to another.

The test statistic measures that change. I remove each model's overall average
and each task's average from the model × task F1 table, then average what is
left over the tasks of each group. If every model's advantage were the same in
every group, these group averages would all be zero. The statistic sums their
squares, weighted by group size. Its p-value comes from randomization
inference: reassign the group labels to tasks at random 5,000 times, keeping
the group sizes, and count how often the statistic is at least as large as
observed.

Within each group, each model's "advantage over the field" is its mean F1 minus
the mean of all models in that group. Its 95% interval comes from resampling the
tasks of the group (2,000 draws).

## The three splits

1. **Language** (task level): 36 English tasks vs 10 tasks that are not
   English only. Eight are non-English (Hebrew, Japanese, Czech, Spanish ×2,
   German, Estonian, German and Italian). Two are mixed: the Global Populism
   Database speeches, in many languages, and politicians' tweets from the US,
   Germany, Italy and Chile. I coded language by hand from the task sources and
   sample texts.
2. **Label structure** (task level): yes/no decisions (14 binary and 4
   multi-label tasks, 18 in all), 3 to 5 classes (16 tasks), 7 or more classes
   (12 tasks, up to 49).
3. **Text length within task** (text level, open-weight models only). Each
   task's 500 texts in the full pool are split at the task's median length. The
   long and short texts of a task share the codebook and the gold-label process,
   so this is the cleanest of the three comparisons. Length is the prompt token
   count under the Qwen3.6 27B tokenizer; the prompt template is the same for all
   texts of a task. The API models coded only 50 to 100 texts per task, too few
   to split, so they are not in this comparison. The p-value comes from random
   halves of the same size within each task, shared by all models.

Assumptions: (i) the group codings are as listed in `task_subgroups.csv`;
(ii) every task counts equally, as in the headline mean; (iii) the tests treat
tasks (or, for length, texts within tasks) as exchangeable under the null.

## Results

| Split | Does the ranking change? | p-value | Rank correlation with overall ranking |
|---|---|---|---|
| Language | Yes, among the 10 tasks that are not English only | 0.037 | English 0.98; not English only 0.58 |
| Language, net of label structure | Weaker | 0.12 | |
| Label structure | Somewhat, at the extremes | 0.081 | yes/no 0.76; 3–5 classes 0.93; 7+ classes 0.94 |
| Length within task (13 open models) | Gaps differ slightly; ranking does not | 0.007 | short vs long 0.97 |

**Language.** On the 10 tasks that are not English only, Gemma 4 31B is 5.7
points above the field (interval +1.9 to +9.9), and Claude Sonnet 5, GLM-5.3 and
Claude Opus 5 are also clearly above it. GPT-6.1 Sol and Claude Opus 5.5, both
among the top three overall, are at the field average. The pattern does not
depend on a single task: when any one of the 10 tasks is dropped, Gemma 4 stays
1st or 2nd and GPT-6.1 Sol stays between 15th and 22nd
([`language_leave_one_out.csv`](../output/sidecar/release_20261119_tasks46/robustness/subgroups/language_leave_one_out.csv)).
The German and Italian manifesto populism task contributes most: there GPT-6.1
Sol and Claude Opus 5.5 are about 25 points below their usual standing. With 10
tasks, the within-group mean F1 intervals are wide (±0.09 to ±0.15), so the
ranking within this group is much less certain than the overall one.

**Label structure.** On tasks with 7 or more classes, the models that reason
before answering (Claude Opus 5.5, Gemini 4 Argon, GPT-6.1 Sol) and Gemini 3.8
Flash lead the field by 6.6 to 7.9 points. On yes/no tasks their lead
disappears, and Gemma 4 31B and GLM-5.3 rank first and second. The small open
models (Gemma 3 12B, Qwen3 30B-A3B 2507, GLM-4.7-Flash, Mistral Small 4) fall
furthest behind on the many-class tasks.

**Length.** Long texts are 1.2 points harder on average for the 13 open models.
Qwen3.6 27B (+0.9) and Gemma 4 31B (+0.2) do not lose anything on long texts,
while Qwen3.5 9B loses 2.9 points. The differences between models are larger
than random halves produce, but they span only about 4 points (−2.9 to +0.9)
and the ranking barely moves (the first three models are the same for short and for long texts).

## Implication

For English text, and for any label structure, the overall ranking is a good
guide. For non-English text the ranking is a weak guide: models that lead
overall because of their strength on many-class English tasks do not lead
here, and Gemma 4 31B is at or near the top. With 10 tasks this is suggestive,
not settled, and some of it is the larger share of yes/no tasks in that group.
A researcher coding non-English text should validate two or three candidates
on their own data, not take the overall leader. Text length is not a reason to
choose a different model.

## Caveats

- **Few tasks per group.** Ten tasks that are not English only and 12 with 7 or
  more classes. The tests can detect large reorderings, not small ones.
- **Groups overlap.** Language and label structure are correlated (6 of the 10
  tasks that are not English only are yes/no tasks). The stratified test
  separates them only roughly.
- **Languages are pooled.** Eight languages in 10 tasks; nothing here says
  anything about a single language.
- **Several comparisons.** Three splits and 25 models per group. Read
  individual models' intervals with that in mind; the permutation p-values are
  per split.
- **Length uses open models only**, and their full pool rather than the
  scored texts on the page.

## Glossary

- **Headline F1**: the per-task score on the page (positive-class F1 for binary
  tasks, mean over labels for multi-label tasks, macro F1 over the classes
  present in the gold labels for categorical tasks).
- **Advantage over the field**: a model's mean F1 in a group minus the mean of
  all models in that group.
- **Randomization inference**: reassign the group labels at random many times
  and ask how unusual the observed statistic is among the reassigned ones.
- **Spearman rank correlation**: the correlation between two rankings; 1 means
  the same order.
