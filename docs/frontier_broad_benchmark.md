# Broad 18-task open-model frontier check

## Current verdict

This analysis asks whether the flat open-model staircase from the smaller task
sets survives when the benchmark covers more of the repository's political
science coding tasks. It compares revision-pinned open checkpoints released
from February 2024 through June 2026 on the same 18 tasks and reports each
checkpoint's mean F1.

The result has two periods. Mean F1 rises from 0.638 for Qwen1.5 72B in
February 2024 to 0.676 for Llama 3 70B in June 2024. Average progress then
nearly stops: Qwen3.6 27B reaches 0.680 in April 2026. Its equal-task gain is
0.005 F1 against Llama 3.1 and
0.015 against Llama 3.3, a second 70B Llama baseline from December 2024.
Changing how the 18 tasks are grouped and weighted raises the point estimates,
but does not change their scale:

| Aggregation | Qwen3.6 minus Llama 3.1 | Qwen3.6 minus Llama 3.3 |
|---|---:|---:|
| Equal task | 0.005 | 0.015 |
| Equal annotation family | 0.007 | 0.018 |
| Equal data-source family | 0.013 | 0.027 |
| Equal complexity stratum | 0.018 | 0.028 |

These are small-to-modest point estimates. Uncertainty about which tasks or
data sources receive weight still permits larger gains in either direction.
Against Llama 3.1, the 95 percent paired-item interval is -0.015 to 0.023 and
the task-bootstrap interval is -0.031 to 0.040. The source-family bootstrap
interval is -0.017 to 0.046. Against Llama 3.3, the corresponding source-family
interval is -0.001 to 0.054. Source-family uncertainty therefore spans zero for
both baselines.

The average masks a change in the mix of capabilities. Relative to Llama 3.1,
Qwen3.6 improves on 11 of 18 tasks and in four of five annotation families. Its
mean gain is 0.056 across the three high-complexity tasks, while its mean score
falls by 0.019 across the nine low-complexity tasks. It gains on issue,
relevance, event, and position coding but declines on claim coding. A newer 27B
model matching a 70B model is also substantial parameter-count efficiency progress,
even if the average F1 gain is small.

The category pattern becomes clearer when the comparison uses all 34 audited
tasks rather than only the frozen 18-task sample. Qwen3.6 improves all seven
event-and-protest tasks, for an average gain of 4.3 F1 points. It improves five
of six policy-and-topic tasks, for an average gain of 2.0 points. Those gains
are offset by average declines of 2.3 points on eight relevance-and-tone tasks,
1.0 point on seven claims-and-relations tasks, and 0.6 points on six
stance-and-sentiment tasks. This broader comparison supports differential
capability change. It does not show that task difficulty or human ambiguity
causes the differences.

Human disagreement is therefore not a main-result graphic. The five datasets
with retained original annotation votes do not show a consistent relationship
between annotator agreement and model improvement. That analysis remains a
useful failed explanation, but it cannot account for the category regressions.

The evidence does not support saying that open models made no progress. A
same-vendor, approximately size-matched comparison shows substantial progress:
Qwen1.5 32B scores 0.632 in April 2024, while Qwen3.6 27B scores 0.680 in April
2026, a gain of 4.8 F1 points with fewer nominal parameters. The Llama 70B
series is flat or declining from Llama 3 through Llama 3.3. The supported
conclusion therefore distinguishes performance at a fixed smaller scale from
the best score observed in the one-GPU deployment envelope:

> Smaller open models improved substantially from 2024 to 2026, but the best
> average score in this tested one-GPU roster moved little beyond the level a
> 70B model reached in June 2024. Newer models improved topic and event coding,
> while regressions elsewhere kept the overall deployment frontier nearly flat.

## What is compared

Each checkpoint classifies the same 3,600 texts: 200 texts from each of 18
frozen tasks. The tasks cover claims, issue topics, relevance, events, and
political positions. The benchmark also varies task complexity: nine tasks are
classified as low complexity, six as medium complexity, and three as high
complexity. Every task contributes one headline F1 score. The headline result
gives all 18 tasks equal weight.

The aggregation check asks whether that choice drives the conclusion. It first
averages task differences within each group and then weights groups equally.
The alternatives give equal weight to five annotation families, 13
data-source families, or three frozen complexity strata. A data-source family
groups tasks built from the same study or underlying source so that related
tasks do not receive multiple separate weights.

The frozen manifest is
[`experiments/frontier_broad_18.yaml`](../experiments/frontier_broad_18.yaml).
It records the exact task and item keys, prompts, output schemas, gold labels,
source and sample fingerprints, scorer version, and benchmark commit. The
manifest selects 200 items per task by ranking a SHA-256 hash of the task name,
item key, and fixed seed `20260821`. This selection is deterministic and does
not use labels or model predictions.

The open roster is
[`experiments/frontier_broad_checkpoints_2026.yaml`](../experiments/frontier_broad_checkpoints_2026.yaml).
It contains 22 dated open checkpoints, of which 21 have eligible results. The
roster is intentionally non-exhaustive:
it includes roughly quarterly models plus additional historically relevant
contenders that fit on one typed 97,887 MiB Hive GPU. Artifact publication
dates determine horizontal placement. A model that fails provenance,
capacity, coverage, or reliability checks remains in the roster but cannot set
the staircase.

The complete roster is descriptive rather than a census or an externally
ranked frontier. The size-fairness check therefore uses two rules that do not
depend on scores from this benchmark. It includes every tested official dense
Qwen chat or instruct generation with 27 to 32 billion parameters (Qwen1.5,
Qwen2.5, Qwen3, and Qwen3.6), and every tested official dense Llama generation
with 70 billion parameters (Llama 3, 3.1, and 3.3). These within-family series
hold the provider and nominal size roughly fixed. They do not establish that
the broader tested roster contains every historically relevant model.

## Score and uncertainty

Binary tasks use positive-class F1, categorical tasks use macro F1 over the
configured labels, and multi-binary tasks average positive-class F1 across
their decisions. Malformed responses remain in the denominator and are wrong
for every scored decision in that row. Exactly 5 percent malformed passes; a
higher rate makes the checkpoint ineligible. The runner never retries an item
because its output was malformed.

The main 95 percent interval uses 10,000 paired bootstrap draws with seed
`20260820`. Each draw resamples items within every task, recomputes all task
scores, and then averages the 18 task scores. These intervals describe
item-level uncertainty conditional on the fixed tasks.

Three sensitivity checks address dependence on benchmark composition. A task
bootstrap resamples the 18 observed task-level differences. A source-family
bootstrap resamples the 13 family-level differences 10,000 times with seed
`20260820`. A leave-one-task-out check recomputes each model difference 18
times, omitting one task at a time. These checks do not turn the tasks or
sources into a probability sample of all possible political-science coding
problems. They show how much the result depends on the selected mix.

## Heterogeneity and possible explanations

The frozen complexity labels give the clearest observed pattern: Qwen3.6 does
better than Llama 3.1 on the higher-complexity tasks and worse on the
low-complexity tasks. The annotation families point in the same direction for
topic and event coding, but the family labels and complexity labels overlap.
These comparisons therefore describe where scores changed; they do not identify
why they changed.

Publication timing weakens a simple claim that Llama 3.1 scores well because it
memorized every exact published benchmark source. Ten of the 18 task entries
come from sources published in 2025 or 2026, after Llama 3.1's August 2024
artifact; the other eight come from earlier sources. Qwen3.6 trails Llama 3.1
by 0.007 F1 on the later-published tasks but leads by 0.020 on the earlier
tasks. Against Llama 3.3, the corresponding differences are -0.003 and 0.039.
The split therefore does not support exact exposure to the later-published
benchmark sources as the explanation for Llama's strong average. This is only
a plausibility check: source publication year is not data vintage. The
underlying texts or datasets may be much older, and related versions may have
appeared in training data.

Task selection also does not explain the flat average in the two anchor models.
Qwen3.6 leads Llama 3.1 by 0.54 F1 points on the selected Broad18 tasks and by
0.29 points on the 16 tasks excluded before either model was scored. Across all
34 repository tasks, the equal-task difference is 0.42 points. On four
out-of-database tasks with separately audited inputs, Qwen3.6 trails by 2.57
points on average. These checks show that the conclusion is not unique to the
chosen 18 tasks. They do not make the repository tasks a probability sample of
all social-science coding work.

| Evidentiary status | Explanation | What the benchmark establishes |
|---|---|---|
| Supported | The capability mix changed | Qwen3.6 gains on 11 tasks and four annotation families, with its largest average gain in the high-complexity stratum. |
| Supported | Parameter efficiency improved | A 27B checkpoint matches the average performance of a 70B checkpoint on the tested tasks. |
| Plausible but unresolved | Prompt compatibility or training-data exposure | The fixed prompt may suit some model families better, and publication dates cannot rule out exposure to older underlying texts. |
| Plausible but unresolved | Model objectives changed | Newer models may devote capacity to reasoning or other behavior that a short structured classification task does not measure. |
| Design limitation | The tested roster is not the global frontier | The benchmark includes only selected immutable checkpoints that fit the one-GPU environment. |

## Execution and evidence reuse

New open runs use one typed Hive Blackwell GPU, vLLM 0.26.0, CUDA 13, an
immutable Hugging Face revision, and offline inference after download. Each run
starts with the first 16 requests in manifest order. A passing pilot is
combined with the remaining 3,584 requests without rerunning the pilot items.
The final evidence must contain exactly 3,600 unique task/item keys.

Existing 18-task or 34-task predictions can be reused only when the builder
matches the sampled keys and independently verifies prompts, schemas, gold
labels, model identity, hardware records, and file hashes. Reuse imports only
the selected predictions, not the old calibration or promotion rules.

The two headline checkpoints do not have identical execution histories. Llama
3.1 ran directly on the Broad18 manifest with seed `20260820`. Qwen3.6 reuses
selected rows from an earlier 34-task run with seed `20260804`. For Qwen3.6,
the builder audited the selected item keys and gold labels and recomputed the
prompt, schema, rendered-input, and configuration fingerprints; its legacy
metadata does not contain the later Broad18 panel fields. Both runs used
temperature zero, a 256-token output cap, structured JSON output, disabled
thinking, the same vLLM 0.26/CUDA 13 runtime lock, and separately verified
immutable model revisions. This supports audit-based comparability, not a claim
that the two execution protocols were identical.

The API roster remains unrun for this broader benchmark. No paid API request
was submitted. Any later API comparison needs a new exact request manifest,
provider-specific token counts, a cost report at or below the approved budget,
and explicit approval.

## Outputs

Authoritative broad outputs live in
[`output/sidecar/frontier_2026/broad18/`](../output/sidecar/frontier_2026/broad18/).
`checkpoint_summary.csv` contains checkpoint scores and eligibility;
`model_by_task.csv` contains the 18 component scores; and
`task_sensitivity.csv` contains paired-item, task-bootstrap, and leave-one-task-out
comparisons. Matching PNG and PDF files show the cumulative-best
staircase.

The stress-test outputs are
[`robustness_summary.csv`](../output/sidecar/frontier_2026/broad18/robustness_summary.csv),
which contains both baselines and all four aggregation rules, and
[`source_heterogeneity.csv`](../output/sidecar/frontier_2026/broad18/source_heterogeneity.csv),
which contains source, annotation-family, complexity, and publication-period
comparisons. The same diagnostic figure is available as
[`robustness_diagnostics.png`](../output/sidecar/frontier_2026/broad18/robustness_diagnostics.png)
and
[`robustness_diagnostics.pdf`](../output/sidecar/frontier_2026/broad18/robustness_diagnostics.pdf).

The all-task category audit is generated by
[`build_full34_category_figure.py`](../code/build_full34_category_figure.py).
Its task-level assignments and scores are in
[`full34_task_categories.csv`](../output/sidecar/frontier_2026/twitter_figures/full34_task_categories.csv),
and the five category means are in
[`full34_category_summary.csv`](../output/sidecar/frontier_2026/twitter_figures/full34_category_summary.csv).
The matching
[`PNG`](../output/sidecar/frontier_2026/twitter_figures/02_full34_categories.png)
and
[`PDF`](../output/sidecar/frontier_2026/twitter_figures/02_full34_categories.pdf)
use all 34 tasks with equal task weight within each analysis-defined category.

The three validity checks are generated by
[`build_validity_checks.py`](../code/build_validity_checks.py). The authoritative
CSVs record
[`task-set sensitivity`](../output/sidecar/frontier_2026/validity_checks/task_set_sensitivity.csv),
[`source-timing evidence`](../output/sidecar/frontier_2026/validity_checks/leakage_timing_check.csv),
and the
[`matched-family roster`](../output/sidecar/frontier_2026/validity_checks/matched_family_roster.csv).
The compact diagnostic is available as
[`PNG`](../output/sidecar/frontier_2026/validity_checks/validity_checks.png) and
[`PDF`](../output/sidecar/frontier_2026/validity_checks/validity_checks.pdf).

## Assumptions and limits

1. The 18 selected tasks define the estimand. They are broader than the four-
   and eight-task versions but are not a random sample of research tasks.
2. Equal task weighting is the headline estimand. Equal annotation-family,
   source-family, and complexity weighting are sensitivity checks rather than
   replacements for it.
3. The 13 source families reduce repeated weighting of closely related tasks;
   the analysis does not assume that the families are statistically
   independent draws from a population.
4. Results depend on the frozen prompts, schemas, scoring definitions, and the
   tested checkpoint roster.
5. The open series describes one 97,887 MiB Hive environment; it does not make
   a laptop-fit claim.
6. A flat cumulative-best line can coexist with gains on some tasks, losses on
   others, and improved performance per parameter. The component and
   sensitivity files are necessary for judging those distinctions.

## Glossary

- **Checkpoint:** one immutable model artifact evaluated with fixed settings.
- **Frontier step:** a checkpoint whose score exceeds every earlier eligible
  checkpoint in the tested series.
- **Headline F1:** the task-specific F1 definition recorded in the benchmark.
- **Annotation family:** tasks grouped by what annotators code: claims, issues,
  relevance, events, or positions.
- **Data-source family:** tasks grouped by their common study or underlying
  source before groups receive equal weight.
- **Paired bootstrap:** resampling that uses the same sampled items for both
  models when estimating their score difference.
