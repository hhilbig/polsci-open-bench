# Draft Twitter thread: open models on social-science coding

## Current verdict

The strongest supported hook is not that smaller models caught up. That result
is already well known on general benchmarks. The useful result is that a
convincing within-benchmark explanation failed an outcome-blind external test.
A 2026 27B model matched the 2024 70B average, but the apparent concentration
of gains in event coding did not generalize.

## Post 1

Have open models actually improved at the text-coding tasks social scientists
use?

I tested 21 qualified immutable checkpoints from February 2024 to June 2026. Each ran on
one 98 GB-class GPU and coded the same 3,600 texts from 18 political-science
tasks.

## Post 2

[Attach `twitter_claims/05_compute_class_staircase.png`.]

Large models established the early tested frontier at 0.679 mean F1.
Medium-compute models caught it in April 2026, reaching 0.680. Low-active
mixture-of-experts models improved after first entering the tested roster but
remained lower at 0.657.

The lines show the best tested score available by each date within three
active-parameter classes. “Active parameters” means the parameters used for
one token, not all weights stored in memory.

This is a deployment comparison, not an architecture experiment. Every
low-active model in the roster is sparse, while the medium and large models are
dense, so active-parameter class and architecture cannot be separated.

## Post 3

[Attach `twitter_claims/02_capability_reallocation.png`.]

Inside the repository, progress looked concentrated. Qwen3.6 improved on all
seven tasks asking “what happened?” (attack, protest, or event type), averaging
+4.3 points. The pattern appeared separately in the frozen 18-task subset and
the 16 tasks excluded beforehand.

But this category was discovered after examining the results. It required an
external test before it could support a general capability claim.

## Post 4

[Attach `event_progress/external_event_confirmation.png`.]

The external test reversed the result. Across four tasks chosen before running
the models, Qwen3.6 averaged 4.4 points below Llama 3.1. It lost on both Arabic
event-presence tasks and on MAVEN, and improved only on RAMS event typing.

So “newer models improved event coding” is not supported as a general claim.
The repository pattern was real but dataset-bound. Language, ontology, and task
format all change in the external set, so this test rejects portability without
identifying which feature caused the reversal.

## Post 5

So the parsimonious verdict is: smaller open models caught the older large-model
average, but neither the aggregate benchmark nor a plausible task-category
story establishes uniform capability progress. Benchmark composition matters
enough to reverse the apparent explanation.

## Post 6

Publication timing also cuts against a simple exact-answer leakage story, but
cannot rule out exposure to older underlying texts.

The all-model roster is non-exhaustive, so I am not calling it the global open
frontier. The historical claim rests on the frozen benchmark, the prespecified
repository holdout, and the same-family dense-model comparison. Sparse versus
dense results remain descriptive.

Methods and task-level results: [LINK]

## Optional robustness reply

[Attach `twitter_claims/04_dense_lineages.png`.]

The compute result is not produced only by mixing model families and sizes.
Among every tested same-developer dense lineage with at least three generations
at roughly fixed size, Qwen 27–32B improves by 4.8 points. Qwen 72B changes by
0.4, and Llama 70B declines by 1.0. This grouping rule was developed after the
initial result and is therefore exploratory.

## Novelty audit

This is a targeted check of adjacent work, not a systematic literature review.

| Possible claim | Status | Use in the post |
|---|---|---|
| Smaller models increasingly match older large models | Well known. The [2025 AI Index](https://hai.stanford.edu/ai-index/2025-ai-index-report/technical-performance) documents a 142-fold reduction in the smallest model exceeding 60% on MMLU. | Context only, not the hook. |
| Top model scores are converging and standard benchmarks saturate | Well known. The [2025](https://hai.stanford.edu/ai-index/2025-ai-index-report/technical-performance) and [2026 AI Index](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance) reports emphasize both patterns. | Do not present the flat average as a general discovery. |
| Models that fit limited hardware approach earlier frontiers | Well known. [Epoch AI](https://epoch.ai/data-insights/consumer-gpu-model-gap) estimates that consumer-GPU models trail the general frontier by roughly 6–12 months. | Our one-GPU staircase is a domain-specific application of this idea. |
| Small or specialized models can classify political text well | Already established. [Political DEBATE](https://www.cambridge.org/core/journals/political-analysis/article/political-debate-efficient-zeroshot-and-fewshot-classifiers-for-political-text/8D0B3E2AAF711F4812E42466DE503A13) shows that much smaller domain-trained models can match large generative models on defined political-text tasks. | Avoid claiming that parameter count is generally unimportant. |
| Historical progress differs systematically across concrete social-science coding decisions | Not supported as a portable claim. The repository event pattern reversed on four outcome-blind external tasks. | Present this as a failed generalization test, not a capability taxonomy. |
| Human disagreement explains the regressions | Not supported by our five human-vote datasets; the relationship is weak and inconsistent. | Do not use as the explanation. |

The strongest short claim is therefore:

> A 2026 27B open model matched the average score of a 2024 70B model on 34
> social-science coding tasks. An apparently clear explanation—all seven event
> tasks improved—then reversed on four outcome-blind external tests. Benchmark
> composition changes not only the score, but the story about what improved.
