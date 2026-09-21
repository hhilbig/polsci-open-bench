# Compact two-year open-versus-API benchmark

This benchmark asks how the best open checkpoint tested on one Hive GPU compares
with the best still-accessible API checkpoint at roughly four-month intervals
from August 2024 through June 2026. It is a compact historical comparison, not a
census of released models.

Public label: **Best tested checkpoints on a frozen eight-task
political-science classification benchmark.**

Required caveat: **Compact and non-exhaustive; not a claim about the global
model frontier.**

## What is compared

The open series contains revision-pinned open-weight checkpoints that complete
on one typed Hive GPU with 97,887 MiB of measured capacity. The API series
contains immutable provider snapshots that remain accessible when requests are
prepared. Each series is the cumulative best score among the checkpoints this
project tested by that date.

The benchmark uses the following eight tasks. Each contributes 500 texts and
one headline F1 score.

| Task | Annotation type | Complexity | Headline score |
|---|---|---:|---|
| `burnham_polnli_entailment` | Claims | Low | Positive-class F1 |
| `dicocco_manifesto_populism` | Claims | Low | Positive-class F1 |
| `cap_crs_policy_topic` | Issues | High | Configured-label macro F1 |
| `erlich_ati_topics` | Issues | High | Mean positive-class F1 |
| `gilardi_relevance` | Relevance | Medium | Positive-class F1 |
| `douglass_icbe_sentence_event_type` | Events | Medium | Configured-label macro F1 |
| `halterman_keith_bfrs` | Events | Medium | Configured-label macro F1 |
| `chae_semeval_stance` | Position | Low | Configured-label macro F1 |

The frozen manifest is
[`experiments/frontier_compact_8.yaml`](../experiments/frontier_compact_8.yaml).
It records the benchmark commit, task counts, prompts, schemas, item keys, gold
labels, rendered inputs, scorer version, and component hashes. The manifest has
exactly 4,000 unique task/item keys. Runners stop on any identity, count,
fingerprint, prompt, schema, gold-label, or duplicate-key mismatch.

## Score and uncertainty

The reported score is the arithmetic mean of the eight task-specific headline
F1 scores. Every task therefore has weight one eighth, regardless of class
balance or label count. The result describes these eight tasks only. It is not
calibrated or extrapolated to the former 34-task suite.

Malformed responses remain in the data and are scored as wrong for every
decision in the row. A checkpoint with exactly 5 percent malformed output
passes the reliability gate; a checkpoint above 5 percent is ineligible. The
runner does not selectively retry malformed items.

Percentile 95 percent intervals use 10,000 draws and seed `20260820`. Each draw
resamples items with replacement independently within every task, recomputes
the eight F1 scores, and averages them. The same item draws are used across
checkpoints, which preserves paired comparisons. These intervals represent
item-level uncertainty conditional on the eight fixed tasks. They do not
represent uncertainty about which political-science tasks could have been
selected.

## Checkpoint roster and dates

The open roster contains seventeen artifacts. The original roughly quarterly
roster contains Llama 3.1 70B Instruct FP8, Llama 3.3 70B Instruct FP8, Qwen3
30B-A3B, Qwen3-Next 80B-A3B Instruct FP8, GLM-4.7 Flash, Qwen3.6 27B FP8, and
Gemma 4 31B QAT. Four additional checkpoints reduce the chance that one
preselected model stands in for an entire period: Qwen2.5 32B Instruct,
DeepSeek R1 Distill Qwen 32B, Qwen3 32B, and Qwen3.5 35B-A3B. [Epoch AI's
historical consumer-hardware analysis](https://epoch.ai/data-insights/consumer-gpu-model-gap)
identifies the first three as leading or near-leading models in their periods.
[Artificial Analysis](https://artificialanalysis.ai/models/qwen3-5-35b-a3b-non-reasoning/)
identifies the non-reasoning Qwen3.5 checkpoint as a leading comparable open
model. These external sources determine candidacy, not performance on this
benchmark.

The graph uses the exact runnable artifact's publication date, not the base
model announcement date. Every open row must have a full Hugging Face revision.
The four selection sources and revisions are recorded in the checkpoint
registry. A second targeted expansion adds six models chosen to test the most
plausible remaining explanation for the flat open staircase: the roster may
have omitted larger, size-matched historical contenders. Those additions are
Qwen2.5 72B Instruct FP8, DeepSeek R1 Distill Llama 70B FP8, Mistral Small 3.1
24B, Gemma 3 27B FP8, GPT-OSS 120B MXFP4, and Mistral Small 4 119B NVFP4.
Gemma uses the public Red Hat AI FP8 conversion because the official Google
artifact is gated on Hive. Mistral Small 4 reuses archived full-suite evidence
only after the compact-row identity audit passes. GPT-OSS uses the vLLM
Responses/Harmony interface because its offline chat output does not separate
reasoning from the final answer reliably.

The API roster contains `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`,
`gpt-4.1-2025-04-14`, `gpt-5-2025-08-07`,
`gpt-5.2-2025-12-11`, `gpt-5.4-2026-03-05`, and
`claude-sonnet-5`. The graph uses the immutable snapshot's publication date.
If a historical snapshot is retired or cannot be called, its row remains in
the tested roster as unavailable and the graph leaves a gap. The benchmark
does not substitute a newer model.

The checkpoint registry is
[`experiments/frontier_compact_checkpoints_2026.yaml`](../experiments/frontier_compact_checkpoints_2026.yaml).
It is the complete tested roster for this comparison, not a claim that no
other relevant checkpoint existed.

## Execution and provenance

Every checkpoint starts with the first 16 logical requests in manifest order.
If the parser and, for open models, model-fit gates pass, the runner processes
requests 17 through 4,000. The two stages are combined into exactly 4,000 rows;
the first 16 are not run again.

Open runs use one typed Hive Blackwell GPU, vLLM 0.26.0, the recorded CUDA 13
runtime lock, an immutable model revision, and offline inference after the
model download. An out-of-memory event, measured use above 97,887 MiB, revision
mismatch, incomplete coverage, or pilot malformed rate above 5 percent makes
the checkpoint ineligible. A failed artifact is recorded without changing its
quantization.

Archived 18-task or 34-task predictions may supply a compact score only when
the selected 4,000 rows pass the same prompt, schema, item-key, gold-label,
model-identity, and file-hash checks. Reuse does not import the archived
calibration or promotion rules. New legacy-format runs are prohibited.

API requests use structured output and a 128-token output cap. GPT-5 uses
minimal reasoning; GPT-5.2 and GPT-5.4 use no reasoning; Claude thinking is
disabled. Every item remains a separate logical request. OpenAI input counts
come from the pinned local tokenizer over the exact serialized request.
Anthropic input counts come from its official token-counting endpoint and are
cached with request and response provenance.

No paid request is authorized by preparing the benchmark. Submission requires
an exact aggregate report whose hashes match the request files and approval
record. The report must include all pilots and have a maximum cost at or below
$50. If the initial roster exceeds $50, preparation drops GPT-5.4 first and the
November 2024 GPT-4o snapshot second. If the remaining exact maximum still
exceeds $50, submission stops.

## Outputs

Compact artifacts live under
[`output/sidecar/frontier_2026/compact8/`](../output/sidecar/frontier_2026/compact8/).
The authoritative score files are `model_by_task.csv` and
`checkpoint_summary.csv`. Matching `frontier_staircase.png` and
`frontier_staircase.pdf` files show cumulative best scores. The API preparation
stage writes hash-pinned JSON and Markdown cost-approval reports in the same
directory. Archived 18-task and 34-task artifacts remain outside the compact
subdirectory and do not define compact results.

## Assumptions and limits

1. The eight selected tasks define the estimand. Their genres and label
   structures are varied, but they do not represent a probability sample of
   political-science classification tasks.
2. Equal task weighting treats every task as equally important.
3. Item-bootstrap intervals condition on the fixed tasks and prompts.
4. The open series describes one 97,887 MiB Hive environment. It makes no
   laptop-fit or broader deployment claim.
5. Missing and retired checkpoints remain missing. The plotted staircase is
   the best among tested, eligible checkpoints only.
