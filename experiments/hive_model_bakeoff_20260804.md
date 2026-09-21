# Hive model bake-off, August 2026

## Current state

The four-model comparison is complete. It does not alter the benchmark's
canonical predictions, report, or model ranking. The machine-readable design
is frozen in
[`hive_model_bakeoff_20260804.yaml`](hive_model_bakeoff_20260804.yaml).
The separate 32 GB Gemma fit job, `19998365`, remains queued because Hive's
RTX 5000 Ada node has a job stuck in `COMPLETING`; this does not affect the
common-hardware comparison.

## Result and recommendation

Use Qwen3.6 27B FP8 as the quality-first general Hive default. It has the
highest equal-task mean F1 and was the safest general model in the blinded
disagreement review. Use Gemma 4 31B QAT W4A16 when throughput matters and a
task-specific pilot shows no harmful threshold drift. Keep Qwen3.6 35B A3B as
an intermediate specialist and the old Qwen3 30B A3B model only for
high-throughput first-pass screening.

| Model | Mean task F1 | Rare-class recall | Generation items/s | Role |
|---|---:|---:|---:|---|
| Qwen3.6 27B FP8 | 0.6539 | 0.6770 | 11.86 | Quality-first default |
| Gemma 4 31B QAT W4A16 | 0.6504 | 0.6963 | 24.47 | Efficiency alternative |
| Qwen3.6 35B A3B FP8 | 0.6374 | 0.6342 | 62.40 | Intermediate specialist |
| Qwen3 30B A3B Instruct 2507 FP8 | 0.6193 | 0.6036 | 206.41 | High-throughput legacy baseline |

All three newer models pass the frozen accuracy promotion gate against the
legacy baseline. Qwen3.6 27B leads the baseline by 0.0346 mean F1 (paired-task
95% interval 0.0161 to 0.0537). Gemma leads it by 0.0311 (0.0081 to 0.0520),
and Qwen3.6 35B leads it by 0.0181 (-0.0001 to 0.0366). All four models have a
zero parse-error rate.

The direct Qwen3.6 27B versus Gemma comparison is a statistical tie. Qwen's
mean F1 advantage is 0.0036, with a paired-task 95% interval from -0.0160 to
0.0240. Gemma wins 19 of 34 tasks, has 0.0193 higher mean rare-class recall,
and is 2.06 times faster. Qwen's small mean edge is driven by offsetting task
differences: removing only the COVID threat-minimization task flips the mean
to Gemma by 0.0030. Recomputing categorical F1 over observed gold classes
does not change the near-tie.

The blinded review breaks the operational tie in Qwen's favor for general
use. Two reviewers examined 102 and 150 stratified disagreement cases without
access to model identities. Qwen3.6 27B and Gemma split their head-to-head task
wins 16 to 16 with two ties in the disagreement-only comparison. Qwen had the
stronger lower tail and fewer pathological label-rate shifts. Gemma won more
tasks overall but showed sharp task-specific overprediction on causal relation
and one incivility definition, and underprediction on COVID threat
minimization. Those errors support task-level validation before using Gemma as
the default for a new codebook.

## Execution record

The full jobs were `19998120` (Qwen3 baseline), `19998174` (Qwen3.6 35B),
`19998275` (Qwen3.6 27B), and `19998572` (Gemma). Each completed all 34 tasks
and all 16,425 rows. The final audit verified exact model revisions, task and
item coverage, paired gold labels, task fingerprints, checkpoint and merged
hashes, timing metadata, the dependency-lock hash, and the common 98 GB
Blackwell GPU type. The four runs therefore contain 65,700 paired predictions
with no unusable output.

The verified runtime was Python 3.12.13, vLLM 0.26.0, PyTorch 2.11.0+cu130,
Transformers 5.14.1, FlashInfer 0.6.14, nvcc 13.3.73, and NVIDIA driver
580.167.08. The runtime-lock SHA-256 is
`fdbea52db4657f2f61eb0aef62a3f0d74a730e04e1688babb8ba9d76ce96c0e4`.

## Implementation corrections

Four problems were found and fixed before the promotion decision:

1. The first baseline pilot failed before model loading because the initial
   remote sync omitted `models/`. The directory was synced and the registered
   retry completed 16/16 rows.
2. The first local coverage audit compared numeric and string item IDs with
   different sort orders. IDs are now normalized to strings before checkpoint
   comparison, with a regression test.
3. An independent report audit found that timing metadata was trusted without
   checking prediction-row latencies, cross-model runtime and GPU equality
   were not enforced, blinded column order leaked config order, and tied rarest
   classes used an arbitrary label. The scorer now fails closed on timing or
   hardware drift, uses random ordered aliases, and averages all minimum-support
   ties. Regression tests cover each correction.
4. The first final scoring command passed its audit but could not create the
   comparison directory under the desktop sandbox. The identical command was
   rerun with repository write permission and produced the audited report.

## Why this comparison is being run

The Hive runbook used Qwen3-30B-A3B-Instruct-2507-FP8 because that checkpoint
had passed an operational production run. That run established GPU fit,
throughput, and valid JSON output, but it did not compare classification
accuracy against newer models. This experiment supplies that comparison before
the runbook promotes a replacement.

## What is compared

Four exact model checkpoints classify the same 16,425 labeled texts from the
benchmark's 34 political-science tasks. Every checkpoint receives the existing
task prompt and input text, its native chat template, the same task-specific
JSON schema, temperature-zero decoding, thinking disabled, and the same output
limit.

The checkpoints are:

- Qwen3-30B-A3B-Instruct-2507-FP8, the legacy high-throughput baseline.
- Qwen3.6-35B-A3B-FP8, the intermediate speed/quality candidate.
- Qwen3.6-27B-FP8, the quality-first candidate.
- Gemma 4 31B IT QAT W4A16, the efficiency and small-GPU candidate.

The Hugging Face revisions are pinned in the YAML file. The models use their
own native tokenizers and chat templates, so the logical system and user
messages are identical but the rendered token sequences are necessarily
model-specific.

## Execution and checks

Each model runs as a separate job on one 98 GB Hive Blackwell GPU for the main
comparison. The runner processes one task-sized
batch at a time, writes the task result atomically, and skips only checkpoints
whose scope, input fingerprint, exact item coverage, and output hash match the
frozen task sample. Pilot and full runs use separate sidecar roots. A restart
therefore retains completed tasks without conditioning the retained sample on
whether individual model outputs parsed. The checkout commit, vLLM version,
resolved Python dependency lock, model revision, and generation settings are
recorded and checked before completed tasks can be reused.

The sequence is:

1. Run a small structural pilot for every checkpoint on the same task.
2. Confirm model load, prompt-length preflight, exact row coverage, JSON parsing,
   runtime metadata, and restart behavior.
3. Run all 34 tasks only after the four pilots pass.
4. Require exact paired item and gold-label coverage across all models before
   computing metrics.
5. Review model disagreements under blinded aliases before changing the
   production default.

The comparison reports task-level F1, rare-class recall, unusable-output rate,
throughput, startup time, and observed peak GPU memory. Cross-task model means
give every task equal weight. Paired uncertainty intervals resample the 34
tasks rather than pooling all items.
Rare-class recall identifies the minimum-support class from the full gold
sample, averages all classes tied at that minimum, and counts an unusable
output as a miss rather than silently removing it.

## Promotion rule

A candidate replaces the baseline only if it satisfies one of two rules:

1. Its mean task F1 is at least 0.01 higher without increasing the parse-error
   rate by more than 0.005.
2. Its mean task F1 is no more than 0.005 lower, its throughput is at least
   1.25 times the baseline, and its parse-error rate does not increase by more
   than 0.005.

These thresholds make the automated decision operational. A newer release or
valid JSON alone does not qualify a model for promotion. All three candidates
passed the accuracy gate; the completed blinded disagreement audit selected
Qwen3.6 27B as the general default and retained Gemma as the efficiency option.

## Assumptions and limitations

1. The benchmark's release sample is fixed and paired, but it was drawn
   deterministically rather than stratified by class. Classes with fewer than
   30 examples remain in the analysis with their exact support reported.
2. The benchmark measures prompt-based classification on these 34 tasks. It
   does not establish performance for long-document extraction, tool use, or
   another project's codebook.
3. The main comparison runs every checkpoint on the same GPU type so its
   throughput ratios are interpretable. Queue conditions can still affect
   startup time, which is reported separately.
4. A separate 32 GB Gemma fit test is an operational check, not a second
   accuracy experiment.

## Outputs

Run artifacts remain under the ignored directory
`output/sidecar/hive_model_bakeoff_20260804/`. Each model directory contains
per-task checkpoints, merged predictions, and runtime metadata. The comparison
directory contains audited task metrics, model summaries, paired contrasts,
class-support counts, per-class recall, a blinded disagreement file and its
separate key, and a short results report.
