# Hive model bake-off extension, August 2026

## Current state and verdict

Keep Qwen3.6 27B FP8 as the quality-first general Hive default. Mistral Small
4 is 12.76 times faster in generation, but its equal-task mean F1 is 0.0447
lower. GLM-4.7-Flash is 10.72 times faster and loses 0.0603 mean F1. Both
losses are much larger than the frozen 0.005 efficiency tolerance, and both
paired-task intervals exclude zero. Neither model qualifies as a general
replacement for Qwen.

Mistral is the stronger of the two new speed candidates: it is both more
accurate and faster than GLM in this matched run. It is suitable only as a
speed-first option after validation on the intended task or as a low-stakes
first pass whose output will be checked. GLM has no general operational role
in the current model set because Mistral dominates it on the benchmark's
aggregate accuracy and generation throughput.

All three checkpoints completed 34 tasks and 16,425 records with no parse
errors. The automated comparison passed its exact-coverage, paired-label,
checkpoint, revision, runtime, hardware, and timing audits. The two-reviewer
blinded disagreement audit independently selected Qwen as the broad default
and did not identify evidence that changes the automated decision. The live
Hive skill now records the tested GLM and Mistral roles, the native-CUTLASS
Mistral recipe, the corrected offline behavior, and the pinned CUDA 13.0-core
runtime.

## Why this extension was run

The August 4 comparison selected Qwen3.6 27B FP8 as the quality-first Hive
default. GLM-4.7-Flash and Mistral Small 4 are prominent open-weight models
whose checkpoints fit one 98 GiB Blackwell GPU, but neither had a matched
political-science classification result. This extension tested whether either
candidate improves classification accuracy enough to replace Qwen or loses
little enough accuracy to be useful as a faster alternative.

## What is compared

The three pinned checkpoints classify the same 16,425 labeled texts from 34
political-science tasks. For each task-item pair, the models receive the same
logical prompt, input text, task-specific JSON schema, temperature-zero
decoding, disabled thinking or reasoning, and 256-token output limit. Each
model uses its native tokenizer and chat template, so the logical messages
are held fixed but the rendered token sequences can differ across model
families.

| Role | Checkpoint | Revision |
|---|---|---|
| Matched baseline | Qwen/Qwen3.6-27B-FP8 | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| Candidate | zai-org/GLM-4.7-Flash | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` |
| Candidate | mistralai/Mistral-Small-4-119B-2603-NVFP4 | `b1a9048590131d38491bd23a7c9f6ed0962f0358` |

The comparison also holds the runtime lock, one typed 98 GiB Blackwell GPU,
two CPUs, and 96 GiB of host memory fixed. GLM uses the `TritonExperts`
mixture-of-experts backend. Mistral uses `TRITON_MLA` for attention and native
vLLM CUTLASS kernels for its mixture-of-experts and linear layers.

## Structural pilot

The structural pilot asks whether each pinned model can load and return
parseable classifications under the final offline runtime before a full
16,425-record job is allowed to run. It compares the three models on the same
deterministic 16-record slice. It is a runtime and output-format check, not an
accuracy comparison.

| Model | Slurm job | Expected and parsed records | Elapsed time | Backend and error evidence | Peak host memory |
|---|---:|---:|---:|---|---:|
| Qwen3.6 27B FP8 | `20008699` | 16/16 | 3m 52s | No network errors | 45.77 GiB |
| GLM-4.7-Flash | `20008711` | 16/16 | 2m 25s | `TritonExperts`; no network errors | 70.19 GiB |
| Mistral Small 4 NVFP4 | `20008681` | 16/16 | 2m 55s | `VLLM_CUTLASS` and `CutlassNvFp4LinearKernel`; no FlashInfer FP4 or mixture-of-experts JIT and no network errors | 76.39 GiB |

All three models passed the structural gate. Mistral's log confirms that the
configuration selected the intended native kernels rather than merely
finishing through a different FlashInfer fallback. The observed pilot memory
peaks fit inside the 96 GiB request, although only the full jobs can establish
the final timing and memory profile.

## Matched results

Qwen has the highest mean and median task F1 and the highest mean rare-class
recall. Mistral is the fastest model. GLM starts slightly faster than Mistral,
but it is slower during generation and less accurate.

| Model | Mean task F1 | Median task F1 | Rare-class recall | Generation items/s | Startup seconds | Parse-error rate | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B FP8 | 0.6560 | 0.6362 | 0.6791 | 11.73 | 183.2 | 0 | Quality-first default |
| Mistral Small 4 NVFP4 | 0.6113 | 0.5933 | 0.6192 | 149.75 | 120.5 | 0 | Speed-first, task-specific validation required |
| GLM-4.7-Flash | 0.5957 | 0.5852 | 0.5914 | 125.73 | 113.2 | 0 | No general role; dominated by Mistral in this comparison |

The paired task comparison confirms that the speed gains do not meet the
benchmark's quality tolerance. Mistral trails Qwen by 0.0447 mean F1, with a
paired-task 95% interval from -0.0647 to -0.0257. It beats Qwen on 6 of 34
tasks. GLM trails Qwen by 0.0603, with an interval from -0.0815 to -0.0414,
and beats Qwen on 4 tasks. Neither candidate increases the parse-error rate.

Mistral's six task wins are concentrated in Ballard incivility (+0.0723),
SCOTUS sentiment (+0.0537), PolNLI entailment (+0.0372), PolNLI event
entailment (+0.0193), CCC protest detection (+0.0183), and creative-ad
classification (+0.0156). These wins do not generalize: Mistral loses on the
other 28 tasks, including losses above 0.10 on political relevance, attack
type, causal relation, COVID threat minimization, manifesto populism, and one
stance task. GLM's four wins are creative-ad classification (+0.0540),
Women's March stance (+0.0229), SCOTUS sentiment (+0.0168), and Spanish
protest toxicity (+0.0063).

The extension reran Qwen under the matched CUDA 13.0-core lock rather than
reusing the August 4 score. Its mean task F1 moved from 0.6539 to 0.6560 and
its throughput from 11.86 to 11.73 items per second. The two Qwen runs agree
on 98.73% of item predictions; 209 of 16,425 predictions changed. The
extension therefore uses only its matched rerun for contrasts and timing. It
does not combine these timing values with the August 4 comparison, which used
a different dependency lock.

The automated accuracy gate requires a mean task F1 gain of at least 0.010
without increasing the parse-error rate by more than 0.005. The efficiency
gate permits a mean task F1 loss no larger than 0.005 when throughput is at
least 1.25 times Qwen's and the parse-error-rate increase is no larger than
0.005. A passing automated gate remains provisional until the blinded
disagreements are reviewed.

## Blinded disagreement review

Two independent reviewers worked only with randomized aliases. Reviewer A
manually examined a deterministic sample of 102 disagreements, three from
each task. Reviewer B examined a separate deterministic sample of 170, five
from each task. Both also computed alias-only diagnostics over all 4,095 rows
on which at least two models disagreed. They completed and froze their reports
before the alias key was opened.

Both reviewers selected `model_3` as the broad default. Across the full
disagreement set, `model_3` exactly matches the gold object on 2,098 of 4,095
rows (51.2%), compared with 1,695 (41.4%) for `model_1` and 1,421 (34.7%) for
`model_2`. The aliases lead 23, 7, and 4 of the 34 tasks, respectively. The
balanced samples agree: reviewer A records 55/102, 50/102, and 39/102 correct;
reviewer B records 101/170, 75/170, and 62/170.

After unblinding, `model_3` is Qwen, `model_1` is Mistral, and `model_2` is
GLM. This strengthens the general recommendation rather than resolving a
close automated result. Qwen is also safer than unqualified majority voting:
when it is the lone dissenter, it is correct on 42.6% of those disagreement
rows, compared with 38.8% for the other two models' shared answer.

The reviews identify plausible specialists, not routing rules ready for
deployment. Mistral performs well on Ballard incivility, the two PolNLI tasks,
CCC protest coding, SCOTUS sentiment, and Twitcivility. GLM performs well on
Women's March stance, Spanish protest toxicity, and creative-ad tone. These
advantages are conditional on disagreement rows and sometimes reverse in the
small manual samples. Any router must first pass a separate held-out test that
includes agreement rows and preserves the intended task's class distribution.

The qualitative findings explain why no global threshold correction works.
Each model becomes conservative or permissive on different datasets. Qwen
overpredicts positives on some causality and incivility definitions. Mistral
collapses toward neutral or negative labels on several stance tasks. GLM
collapses toward negative or `NONE` on several entailment, relevance, and
stance tasks. Fine-grained policy and event codebooks are the common lower-tail
problem for all three models, and 794 of 4,095 disagreements (19.4%) have no
correct alias.

These disagreement-conditional percentages are not benchmark accuracies.
They deliberately overrepresent hard boundary cases, and several samples
expose plausible gold errors, missing context, or annotation conventions that
cannot be recovered from ordinary semantics alone. The final model ranking
therefore rests on the audited full benchmark; the blinded review checks the
operational meaning of its errors.

The 102-row sample hash is
`64634ef426eff85988ebbe38bb6b71ac5eeabda14b67bab89fec8601bb65c1b6` and
reviewer A's report hash is
`282e6e9bf17ca3c4ef451b0efc069b54126bfed6215a04d36c1f71a5961bf886`.
The 170-row sample hash is
`db46b5e4eb4ede0a40db2d9357e95aad63629738dd489c5f45ba51c9c439d041`
and reviewer B's report hash is
`b5fba0a0f294b2deacff945739265f8fa331748f8aa6a58ba07f8eabc9093bfd`.

## Execution and audit record

The authoritative benchmark commit is
`3c7ad0756d447b1d57ed4daf26bbb85f5f296042`. The final design uses run ID
`hive_model_bakeoff_extension_cuda130_native_cutlass_20260805`.

| Audited artifact | SHA-256 or test result |
|---|---|
| `experiments/hive_model_bakeoff_extension_cuda130_20260805.yaml` | `5e0e0b8d5498b2935a21370912e9f47df482a9ef82bf95c21efcbfbd38938630` |
| Runtime lock | `4f46101fa493e38b312af034b8d740c612bd0001bab4cb0f624f4a9e1c49f5b3` |
| `experiments/hive_model_bakeoff_20260804.sbatch` | `052cc4e660b9aa7ae6e256c2d7f1396e1f898a34ccaea5910c268e03d9b04c7d` |
| `code/hive_vllm_benchmark.py` | `c5845ec49ba1cbdabcb98ab14bfcd4e1f60bf392f30ddb7a592cb03ad1c109f5` |
| Local test suite | 71/71 passed |
| Installed `hive-llm` skill | `94986e12a425d2930f64ac5acb6ad8396bca08de1d157f7e531a2066e15a7e39`; structural validation and two-scenario forward test passed |

The matched runtime uses Python 3.12, vLLM 0.26.0, PyTorch 2.11.0+cu130,
Transformers 5.14.1, FlashInfer 0.6.14, and the runtime-lock hash shown above.
Its CUDA compiler, header, and runtime core pins toolkit 13.0.2 with cccl
13.0.85; crt, nvcc, nvrtc, and nvvm 13.0.88; and runtime 13.0.96. Auxiliary
CUDA Python packages remain at 13.3, so this is a pinned CUDA 13.0 core rather
than an entirely CUDA 13.0 environment.

Every accepted full run must contain 34 task checkpoints and 16,425 unique
task-item keys with identical paired gold labels and no duplicate keys. The
task, checkpoint, merged-prediction, config, model-revision, runner, wrapper,
and runtime-lock hashes must match the registered run. The timing audit must
also show that each task's reported generation time equals the sum of its
prediction-row latencies. No full-run result is accepted in this report until
those checks pass.

All three full runs passed those checks:

| Model | Slurm job | Elapsed | Parsed rows | Generation seconds | Peak GPU memory | Peak host memory | Prediction SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B FP8 | `20008847` | 27m 14s | 16,425/16,425 | 1,399.8 | 89,578 MiB | 13.60 GiB | `a622314e6dba136fc66700a01e8e4c2962637d545e53e7fc788d32a593daa36b` |
| GLM-4.7-Flash | `20008862` | 4m 53s | 16,425/16,425 | 130.6 | 92,844 MiB | 37.51 GiB | `992cf230ff98facb42c6b546df78ab6d10cb5687e729be5242ebed885136ee23` |
| Mistral Small 4 NVFP4 | `20008875` | 4m 26s | 16,425/16,425 | 109.7 | 92,550 MiB | 61.47 GiB | `b79e6e217ec9d9128990eb75b889d786b784686a9080ecca5c68bce38e978add` |

The one-shot comparison scorer wrote eight files after the cross-model audit.
The model summary hash is
`a23d6c717babd80106fa25ef200da88ca1df29b34ca8c4b4661c23702ad99002`;
the paired-contrast hash is
`7e4541f7f9d3a7bc57f410ad6eb1519af91f01aa2ac799def2c21c83089bc6e2`;
the blinded-disagreement hash is
`5efb4d5df4bd0ddcbbc9bd1eea254ebe451acc3391397f85febb5fb275933864`;
and the separate blinding-key hash is
`e6f3ab0581143c12f058448e1dff2d21ad967e7c8b71a9eee09b128e72760706`.

An earlier Mistral attempt used FlashInfer's job-local cold JIT path. After 33
minutes, it had compiled only 27 of 90 `fused_moe` objects, so that route was
abandoned. The final configuration explicitly selects native
`VLLM_CUTLASS` and `CutlassNvFp4LinearKernel` kernels. The operational lesson
is to avoid job-local FlashInfer FP4 and mixture-of-experts compilation for
this model under Hive's short GPU allocations.

The final pilot initially made one avoidable Hugging Face network check before
using its cached files. The shared wrapper now exports `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` whenever `OFFLINE_INSTALL=1`. The exact final pilots
and all full runs then completed with zero network-error lines. A preliminary
two-model scoring check initially failed closed because the frozen three-model
configuration correctly required the absent Mistral result. The check was
rerun with an in-memory two-model diagnostic configuration; the official
scorer was run only once, after all three accepted full results existed.

## Assumptions and limitations

1. The 34-task release sample is deterministic and paired across models, but
   it is not class-stratified. Results for classes with few labeled examples
   remain descriptive.
2. The structural pilot establishes model loading, offline execution, and
   parseable output on 16 records. The accepted full runs, not the pilot,
   establish comparative accuracy and throughput.
3. The benchmark measures prompt-based classification on these 34 tasks. It
   does not establish performance for long-document extraction, tool use, or
   another project's codebook.
4. Throughput ratios are valid only within this matched three-model extension.
   The original four-model comparison used a different runtime lock.
5. Generation throughput excludes startup. Startup and kernel compilation
   times are reported separately and are not treated as clean model-speed
   contrasts.
6. Native tokenizers and chat templates differ even though the logical
   prompts and decoding constraints are held fixed.
7. The full jobs remained within the 96 GiB host-memory request and one 98 GiB
   Blackwell GPU. Mistral used 61.47 GiB of host memory and all three models
   used at least 89,578 MiB of GPU memory, so smaller-GPU fit must be measured
   directly rather than inferred.
8. Temperature-zero decoding did not make outputs bit-identical across the two
   Qwen runtime stacks. The 98.73% prediction agreement is high, but exact
   reproducibility still requires pinning and recording the full runtime.

## Outputs

Pilot artifacts are stored under the ignored directory
`output/sidecar/hive_model_bakeoff_extension_cuda130_native_cutlass_20260805_pilot_final/`.
Accepted full-run artifacts are stored under
`output/sidecar/hive_model_bakeoff_extension_cuda130_native_cutlass_20260805/`.
Each model directory contains per-task checkpoints, merged predictions, and
runtime metadata. The audited comparison directory contains
task metrics, model summaries, paired contrasts, class-support counts,
per-class recall, and the frozen blinded-disagreement artifacts. Its
`blind_review/` subdirectory retains both deterministic samples, their
manifest, and the two pre-unblinding reviewer reports.

## Glossary

- **F1:** The harmonic mean of precision and recall. Mean task F1 gives each
  benchmark task equal weight.
- **FP8 / NVFP4:** Eight-bit and NVIDIA four-bit floating-point checkpoint
  formats used to reduce GPU memory and computation.
- **JIT:** Just-in-time compilation of a kernel when a job starts.
- **Native CUTLASS:** Precompiled vLLM CUDA kernels selected directly through
  the runtime configuration, without FlashInfer compiling equivalent kernels
  inside the job.
- **Parse error:** A model response that cannot be converted into the task's
  required structured output.
- **Structural pilot:** A small run that verifies loading, inference, and
  output structure before the full benchmark is launched.
