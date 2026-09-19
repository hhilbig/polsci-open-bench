# Changelog

Notable changes to the benchmark: task set, gold labels, prompts, metrics, and
anything that moves published numbers. Entries are newest first.

Two conventions hold throughout:

- **Published results are not rewritten in place.** When a task definition
  changes, the v1 manifest, prompt and predictions stay as they were and the
  corrected version ships alongside under `tasks_v2/` and `prompts_v2/`. When a
  metric changes, the previous output is preserved next to the new one and a flag
  reproduces it exactly.
- **A change that moves a published number says so, with the number.**

## 2026-09-19

### Added

- **Supervised baselines as a function of training-set size.** The paper
  previously conceded in its limitations that supervised classifiers "may be
  cheaper and more reliable" without testing it. `code/build_supervised_baseline.py`
  now measures the crossover: how many hand-coded labels a classifier needs before
  it matches zero-shot LLM coding.

  Design. Each task's unsampled remainder is labelled training data and the test
  set is the exact 500 items every LLM was scored on, rendered through the same
  `task_registry` template, scored through `code/scoring.py` with the same
  support-only rule and pinned label set. The supervised and LLM columns are
  therefore on one metric over one test set. Two classifiers, both logistic
  regression on identical training draws: TF-IDF features, and frozen
  `intfloat/multilingual-e5-large` embeddings. Six of the tasks are not English,
  which rules out English-only encoders. Training sizes 50 to 2,000 with five draws
  each, over the 30 tasks with a remainder.

  | Hand-coded labels | TF-IDF | E5 embeddings |
  |---:|---:|---:|
  | 50 | 0.327 | 0.447 |
  | 250 | 0.435 | 0.595 |
  | 1,000 | 0.519 | 0.660 |
  | 2,000 | 0.549 | 0.672 |

  Zero-shot reference on the same 30 tasks: 0.697 best local, 0.711 best API.

  So zero-shot coding still wins at 2,000 labels per task. The embedding
  classifier matches or beats the best local model on 10 of 30 tasks at that size
  and on 13 of 30 at some size, with a median crossover near 1,000 labels;
  bag-of-words manages four tasks at any size. Representation dominates the
  classifier: E5 adds 0.14 mean F1 at every training size over the same logistic
  regression, and on the Japanese task TF-IDF reaches 0.098 against 0.736, because
  word-level features cannot segment Japanese.

  The classifier wins where the label follows a corpus-specific convention
  (Japanese policy areas, cross-domain topics, Spanish request topics, GTD attack
  types) and loses on semantic judgment (entailment, stance, incivility). Three of
  the tasks it wins were flagged in the source-fidelity audit for exactly that
  property.

  Reported in `output/report_pdf_v2.qmd` section 3.7 with
  `output/figures/fig-supervised-curve.png`; per-task crossovers in
  `output/supervised_crossover.csv`.

- **Training pool capped at 3,000 rows per task** (`POOL_CAP`). The original design
  embedded all 246,532 frame rows and evaluated a full-remainder point. That point
  answered a question nobody asks, since no one hand-codes 36,838 examples, and it
  was the expensive one: it required a 505 MB embedding cache on a disk with
  118 MB free, and TF-IDF bigram fits over 36,838 documents inside a grid search
  dominated the runtime. Capping cuts the cache to 192 MB, and has the side benefit
  that both classifiers now train on identical draws, so the two curves differ only
  by representation. The cost is that no supervised ceiling at full data is
  reported, which the report states rather than implies.

### Changed

- **`output/report_pdf.qmd` and the PDF are now the revised report.** The 33-task
  set, the corrected metric, the constrained-decoding experiment and the supervised
  baseline are all in the canonical report rather than a parallel v2 file. The
  previous version remains in git history. New prose was revised against
  `hanno-voice`, `clarity-reviewer` and `avoid-ai-writing`: a bootstrap interval
  whose sign convention was unreadable now states its direction in words, a
  paragraph carrying five decimals was split into claim and detail, two rhetorical
  contrasts were replaced with plain statements, and a reference to tasks "flagged
  earlier in this paper or its companion audit" now names which tasks and why.

### Known issues

- Multilingual-E5 truncates at 512 tokens while the LLM sees the whole item, so the
  embedding classifier is understated on long-document tasks. The TF-IDF line has
  no such limit and does not overtake it, which bounds the concern without removing
  it.
- A fine-tuned encoder is untested and would likely sit above the frozen-embedding
  curve. These results locate the crossover for a cheap supervised pipeline, not a
  ceiling for supervised methods.
- `output/embeddings/` is gitignored and regenerable; rebuilding it takes roughly
  an hour on an Apple Silicon GPU.

## 2026-09-18

### Changed

- **`halterman_ccc_protest` is excluded from the benchmark**, pending the source
  pairing file. It remains in `tasks/` with `status: excluded`, so the manifest,
  prompt, data and past predictions are all intact and the decision is reversible
  by deleting that key.

  CCC's `type` field is a free-text, semicolon-separated descriptor, not a
  mutually exclusive category. In the public release (Harvard Dataverse
  `doi:10.7910/DVN/6OPP7H`, 72,181 events) 6.8 percent of events carry several
  types at once, and among the 2,883 whose type includes "march", 1,546
  (54 percent) list `protest` or `rally` first. Collapsing to the first listed
  type therefore labels a march as PROTEST about half the time, which is both
  what the gold does and what contradicts the codebook printed in the task's own
  prompt. The task cannot be rebuilt from public data: CCC ships terse curator
  notes, not the news-story prose the benchmark uses, and the pairing of events
  to articles is Halterman and Keith's own unpublished work.

  Effect on the headline: local models match or exceed the best API model on
  **9 of 33** tasks rather than 10 of 34, and the mean best-API-minus-best-local
  gap moves from +0.0112 to +0.0130. Model ordering is unchanged. CCC was one of
  the tasks local models won, so excluding it slightly favours the API side.

- **Macro F1 no longer counts labels with zero gold support as zeros.** The
  categorical metric averaged over every label declared in a manifest; a label
  absent from the sample can only score 0, so it entered the mean as one.
  `code/scoring.py` is now the single implementation for every summary and
  analysis script.

  Three tasks change, the other 30 are identical:

  | Task | Before | After | Cause |
  |---|---|---|---|
  | `mellon_bes_mii_2024` | 0.505 | 0.721 | 15 of 50 declared labels have no gold support |
  | `cap_crs_policy_topic` | 0.669 | 0.702 | 1 of 21 |
  | `haunss_papea_claims` | 0.396 | 0.426 | 2 of 28 |

  All ten model means rise by roughly 0.009 and the ordering is unchanged,
  because the correction removed a penalty that applied to every model equally.
  `python3 code/build_summary.py --legacy-all-labels` reproduces the published
  numbers exactly; they are also kept at
  `output/summary_legacy_all_labels.csv` and
  `output/summary_batched_legacy_all_labels.csv`.

  `code/build_frontier_2026.py`, `code/summarize_hive_bakeoff.py` and the frozen
  18-task frontier panel calibration stay pinned to the legacy metric on purpose,
  so existing frontier and refresh panels keep reproducing. Recalibrating that
  panel is a deliberate analysis decision and has not been made.

- **Report prose and PDF rebuilt** against the corrected metric. They are NOT yet
  updated for the CCC exclusion: `output/report_pdf.pdf` still describes 34 tasks
  and "10 of 34", because that is a substantive claim change rather than a
  rounding update and needs sign-off. The CSV summaries, figures and appendix
  tables in `output/` do reflect the 33-task set, so the PDF and the rest of
  `output/` currently disagree. README carries a note to that effect.

### Added

- **New task `yan_bernhard_offensive`, in `tasks_ext/` rather than `tasks/`.**
  Replies to political canvassing text messages, labelled offensive or not.
  Source: Yan and Bernhard (2024), Harvard Dataverse `doi:10.7910/DVN/UYKE7T`.
  3,376 texts after cleaning (2.0 percent dropped: one text whose duplicates
  disagreed on gold, plus deduplication).

  Gold is the majority of three crowd rating slots. Those slots are not three
  named coders -- each text was rated by three Prolific respondents, ten texts
  per respondent -- which is why slot 3 agrees poorly with the other two
  (Cohen's kappa 0.27-0.33, against 0.73 between slots 1 and 2). Raters saw a
  five-point ordinal scale and the binary is the authors' own collapse of it;
  the exact cut point is not recoverable from the released file, so the benchmark
  treats the authors' binary as the direct source label. Unanimity across the
  three slots is 0.809 and should be reported alongside model scores.

  Sampled at 1,000 items rather than the usual 500: the positive class is
  6.6 percent of the frame, so a 500-item draw yields only about 33 offensive
  replies, which makes positive-class F1 unstable. The realised sample has 59
  positives in 1,000 items.

  It lives in `tasks_ext/` because adding a manifest to `tasks/` raises the
  all-manifests count to 35 and breaks every frozen panel pinning
  `expected_tasks: 34`. It is scored on the four Hive checkpoints only and
  carries no local-vs-API comparison: the six published Ollama models cannot be
  run (mac2 has no ollama service and no pulled models) and the API models cost
  money. It does not enter the 33-task headline.

  This does not replace `halterman_ccc_protest`. CCC was protest-form
  classification and the event-coding family already holds six tasks; no
  candidate in the expansion list measures what it measured.

- **Candidate list verified.** `docs/task_expansion_candidates.csv` had stale
  access flags. Of seven candidates with DOIs, one is ingestible
  (`yan_bernhard`), one is real but only 100 rows
  (`haunss_papea_full_article_claims`), and four fail outright:
  `where_do_parties_talk_about_what` (corpus text column 100 percent NA across
  41,497 rows), `policlim_climate_manifestos` (no text column; Manifesto Project
  terms restrict redistribution), `ziegler_manipulation_checks` (no data files at
  all), `radford_2021_automated_dictionary_generation` (word2vec embeddings, not
  a labelled corpus). `hobbs_green_open_ended_attitudes` is unverified; its
  archive is a single 12.8 GB capsule and was not downloaded. Three rows that
  read as unimplemented were corrected to `implemented`.

- **`tasks_v2/` and `prompts_v2/`** hold corrected definitions for five tasks.
  v1 is untouched, so published results stay reproducible. Four are item-paired
  with v1; `brandt_gtd_attack_type` is not, because it drops rows.

  | v2 task | Fix |
  |---|---|
  | `brandt_gtd_attack_type` | Drops the `Unknown` class. GTD assigns it when the source report did not specify the method, which the model cannot see: 47 of 500 items, 3.2 percent accuracy across all ten models. |
  | `cap_crs_policy_topic` | Reads the prebuilt `text` column so the 190 of 500 items with no summary stop rendering an empty `Summary:` heading. Topic 5 renamed to `Labor`. Codebook definitions added. |
  | `cap_party_platform_policy_topic` | Topic 5 renamed to `Labor`. Codebook definitions added. |
  | `halterman_keith_bfrs` | Removes the source header telling the model to write the bare label "with no other text", which contradicted the prompt's own JSON-output block. |
  | `halterman_keith_cmp` | Same header removal. |

- **Two manifest mechanisms.** `ground_truth.exclude_labels` drops gold classes
  that record coder uncertainty rather than a property of the text.
  `ground_truth.label_map` renames gold values at load time, so a label name can
  be corrected without regenerating a cleaned CSV from a source archive. CAP
  major topic 5 carried its pre-split name "Labor and Immigration" alongside a
  separate "Immigration" topic, giving the model two buckets for one thing.

- **`status: excluded`** in a task manifest holds a task out of the benchmark
  without deleting anything. Loading returns every manifest by default and the
  current release view opts in with `active_only=True`, rather than the reverse.
  That choice is deliberate: 27 files load tasks and most of them are frozen
  panels whose scope is fixed by a config or a published artifact. Making the
  exclusion the default broke 19 tests with errors like `frozen scope mismatch:
  observed 33 tasks/15925 items, expected 34/16425` and `category taxonomy must
  partition exactly 34 tasks`. The permissive default keeps those panels
  resolving to the task set they were built on. The cost is that a new consumer
  which forgets `active_only=True` will silently include a held-out task.

- **Constrained-decoding experiment.** `code/hive_vllm_benchmark.py` accepts
  `generation.structured_outputs: none`, and
  `experiments/schema_parity_20260918_{guided,plain}.yaml` are byte-identical
  apart from that key. The published run gave API models server-side JSON-schema
  enforcement and local Ollama models none, so the local-versus-API gap confounds
  model quality with harness parity. The two arms vary only constrained decoding,
  with model, GPU, prompt, items and seed fixed. Not yet run.

- **`osnabruegge_cross_domain_topic` human ceiling.** The replication capsule
  carries three independent validation coders on 250 rows. They match the
  published label only 61.6 / 65.2 / 65.2 percent of the time and agree with each
  other on 54.8 percent (Cohen's kappa 0.57-0.65). The archive has no speech,
  debate, date or speaker column, so the coders saw exactly what the model is
  shown. Model accuracy of 0.453 on this task should be read against a ceiling
  near 0.65, not against 1.0. The coder columns are now carried into
  `data/osnabruegge_cross_domain_topic.csv` and the build script records its
  source URL.

### Fixed

- **The frozen frontier panel read a mutable artifact.**
  `experiments/frontier_checkpoints_2026.yaml` pointed three API checkpoints
  (`claude_sonnet_4_6_api`, `gpt_5_5_api`, `deepseek_v4_pro_api`) at
  `output/summary.csv` with `result_status: full34_confirmed`. Rebuilding that
  file at 33 tasks broke their declared coverage. They now read
  `output/summary_legacy_all_labels.csv`, the preserved 34-task legacy-metric
  artifact they were built on, matching the `support_only=False` pin already in
  `code/build_frontier_2026.py`. The underlying fragility remains: a frozen panel
  should hold its own copy of its evidence rather than depend on a live release
  file, as the Hive checkpoints do.

- `output/summary.csv` was missing `gpu_hours_per_1000` for
  `qwen3.5:35b-a3b-q4_K_M` across all 34 tasks, although its manifest sets
  `compute_class: local`. The published file predated the manifest.

### Known issues

- **Duplicate items.** Several tasks sample the same text many times over:
  `mellon_bes_mii_2024` 56 percent of 500 items ("Open-ended response: Covid"
  appears 74 times), `haunss_papea_fgz_forms` 50 percent (single German words),
  `bestvater_kavanaugh_stance` 31 percent. Effective unique items are 222 and 249
  rather than 500. The paired bootstrap resamples by item, so repeated texts count
  as independent draws and the published confidence intervals are narrower than
  the effective sample supports. Documented, deliberately not yet fixed.
- `data/gilardi_relevance.csv` rows 242 and 1588 hold identical text with
  `gt_relevant` 1 and 0, against the entry rule in `TODO.md`.
- **Codebook depth is not comparable across tasks.** Prompt body length per label
  ranges from 24 characters (`haunss_papea_claims`, 28 labels) to 1,130
  (`halterman_keith_bfrs`, 12 labels). `plover_cameo_event` and
  `haunss_papea_claims` still give a bare list with no definitions; writing them
  requires the PLOVER and PAPEA codebooks, which are not in the repo.
- The CAP definitions added to the v2 prompts were written from the CAP
  major-topic scheme rather than transcribed from a downloaded copy of the master
  codebook, and should be checked against it before the v2 run is treated as
  final.
- Eleven tasks have no `code/build_*_task.py`, so their cleaned CSVs cannot be
  regenerated and their import cannot be verified.
- 175 `item_id` values are reused across different tasks, so any join over
  predictions must key on `(task, item_id)`.
- **The refresh configs pin `benchmark_commit: 3c7ad07`,** so
  `tests/test_refresh_backfills.py::test_actual_refresh_harmony_run_and_resume`
  fails against any newer HEAD with `benchmark checkout is ..., but the frozen
  config requires ...`. This is the pin working as designed: the refresh panels
  genuinely cannot be reproduced from current HEAD now that the task set and the
  metric have changed. Silencing it would defeat its purpose. It needs a decision
  -- re-pin the configs to a new commit and rebuild those panels, or declare the
  refresh track frozen against `3c7ad07`.

## 2026-06-13

- Added Qwen3.5 35B-A3B as a sixth local model, serial and 10-item batched across
  all 34 tasks, bringing the benchmark to ten models. Qwen3 30B-A3B is retained as
  a prior-generation reference. The batch parser was extended to accept
  newline-delimited JSON, which Qwen3.5 emits for batched prompts.

## 2026-06-06

- Email addresses embedded in dataset text were replaced with `[EMAIL]` across
  nine corpora. The published predictions predate this and were generated on the
  pre-redaction text, so the released corpora differ from the exact scored inputs
  for a small number of items. Predictions were not regenerated.

## 2026-05-20

- Report assets rebuilt. The two CAP task manifests had their `source:` citation
  strings corrected; data, labels and prompts were unchanged, so no rerun was
  required.

## 2026-05-18

- First public release: 34 tasks, 10 models (6 local Ollama, 4 commercial API),
  340 task-model pairs, 164,250 serial classifications.
