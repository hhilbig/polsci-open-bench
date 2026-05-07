# Post-release changes

Last updated: 2026-05-02

This note inventories the current working-tree changes relative to
`HEAD` commit `1d20dce` (2026-04-29), which appears to be the last
pre-release repo state.

## Bottom line

- The public, scored benchmark artifacts still correspond to the original
  10-task release.
- Since release, the repo has been upgraded in four main ways:
  - benchmark infrastructure is now manifest driven
  - raw artifact integrity issues were repaired
  - four new live tasks were added for future reruns
  - reporting, docs, and tests were expanded substantially

## 1. Infrastructure and reproducibility

- Added manifest-driven task loading:
  - [`code/task_registry.py`](../code/task_registry.py)
  - [`tasks/`](../tasks)
- Added manifest-driven model loading:
  - [`code/model_registry.py`](../code/model_registry.py)
  - [`models/`](../models)
- Added a built-in DeepSeek API model manifest:
  - [`models/deepseek_v4_pro.yaml`](../models/deepseek_v4_pro.yaml)
- Added provider-specific OpenAI-compatible manifest fields for cases like
  DeepSeek:
  - `provider`
  - `response_format_type`
  - `thinking_mode`
- Updated the main runners and summary builders to use the registries:
  - [`code/benchmark.py`](../code/benchmark.py)
  - [`code/batch_benchmark.py`](../code/batch_benchmark.py)
  - [`code/build_summary.py`](../code/build_summary.py)
  - [`code/build_summary_batched.py`](../code/build_summary_batched.py)
- Added pinned Python dependencies:
  - [`requirements.txt`](../requirements.txt)
- Added custom extension docs and minimal examples:
  - [`docs/custom_tasks.md`](custom_tasks.md)
  - [`docs/custom_models.md`](custom_models.md)
  - [`examples/minimal_custom_task/`](../examples/minimal_custom_task)
  - [`examples/minimal_custom_models/`](../examples/minimal_custom_models)
- Hardened built-in API model loading so OpenAI-compatible or Anthropic models
  without configured keys are skipped with a notice rather than run into the
  benchmark as error rows.
- Added a DeepSeek-specific compatibility path in the OpenAI runner:
  - `deepseek-v4-pro` now uses `response_format={"type": "json_object"}`
  - the runner appends explicit JSON-only output reminders for that path
  - the live DeepSeek manifest forces non-thinking mode

## 2. Integrity repairs to released artifacts

- Sampled `item_id` values are now guaranteed unique within task.
- The checked-in raw prediction files were repaired to use the same
  deterministic duplicate-ID suffixing as the live loaders:
  - [`output/predictions.csv`](../output/predictions.csv)
  - [`output/predictions_batched.csv`](../output/predictions_batched.csv)
- Stale `state_adaptation` columns were removed from the batched raw file.
- The serial summary was rebuilt after the repair:
  - [`output/summary.csv`](../output/summary.csv)
- The substantive benchmark results did not change in scope:
  - the checked-in scored benchmark still covers the original 10 tasks
  - no new model calls were made during this post-release maintenance wave
- The only metric changes from the integrity repair were small paired-bootstrap
  CI updates, concentrated in `gilardi_relevance`.

## 3. New live tasks added after release

These tasks are now implemented in the live manifests and ready for future
benchmark reruns, but they are not yet included in the checked-in scored
prediction artifacts.

1. `osnabruegge_cross_domain_topic`
   - Source: Osnabruegge, Ash, and Morelli (2023)
   - Family: `Policy-topic coding`
   - Files:
     - [`code/build_osnabruegge_cross_domain_topic_task.py`](../code/build_osnabruegge_cross_domain_topic_task.py)
     - [`data/osnabruegge_cross_domain_topic.csv`](../data/osnabruegge_cross_domain_topic.csv)
     - [`tasks/osnabruegge_cross_domain_topic.yaml`](../tasks/osnabruegge_cross_domain_topic.yaml)
     - [`prompts/osnabruegge_cross_domain_topic.txt`](../prompts/osnabruegge_cross_domain_topic.txt)

2. `rheault_line_of_fire_incivility`
   - Source: Rheault, Rayment, and Musulan (2019)
   - Family: `Relevance / Incivility`
   - Files:
     - [`code/build_rheault_line_of_fire_task.py`](../code/build_rheault_line_of_fire_task.py)
     - [`data/rheault_line_of_fire_incivility.csv`](../data/rheault_line_of_fire_incivility.csv)
     - [`tasks/rheault_line_of_fire_incivility.yaml`](../tasks/rheault_line_of_fire_incivility.yaml)
     - [`prompts/rheault_line_of_fire_incivility.txt`](../prompts/rheault_line_of_fire_incivility.txt)

3. `haunss_papea_fgz_forms`
   - Source: Haunss et al. (2025)
   - Family: `Event coding`
   - Files:
     - [`code/build_haunss_papea_fgz_forms_task.py`](../code/build_haunss_papea_fgz_forms_task.py)
     - [`data/haunss_papea_fgz_forms.csv`](../data/haunss_papea_fgz_forms.csv)
     - [`tasks/haunss_papea_fgz_forms.yaml`](../tasks/haunss_papea_fgz_forms.yaml)
     - [`prompts/haunss_papea_fgz_forms.txt`](../prompts/haunss_papea_fgz_forms.txt)

4. `brandt_political_relevance`
   - Source: Brandt et al. (2025)
   - Family: `Relevance / Incivility`
   - Files:
     - [`code/build_brandt_political_relevance_task.py`](../code/build_brandt_political_relevance_task.py)
     - [`data/brandt_political_relevance.csv`](../data/brandt_political_relevance.csv)
     - [`tasks/brandt_political_relevance.yaml`](../tasks/brandt_political_relevance.yaml)
     - [`prompts/brandt_political_relevance.txt`](../prompts/brandt_political_relevance.txt)
   - Note:
     - this public corpus has 320 usable rows after exact-text deduplication, so
       the live loader uses all available rows rather than forcing a 500-item sample

5. `theocharis_dynamics_incivility`
   - Source: Theocharis et al. (2020)
   - Family: `Relevance / Incivility`
   - Files:
     - [`code/build_theocharis_dynamics_incivility_task.py`](../code/build_theocharis_dynamics_incivility_task.py)
     - [`data/theocharis_dynamics_incivility.csv`](../data/theocharis_dynamics_incivility.csv)
     - [`tasks/theocharis_dynamics_incivility.yaml`](../tasks/theocharis_dynamics_incivility.yaml)
     - [`prompts/theocharis_dynamics_incivility.txt`](../prompts/theocharis_dynamics_incivility.txt)
   - Note:
     - this public replication training file has 4,000 labeled rows; the live
       build drops 3 rows during exact-text conflict and duplicate handling,
       leaving 3,997 usable tweets

## 4. Diagnostics, tests, and reporting

- Added a task-length audit build:
  - [`code/build_task_length_audit.py`](../code/build_task_length_audit.py)
  - [`output/task_length_audit.csv`](../output/task_length_audit.csv)
  - [`output/task_length_analysis.md`](../output/task_length_analysis.md)
- Added loader, manifest, and artifact integrity tests:
  - [`tests/test_benchmark_integrity.py`](../tests/test_benchmark_integrity.py)
- Updated report inputs and rerendered public-facing artifacts:
  - [`output/report_pdf.qmd`](../output/report_pdf.qmd)
  - [`output/report_pdf.pdf`](../output/report_pdf.pdf)
  - [`output/figures/fig-mean-f1.png`](../output/figures/fig-mean-f1.png)
  - [`output/figures/fig-speed.png`](../output/figures/fig-speed.png)

## 5. Documentation and roadmap

- Expanded core docs:
  - [`README.md`](../README.md)
  - [`docs/reproduce.md`](reproduce.md)
  - [`docs/schema.md`](schema.md)
  - [`docs/prompts_provenance.md`](prompts_provenance.md)
  - [`docs/status.md`](status.md)
  - cost-tracking guidance in the scaffolding docs now distinguishes
    provider-reconciled billing from manifest-level benchmark approximations
- Added ongoing roadmap / maintenance tracking:
  - [`TODO.md`](../TODO.md)
- Added an explicit branch / sidecar / canonical workflow note:
  - [`docs/release_workflow.md`](release_workflow.md)
- Added sourcing notes for benchmark expansion:
  - [`docs/task_expansion_candidates.md`](task_expansion_candidates.md)
  - [`docs/task_expansion_candidates.csv`](task_expansion_candidates.csv)
- The roadmap now also records a possible future local-model expansion pass,
  including a concrete shortlist:
  - add next: run the prepared built-in `gemma4:26b` manifest
  - sidecar only or conditional: `LFM2-24B-A2B`, `gpt-oss:20b`,
    `glm-4.7-flash`
  - skip for now: larger or more overlapping options such as
    `DeepSeek-R1-Distill-Qwen-32B`, `gpt-oss:120b`, and `Qwen 2.5-72B`
  - the new built-in manifest lives at
    [`models/gemma4_26b_a4b.yaml`](../models/gemma4_26b_a4b.yaml)
- A second parallel sourcing pass on 2026-05-03 expanded the structured task
  candidate queue with new `Relevance / Incivility`, `Event coding`,
  `Policy-topic coding`, and `new family` leads, including `Clicks and Stones`,
  `Political DEBATE / PolNLI`, and `Campaign communication and legislative leadership`.
- That sourcing pass also produced a locked broader next verification wave of
  four targets:
  - `ICBe`
  - `Campaign communication and legislative leadership`
  - `Clicks and Stones`
  - `Political DEBATE / PolNLI`
- `ICBe` has now been replication-verified as the first target in that wave.
  Direct inspection of the public files shows a clean path to a sentence-level
  event task using the aligned agreed-event file plus the public sentence
  corpus. The current recommended first slice is a `5`-class sentence task:
  `No Event`, `Action`, `Speech`, `Thought`, and `Mixed`.
- That `ICBe` sentence task is now live in the repo as
  [`douglass_icbe_sentence_event_type`](../tasks/douglass_icbe_sentence_event_type.yaml),
  built from the public RDS files with `12,676` cleaned sentence rows after
  dropping `4` malformed sentence rows from the public corpus.
- `Campaign communication and legislative leadership` has now also been
  implemented in the live manifests as
  [`muller_fujimura_campaign_policy_area`](../tasks/muller_fujimura_campaign_policy_area.yaml),
  built from the public supervised sentence splits. The live task keeps `12`
  policy-area labels, drops `12` rows tied to exact-text label conflicts, then
  deduplicates exact repeated text-label pairs to leave `2,915` cleaned rows.
- `Clicks and Stones` failed the direct-ingest screen. The public archive
  exposes legislator-level aggregate hostility rates and replication code for
  those aggregates, but not tweet-level public text plus labels.
- `Political DEBATE / PolNLI` has now been implemented as
  [`burnham_polnli_entailment`](../tasks/burnham_polnli_entailment.yaml), using
  the public PolNLI test split. The live task keeps `15,314` cleaned
  premise-hypothesis pairs after dropping `52` duplicate or conflicting pairs
  and maps the source coding so `gt_entails = 1` means the hypothesis is
  supported by the premise.

## 6. Current live state after the post-release wave

- Checked-in scored benchmark: still 10 tasks
- Live built-in task library: now 18 tasks
- Live built-in model manifests: now 9 models, including `deepseek-v4-pro`
  and the prepared `gemma4:26b` local manifest
- New task families are more balanced than at release:
  - `Relevance / Incivility`: now 5 live tasks
  - `Sentiment / Stance / Tone`: 4 live tasks
  - `Event coding`: 4 live tasks
  - `Policy-topic coding`: 4 live tasks
  - `Hypothesis-conditioned classification`: 1 live pilot task

## 7. Not done yet

- The four newly added live tasks have now been run for the four local
  open-weight models in a sidecar artifact, but that sidecar has not been
  merged into the canonical public outputs.
- No post-release full serial rerun with the main API models has been committed
  yet.
- No post-release batched rerun has been committed yet.
- A 2026-05-04 replacement screen for the failed `Clicks and Stones` slot
  rejected three additional `Relevance / Incivility` candidates for the live
  benchmark:
  - `Super-Unsupervised Classification for Labelling Text`: label/score files
    do not expose text, and the only public text example file has 35 rows.
  - `Citizens' Perceptions of Online Abuse Directed at Politicians`: public
    data expose numeric message IDs and respondent ratings, but not message text.
  - `Online Abuse of Politicians`: public data expose numeric message IDs and
    politician ratings, but not message text.
- The same screen verified three direct-label alternatives:
  - `The Dynamics of Political Incivility on Twitter`: `4,000` public English
    tweets with direct `uncivil` labels in a paper-backed replication repo,
    now implemented as `theocharis_dynamics_incivility`.
  - `toxicity-protests-ES`: `1,000` public Spanish protest-discourse rows with
    human coder labels, useful for multilingual coverage.
  - `TwitCivility`: `13,124` public rows with direct `impoliteness` and
    `intolerance` labels, but weaker on the published-political-science-paper
    criterion.
- Non-English tasks are acceptable for future benchmark expansion if they
  otherwise satisfy the task-selection criteria.
- The next recommended task-collection step is to either implement the
  multilingual `toxicity-protests-ES` fallback or return to balancing additions
  across the other original task families.

## 8. DeepSeek Sidecar Runs

- Two post-release DeepSeek serial runs were completed on `mac2` and copied back
  into a dated sidecar archive:
  - [`output/archive/deepseek_sidecar_2026-05-02/`](../output/archive/deepseek_sidecar_2026-05-02)
- These artifacts are explicitly not merged into the canonical public outputs in
  `output/predictions.csv`, `output/summary.csv`, or `output/report_pdf.pdf`.
- Sidecar contents:
  - `deepseek_full_predictions_nonthinking.csv`
  - `deepseek_full_summary_nonthinking.csv`
  - `deepseek_full_predictions_thinking.csv`
  - `deepseek_full_summary_thinking.csv`
- Non-thinking DeepSeek sidecar:
  - 14 live tasks
  - 6,820 rows
  - 0 parse errors
  - mean latency about 1.59 s/item
- Thinking DeepSeek sidecar:
  - 14 live tasks
  - 6,820 rows
  - 152 parse errors
  - mean latency about 8.17 s/item
- Substantive read:
  - thinking mode improved DeepSeek's mean macro F1 on the original 10 release
    tasks from about 0.591 to about 0.609
  - but it also made the model much slower and much less robust on structured
    output, so both conditions are being retained as sidecar artifacts rather
    than merged into the canonical benchmark

## 9. Open-weight sidecar run for new tasks

- A local-model rerun for the four post-release tasks was completed on `mac2`
  and copied back into a dated sidecar archive:
  - [`output/archive/openweight_new_tasks_2026-05-02/`](../output/archive/openweight_new_tasks_2026-05-02)
- Sidecar contents:
  - `new_tasks_openweight_predictions.csv`
  - `new_tasks_openweight_summary.csv`
- Coverage:
  - 4 tasks
  - 4 local open-weight models
  - 7,280 scored rows
  - 0 parse errors
- Best sidecar model by task:
  - `brandt_political_relevance`: `qwen3:14b-q4_K_M` (`0.739`)
  - `haunss_papea_fgz_forms`: `gemma4:31b-it-q4_K_M` (`0.958`)
  - `osnabruegge_cross_domain_topic`: `gemma4:31b-it-q4_K_M` (`0.435`)
  - `rheault_line_of_fire_incivility`: `gemma4:31b-it-q4_K_M` (`0.582`)
- Merge decision:
  - these results are being kept as sidecar artifacts for now
  - they are not merged into `output/predictions.csv`, `output/summary.csv`, or
    `output/report_pdf.pdf` because the main API models have not yet been rerun
    on the same four tasks

## 10. OpenAI-only proprietary sidecar for new tasks

- An OpenAI-only rerun for the four post-release tasks was completed and stored
  as a dated sidecar archive:
  - [`output/archive/proprietary_openai_only_new_tasks_2026-05-03/`](../output/archive/proprietary_openai_only_new_tasks_2026-05-03)
- Sidecar contents:
  - `proprietary_openai_only_predictions.csv`
  - `proprietary_openai_only_summary.csv`
  - `proprietary_openai_only.log`
- Coverage:
  - 4 tasks
  - 2 OpenAI models
  - 3,640 scored rows
  - 0 parse errors
- Best sidecar model by task:
  - `brandt_political_relevance`: `gpt-5.4-nano` (`0.646`)
  - `haunss_papea_fgz_forms`: `gpt-5.5` (`0.965`)
  - `osnabruegge_cross_domain_topic`: `gpt-5.5` (`0.443`)
  - `rheault_line_of_fire_incivility`: `gpt-5.4-nano` (`0.607`)
- Merge decision:
  - these results are being kept as sidecar artifacts for now
  - they are not merged into `output/predictions.csv`, `output/summary.csv`, or
    `output/report_pdf.pdf` because the Claude run is still missing for the
    same four tasks

## 11. Combined six-model comparison table for new tasks

- Added a helper archive that combines the existing open-weight and OpenAI-only
  sidecars for the four post-release tasks:
  - [`output/archive/new_tasks_full_comparison_2026-05-03/`](../output/archive/new_tasks_full_comparison_2026-05-03)
- Files:
  - `combined_new_tasks_summary_long.csv`
  - `combined_new_tasks_headline_f1_wide.csv`
  - `combined_new_tasks_headline_f1.md`
- Scope:
  - 4 tasks
  - 6 models
  - 24 `(task, model)` cells
- Purpose:
  - provide one compact comparison table across the currently completed local
    and OpenAI reruns
  - remain non-canonical until the same task set is completed for Claude
