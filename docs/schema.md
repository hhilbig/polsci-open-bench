# Output schema contract

The benchmark writes two raw prediction files and three summary files to
`output/`. The intended schema is defined by the live task manifests in
[`tasks/`](../tasks). Model-level summary semantics are defined by the live
model manifests in [`models/`](../models).

## Raw predictions

### `output/predictions.csv`

One row per `(task, model, item_id)` from the serial benchmark.

Core columns:

| Column | Type | Required | Description |
|---|---|---|---|
| `task` | string | yes | Task name from the loaded task manifest |
| `model` | string | yes | Model identifier |
| `item_id` | string | yes | Stable sampled item identifier within task |
| `latency_s` | float | yes | Wall-clock latency for that item call |
| `eval_count` | int | optional | Output tokens if reported by the backend |
| `parse_error` | string | optional | Null on clean parse, else parse or API error |
| `raw_content_preview` | string | optional | First 200 chars of raw model output |
| `source_file_label` | string | optional | Source split or source file label when a task builder keeps that provenance |

Task-specific label columns are wide and sparse: each row carries only the
`pred_*` and `gt_*` columns relevant to its task, and all unrelated task columns
are null.

Latency contract:

- `latency_s` is a required benchmark field, not auxiliary metadata.
- For serial rows, `latency_s` is the wall-clock time for exactly one item-level
  model call, measured around the backend request and response.
- For API models, this includes provider/network request time as observed by the
  benchmark runner. For local models, it includes the local serving request time
  as observed by the runner.
- Cleanly parsed rows must have positive, non-missing `latency_s`. Rows that fail
  before receiving a model response may have missing `latency_s`, but those rows
  must carry a non-null `parse_error`.
- New backends and custom runners should preserve this field so downstream users
  can compare runtime, not just classification accuracy.

Active task label columns:

| Task | Label kind | Prediction / ground-truth columns |
|---|---|---|
| `gilardi_relevance` | binary | `pred_relevant`, `gt_relevant` |
| `ballard_incivility` | binary | `pred_uncivil`, `gt_uncivil` |
| `gilardi_stance` | categorical | `pred_stance`, `gt_stance` |
| `chae_semeval_stance` | categorical | `pred_stance`, `gt_stance` |
| `ornstein_scotus_sentiment` | categorical | `pred_sentiment`, `gt_sentiment` |
| `wesleyan_creative_ads_2022` | categorical | `pred_tone`, `gt_tone` |
| `halterman_ccc_protest` | categorical | `pred_protest_type`, `gt_protest_type` |
| `halterman_keith_bfrs` | categorical | `pred_event_type`, `gt_event_type` |
| `halterman_keith_cmp` | categorical | `pred_policy_domain`, `gt_policy_domain` |
| `mellon_bes_mii_2024` | categorical | `pred_issue`, `gt_issue` |
| `osnabruegge_cross_domain_topic` | categorical | `pred_policy_domain`, `gt_policy_domain` |
| `rheault_line_of_fire_incivility` | binary | `pred_uncivil`, `gt_uncivil` |
| `haunss_papea_fgz_forms` | categorical | `pred_protest_form`, `gt_protest_form` |
| `brandt_political_relevance` | binary | `pred_relevant`, `gt_relevant` |
| `douglass_icbe_sentence_event_type` | categorical | `pred_event_type`, `gt_event_type` |
| `muller_fujimura_campaign_policy_area` | categorical | `pred_policy_area`, `gt_policy_area` |
| `burnham_polnli_entailment` | binary | `pred_entails`, `gt_entails` |
| `theocharis_dynamics_incivility` | binary | `pred_uncivil`, `gt_uncivil` |
| `toxicity_protests_es` | binary | `pred_toxic`, `gt_toxic` |
| `brandt_gtd_attack_type` | categorical | `pred_attack_type`, `gt_attack_type` |
| `haunss_papea_claims` | categorical | `pred_protest_claim`, `gt_protest_claim` |
| `twitcivility_impoliteness` | binary | `pred_impolite`, `gt_impolite` |
| `bestvater_wm_stance` | binary | `pred_pro_womens_march`, `gt_pro_womens_march` |
| `erlich_ati_topics` | multi-binary | `pred_Activities`, `gt_Activities`, `pred_Budget`, `gt_Budget`, `pred_Evaluation`, `gt_Evaluation`, `pred_External Contracts`, `gt_External Contracts`, `pred_Institutional Structure`, `gt_Institutional Structure`, `pred_Other`, `gt_Other`, `pred_Regulatory`, `gt_Regulatory` |
| `plover_cameo_event` | categorical | `pred_event_type`, `gt_event_type` |
| `burnham_polnli_event_entailment` | binary | `pred_entails`, `gt_entails` |
| `burnham_trump_stance` | categorical | `pred_stance_toward_trump`, `gt_stance_toward_trump` |
| `burnham_covid_threat_minimization` | binary | `pred_threat_minimizing`, `gt_threat_minimizing` |
| `dicocco_manifesto_populism` | binary | `pred_populist`, `gt_populist` |
| `bestvater_kavanaugh_stance` | binary | `pred_pro_kavanaugh`, `gt_pro_kavanaugh` |
| `politicause_causal_relation` | binary | `pred_causal_relation`, `gt_causal_relation` |
| `cap_party_platform_policy_topic` | categorical | `pred_policy_topic`, `gt_policy_topic` |
| `cap_crs_policy_topic` | categorical | `pred_policy_topic`, `gt_policy_topic` |
| `agoraspeech_criticism_agenda` | categorical | `pred_criticism_or_agenda`, `gt_criticism_or_agenda` |

### `output/predictions_batched.csv`

One row per `(task, model, batch_size, item_id)` from the batched benchmark.

Core columns:

| Column | Type | Required | Description |
|---|---|---|---|
| `task` | string | yes | Task name from the loaded task manifest |
| `model` | string | yes | Model identifier |
| `batch_size` | int | yes | Requested batch size for that cell |
| `item_id` | string | yes | Stable sampled item identifier within task |
| `latency_s` | float | yes | Per-item latency, defined as batch wall time divided by batch size |
| `batch_latency_s` | float | yes | Full wall time for the batch call |
| `eval_count` | int | optional | Output tokens if reported by the backend |
| `parse_error` | string | optional | Null on clean parse, else parse or API error |
| `raw_content_preview` | string | optional | First 200 chars of raw model output |

The task-specific `pred_*` and `gt_*` columns follow the same live task schema
as `predictions.csv`.

For batched runs, `batch_latency_s` is the full wall-clock time for the backend
call, and `latency_s` is the per-item latency defined as `batch_latency_s`
divided by the actual number of items in that batch. Both fields are required
for cleanly parsed rows.

## Summary files

### `output/summary.csv`

One row per `(task, model)` from the serial benchmark.

| Column | Type | Description |
|---|---|---|
| `task`, `model` | string | Identifiers |
| `n` | int | Total attempted items |
| `parse_ok` | int | Items with clean parse |
| `parse_err_rate` | float | Share of items with parse errors |
| `mean_latency_s`, `median_latency_s` | float | Per-item latency summaries |
| `gpu_hours_per_1000` | float | Models whose manifest has `compute_class: local`; benchmark-side compute proxy, not provider billing |
| `gpu_hours_per_1000_correct` | float | Same as above, scaled by share correct |
| `usd_per_1000` | float | Models whose manifest provides `cost_per_call_usd`; benchmark-side approximation, not provider-reconciled billing |
| `usd_per_1000_correct` | float | Same as above, scaled by share correct |
| `f1_*` | float | Per-label F1 on cleanly parsed items |
| `avg_f1` | float | Mean across the available `f1_*` label columns for that row |
| `accuracy` | float | Exact-match accuracy where defined |
| `mcc` | float | Matthews correlation coefficient |
| `headline_f1` | float | Unified task-level main F1 metric |
| `headline_f1_lo`, `headline_f1_hi` | float | 95% paired bootstrap interval |

The report calls `headline_f1` the main F1 score. The column name is retained
for backward compatibility. It is defined as:

- positive-class F1 for binary tasks
- mean per-label F1 for multi-binary tasks
- macro F1 across the task label set for categorical tasks

Use `headline_f1` to reproduce the report. `avg_f1` is a lower-level summary of
the row's label-specific `f1_*` columns. The two often match for categorical
tasks, but `headline_f1` is the stable report-level outcome because it also
handles binary and multi-binary tasks consistently.

The `f1_*` columns are generated from task labels. Some labels contain spaces,
slashes, parentheses, commas, or mixed capitalization. In R or Python, quote
these column names or clean them before using them in formulas or attribute
access.

Cost note:

- `usd_per_1000` and `usd_per_1000_correct` are only as exact as the
  manifest-level `cost_per_call_usd` supplied for that model.
- For token-priced APIs such as OpenAI, Anthropic, and DeepSeek, exact billing
  should be taken from provider usage/cost reporting rather than inferred from
  these columns.

### `output/summary_batched.csv`

One row per `(task, model, batch_size)` in the batching comparison summary.
Rows with `batch_size == 1` are one-item-at-a-time baselines copied into the
same summary so that `agreement_vs_b1` and speed comparisons can be computed in
one table. They do not imply that `output/predictions_batched.csv` contains raw
batch-size-1 predictions.

| Column | Type | Description |
|---|---|---|
| `task`, `model`, `batch_size` | identifiers | Cell identifiers |
| `n` | int | Total attempted items |
| `parse_ok` | int | Items with clean parse |
| `parse_err_rate` | float | Share of items with parse errors |
| `mean_latency_s`, `median_latency_s` | float | Per-item latency summaries |
| `median_batch_latency_s` | float | Median wall time for the actual batched call |
| `f1_*` | float | Per-label F1 on cleanly parsed items |
| `avg_f1` | float | Mean across the available `f1_*` label columns for that row |
| `accuracy` | float | Exact-match accuracy where defined |
| `headline_f1` | float | Unified task-level main F1 metric |
| `agreement_vs_b1` | float | Share of items whose prediction matches the same model at `batch_size == 1` |

### `output/summary_batched_local_b10.csv`

Subset view of `summary_batched.csv` used for the local 10-item prompt
comparison. It includes:

- `batch_size == 10` rows for the five local models across all 34 tasks;
- `batch_size == 1` baseline rows for all nine serial models; and
- the same columns as `summary_batched.csv`.

This file is convenient for local speed and reliability plots. For the raw
10-item prompt outputs, use `output/predictions_batched.csv`.

## Merging semantics

`--merge-into PATH` in [`code/benchmark.py`](../code/benchmark.py) replaces any
rows in the existing CSV that share `(task, model, item_id)` with the newly
generated rows, preserving all other rows. This supports selective reruns after
prompt or parser changes without redoing the full serial grid.
