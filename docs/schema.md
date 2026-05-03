# Output schema contract

The benchmark writes two raw prediction files and two summary files to
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

Task-specific label columns are wide and sparse: each row carries only the
`pred_*` and `gt_*` columns relevant to its task, and all unrelated task columns
are null.

Active task label columns:

| Task | Label kind | Prediction / ground-truth columns |
|---|---|---|
| `gilardi_relevance` | binary | `pred_relevant`, `gt_relevant` |
| `ballard_incivility` | binary | `pred_uncivil`, `gt_uncivil` |
| `rheault_line_of_fire_incivility` | binary | `pred_uncivil`, `gt_uncivil` |
| `brandt_political_relevance` | binary | `pred_relevant`, `gt_relevant` |
| `gilardi_stance` | categorical | `pred_stance`, `gt_stance` |
| `chae_semeval_stance` | categorical | `pred_stance`, `gt_stance` |
| `ornstein_scotus_sentiment` | categorical | `pred_sentiment`, `gt_sentiment` |
| `halterman_ccc_protest` | categorical | `pred_protest_type`, `gt_protest_type` |
| `halterman_keith_bfrs` | categorical | `pred_event_type`, `gt_event_type` |
| `haunss_papea_fgz_forms` | categorical | `pred_protest_form`, `gt_protest_form` |
| `halterman_keith_cmp` | categorical | `pred_policy_domain`, `gt_policy_domain` |
| `osnabruegge_cross_domain_topic` | categorical | `pred_policy_domain`, `gt_policy_domain` |
| `wesleyan_creative_ads_2022` | categorical | `pred_tone`, `gt_tone` |
| `mellon_bes_mii_2024` | categorical | `pred_issue`, `gt_issue` |

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
| `avg_f1` | float | Mean F1 across labels for multi-class tasks |
| `accuracy` | float | Exact-match accuracy where defined |
| `mcc` | float | Matthews correlation coefficient |
| `headline_f1` | float | Unified task-level primary metric |
| `headline_f1_lo`, `headline_f1_hi` | float | 95% paired bootstrap interval |

`headline_f1` is defined as:

- positive-class F1 for binary tasks
- macro F1 across the task label set for categorical tasks

Cost note:

- `usd_per_1000` and `usd_per_1000_correct` are only as exact as the
  manifest-level `cost_per_call_usd` supplied for that model.
- For token-priced APIs such as OpenAI, Anthropic, and DeepSeek, exact billing
  should be taken from provider usage/cost reporting rather than inferred from
  these columns.

### `output/summary_batched.csv`

One row per `(task, model, batch_size)` from the batched benchmark.

| Column | Type | Description |
|---|---|---|
| `task`, `model`, `batch_size` | identifiers | Cell identifiers |
| `n` | int | Total attempted items |
| `parse_ok` | int | Items with clean parse |
| `parse_err_rate` | float | Share of items with parse errors |
| `mean_latency_s`, `median_latency_s` | float | Per-item latency summaries |
| `median_batch_latency_s` | float | Median wall time for the actual batched call |
| `f1_*` | float | Per-label F1 on cleanly parsed items |
| `avg_f1` | float | Mean F1 across labels for multi-class tasks |
| `accuracy` | float | Exact-match accuracy where defined |
| `headline_f1` | float | Unified task-level primary metric |
| `agreement_vs_b1` | float | Share of items whose prediction matches the same model at `batch_size == 1` |

## Merging semantics

`--merge-into PATH` in [`code/benchmark.py`](../code/benchmark.py) replaces any
rows in the existing CSV that share `(task, model, item_id)` with the newly
generated rows, preserving all other rows. This supports selective reruns after
prompt or parser changes without redoing the full serial grid.
