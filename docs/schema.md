# Output schema contract

The benchmark runner writes two CSVs to `output/`. Their column contracts are fixed (downstream tooling and the report depend on them).

## `output/predictions.csv` — long format, one row per (task × model × item)

| Column | Type | Required | Description |
|---|---|---|---|
| `task` | string | yes | Task name, one of `TASKS[i]["name"]` in `code/benchmark.py` |
| `model` | string | yes | Model name (Ollama tag or OpenAI model id) |
| `item_id` | string | yes | Stable per-task item id (e.g., tweet_id or `ccc_000`) |
| `latency_s` | float | yes | Wall-clock latency in seconds for this single API call |
| `eval_count` | int | opt | Output tokens generated (if the backend reports it) |
| `parse_error` | string | opt | NaN if parsed cleanly; else the error/exception string |
| `raw_content_preview` | string | opt | First 200 chars of the model's raw output (truncated) |
| `pred_<label>` | 0/1 or string or NaN | yes | Model's prediction for label. One column per label in the task's label schema. NaN if parse failed. |
| `gt_<label>` | 0/1 or string | yes | Ground-truth label (from the task's data file). One column per label. |

Task label schemas:

| Task | Label kind | Label columns |
|---|---|---|
| `state_adaptation` | multi_binary | `pred_adaptation`, `pred_mitigation`, …, `pred_reversal` (8 total) |
| `gilardi_relevance` | binary | `pred_relevant` |
| `gilardi_stance` | categorical | `pred_stance` ∈ {pro, neutral, contra} |
| `ballard_incivility` | binary | `pred_uncivil` |
| `ornstein_scotus_sentiment` | categorical | `pred_sentiment` ∈ {Positive, Negative, Neutral} |
| `halterman_ccc_protest` | categorical | `pred_protest_type` ∈ {PROTEST, RALLY, …, COUNTER_PROTEST} (8 total) |
| `chae_semeval_stance` | categorical | `pred_stance` ∈ {FAVOR, AGAINST, NONE} |

## `output/summary.csv` — one row per (task, model)

| Column | Type | Description |
|---|---|---|
| `task`, `model` | string | Identifiers |
| `n` | int | Total items attempted |
| `parse_ok` | int | Items with clean output (`parse_error` NaN) |
| `parse_err_rate` | float | Share of items with parse errors |
| `mean_latency_s` | float | Mean of per-item `latency_s` |
| `median_latency_s` | float | Median of per-item `latency_s` |
| `f1_<label>` | float | Per-label F1 on items with clean output |
| `avg_f1` | float | Mean F1 across the task's labels (for multi-label or multi-class tasks) |
| `accuracy` | float | Categorical tasks only — share of items where `pred == gt` |
| `headline_f1` | float | Unified primary metric for cross-task plotting (see below) |

**Headline F1 unification**:

| Task | Headline F1 column |
|---|---|
| `state_adaptation` | `avg_f1` across 8 binary labels |
| `gilardi_relevance` | `f1_relevant` (positive-class F1) |
| `ballard_incivility` | `f1_uncivil` (positive-class F1) |
| `gilardi_stance` | `avg_f1` across 3 stance classes |
| `ornstein_scotus_sentiment` | `avg_f1` across 3 sentiment classes |
| `halterman_ccc_protest` | `avg_f1` across 8 protest classes |
| `chae_semeval_stance` | `avg_f1` across 3 stance classes |

This metric is imperfect for cross-task comparison (different baselines, different chance rates). The report uses **ΔF1 vs gpt-5.4-nano** within each task as the primary comparison metric for cross-task plots.

## Merging semantics

`--merge-into PATH` in `code/benchmark.py` replaces any rows in the existing CSV that share `(task, model, item_id)` with the newly-generated rows, preserving all other rows untouched. This supports selective reruns (e.g., after updating a single task's prompt) without redoing the full benchmark.
