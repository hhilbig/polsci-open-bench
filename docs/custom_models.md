# Custom models

The benchmark can now load models from YAML manifests instead of requiring
edits to `code/benchmark.py` or `code/batch_benchmark.py`.

## Minimal interface

Use either:

- `--model-manifest path/to/model.yaml`
- `--models-dir path/to/directory_of_model_manifests`

The serial benchmark, batched benchmark, serial summary builder, and
task-length audit all accept these flags.

## Manifest schema

Required fields:

- `name`: model identifier passed to the backend
- `backend`: `ollama`, `openai`, or `anthropic`

Common optional fields:

- `display_name`: human-readable label
- `order`: sort order within the loaded model set
- `compute_class`: `local` or `api`
- `think`: whether to enable the model's internal thinking mode (Ollama only)
- `reasoning_effort`: passed through to OpenAI chat completions when supported
- `cost_per_call_usd`: optional benchmark-specific per-call cost used in `summary.csv`
- `ollama_url`: override Ollama host for this model
- `provider`: optional provider tag such as `deepseek`
- `base_url`: OpenAI-compatible endpoint for self-hosted servers
- `api_key_env`: environment variable name for the backend API key
- `api_key_file`: fallback file path for the API key
- `response_format_type`: optional OpenAI-compatible output mode, currently `json_schema` or `json_object`
- `thinking_mode`: optional provider-specific thinking switch, currently `enabled` or `disabled`

`compute_class` controls how `build_summary.py` and the task-length audit treat
the model:

- `local`: include in local-vs-API comparisons and compute `gpu_hours_per_1000`
- `api`: include in the API side of those comparisons

`backend` controls how the runner calls the model:

- `ollama`: local `/api/chat`
- `openai`: OpenAI chat completions, including OpenAI-compatible servers via `base_url`
- `anthropic`: Anthropic Messages API with tool-use forcing

## Latency tracking

Per-item latency is a core benchmark outcome. Every supported backend must
return a wall-clock latency measurement for each successful call so raw
prediction rows can populate `latency_s`.

Backend expectations:

- serial runs: `latency_s` is one item-level request-response wall time
- batched runs: `batch_latency_s` is one batch request-response wall time, and
  `latency_s` is `batch_latency_s / actual_batch_size`
- cleanly parsed rows must have positive, non-missing latency values
- rows that fail before a model response may have missing latency, but must keep
  the error in `parse_error`

If you add a new backend adapter later, treat latency capture as part of the
minimum interface alongside `content` and token-count metadata. Accuracy without
timing is not enough for this benchmark.

For remote OpenAI-compatible providers, set the provider-specific API key in
the manifest via `api_key_env`. If that key is unavailable at runtime, the
built-in benchmark runners skip the model with a notice. Local OpenAI-compatible
servers on `localhost` still work without a real API key.

DeepSeek is the main built-in case where extra provider metadata matters:

- `provider: deepseek`
- `response_format_type: json_object`
- `thinking_mode: disabled`

That keeps the runner on the compatibility path documented by DeepSeek's
official API docs, instead of assuming OpenAI's strict `json_schema` mode.

## Cost tracking

There are two different cost concepts in this repo:

- exact provider billing
- the benchmark's own `usd_per_1000` summary columns

They are not the same thing.

`cost_per_call_usd` is only a manifest-level average. It is useful when you
want a rough benchmark-side USD column in `summary.csv`, but it is not a direct
reconciliation against provider billing for token-priced APIs.

Provider-side guidance:

- `openai`
  - Per-request responses include token `usage`.
  - For reconciled spend, use OpenAI's organization Usage / Costs API or the
    Usage Dashboard Costs tab.
  - Pricing is token-based and model-specific, so exact spend should usually be
    computed from provider reporting rather than a static `cost_per_call_usd`.

- `anthropic`
  - Messages responses include `usage.input_tokens` and `usage.output_tokens`.
  - For reconciled spend, use Anthropic's Usage & Cost Admin API or the Console.
  - Anthropic's Admin API is organization-only, not available for individual
    accounts.
  - Claude Sonnet is token-priced, so exact billing is again better treated as
    provider-side accounting than as a fixed per-call constant.

- `openai` with `provider: deepseek`
  - DeepSeek's chat completion responses include a `usage` object with prompt
    and completion tokens, plus prompt cache hit / miss breakdowns.
  - DeepSeek bills per input and output token, and current pricing also differs
    by cache hit versus cache miss.
  - Use the returned `usage` object plus the current pricing page for exact
    run-level accounting.

- `ollama`
  - There is no provider bill. The benchmark reports `gpu_hours_per_1000` for
    local models instead.
  - If you want local USD columns, set a benchmark-specific `cost_per_call_usd`
    yourself.

Practical recommendation:

- leave `cost_per_call_usd` blank when you want provider-exact accounting to
  live outside the benchmark
- set `cost_per_call_usd` only when you intentionally want a rough benchmark
  summary column for cross-model comparison

Official references:

- OpenAI pricing and usage: https://platform.openai.com/docs/pricing/ and https://platform.openai.com/docs/api-reference/usage/costs
- Anthropic pricing and usage-cost APIs: https://docs.anthropic.com/en/docs/about-claude/pricing and https://docs.anthropic.com/en/api/usage-cost-api
- DeepSeek pricing and token usage: https://api-docs.deepseek.com/quick_start/pricing and https://api-docs.deepseek.com/quick_start/token_usage

## Examples

The bundled example in
[`examples/minimal_custom_models`](../examples/minimal_custom_models)
contains:

```yaml
name: my-local-ollama-model
display_name: My Local Ollama Model
backend: ollama
compute_class: local
think: false
```

Example for an OpenAI-compatible local server:

```yaml
name: qwen-openai-compatible
display_name: Qwen via local server
backend: openai
compute_class: local
base_url: http://localhost:8000/v1
api_key_env: LOCAL_OPENAI_KEY
```

## Commands

Run the bundled custom task against the bundled custom model:

```bash
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv
```

Run the batched benchmark with the same model:

```bash
python3 code/batch_benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --output output/custom_predictions_batched.csv

python3 code/build_summary_batched.py \
  --task-dir examples/minimal_custom_task \
  --predictions output/custom_predictions_batched.csv \
  --serial-predictions output/custom_predictions.csv \
  --output output/custom_summary_batched.csv
```
