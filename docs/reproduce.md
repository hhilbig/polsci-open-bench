# Reproduction guide

Detailed setup, model pulls, report dependencies, custom-task support, and
instructions for adding a new task.

## Verified local tooling

The current release was verified locally with:

- `Python 3.14.3`
- `R 4.5.2`
- `Quarto 1.9.27`

The checked-in benchmark outputs were produced on Apple Silicon with Ollama
0.19.0 and the hardware noted in the README.

## Python environment

Create and activate a virtual environment, then install the pinned benchmark
dependencies from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The serial and batched benchmark runners depend on the packages in
[`requirements.txt`](../requirements.txt).

## Task manifests

Built-in benchmark tasks live in YAML manifests under [`tasks/`](../tasks).
The generated task-count inventory is [`docs/task_inventory.md`](task_inventory.md):
it records the public 34-task benchmark and output coverage.

You can also run a custom task without editing core code by supplying either:

- `--task-manifest path/to/task.yaml`
- `--task-dir path/to/task_directory`
- `--tasks-dir path/to/directory_of_manifests`

See [`docs/custom_tasks.md`](custom_tasks.md) for the manifest schema and
[`examples/minimal_custom_task`](../examples/minimal_custom_task) for a minimal
working example.

After adding, removing, or changing task manifests, refresh the inventory:

```bash
python3 code/task_inventory.py --write
python3 code/task_inventory.py --check
```

## Model manifests

Built-in benchmark models now live in YAML manifests under [`models/`](../models).
You can also run a custom model set without editing core code by supplying
either:

- `--model-manifest path/to/model.yaml`
- `--models-dir path/to/directory_of_model_manifests`

See [`docs/custom_models.md`](custom_models.md) for the manifest schema and
[`examples/minimal_custom_models`](../examples/minimal_custom_models) for a
minimal working example.

## Report environment

The PDF report is rendered from [`output/report_pdf.qmd`](../output/report_pdf.qmd)
with Quarto plus these R packages:

- `tidyverse`
- `knitr`
- `ggrepel`
- `scales`
- `haschaR`
- `kableExtra`

Install them with:

```bash
Rscript -e "install.packages(c('tidyverse','knitr','ggrepel','scales','haschaR','kableExtra'), repos='https://cloud.r-project.org')"
```

The report QMD is intentionally prose-only: figures and tables are generated
first, then loaded by the QMD. Rebuild the report assets and render with:

```bash
Rscript code/build_report_assets.R
quarto render output/report_pdf.qmd
```

## Backends

### Ollama (local)

The benchmark was run with Ollama 0.19.0 on Apple Silicon (M2 Pro, 32 GB
unified memory). Most local manifests use `Q4_K_M`; `gemma4:26b` uses its
published local Ollama tag.

Pull the five local models used in the checked-in release benchmark:

```bash
ollama pull gemma4:26b
ollama pull gemma4:31b-it-q4_K_M
ollama pull qwen3:14b-q4_K_M
ollama pull qwen3:30b-a3b-q4_K_M
ollama pull mistral-small:24b-instruct-2501-q4_K_M
```

The runner reads `OLLAMA_URL` if you need to point it at a non-default host:

```bash
export OLLAMA_URL=http://localhost:11434
```

Local calls run with `temperature=0.1` and the model's internal thinking mode
disabled.

### OpenAI

```bash
export OPENAI_API_KEY=...
```

Both OpenAI tiers (`gpt-5.5`, `gpt-5.4-nano`) run with
`reasoning_effort=medium` and a strict JSON schema.

OpenAI-compatible local servers can also be used through the same backend by
setting `backend: openai` plus a `base_url` in the model manifest.

For cost tracking:

- per-request responses include token `usage`
- reconciled spend is better taken from OpenAI's organization Usage / Costs API
  or the Usage Dashboard Costs tab
- the benchmark's `usd_per_1000` columns are not direct OpenAI billing unless
  you deliberately set a benchmark-specific `cost_per_call_usd` in the manifest

### DeepSeek

The built-in DeepSeek manifest uses the same `openai` backend with DeepSeek's
OpenAI-compatible endpoint:

```bash
export DEEPSEEK_API_KEY=...
```

If `DEEPSEEK_API_KEY` is not set, the runner skips the built-in DeepSeek model
with a notice instead of writing error rows.

The live manifest runs `deepseek-v4-pro` in non-thinking mode and uses
`response_format={"type": "json_object"}` rather than OpenAI-style strict
`json_schema`, because that is the compatibility path documented in DeepSeek's
official API docs.

For cost tracking:

- DeepSeek responses include a `usage` object
- pricing is token-based and distinguishes prompt cache hits from misses
- use the returned `usage` plus the current DeepSeek pricing page for exact
  run-level accounting
- as with other token-priced APIs, `cost_per_call_usd` in a manifest is only a
  benchmark-side approximation

### Anthropic

```bash
export ANTHROPIC_API_KEY=...
```

If `ANTHROPIC_API_KEY` is unset, the runner falls back to
`~/.anthropic_api_key` (file mode 0600). Anthropic calls use tool-use forcing
to constrain outputs to the same schema as the OpenAI path.

For cost tracking:

- Messages responses include `usage.input_tokens` and `usage.output_tokens`
- reconciled spend is available from Anthropic's Usage & Cost Admin API or the
  Console
- Anthropic's Admin API is organization-only, not available for individual
  accounts
- the benchmark's `usd_per_1000` columns are not direct Anthropic billing
  unless you intentionally supply a benchmark-specific `cost_per_call_usd`

### Cost columns inside the repo

`build_summary.py` computes two cost-like columns:

- `gpu_hours_per_1000` for models with `compute_class: local`
- `usd_per_1000` for models whose manifest provides `cost_per_call_usd`

These are useful benchmark summaries, but they should not be confused with
provider-reconciled spend for token-priced APIs. For exact billing, prefer the
provider-side paths above.

## Main commands

Run the full serial benchmark and rebuild the serial summary:

These commands overwrite the checked-in release outputs and can call paid API
models if API keys are set. Use a custom `--output` path and explicit
`--model-manifest` for smoke tests or custom benchmarks.

```bash
python3 code/benchmark.py
python3 code/build_summary.py
```

Run the batched benchmark and rebuild the batched summary:

```bash
python3 code/batch_benchmark.py
python3 code/build_summary_batched.py
```

Build the task-length audit and lightweight relative-performance note:

```bash
python3 code/build_task_length_audit.py
```

## Selective reruns

Useful when you have edited one prompt or want to fill in a single
`(task, model)` cell rather than rerunning the whole grid:

Commands that omit `--model-manifest` use the built-in model set. If API keys
are set, those commands can call paid API models. For no-cost checks, use an
explicit local manifest and a custom `--output` path.

```bash
# Run one task across all models
python3 code/benchmark.py --only-task gilardi_stance

# Run one (task, model) cell and merge into the existing predictions CSV
python3 code/benchmark.py \
  --only-model qwen3:30b-a3b-q4_K_M \
  --only-task halterman_ccc_protest \
  --merge-into output/predictions.csv

# Run only newly sampled items for a task/model cell
python3 code/benchmark.py \
  --only-model qwen3:30b-a3b-q4_K_M \
  --only-task halterman_ccc_protest \
  --only-new-items \
  --merge-into output/predictions.csv

# Batched run for a single task
python3 code/batch_benchmark.py \
  --only-task gilardi_relevance \
  --batch-sizes 10,20

# Run a custom task directory through the same pipeline
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv

# Run a custom task against a custom model manifest
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv

# Batched run with a custom model manifest
python3 code/batch_benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/my_ollama_model.yaml \
  --output output/custom_predictions_batched.csv
```

`examples/minimal_custom_models/my_ollama_model.yaml` is a template. Replace
`my-local-ollama-model` with an Ollama model installed on your machine before
using that manifest.

For the bundled stub smoke test, success means four rows in
`output/custom_predictions.csv` and one row in `output/custom_summary.csv`.
The most important summary fields are `parse_ok`, `parse_err_rate`,
`headline_f1`, `median_latency_s`, and `usd_per_1000`.

Rebuild summaries after any rerun:

```bash
python3 code/build_summary.py
python3 code/build_summary_batched.py
```

## Output schema

Column definitions for the predictions and summary CSVs are in
[`schema.md`](schema.md).

Latency is part of the reproducibility contract. After any serial or batched
rerun, clean prediction rows should have positive `latency_s`; batched rows
should also have positive `batch_latency_s`. The summary builders use these
fields for `mean_latency_s`, `median_latency_s`, and batched wall-time summaries.

## Adding a new task

1. Add a cleaned task CSV to `data/`.
2. Add a prompt to `prompts/`.
3. Add a task manifest under `tasks/` or a self-contained task directory with `task.yaml`.
4. Run `python3 code/benchmark.py --only-task {task_name}`.
5. Rebuild summaries with `python3 code/build_summary.py`.
6. Update [`docs/prompts_provenance.md`](prompts_provenance.md) and
   [`docs/schema.md`](schema.md) if the task changes prompt provenance or
   output columns.

See the built-in manifests in [`tasks/`](../tasks) for examples.

For a no-cost smoke test, first run
`python3 examples/local_openai_stub_server.py --max-requests 4` in another shell,
then run the custom-task commands above.

## Adding a new model

1. Add a model manifest under `models/` or create a self-contained model YAML elsewhere.
2. If needed, set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or a backend-specific key env var named in the manifest.
3. Run `python3 code/benchmark.py --only-model {model_name}` or point the runner at the new manifest with `--model-manifest`.
4. Rebuild summaries with `python3 code/build_summary.py`.
5. Confirm the resulting clean prediction rows have positive `latency_s`; this
   is required for model comparisons.
6. If you want rough benchmark USD columns populated, add
   `cost_per_call_usd` to the model manifest.
7. If the model is token-priced and you need exact spend, keep provider-side
   accounting separately from the manifest-level `cost_per_call_usd`.

See the built-in manifests in [`models/`](../models) for examples.
