# polsci-open-bench

A small benchmark of LLMs for political science text classification.

The checked-in benchmark outputs compare four local open-weight models, run through Ollama, against three commercial API models, two from OpenAI and one from Anthropic, on ten classification tasks from published political science replication archives. The live built-in task library now includes four additional tasks, `osnabruegge_cross_domain_topic`, `rheault_line_of_fire_incivility`, `haunss_papea_fgz_forms`, and `brandt_political_relevance`, which are ready for future reruns but are not yet included in the committed prediction and report artifacts. The live built-in model library now also includes `deepseek-v4-pro` for future API-side comparisons and a prepared `gemma4:26b` local manifest for the `v2` expansion branch.

The goal is narrower than a general leaderboard: can local models be useful for applied political-text coding, and what tradeoffs appear in practice?

## Main outputs

- PDF report: [`output/report_pdf.pdf`](output/report_pdf.pdf)
- Serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)

## Headline findings

Main takeaway: local open-weight models are competitive on many tasks, but the benchmark does not support a simple local-versus-API ranking. Task family, metric choice, and batching strategy matter more.

**Figure 1. Mean macro F1 across ten tasks per model.**

![Mean macro F1 across ten tasks per model](output/figures/fig-mean-f1.png)

- **Local models are competitive on average.** The top three models, `gpt-5.5`, `claude-sonnet-4-6`, and `gemma4:31b`, are within 0.002 mean macro F1, and all seven within roughly 0.05.
- **No model wins everywhere.** Top point estimates are split across four of the seven models. Task-family averages also do not separate local and API models cleanly.

**Figure 2. Mean macro F1 within task family, by model.**

![Mean macro F1 within task family, by model](output/figures/fig-family.png)

- **Local inference is not uniformly slower.** On the M2 Pro machine used here, local latency ranges from about 0.6 s/item for `qwen3:30b-a3b` to about 4.7 s/item for `gemma4:31b`.
- **Batching helps on short prompts.** At `b=10`, short prompt tasks usually see 1.2-2.7x speedups with limited F1 loss.
- **Batching is fragile on long codebooks.** Malformed output rates rise sharply on some long, multi-class tasks; `gpt-5.5` reaches 68% malformed output on Halterman CCC at `b=10`.
- **API costs differ sharply.** `gpt-5.4-nano` costs about $1.20 for 5,000 predictions, compared with $21.66 for `gpt-5.5`, with a 0.021 mean F1 lag.

## Scope

- **Tasks in committed outputs:** 10 political science classification tasks
- **Built-in task manifests:** 14 tasks
- **Items:** typically 500 per task, fixed random seed; smaller public extension tasks use all available rows
- **Models in committed outputs:** 4 local Ollama models + 3 commercial API models
- **Built-in model manifests:** 8 models
- **Predictions:** 35,000 serial + 44,500 batched
- **Metrics:** macro F1, accuracy, MCC, latency, malformed-output rate
- **Local hardware:** Apple M2 Pro, 32 GB unified memory, macOS Tahoe 26.1, Ollama 0.19.0

## What this benchmark is, and is not

This benchmark is for applied researchers considering local LLMs for political-text classification. It is not a universal leaderboard: the model set is practical rather than exhaustive, prompts are not optimized separately for each model, and the cross-task average is descriptive. For a new project, validate a few candidate models on labeled examples from the target task before scaling up.

## Tasks

| Task | Source | Family | Labels |
|---|---|---|---|
| `gilardi_relevance` | Gilardi et al. 2023 | Relevance / Incivility | binary |
| `ballard_incivility` | Ballard 2022 | Relevance / Incivility | binary |
| `rheault_line_of_fire_incivility` | Rheault, Rayment, & Musulan 2019 | Relevance / Incivility | binary |
| `brandt_political_relevance` | Brandt et al. 2025 | Relevance / Incivility | binary |
| `gilardi_stance` | Gilardi et al. 2023 | Sentiment / Stance / Tone | 3-class |
| `chae_semeval_stance` | Chae & Davidson 2025 | Sentiment / Stance / Tone | 3-class |
| `ornstein_scotus_sentiment` | Ornstein et al. 2025 | Sentiment / Stance / Tone | 3-class |
| `wesleyan_creative_ads_2022` | Zhang et al. 2025 | Sentiment / Stance / Tone | 3-class |
| `halterman_ccc_protest` | Halterman & Keith 2025 | Event coding | 4-class |
| `halterman_keith_bfrs` | Halterman & Keith 2025 | Event coding | 12-class |
| `haunss_papea_fgz_forms` | Haunss et al. 2025 | Event coding | 7-class |
| `halterman_keith_cmp` | Halterman & Keith 2025 + CMP | Policy-topic coding | 7-class |
| `osnabruegge_cross_domain_topic` | Osnabruegge, Ash, & Morelli 2023 | Policy-topic coding | 8-class |
| `mellon_bes_mii_2024` | Mellon et al. 2024 | Policy-topic coding | 50-class |

Prompt provenance is documented in [`docs/prompts_provenance.md`](docs/prompts_provenance.md).
The committed benchmark outputs still cover the original 10-task run; the new
Cross-Domain, Line-of-Fire, PAPEA, and Brandt relevance tasks are present in
the live manifests for future reruns.

## Models

| Model | Backend | Notes |
|---|---|---|
| `gemma4:31b-it-q4_K_M` | Ollama | 31B dense, 4-bit |
| `gemma4:26b` | Ollama | 26B A4B Gemma 4 MoE, prepared on `v2-benchmark-expansion` but not yet scored |
| `qwen3:14b-q4_K_M` | Ollama | 14B dense, 4-bit |
| `qwen3:30b-a3b-q4_K_M` | Ollama | 30B MoE, about 3B active, 4-bit |
| `mistral-small:24b-instruct-2501-q4_K_M` | Ollama | 24B dense, 4-bit |
| `gpt-5.5` | OpenAI | flagship, `reasoning_effort=medium` |
| `gpt-5.4-nano` | OpenAI | small/cheap, `reasoning_effort=medium` |
| `deepseek-v4-pro` | DeepSeek API | current DeepSeek flagship via OpenAI-compatible endpoint, run in non-thinking mode with JSON-object output |
| `claude-sonnet-4-6` | Anthropic | flagship-tier |

Local models ran with thinking off. OpenAI calls used strict structured JSON outputs; Anthropic calls used tool-use forcing for the same schema.

## Cost notes

- The benchmark's `usd_per_1000` columns are manifest-derived summaries, not
  direct billing reconciliation. They are populated only when a model manifest
  provides `cost_per_call_usd`.
- For token-priced API models, exact billed spend is better taken from the
  provider directly:
  - OpenAI: use the API `usage` fields for token counts and the organization
    Usage / Costs endpoints or dashboard for reconciled spend.
  - Anthropic: use the Messages API `usage` fields for token counts and the
    Usage & Cost Admin API or Console for reconciled spend. Anthropic's Admin
    API is organization-only, not individual-account.
  - DeepSeek: use the returned `usage` object plus the current pricing page;
    DeepSeek bills per input and output token and also distinguishes prompt
    cache hits from cache misses.
- For local models, the benchmark reports `gpu_hours_per_1000` instead of USD.
  If you want local dollar estimates, add your own benchmark-specific
  `cost_per_call_usd` to the manifest or compute a hardware/electricity cost
  outside the benchmark.

Built-in models are defined in YAML manifests under [`models/`](models), and
the benchmark can also load custom model manifests at the command line. See
[`docs/custom_models.md`](docs/custom_models.md).

## Reproduce

Create a Python environment, install dependencies, and set API keys:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...   # optional; required only to include DeepSeek
```

Run the serial benchmark and rebuild summaries:

```bash
python3 code/benchmark.py
python3 code/build_summary.py
```

Run the batched benchmark:

```bash
python3 code/batch_benchmark.py
python3 code/build_summary_batched.py
```

Run your own task from a local task directory:

```bash
python3 code/benchmark.py --task-dir examples/minimal_custom_task --output output/custom_predictions.csv
python3 code/build_summary.py --task-dir examples/minimal_custom_task --predictions output/custom_predictions.csv --output output/custom_summary.csv
```

Run your own task against your own model manifest:

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

Build the task-length audit:

```bash
python3 code/build_task_length_audit.py
```

Render the report:

```bash
quarto render output/report_pdf.qmd
```

The PDF report also needs Quarto plus the R packages loaded in
[`output/report_pdf.qmd`](output/report_pdf.qmd).

See [`docs/reproduce.md`](docs/reproduce.md) for setup, selective reruns, report
dependencies, output schema, and adding new tasks.

## Repo layout

```
code/      benchmark runners and summary builders
data/      cleaned task files
models/    model manifests for the built-in benchmark
tasks/     task manifests for the built-in benchmark
prompts/   task prompts
output/    predictions, summaries, reports
docs/      provenance, schema, reproduction notes
examples/  minimal custom-task and custom-model examples
```

## Roadmap

Near-term next steps are tracked in [`TODO.md`](TODO.md). The main planned
extensions are broader task coverage within each existing family, a new
long-document family, and a more unified evaluation framework across political
science benchmarks.

## Citation

If you use this benchmark in academic work, please cite both the report and the source papers for the individual tasks (listed in [`docs/prompts_provenance.md`](docs/prompts_provenance.md)).

## License

Code: MIT (see [`LICENSE`](LICENSE)).

Data: each task CSV is derived from a publicly distributed replication archive. Task-level licenses inherit from the source paper.
