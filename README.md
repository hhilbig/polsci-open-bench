# polsci-open-bench

A small benchmark of LLMs for political-science text classification.

The benchmark compares four local open-weight models, run through Ollama, against two OpenAI API models on ten classification tasks from published political-science replication archives. The goal is practical guidance for applied researchers, not a definitive model leaderboard.

## Main outputs

- Short PDF report: [`output/report_pdf.pdf`](output/report_pdf.pdf)
- Full HTML report: [`output/report.html`](output/report.html)
- Serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)

## Headline findings

Across the ten tasks, the top three models (gpt-5.5, Claude Sonnet 4.6, Gemma 4 31B) sit within 0.002 macro F1 of each other on the cross-task average, with per-task winners split across four of the seven models. Local inference can match or beat the commercial API tier on speed, and batching helps on short prompt tasks but raises the rate of malformed output on tasks with long, multi-class codebooks.

![Mean macro F1 across ten tasks per model. Large filled circles show cross-task means; small dots show per-task macro F1 values.](output/figures/fig-mean-f1.png)

- **Local models are competitive on average.** Top three (gpt-5.5, Claude Sonnet 4.6, Gemma 4 31B) within 0.002 mean macro F1; all seven within roughly 0.05.
- **No model wins everywhere.** Per-task winners split across four of the seven models; family-level patterns do not show a simple local-versus-API hierarchy.
- **Local inference is not uniformly slower.** Per-item latency on the M2 Pro ranges from about 0.6 s (Qwen3-30B-A3B, MoE) to about 4.7 s (Gemma 4 31B, dense). Parameter count is a poor guide to speed.
- **Batching helps on short prompts, hurts on long codebooks.** At b=10 short prompt tasks see 1.2-2.7x speedup with little F1 loss. On long codebooks, malformed-JSON rates climb sharply (gpt-5.5 hits 68% on Halterman CCC at b=10).
- **Cost.** gpt-5.4-nano is roughly 17 times cheaper than gpt-5.5 ($1.20 vs $21.66 for 5,000 predictions) at a 0.021 F1 lag. Local models avoid per-item charges.

Full discussion, metric robustness checks, and per-task tables are in [`output/report_pdf.pdf`](output/report_pdf.pdf).

## Scope

- **Tasks:** 10 political-science classification tasks
- **Items:** 500 per task (random seed 20260422)
- **Models:** 4 local Ollama models + 3 commercial API models (2 OpenAI, 1 Anthropic)
- **Predictions:** 35,000 serial + 44,500 batched
- **Metrics:** macro F1, accuracy, MCC, latency, parse-error rate
- **Local hardware:** Apple M2 Pro, 32 GB unified memory, macOS Tahoe 26.1, Ollama

## What this benchmark is for

This benchmark is intended to help applied researchers decide whether local models are worth testing for political-text classification workflows. It is most useful for questions such as:

- Are local models competitive with commercial ones on common political science classification tasks?
- How long do they take, and can they run on a single laptop?
- Does batching speed them up?
- How much does the answer vary across tasks?

## What this benchmark is not

This is not a universal LLM leaderboard. The model set is convenience-selected, prompts are not separately optimized for each model, and the task average is descriptive rather than an estimand. For a new project, the safest workflow is still to validate a few candidate models on a task-specific labeled sample.

## Tasks

| Task | Source | Family | Labels |
|---|---|---|---|
| `gilardi_relevance` | Gilardi et al. 2023 | Relevance / Incivility | binary |
| `ballard_incivility` | Ballard 2022 | Relevance / Incivility | binary |
| `gilardi_stance` | Gilardi et al. 2023 | Sentiment / Stance / Tone | 3-class |
| `chae_semeval_stance` | Chae & Davidson 2025 | Sentiment / Stance / Tone | 3-class |
| `ornstein_scotus_sentiment` | Ornstein et al. 2025 | Sentiment / Stance / Tone | 3-class |
| `wesleyan_creative_ads_2022` | Zhang et al. 2025 | Sentiment / Stance / Tone | 3-class |
| `halterman_ccc_protest` | Halterman & Keith 2025 | Event coding | 4-class |
| `halterman_keith_bfrs` | Halterman & Keith 2025 | Event coding | 12-class |
| `halterman_keith_cmp` | Halterman & Keith 2025 + CMP | Policy-topic coding | 7-class |
| `mellon_bes_mii_2024` | Mellon et al. 2024 | Policy-topic coding | 50-class |

Prompt provenance is documented in [`docs/prompts_provenance.md`](docs/prompts_provenance.md).

## Models

| Model | Backend | Notes |
|---|---|---|
| `gemma4:31b-it-q4_K_M` | Ollama | 31B dense, 4-bit |
| `qwen3:14b-q4_K_M` | Ollama | 14B dense, 4-bit |
| `qwen3:30b-a3b-q4_K_M` | Ollama | 30B MoE, about 3B active, 4-bit |
| `mistral-small:24b-instruct-2501-q4_K_M` | Ollama | 24B dense, 4-bit |
| `gpt-5.5` | OpenAI | flagship, `reasoning_effort=medium` |
| `gpt-5.4-nano` | OpenAI | small/cheap, `reasoning_effort=medium` |
| `claude-sonnet-4-6` | Anthropic | flagship-tier |

Local models ran with thinking off. OpenAI calls used strict structured JSON outputs; Anthropic calls used tool-use forcing for the same schema.

## Reproduce

Install dependencies:

```bash
pip install openai httpx openpyxl pandas pyyaml scikit-learn
```

Set up backends:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...   # or save to ~/.anthropic_api_key (mode 0600)
export OLLAMA_URL=http://localhost:11434  # optional; default shown
ollama pull gemma4:31b-it-q4_K_M
ollama pull qwen3:14b-q4_K_M
ollama pull qwen3:30b-a3b-q4_K_M
ollama pull mistral-small:24b-instruct-2501-q4_K_M
```

The benchmark was run with Ollama 0.19.0; quantization is `Q4_K_M` for all four local models (the suffix in the model names).

```bash
pip install anthropic    # in addition to the dependencies above
```

Run the serial benchmark:

```bash
python code/benchmark.py
python code/build_summary.py
```

Run the batched benchmark:

```bash
python code/batch_benchmark.py
python code/build_summary_batched.py
```

Render reports:

```bash
quarto render output/report.qmd
quarto render output/report_pdf.qmd
```

<details>
<summary>Advanced: selective reruns</summary>

```bash
# Run one task across all models (after editing a prompt)
python code/benchmark.py --only-task gilardi_stance

# Run one (task, model) cell and merge into the existing predictions CSV
python code/benchmark.py \
  --only-model qwen3:30b-a3b-q4_K_M \
  --only-task halterman_ccc_protest \
  --merge-into output/predictions.csv

# Batched run for a single (task, model) cell
python code/batch_benchmark.py \
  --only-task gilardi_relevance \
  --batch-sizes 10,20
```
</details>

See [`docs/schema.md`](docs/schema.md) for output CSV column definitions.

## Repo layout

```
code/
  benchmark.py              serial runner
  batch_benchmark.py        batched runner (b in {1, 10, 20})
  build_summary.py          builds summary.csv
  build_summary_batched.py  builds summary_batched.csv
data/                       cleaned per-task CSVs
prompts/                    one prompt file per task
output/                     predictions, summaries, reports
docs/                       prompt provenance, schema, twitter thread
```

## Add a task

1. Add a cleaned task CSV to `data/`.
2. Add a prompt to `prompts/`.
3. Add a task config and loader in `code/benchmark.py`.
4. Run `python code/benchmark.py --only-task {task_name}`.
5. Rebuild summaries with `python code/build_summary.py`.

See existing task loaders for examples.

## Citation

If you use this benchmark in academic work, please cite both the report and the source papers for the individual tasks (listed in the Tasks table above and in [`docs/prompts_provenance.md`](docs/prompts_provenance.md)).

## License

Code: MIT (see [`LICENSE`](LICENSE)).

Data: each task CSV is derived from a publicly distributed replication archive. Task-level licenses inherit from the source paper.
