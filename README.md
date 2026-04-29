# polsci-open-bench

A small benchmark of LLMs for political science text classification.

The benchmark compares four local open-weight models, run through Ollama, against three commercial API models, two from OpenAI and one from Anthropic, on ten classification tasks from published political science replication archives.

The goal is narrower than a general leaderboard: can local models be useful for applied political-text coding, and what tradeoffs appear in practice?

## Main outputs

- PDF report: [`output/report_pdf.pdf`](output/report_pdf.pdf)
- Serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)

## Headline findings

Main takeaway: local open-weight models are competitive on many tasks, but the benchmark does not support a simple local-versus-API ranking. Task family, metric choice, and batching strategy matter more.

![Mean macro F1 across ten tasks per model](output/figures/fig-mean-f1.png)

- **Local models are competitive on average.** The top three models, `gpt-5.5`, `claude-sonnet-4-6`, and `gemma4:31b`, are within 0.002 mean macro F1, and all seven within roughly 0.05.
- **No model wins everywhere.** Top point estimates are split across four of the seven models. Task-family averages also do not separate local and API models cleanly.

![Mean macro F1 within task family, by model](output/figures/fig-family.png)

- **Local inference is not uniformly slower.** On the M2 Pro machine used here, local latency ranges from about 0.6 s/item for `qwen3:30b-a3b` to about 4.7 s/item for `gemma4:31b`.
- **Batching helps on short prompts.** At `b=10`, short prompt tasks usually see 1.2-2.7x speedups with limited F1 loss.
- **Batching is fragile on long codebooks.** Malformed output rates rise sharply on some long, multi-class tasks; `gpt-5.5` reaches 68% malformed output on Halterman CCC at `b=10`.
- **API costs differ sharply.** `gpt-5.4-nano` costs about $1.20 for 5,000 predictions, compared with $21.66 for `gpt-5.5`, with a 0.021 mean F1 lag.

## Scope

- **Tasks:** 10 political science classification tasks
- **Items:** 500 per task, fixed random seed
- **Models:** 4 local Ollama models + 3 commercial API models
- **Predictions:** 35,000 serial + 44,500 batched
- **Metrics:** macro F1, accuracy, MCC, latency, malformed-output rate
- **Local hardware:** Apple M2 Pro, 32 GB unified memory, macOS Tahoe 26.1, Ollama 0.19.0

## What this benchmark is, and is not

This benchmark is for applied researchers considering local LLMs for political-text classification. It focuses on accuracy, speed, batching, and task heterogeneity under a realistic laptop setup.

It is not a universal LLM leaderboard. The model set is practical rather than exhaustive, prompts are not optimized separately for each model, and the cross-task average is descriptive. For a new project, validate a few candidate models on labeled examples from the target task before scaling up.

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

Install dependencies and set API keys:

```bash
pip install openai anthropic httpx openpyxl pandas pyyaml scikit-learn
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Run the serial benchmark and rebuild summaries:

```bash
python code/benchmark.py
python code/build_summary.py
```

Run the batched benchmark:

```bash
python code/batch_benchmark.py
python code/build_summary_batched.py
```

Render the report:

```bash
quarto render output/report_pdf.qmd
```

See [`docs/reproduce.md`](docs/reproduce.md) for Ollama setup, model pulls, selective reruns, and output schema.

## Repo layout

```
code/      benchmark runners and summary builders
data/      cleaned per-task CSVs
prompts/   one prompt file per task
output/    predictions, summaries, and reports
docs/      prompt provenance, schema, reproduction notes, public thread
```

## Add a task

1. Add a cleaned task CSV to `data/`.
2. Add a prompt to `prompts/`.
3. Add a task config and loader in `code/benchmark.py`.
4. Run `python code/benchmark.py --only-task {task_name}`.
5. Rebuild summaries with `python code/build_summary.py`.

## Citation

If you use this benchmark in academic work, please cite both the report and the source papers for the individual tasks (listed in [`docs/prompts_provenance.md`](docs/prompts_provenance.md)).

## License

Code: MIT (see [`LICENSE`](LICENSE)).

Data: each task CSV is derived from a publicly distributed replication archive. Task-level licenses inherit from the source paper.
