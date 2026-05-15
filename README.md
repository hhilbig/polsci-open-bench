# polsci-open-bench

A benchmark of local open-weight and commercial API LLMs for political science
text classification.

The benchmark compares five local models, run through Ollama, against four
commercial API models from OpenAI, Anthropic, and DeepSeek on 34 classification
tasks from published political science replication archives. The goal is not a
general LLM leaderboard. The goal is to help applied researchers decide whether
local models are useful for real text-coding workflows, and when API models,
batching, or task-specific validation still matter.

## Main Outputs

- PDF report: [`output/report_pdf.pdf`](output/report_pdf.pdf)
- Canonical serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Canonical serial summary: [`output/summary.csv`](output/summary.csv)
- Canonical prompt-batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)
- Canonical prompt-batched summary: [`output/summary_batched.csv`](output/summary_batched.csv)
- Local b=10 summary: [`output/summary_batched_local_b10.csv`](output/summary_batched_local_b10.csv)
- Task inventory: [`docs/task_inventory.md`](docs/task_inventory.md)
- Reproduction guide: [`docs/reproduce.md`](docs/reproduce.md)

## Headline Findings

Main takeaway: local open-weight models are competitive on many tasks, but the
benchmark does not support a single global model ranking. Task-specific
validation matters.

**Figure 1. Mean headline F1 across 34 tasks per model.**

![Mean headline F1 across thirty-four tasks per model](output/figures/fig-mean-f1.png)

- The four strongest models, `claude-sonnet-4-6`, `gpt-5.5`,
  `gemma4:31b-it-q4_K_M`, and `deepseek-v4-pro`, are within 0.021 mean
  headline F1.
- The best local model matches or exceeds the best API model on 9 of 34 tasks.
  The average best-local-minus-best-API gap is -0.015 headline F1.
- Top point estimates are split across eight of the nine models, so average
  rank is a weak guide to model choice for a specific task.

**Figure 2. Best API minus best local headline F1 by task.**

![Best API minus best local headline F1 by task](output/figures/fig-best-local-api-gap.png)

- API models have their clearest edge on complex tasks with many active labels,
  long codebooks, or multi-output schemas.
- Local inference is practical on one 32 GB Apple Silicon laptop. On a typical
  500-item task, median single-item runtimes range from about 5 minutes for
  Qwen3 30B-A3B to about 29 minutes for Gemma 4 31B.
- Prompt batching at b=10 covers all 34 tasks for the five local models. It
  improves median local throughput by about 1.23x to 1.86x, but some
  model-task cells have high schema or label failure rates.

## Scope

- **Tasks:** 34 canonical task manifests in [`tasks/`](tasks)
- **Models:** 5 local Ollama models and 4 commercial API models
- **Serial coverage:** 34 tasks x 9 models = 306 model-task cells
- **Local b=10 coverage:** 34 tasks x 5 local models = 170 model-task cells
- **Items:** 293 to 500 per task, depending on cleaned data availability
- **Metrics:** headline F1, accuracy, MCC, per-item latency, schema/label
  failure rate
- **Local hardware:** Apple M2 Pro, 32 GB unified memory, macOS Tahoe 26.1,
  Ollama 0.19.0

The generated source of truth for task counts is
[`docs/task_inventory.md`](docs/task_inventory.md). Regenerate it with:

```bash
python3 code/task_inventory.py --write
python3 code/task_inventory.py --check
```

## What This Benchmark Is

This benchmark is for prompt-based annotation workflows in political science
and adjacent social-science text-as-data work. It asks whether researchers can
often get API-level performance locally, especially when cost, restricted data,
privacy, or reproducibility make commercial APIs less attractive.

It is not a replacement for supervised baselines when large labeled datasets
exist. It also does not prove that one model is best in general. For a new
project, screen several candidate models on labeled examples from the target
task before production coding.

## Models

| Model | Backend | Notes |
|---|---|---|
| `gemma4:31b-it-q4_K_M` | Ollama | 31B dense, 4-bit |
| `gemma4:26b` | Ollama | 26B A4B Gemma 4 MoE |
| `qwen3:14b-q4_K_M` | Ollama | 14B dense, 4-bit |
| `qwen3:30b-a3b-q4_K_M` | Ollama | 30B MoE, about 3B active, 4-bit |
| `mistral-small:24b-instruct-2501-q4_K_M` | Ollama | 24B dense, 4-bit |
| `gpt-5.5` | OpenAI | flagship, `reasoning_effort=medium` |
| `gpt-5.4-nano` | OpenAI | low-cost tier, `reasoning_effort=medium` |
| `deepseek-v4-pro` | DeepSeek API | non-thinking mode, JSON-object output |
| `claude-sonnet-4-6` | Anthropic | flagship-tier |

Local models ran with thinking disabled. OpenAI calls used strict structured
JSON outputs; Anthropic calls used tool-use forcing for the same schema.

## Reproduce

Create a Python environment, install dependencies, and set API keys as needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Run the serial benchmark and rebuild canonical summaries:

```bash
python3 code/benchmark.py
python3 code/build_summary.py
```

Run prompt batching and rebuild batched summaries:

```bash
python3 code/batch_benchmark.py
python3 code/build_summary_batched.py
```

Build report assets and render the PDF:

```bash
Rscript code/build_report_assets.R
quarto render output/report_pdf.qmd
```

See [`docs/reproduce.md`](docs/reproduce.md) for backend setup, exact report
dependencies, cost notes, custom tasks, and custom model manifests.

## Repo Layout

```text
code/      benchmark runners, summary builders, report-asset builders
data/      cleaned task files
models/    model manifests for the built-in benchmark
tasks/     canonical task manifests for the built-in benchmark
prompts/   task prompts
output/    predictions, summaries, figures, tables, reports
docs/      provenance, schema, reproduction notes, project status
examples/  minimal custom-task and custom-model examples
```

## Citation

If you use this benchmark in academic work, please cite both the report and the
source papers for the individual tasks listed in
[`docs/prompts_provenance.md`](docs/prompts_provenance.md).

## License

Code: MIT (see [`LICENSE`](LICENSE)).

Data: each task CSV is derived from a publicly distributed replication archive.
Task-level licenses inherit from the source paper or source dataset.
