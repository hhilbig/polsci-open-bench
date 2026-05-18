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
- Serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Serial summary: [`output/summary.csv`](output/summary.csv)
- Prompt-batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)
- Prompt-batched summary: [`output/summary_batched.csv`](output/summary_batched.csv)
- Local 10-item comparison summary: [`output/summary_batched_local_b10.csv`](output/summary_batched_local_b10.csv)
- Task inventory: [`docs/task_inventory.md`](docs/task_inventory.md)
- Reproduction guide: [`docs/reproduce.md`](docs/reproduce.md)

The raw prompt-batched file has mixed scope: it contains the full 34-task local
10-item prompt grid plus OpenAI diagnostic prompt-batching rows. The
report-ready local batching view is `output/summary_batched_local_b10.csv`.
`output/summary_batched.csv` also contains copied one-item-at-a-time baselines
for comparison, which is why its model and cell counts are larger than the raw
batched file.

## Headline Findings

Main takeaway: local open-weight models are competitive on many tasks, but the
benchmark does not support a single global model ranking. Task-specific
validation matters.

**Figure 1. Mean F1 across 34 tasks per model.**

![Mean F1 across thirty-four tasks per model](output/figures/fig-mean-f1.png)

- The four strongest models, `claude-sonnet-4-6`, `gpt-5.5`,
  `gemma4:31b-it-q4_K_M`, and `deepseek-v4-pro`, are within 0.021 mean
  F1.
- The best local model matches or exceeds the best API model on 9 of 34 tasks.
  On average, the best API model exceeds the best local model by 0.015 F1.
- Top point estimates are split across eight of the nine models, so average
  rank is a weak guide to model choice for a specific task.

**Figure 2. Best API minus best local F1 by task.**

![Best API minus best local F1 by task](output/figures/fig-best-local-api-gap.png)

- API models have their clearest edge on complex tasks with many active labels,
  long codebooks, or multiple outputs per item.
- Local inference is practical on one 32 GB Apple Silicon laptop. On a typical
  500-item task, median one-item-at-a-time runtimes range from about 5 minutes for
  Qwen3 30B-A3B to about 29 minutes for Gemma 4 31B.
- Prompt batching with 10 items per prompt covers all 34 tasks for the five local models. It
  reduces median local per-item runtime by about 1.23x to 1.86x, but some
  task-model pairs have high response-format or label failure rates.

## Scope

- **Tasks:** 34 public benchmark task manifests in [`tasks/`](tasks)
- **Models:** 5 local Ollama models and 4 commercial API models
- **Serial coverage:** 34 tasks x 9 models = 306 task-model pairs
- **Serial classifications:** 147,825 model-item classifications
- **Local 10-item prompt coverage:** 34 tasks x 5 local models = 170 task-model pairs
- **Items:** 293 to 500 per task, depending on cleaned data availability
- **Metrics:** main F1, accuracy, MCC, time per item, response-format/label
  failure rate
- **Local hardware:** Apple M2 Pro, 32 GB unified memory, macOS Tahoe 26.1,
  Ollama 0.19.0

The generated source of truth for task counts is
[`docs/task_inventory.md`](docs/task_inventory.md). Regenerate it with:

```bash
python3 code/task_inventory.py --write
python3 code/task_inventory.py --check
```

In the CSVs, the report's "main F1" is the column `headline_f1`. The older
column name is retained for compatibility. Use `headline_f1` rather than
`avg_f1` when reproducing the report figures.

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

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Set API keys only if you intend to run paid API models:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
```

For a no-cost smoke test of the custom benchmark path, run the local stub server
in one shell:

```bash
python3 examples/local_openai_stub_server.py --max-requests 4
```

Then run the bundled custom task against that local stub in another shell:

```bash
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv
```

The smoke test succeeds when `output/custom_predictions.csv` has four rows and
`output/custom_summary.csv` has one row with `parse_err_rate = 0` and
`headline_f1 = 1`.

Run the serial benchmark and rebuild summaries:

```bash
python3 code/benchmark.py
python3 code/build_summary.py
```

These commands overwrite the checked-in release outputs. They also call paid API
models if API keys are present.

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
tasks/     task manifests for the built-in benchmark
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
