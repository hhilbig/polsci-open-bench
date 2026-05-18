# polsci-open-bench

A benchmark of local open-weight and commercial API LLMs for political science
text classification.

The current release compares five local Ollama models with four commercial API
models from OpenAI, Anthropic, and DeepSeek on 34 classification tasks from
published political science replication archives. The report is the
authoritative project summary.

## Paper and Data

- PDF report: [`output/report_pdf.pdf`](output/report_pdf.pdf)
- Serial predictions: [`output/predictions.csv`](output/predictions.csv)
- Serial summary: [`output/summary.csv`](output/summary.csv)
- Prompt-batched predictions: [`output/predictions_batched.csv`](output/predictions_batched.csv)
- Prompt-batched summary: [`output/summary_batched.csv`](output/summary_batched.csv)
- Local 10-item batching summary: [`output/summary_batched_local_b10.csv`](output/summary_batched_local_b10.csv)
- Task inventory: [`docs/task_inventory.md`](docs/task_inventory.md)

## Main Result

Local open-weight models are often competitive with commercial API models, but
the benchmark does not support a single global model ranking. The best local
model matches or exceeds the best API model on 9 of 34 tasks; on average, the
best API model exceeds the best local model by 0.015 F1. API models have their
clearest edge on complex tasks with many active labels, long codebooks, or
multiple outputs per item.

For applied work, the practical recommendation is to test candidate models on
labeled examples from the target task and report both performance and unusable
output rates. Prompt batching can make local models faster, but it needs
task-specific reliability checks. This is a prompt-based annotation benchmark,
not a supervised-learning baseline suite.

## Benchmark Scope

- 34 task manifests in [`tasks/`](tasks)
- 9 serial models: 5 local Ollama models and 4 commercial API models
- 306 serial task-model comparisons
- 147,825 serial model-item classifications
- 170 local prompt-batched task-model comparisons with 10 items per prompt
- 293 to 500 items per task
- Metrics: main F1, accuracy, MCC, time per item, and unusable-output rate
- Local hardware: Apple M2 Pro with 32 GB unified memory

The report's "main F1" column is named `headline_f1` in the CSV outputs.

## Reproduce

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Basic checks:

```bash
python3 code/task_inventory.py --check
python3 -m unittest discover -s tests
```

Rebuild report assets and the PDF:

```bash
Rscript code/build_report_assets.R
quarto render output/report_pdf.qmd --to pdf
```

Running the full benchmark can call paid APIs if API keys are present. See
[`docs/reproduce.md`](docs/reproduce.md) for backend setup, model pulls, custom
tasks, custom models, cost notes, and full rerun commands.

## Documentation

- [`docs/status.md`](docs/status.md): current release state
- [`docs/reproduce.md`](docs/reproduce.md): setup and rerun instructions
- [`docs/schema.md`](docs/schema.md): output schema
- [`docs/prompts_provenance.md`](docs/prompts_provenance.md): prompt and task provenance
- [`docs/custom_tasks.md`](docs/custom_tasks.md): custom task manifests
- [`docs/custom_models.md`](docs/custom_models.md): custom model manifests
- [`docs/release_workflow.md`](docs/release_workflow.md): release and arXiv workflow

## Repo Layout

```text
code/      benchmark runners and report builders
data/      cleaned task files
models/    model manifests
tasks/     task manifests
prompts/   task prompts
output/    predictions, summaries, figures, tables, reports
docs/      documentation
examples/  minimal custom-task and custom-model examples
```

## Citation

If you use this benchmark in academic work, please cite the report and the
source papers for the individual tasks listed in
[`docs/prompts_provenance.md`](docs/prompts_provenance.md).

## License

Code: MIT, see [`LICENSE`](LICENSE).

Data: task CSVs are derived from public replication archives. Task-level data
licenses inherit from the source paper or source dataset.
