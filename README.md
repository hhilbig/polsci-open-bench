# polsci-open-bench

A benchmark of local open-weight and commercial API language models for
political science text classification.

The main release compares six local Ollama models with four commercial API
models on 33 classification tasks drawn from political science papers,
replication archives and documented public datasets. The
[report](output/report_pdf.pdf) is the authoritative summary.

## Main Result

Local open-weight models are often competitive with commercial API models, and
no single model ranks first across tasks. The best local model matches or
exceeds the best API model on 9 of 33 tasks. On average, the best API model
leads by 0.013 F1, with its clearest advantage on tasks that have many active
labels, long codebooks or several outputs per item.

Two further experiments test whether this gap is an artifact of the benchmark.
The published run gave API models server-side JSON-schema enforcement and local
models none. Running four open-weight models twice over the same 16,425 items,
once with schema-constrained decoding and once without, moves mean F1 by 0.0014,
an order of magnitude less than the gap. The difference between the two groups
therefore reflects the models themselves.

The second experiment asks how many hand-coded labels a supervised classifier
needs to match zero-shot coding. Logistic regression on frozen multilingual-E5
embeddings, trained on 2,000 labels per task, reaches 0.672 mean F1 against
0.697 for the best local model with no labels at all, and the learning curve has
flattened. The crossover thus lies in the low thousands of labels per task.

Researchers should test candidate models on labeled examples from their own task
and report both performance and the rate of unusable output.

## API Model Comparison, September 2026

A second panel runs 11 commercial API models on the same 33 tasks, with 100
frozen items per task, and measures what each model costs and how long a request
takes.

![Cost and latency against accuracy](output/figures/fig-jev-cost-latency.png)

Mean task F1 ranges from 0.661 for Jev 1.13 to 0.714 for Claude Opus 5. On a
paired task bootstrap, six of the ten other models outperform Jev: both Claude
models, both Gemini Flash models and two GPT-5.6 models. The remaining four are
indistinguishable from it. Jev is the cheapest and fastest model on the panel.
The cheapest model that outperforms it is Gemini 3.1 Flash-Lite, which costs
about four times as much per item and takes twice as long per request, for a gain
of 0.021 F1.

![Real-time against batch pricing](output/figures/fig-cost-realtime-batch.png)

OpenAI, Anthropic and Google discount batch requests by 50%, while DeepSeek and
Jev offer no batch option. At batch rates, Flash-Lite costs about twice as much
as Jev.

Prompt optimisation does not close the gap for Jev. GEPA, run on 30 tasks with
prompts selected on rows the benchmark never sampled, raises held-out F1 by
0.011, with a paired interval that includes zero. The same prompts gained 0.048
on the rows used to select them, so 78% of the apparent improvement does not
carry over to new items.

Costs for the OpenAI, DeepSeek and Jev models come from provider billing
records. Costs for Gemini and Anthropic are token counts multiplied by published
prices, and the figures draw these models with hollow markers.

## Scope

- 33 active tasks in [`tasks/`](tasks), one held out (see [`CHANGELOG.md`](CHANGELOG.md))
- Main release: 6 local and 4 API models, 293 to 500 items per task
- September panel: 11 API models, 100 items per task
- Metrics: main F1 (`headline_f1` in the CSVs), accuracy, MCC, time per item and
  unusable-output rate
- Local hardware: Apple M2 Pro, 32 GB

## Outputs

- Predictions and summary: [`output/predictions.csv`](output/predictions.csv),
  [`output/summary.csv`](output/summary.csv)
- Prompt-batched runs: [`output/predictions_batched.csv`](output/predictions_batched.csv),
  [`output/summary_batched.csv`](output/summary_batched.csv)
- Supervised learning curves: [`output/supervised_baseline_e5.csv`](output/supervised_baseline_e5.csv)
- September panel tables: [`output/sidecar/jev_sidecar/`](output/sidecar/jev_sidecar)
- GEPA results, including each optimised prompt: [`output/sidecar/gepa_jev_20260920/`](output/sidecar/gepa_jev_20260920)

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 code/task_inventory.py --check
python3 -m unittest discover -s tests
```

Rebuild the report:

```bash
Rscript code/build_report_assets.R
quarto render output/report_pdf.qmd --to pdf
```

A full rerun calls paid APIs when keys are present. See
[`docs/reproduce.md`](docs/reproduce.md) for setup, costs and rerun commands.

## Documentation

- [`docs/status.md`](docs/status.md): current release state
- [`docs/schema.md`](docs/schema.md): output schema
- [`docs/prompts_provenance.md`](docs/prompts_provenance.md): task and prompt provenance
- [`docs/task_source_fidelity_audit.md`](docs/task_source_fidelity_audit.md): source-fidelity audit
- [`docs/custom_tasks.md`](docs/custom_tasks.md), [`docs/custom_models.md`](docs/custom_models.md): adding tasks and models
- [`CHANGELOG.md`](CHANGELOG.md): changes to tasks, labels, prompts and metrics

## Layout

```text
code/         runners, analysis scripts and report builders
data/         cleaned task files
tasks/        task manifests (tasks_v2/ supersedes corrected v1 tasks)
prompts/      task prompts (prompts_v2/ belongs to tasks_v2/)
models/       model manifests
experiments/  run configurations for the side experiments
output/       predictions, summaries, figures and reports
docs/         documentation
tests/        unit tests
```

Raw per-request responses under `output/sidecar/` are not tracked, since they
run to tens of thousands of files. The derived tables the figure scripts read
are tracked, so the figures rebuild from a clone.

## Citation

Please cite the report and the source papers for the individual tasks, listed in
[`docs/prompts_provenance.md`](docs/prompts_provenance.md).

## License

Code is MIT licensed, see [`LICENSE`](LICENSE). Task CSVs derive from public
replication archives and documented public datasets, and each inherits the
license of its source.
