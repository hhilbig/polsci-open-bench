# polsci-open-bench

Code, tasks and results for the
[Political Science LLM Benchmark](https://www.hannohilbig.com/llm-benchmark/).

This is a benchmark of 43 models, among them 21 open-weight and 22 API, on a
total of 33 political science coding tasks. I use 100 texts per task, scored
against human labels. Human scoring was done by the original authors of the
tasks. I compare model performance using mean F1. Claude Opus 5 scores highest
here. The best open-weight model is MiMo V2.6 Flash. Differences between models
are typically small but do vary by task. The full results are below.

- Results, costs and methods: [hannohilbig.com/llm-benchmark](https://www.hannohilbig.com/llm-benchmark/)
- Release data: [`output/sidecar/refresh_20261106_release_33/`](output/sidecar/refresh_20261106_release_33)
- Tasks: [`tasks/`](tasks), sources in [`docs/prompts_provenance.md`](docs/prompts_provenance.md)

## Earlier release

- [Report (PDF)](output/report_pdf.pdf): 6 local Ollama models and 4 API models
- 293 to 500 items per task; unusable answers excluded from F1
- Not comparable to the page
- Data: [`output/summary.csv`](output/summary.csv), [`output/predictions.csv`](output/predictions.csv)

## Running the code

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests   # about 65 skips on a fresh clone

# no-cost smoke test: stub server in one terminal ...
python3 examples/local_openai_stub_server.py --max-requests 4
# ... two-item example task in another (expect 4 prediction rows)
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --output /tmp/polsci-smoke/predictions.csv

# list tasks, models and call counts without calling anything
python3 code/benchmark.py --dry-run
```

A real run calls every API model whose key is set, and those calls are billed.

- Setup and reruns: [`docs/reproduce.md`](docs/reproduce.md)
- Adding tasks or models: [`docs/custom_tasks.md`](docs/custom_tasks.md), [`docs/custom_models.md`](docs/custom_models.md)
- Output files: [`docs/schema.md`](docs/schema.md)
- Rebuilding the page: [`docs/update_benchmark_page.md`](docs/update_benchmark_page.md)
- Changes to tasks and scoring: [`CHANGELOG.md`](CHANGELOG.md)

## Citation and license

- Cite the benchmark: [`CITATION.cff`](CITATION.cff), plus the source papers of the tasks you use
- Code: MIT ([`LICENSE`](LICENSE))
- Task data: license of each source ([`DATA_LICENSE.md`](DATA_LICENSE.md))
