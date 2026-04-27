# polsci-open-bench

A small political-science text-classification benchmark comparing four local
open-weight LLMs (via Ollama) against two OpenAI API tiers on ten tasks
drawn from published replication archives.

The goal is not a definitive model ranking. It is practical guidance for
applied researchers asking "is the local model good enough on the specific
task I want to run?" Headline numbers, per-task figures, batched-inference
results, and reproduction notes are in
[`output/report.html`](output/report.html) (HTML companion) and
[`output/report_pdf.pdf`](output/report_pdf.pdf) (six-to-eight page PDF).

## Scope

- 10 classification tasks, drawn from public replication archives.
- 6 models: 4 local (Ollama, 4-bit quantized) + 2 OpenAI tiers (gpt-5.5,
  gpt-5.4-nano), both with `reasoning_effort=medium`.
- N=500 items per task, fixed random seed (20260422).
- 30,000 serial predictions plus 44,500 batched predictions across b in
  {1, 10, 20}.
- Metrics: macro F1, accuracy, MCC, latency, parse-error rate. Bootstrap
  confidence intervals (1000 iterations, paired-by-item, 95%).

## Tasks

| Task | Source | Family | Labels | Coded object |
|---|---|---|---|---|
| `gilardi_relevance` | Gilardi et al. 2023 PNAS | Relevance / Incivility | binary | tweets, classified as about content moderation or not |
| `ballard_incivility` | Ballard 2022 LSQ | Relevance / Incivility | binary | tweets by U.S. Congress members |
| `gilardi_stance` | Gilardi et al. 2023 PNAS | Sentiment / Stance / Tone | 3-class | tweets about content moderation |
| `chae_semeval_stance` | Chae & Davidson 2025 SMR | Sentiment / Stance / Tone | 3-class | tweets toward a named political target |
| `ornstein_scotus_sentiment` | Ornstein et al. 2025 PSRM | Sentiment / Stance / Tone | 3-class | tweets about a Supreme Court ruling |
| `wesleyan_creative_ads_2022` | Zhang et al. 2025 Sci Data | Sentiment / Stance / Tone | 3-class | political ads on Meta toward a candidate |
| `halterman_ccc_protest` | Halterman & Keith 2025 PA | Event coding | 4-class | news stories about U.S. protest events |
| `halterman_keith_bfrs` | Halterman & Keith 2025 PA | Event coding | 12-class | news stories about Pakistani political violence |
| `halterman_keith_cmp` | Halterman & Keith 2025 PA + CMP | Policy-topic coding | 7-class | quasi-sentences from political party manifestos |
| `mellon_bes_mii_2024` | Mellon et al. 2024 R&P | Policy-topic coding | 50-class | open-ended British Election Study survey responses |

Per-task prompt provenance (verbatim from source paper, verbatim with
format adaptations, or derived from the published codebook) is documented
in [`docs/prompts_provenance.md`](docs/prompts_provenance.md).

## Models

| Model | Backend | Size / role |
|---|---|---|
| `gemma4:31b-it-q4_K_M` | Ollama | 31B dense |
| `qwen3:14b-q4_K_M` | Ollama | 14B dense |
| `qwen3:30b-a3b-q4_K_M` | Ollama | 30B MoE (~3B active) |
| `mistral-small:24b-instruct-2501-q4_K_M` | Ollama | 24B dense |
| `gpt-5.5` | OpenAI | flagship; `reasoning_effort=medium` |
| `gpt-5.4-nano` | OpenAI | small / cheap; `reasoning_effort=medium` |

All local models ran with thinking off. OpenAI calls use structured JSON
outputs (strict schema).

## How to reproduce

Prerequisites:

- Python 3.10+
- `pip install openai httpx openpyxl pandas pyyaml scikit-learn`
- Ollama running at `http://localhost:11434` (or set `OLLAMA_URL`)
- The four Ollama models installed (`ollama pull <model>`)
- `OPENAI_API_KEY` exported in your shell environment

Run the full grid (10 tasks x 6 models x N=500). On Apple Silicon with
32 GB unified memory the full grid takes roughly 6-8 hours of wall-clock
time, dominated by local Ollama inference:

```bash
python code/benchmark.py
```

Selective reruns and merging back into the canonical CSV:

```bash
# Run one task across all models (e.g. after editing a prompt)
python code/benchmark.py --only-task gilardi_stance

# Run one (task, model) cell and merge into the existing predictions CSV
python code/benchmark.py \
  --only-model qwen3:30b-a3b-q4_K_M \
  --only-task halterman_ccc_protest \
  --merge-into output/predictions.csv
```

Rebuild the summary table and the report:

```bash
python code/build_summary.py             # output/summary.csv
python code/build_summary_batched.py     # output/summary_batched.csv
quarto render output/report.qmd          # output/report.html
quarto render output/report_pdf.qmd      # output/report_pdf.pdf
```

Batched-inference study (separate from the serial run):

```bash
# Default grid: 10 tasks x 6 models x b in {1, 10, 20}
python code/batch_benchmark.py

# Single cell with smaller batch sizes
python code/batch_benchmark.py \
  --only-task gilardi_relevance \
  --batch-sizes 10,20
```

## Output schema

`output/predictions.csv` (one row per (task, model, item)):

- `task`, `model`, `item_id`
- `latency_s`, `eval_count`
- `parse_error` (NaN when parsed cleanly)
- `raw_content_preview` (first 200 chars of model output)
- `pred_<label>` and `gt_<label>` columns per the task's label schema

`output/summary.csv` has one row per (task, model) with per-label F1,
`accuracy`, `mcc`, `median_latency_s`, `parse_err_rate`, `headline_f1`,
bootstrap CI bounds, and per-correct-prediction cost columns. Rebuild
from predictions with `python code/build_summary.py`.

`output/predictions_batched.csv` and `output/summary_batched.csv` are
the analogous outputs for the batched-inference study, keyed by
`(task, model, batch_size)`.

Full schema reference: [`docs/schema.md`](docs/schema.md).

## Repo layout

```
code/
  benchmark.py              # serial runner
  batch_benchmark.py        # batched runner (b in {1, 10, 20})
  build_summary.py          # builds summary.csv from predictions.csv
  build_summary_batched.py  # builds summary_batched.csv
data/
  *.csv                     # cleaned per-task data files (10 tasks)
prompts/
  *.txt                     # one prompt per task
output/
  predictions.csv           # 30,000 serial predictions
  summary.csv               # 60 (task, model) summary rows
  predictions_batched.csv   # batched-inference predictions
  summary_batched.csv       # batched-inference summary
  report.qmd / report.html  # full HTML report (Quarto)
  report_pdf.qmd            # PDF source
  report_pdf.pdf            # six-to-eight page applied summary
docs/
  prompts_provenance.md     # per-task provenance ratings
  schema.md                 # CSV column reference
  inventory_schema.md       # task-candidate yaml schema
  twitter_thread.md         # six-post public summary
```

## Extending the benchmark

To add a new task:

1. Prepare a clean CSV at `data/{task_name}.csv` with a text column, an
   id column, and gold-label columns.
2. Write the prompt at `prompts/{task_name}.txt`. Paste verbatim from the
   source paper when possible; derive from the codebook otherwise.
3. Add a loader function and task config entry in `code/benchmark.py`
   (`_sample_csv` covers most patterns).
4. Run `python code/benchmark.py --only-task {task_name}` to populate the
   predictions, then `python code/build_summary.py`.

See existing loaders for the expected item shape: `{item_id, user_content, gt: {label_key: value, ...}}`.

## Citation

If you use this benchmark in academic work, please cite both the report
and the source papers for the individual tasks (listed in the Tasks table
above and in [`docs/prompts_provenance.md`](docs/prompts_provenance.md)).

## License

Code: MIT (see [`LICENSE`](LICENSE)).

Data: each task's CSV is derived from a publicly distributed replication
archive. Task-level licenses inherit from the source paper. Re-derivation
recipes and source URLs are in [`docs/prompts_provenance.md`](docs/prompts_provenance.md).
