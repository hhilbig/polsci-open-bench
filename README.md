# polsci-open-bench

A **political-science text-classification benchmark** comparing local open-weight LLMs (via Ollama) against the OpenAI API on tasks drawn from published replication archives.

## Current state

- **6 tasks × 6 models × N=250 per task (halterman_ccc capped at N=50 by data availability) = 7,800 predictions**, run 2026-04-23 on mac2.
- `state_adaptation` (the user's own 24 KB-prompt bill classification) is temporarily deactivated in `TASKS` — Ollama prefill at ~150 s/item made the cell unusable on gemma4:31b. Task definition is preserved at module level; reinstate for a dedicated single-task run.
- `gpt-5.4-mini` dropped from the 2026-04-23 run (rarely Pareto-optimal in prior rounds).
- Task prompts are either verbatim from the original paper or derived from the codebook when the paper didn't use an LLM; provenance is documented per task.

See `output/report.html` for the current results summary (self-contained Quarto report).

## Tasks

| Task | Paper | Labels | Text | Prompt source |
|---|---|---|---|---|
| `state_adaptation` | User's own bill classification | 8 binary (adaptation, mitigation, …) | bill title + abstract | Production few-shot prompt |
| `gilardi_relevance` | Gilardi, Alizadeh & Kubli 2023 PNAS | binary | tweet | Verbatim from replication archive |
| `gilardi_stance` | Gilardi et al. 2023 PNAS | 3-class (pro/neutral/contra) | tweet | Derived (PNAS SI §S1 paywalled) |
| `ballard_incivility` | Ballard 2022 LSQ | binary | tweet | Derived (Ballard used BERT) |
| `ornstein_scotus_sentiment` | Ornstein et al. 2025 PSRM | 3-class | tweet | Derived from `promptr` codebook |
| `halterman_ccc_protest` | Halterman & Keith 2025 PA | 8-class | news story | Verbatim from archive codebook |
| `chae_semeval_stance` | Chae & Davidson 2025 SMR (SemEval-2016) | 3-class | tweet | Derived from SemEval codebook |

See `candidates.yaml` for the full inventory of 35 candidate tasks (ready, partial, blocked, deprioritized), and `candidates.md` for a quick-scan summary.

## Models

| Model | Backend | Size |
|---|---|---|
| `gemma4:31b-it-q4_K_M` | Ollama | 31B dense |
| `qwen3:14b-q4_K_M` | Ollama | 14B dense |
| `qwen3:30b-a3b-q4_K_M` | Ollama | 30B MoE (~3B active) |
| `mistral-small:24b-instruct-2501-q4_K_M` | Ollama | 24B dense |
| `gpt-5.4-nano` | OpenAI | — |
| `gpt-5.4` | OpenAI | — |

All local models ran with thinking OFF. API models use structured JSON outputs. `gpt-5.4-mini` is available in the MODELS list structurally but was dropped from the 2026-04-23 run.

## Usage

Prerequisites: Python 3.9+, `pip install openai httpx openpyxl pandas pyreadr pyyaml scikit-learn`. Ollama running locally (or reachable via `OLLAMA_URL`). `OPENAI_API_KEY` set. The four Ollama models are installed on mac2; benchmark runs there — see `CLAUDE.md` for the mac2 workflow.

```bash
# Full benchmark (6 tasks × 6 models — state_adaptation is currently commented out in TASKS)
python code/benchmark.py

# Selective rerun of one (task, model) cell, merge into existing predictions
# NOTE: --merge-into has known bugs (documented in MEMORY.md and KNOWLEDGE_BASE §6);
# prefer a fresh full run or a task-only run until they're fixed.
python code/benchmark.py \
  --only-model qwen3:30b-a3b-q4_K_M \
  --only-task halterman_ccc_protest

# Run all models on one task (e.g. after updating a prompt)
python code/benchmark.py --only-task gilardi_stance

# Rebuild output/summary.csv from output/predictions.csv
python code/build_summary.py

# Re-render the Quarto report (from the repo root)
quarto render output/report.qmd --execute-dir .
```

## Output schema

`output/predictions.csv` columns (long format; one row per task × model × item):

- `task`, `model`, `item_id`
- `latency_s`, `eval_count`
- `parse_error` (NaN if parsed cleanly)
- `raw_content_preview` (first 200 chars of model output)
- `pred_<label>` per task's label schema
- `gt_<label>` per task's ground truth

`output/summary.csv` has per (task, model) metrics: F1 per label, `avg_f1`, `accuracy` (categorical tasks), `median_latency_s`, `parse_err_rate`, `headline_f1` (unified per-task primary metric). Rebuild it from `predictions.csv` with `python code/build_summary.py`.

## Private data

`data/private/bills_for_matthew.xlsx` is a single-RA's personal coding of US state bills. It is kept in the private repo but is **not for redistribution**; if this repo is ever made public, this file must be removed. The `state_adaptation` loader will fail clearly if the file is missing.

## Repo structure

```
code/
  benchmark.py          # unified runner (--only-task, --only-model, --merge-into)
data/
  *.csv                 # public clean per-task data
  private/              # bills_for_matthew.xlsx (gitignored pattern on release)
prompts/                # per-task LLM prompts (verbatim or derived)
output/
  predictions.csv       # raw per-item predictions
  summary.csv           # per (task, model) F1 and latency
  report.qmd            # Quarto source for the report
  report.html           # rendered self-contained report
candidates.yaml         # 35-entry inventory of task candidates
candidates.md           # summary markdown rendering of the yaml
docs/
  inventory_schema.md   # field definitions for candidates.yaml
```

## Extending the benchmark

To add a new task:

1. Prepare a clean CSV at `data/{task_name}.csv` with a text column, an id column, and a gold-label column.
2. Write the prompt at `prompts/{task_name}.txt`. Paste verbatim from the paper when possible; derive from the codebook otherwise.
3. Add a loader function + task config entry in `code/benchmark.py` (the TASKS list).
4. Add the candidate to `candidates.yaml` and regenerate `candidates.md`.
5. Run `python code/benchmark.py --only-task {task_name}`.

See the existing loaders for the expected item shape (`item_id`, `user_content`, `gt` dict).

## License

Code: MIT. Data: each task inherits its source paper's license — see `candidates.yaml` `license` field per task.
