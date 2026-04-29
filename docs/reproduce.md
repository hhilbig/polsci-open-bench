# Reproduction guide

Detailed setup, model pulls, and selective reruns for `polsci-open-bench`.

## Backends

### Ollama (local)

The benchmark was run with Ollama 0.19.0 on Apple Silicon (M2 Pro, 32 GB unified memory). Quantization is `Q4_K_M` for all four local models (the suffix in the model names).

Pull the four local models:

```bash
ollama pull gemma4:31b-it-q4_K_M
ollama pull qwen3:14b-q4_K_M
ollama pull qwen3:30b-a3b-q4_K_M
ollama pull mistral-small:24b-instruct-2501-q4_K_M
```

The runner reads `OLLAMA_URL` if you need to point it at a non-default host:

```bash
export OLLAMA_URL=http://localhost:11434  # default
```

Local calls run with `temperature=0.1` and the model's internal "thinking" mode disabled.

### OpenAI

```bash
pip install openai
export OPENAI_API_KEY=...
```

Both OpenAI tiers (`gpt-5.5`, `gpt-5.4-nano`) run with `reasoning_effort=medium` and a strict JSON schema.

### Anthropic

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
```

If `ANTHROPIC_API_KEY` is unset, the runner falls back to `~/.anthropic_api_key` (file mode 0600). Anthropic calls use tool-use forcing to constrain outputs to the same schema as the OpenAI path.

## Selective reruns

Useful when you have edited one prompt or want to fill in a single (task, model) cell rather than rerunning the whole grid:

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

Rebuild summaries after any rerun:

```bash
python code/build_summary.py
python code/build_summary_batched.py
```

## Output schema

Column definitions for the predictions and summary CSVs are in [`schema.md`](schema.md).
