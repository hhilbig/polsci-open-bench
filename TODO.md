# TODO

## Maintenance

- Keep the generated task inventory synchronized with the 34 canonical manifests
  in [`tasks/`](tasks):

  ```bash
  python3 code/task_inventory.py --write
  python3 code/task_inventory.py --check
  ```

- After any benchmark run, rebuild the relevant summaries, report assets, and
  PDF before treating the output as current.
- Register every long-running local, API, droplet, or `mac2` run in
  [`output/run_registry.jsonl`](output/run_registry.jsonl), then refresh
  [`docs/run_status.md`](docs/run_status.md).

## Benchmark Extensions

- Add more tasks only when they satisfy the hard entry threshold: public text
  plus labels are directly ingestible, labels are direct or transparently
  collapsed from direct codebook labels, any cleaning drop above 20 percent is
  flagged, conflicting duplicate labels are removed, and sampled `item_id`
  values are unique.
- Broaden coverage in underrepresented areas, especially longer documents,
  structured-input tasks, and coding tasks with richer label spaces.
- Avoid adding near-duplicate tasks from the same source family unless they
  test a clearly different prediction target.

## Model Watchlist

- Tabled due to error rate: `ibm/granite4.1:8b`. The completed run had
  470 / 8,820 parse errors (5.3%), above the clean-baseline threshold. Do not
  promote it unless parser or prompt behavior is revisited.
- Tested but not retained: `phi4-mini`. It was fast, but materially weaker than
  the retained local baselines.
- Consider future local models only when they add a distinct architecture,
  deployment tier, or practical cost/speed point relative to Gemma 4, Qwen3,
  and Mistral Small.

## Deferred Engineering

- Refactor inference backends into adapters only if custom provider support
  becomes a real user need. The current framework already supports custom tasks
  and custom models for the built-in `ollama`, `openai`, and `anthropic`
  backends.
- Explore whether this benchmark can interoperate with adjacent social-science
  text-as-data benchmarks, including OCR-heavy or long-document settings.
