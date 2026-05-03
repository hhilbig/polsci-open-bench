# TODO

## Next iterations

- Add more tasks to each of the four existing task families: Relevance / Incivility, Sentiment / Stance / Tone, Event coding, and Policy-topic coding. This should be treated as the main content-expansion track for the next benchmark iteration, with the goal of adding multiple new tasks within each family rather than deepening only one family at a time. The current family averages are based on a small number of tasks, so broader within-family coverage should make cross-family comparisons more stable and reduce the influence of any single dataset.
  Selection criteria: politically relevant, replication-accessible, prompt or codebook preferably directly obtainable, non-redundant within the family, and practical to benchmark cleanly.
  Balancing rule: prioritize the currently underrepresented families, and do not add two new tasks to the same family before adding at least one to each lower-count family.
  Implemented in the live manifests: `Cross-Domain Topic Classification`, `Politicians in the Line of Fire`, `PAPEA`, and `brandt_political_relevance`.
  Refreshed next-target queue after that first implementation wave:
  1. `ICBe` for `Event coding`
  2. Brandt et al. GTD attack-type classification or PAPEA protest-claim classification as easier event-side fallbacks
  3. Revisit `The Silenced Text` only if a derived rating threshold is acceptable for `Relevance / Incivility`
- Add a new long-document task family. The current benchmark is dominated by short inputs, so it does not say much about performance on longer passages, full documents, or more complex structured-input problems. Candidate additions include longer political texts, research-paper passages, OCR-heavy material, or table-structure tasks.
- Maybe add `ling-2.6-1T` to the benchmark model set in a future model-expansion pass.
- Keep a concrete local-model shortlist for the `v2` benchmark expansion rather than adding popular releases ad hoc.
  Add next: run the prepared `gemma4:26b` / `Gemma 4 26B A4B` manifest as a distinct Gemma efficiency point relative to the current `gemma4:31b`.
  Sidecar only or conditional: `LFM2-24B-A2B`, `gpt-oss:20b`, and `glm-4.7-flash` if the local deployment path is stable and clean.
  Skip for now: `DeepSeek-R1-Distill-Qwen-32B`, `Kimi K2.5` / `K2.6`, `GLM-5` / `GLM-5.1`, `gpt-oss:120b`, `Qwen 2.5-72B` / `Qwen 3.5`, `MiMo-V2-Flash`, `DeepSeek-V3`, `Gemma 3 27B`, and `Mistral Small 3.1`.
  Rule: prefer additions that are both currently popular and not too overlapping with the existing `Gemma 4 31B`, `Qwen 3 14B`, `Qwen 3 30B-A3B`, and `Mistral Small 24B` lineup. Favor new architecture or deployment-tier coverage over near-duplicates from the same family.
- Modularize the repo so other researchers can run their own evaluations against the same framework. The longer-run goal is a reusable political-science benchmark suite with pluggable task definitions, model adapters, and reporting, so labs can add models or benchmarks without rewriting the pipeline.
- Defer a backend-adapter refactor that would make unsupported providers pluggable without core-code edits. Current state: outside users can already supply custom tasks and custom models within the supported `ollama`, `openai`, and `anthropic` backends. Deferred next module:
  1. Extract the current built-in inference paths into backend adapters with no behavior change.
  2. Add a backend registry so runners dispatch by adapter key rather than hard-coded backend branches.
  3. Add a `subprocess` adapter backend so unsupported providers can be wrapped externally.
  4. Add one example external adapter plus a smoke test and docs.
- Explore whether this benchmark can interoperate with adjacent field benchmarks, including OCR-oriented work, so the field moves toward a unified evaluation suite rather than isolated one-off comparisons.

## Motivation

- The current results are most informative for short classification inputs. They likely understate performance gaps that appear when models need to process longer or more structurally complex material.
- A modular benchmark suite would lower replication costs for other political scientists and make it easier to compare local open-weight models against commercial APIs on shared tasks.
