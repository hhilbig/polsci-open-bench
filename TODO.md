# TODO

## Next iterations

- Complete full-grid batched inference. Current batched artifacts cover the original 10-task grid for six models at `b=10` and `b=20`, with Claude Sonnet 4.6 only present as a serial `b=1` baseline in `summary_batched.csv`. Batched inference is still outstanding for the expanded 18-task x 9-model benchmark, including `deepseek-v4-pro`, `gemma4:26b`, Claude Sonnet `b=10/20`, and all eight expanded tasks.
- Add more tasks to each of the four existing task families: Relevance / Incivility, Sentiment / Stance / Tone, Event coding, and Policy-topic coding. This should be treated as the main content-expansion track for the next benchmark iteration, with the goal of adding multiple new tasks within each family rather than deepening only one family at a time. The current family averages are based on a small number of tasks, so broader within-family coverage should make cross-family comparisons more stable and reduce the influence of any single dataset.
  Selection criteria: substantively or methodologically relevant to political-science research, replication-accessible, prompt or codebook preferably directly obtainable, non-redundant within the family, and practical to benchmark cleanly.
  Balancing rule: prioritize the currently underrepresented families, and do not add two new tasks to the same family before adding at least one to each lower-count family.
  Implemented in the live manifests: `Cross-Domain Topic Classification`, `Politicians in the Line of Fire`, `PAPEA`, `brandt_political_relevance`, `douglass_icbe_sentence_event_type`, `muller_fujimura_campaign_policy_area`, and `burnham_polnli_entailment`.
  Refreshed next-target queue after that first implementation wave:
  Result of the broader next-target wave:
  1. `Clicks and Stones` failed the direct-ingest screen because the public archive exposes aggregate legislator-level hostility rates rather than tweet-level public text and labels.
  2. `Political DEBATE / PolNLI` is implemented as `burnham_polnli_entailment`, a deliberate broader-direction pilot for hypothesis-conditioned political text classification.
  Replacement `Relevance / Incivility` screen on 2026-05-04: `Super-Unsupervised Classification for Labelling Text`, `Citizens' Perceptions of Online Abuse Directed at Politicians`, and `Online Abuse of Politicians` all failed the direct-ingest screen because the public archives do not expose directly usable public text plus labels.
  Verified replacement candidates from the same screen: `The Dynamics of Political Incivility on Twitter` is implemented as `theocharis_dynamics_incivility`; `toxicity-protests-ES` is now staged as `tasks_next/toxicity_protests_es.yaml` with a cleaned 972-row data file; `TwitCivility` is now staged as `tasks_next/twitcivility_impoliteness.yaml` with a cleaned 13,124-row data file. Non-English benchmark tasks are acceptable if they otherwise satisfy the task-selection criteria.
  Prepared but not live-scored candidates for the next benchmark wave:
  1. `toxicity_protests_es`, a Spanish protest-toxicity binary task built from coder-agreement rows.
  2. `brandt_gtd_attack_type`, a nine-class GTD primary attack-type task built from Brandt et al.'s public GTD multi-label file.
  3. `haunss_papea_claims`, a 28-class PAPEA protest-claim task built from the public FGZ claim annotations.
  4. `twitcivility_impoliteness`, a binary impoliteness task built from the public TwitCivility train/test splits.
  Additional prepared but not live-scored candidates:
  5. `bestvater_wm_stance`, a Women's March stance task built from Bestvater and Monroe's public ground-truth tweets.
  6. `erlich_ati_topics`, a seven-label multi-binary ATI request-topic task built from Erlich et al.'s public human-coded Mexican access-to-information requests.
  7. `plover_cameo_event`, an 18-class event-type task built from PLOVER gold-standard CAMEO records. This is usable but small, with 312 clean examples.
  8. `burnham_polnli_event_entailment`, an event-extraction-only PolNLI entailment subset.
  9. `burnham_trump_stance`, a three-class Trump stance task built from Burnham's public train file and adjudicated test file.
  10. `burnham_covid_threat_minimization`, a binary COVID threat-minimization task built from Burnham's public labeled sample.
  11. `dicocco_manifesto_populism`, a binary manifesto-sentence populism task built from Di Cocco and Monechi's public Italian manual annotations.
  12. `bestvater_kavanaugh_stance`, a binary Brett Kavanaugh stance task built from Bestvater and Monroe's public ground-truth tweets. Exact same-label repeats are retained because deduplicating them would discard more than 20% of source rows; conflicting duplicate texts are removed.
  13. `politicause_causal_relation`, a binary causal-relation sentence task built from the public PolitiCAUSE train/validation/test splits.
  14. `cap_party_platform_policy_topic`, a 21-class CAP major-topic task built from public Democratic and Republican Party Platform quasi-statements.
  15. `cap_crs_policy_topic`, a 21-class CAP major-topic task built from public Congressional Research Service report titles and summaries.
  16. `agoraspeech_criticism_agenda`, a two-class Greek campaign-speech paragraph task built from AgoraSpeech's human-validated criticism-or-agenda labels and English translations.
  These candidates are intentionally staged under `tasks_next/` rather than `tasks/`, so the current 18-task complete matrix remains unchanged until scoring starts.
  Hard entry threshold for future task additions: public text plus labels must be directly ingestible, labels must be direct or transparently collapsed from direct codebook labels, any >20% cleaning or unique-unit drop must be flagged and justified before staging, conflicting duplicate labels must be removed, and the manifest must load with unique sampled `item_id` values. Low downstream F1 is not a rejection criterion by itself.
  Candidate screened but not staged: `wang_2023_topic_classification_pretrained_lms`; the Dataverse archive notebooks reference `data_and_models.zip`, but that zip is absent from the visible Dataverse file list, so the labeled corpus is not directly ingestible from the archive as inspected.
  Additional candidates screened but not staged in the 2026-05-06 pass: Policlim is locked in Dataverse review; Ziegler's manipulation-check archive did not expose a direct text-plus-label response file; Hobbs and Green required impractical Code Ocean ingestion rather than a direct benchmark table; Ivanusch et al. exposed manual labels but the public corpora inspected had empty text fields; MaML did not expose a clean direct text-label validation set.
  Additional candidates screened but not staged in the 2026-05-07 pass: Media Frames Corpus annotations require separate LexisNexis access for article text; CAP State of the Union required dropping just over 20% of rows to remove non-policy or unsupported-topic material; AgoraSpeech sentiment, polarization, and populism are continuous scores that would require thresholding rather than direct categorical labels.
  Next task-collection step: once mac2 or another runner is available, either start scoring the staged candidates or continue balancing additions across the other original task families.
  Immediate fallbacks if one of the locked four fails verification:
  1. `The Silenced Text`, only if a derived rating threshold is acceptable
  2. PLOVER gold-standard records
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
