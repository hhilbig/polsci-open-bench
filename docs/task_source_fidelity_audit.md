# Task Source-Fidelity Audit

Last checked: 2026-05-18.

This audit checks whether each live benchmark task stays close to its source
construct. It is separate from citation coverage. The main question is whether
the benchmark target, label direction, input unit, and documented prompt
provenance match what the source data or authors were trying to measure.

## Summary

- The 34 live task manifests are internally consistent: every task has a
  non-missing gold label, the observed gold labels fit the manifest schema, and
  the prompt mentions the allowed output labels.
- Most tasks use a direct source label, a source codebook, or a transparent
  label collapse. No task currently appears to use a speculative outcome that
  is unrelated to the source data.
- The largest source-fidelity risks are not label errors. They are scope and
  wording issues: some tasks use subsets of source labels, some prompts are
  derived rather than author-provided, and a few source corpora are documented
  public datasets rather than political-science replication archives.
- The report now describes the task sources as political science papers,
  public replication archives, and documented public datasets. That is more
  accurate than saying all tasks come from published political science
  replication archives.

## Audit Rules

A task is source-faithful for this benchmark if:

1. the input unit matches a source unit or a clearly documented source-derived
   unit;
2. the gold label is a direct source label, a source-provided aggregation, or a
   transparent codebook-level collapse;
3. the prompt asks for the same construct as the source label;
4. any filtering, deduplication, agreement restriction, label restriction, or
   translation choice is documented.

## Task-Level Read

| Task | Source-fidelity read | Notes |
|---|---|---|
| `gilardi_relevance` | Pass | Prompt is essentially source wording, with JSON output added. Target remains content-moderation relevance. |
| `gilardi_stance` | Pass with label-name adaptation | Prompt preserves the Section 230 stance task. Label names use the replication-data wording (`pro`, `neutral`, `contra`). |
| `ballard_incivility` | Review before next rerun | The target is a direct incivility label, but the prompt is fully derived. The first sentence says tweets were pre-filtered as divisive; the cleaned data do not store a divisiveness filter. If strict source fidelity is required, rerun this task with a neutral incivility prompt that does not claim prefiltering. |
| `brandt_political_relevance` | Pass with wording clarification | Uses the direct binary source label. The table now describes this as conflict- or politics-relevance, which better matches the source task than generic political relevance. |
| `rheault_line_of_fire_incivility` | Pass | Prompt tracks the source definition: swear words, vulgarities, insults, threats, private-life attacks, and group-targeted attacks. |
| `theocharis_dynamics_incivility` | Pass | Uses the public tweet text and direct yes/no incivility label. Prompt is derived from the source construct. |
| `ornstein_scotus_sentiment` | Pass | Uses the source sentiment framing for tweets about Supreme Court rulings. The prompt is derived from the task framing rather than copied as an author prompt. |
| `chae_semeval_stance` | Pass with minor provenance issue | Uses the SemEval stance construct: target-specific FAVOR, AGAINST, NONE. The prompt footer still says Chae and Davidson (2025); docs and references now use 2026. Changing the prompt footer would be non-substantive but would mean the checked-in prompt differs from the run prompt. |
| `halterman_ccc_protest` | Pass with label restriction | Uses a verbatim-adapted source codebook subset and keeps the four labels present in the cleaned benchmark file. |
| `halterman_keith_bfrs` | Pass | Uses the source BFRS event codebook with JSON output added. |
| `douglass_icbe_sentence_event_type` | Pass with transparent conversion | Converts aligned source event annotations into sentence-level `No Event`, `Action`, `Speech`, `Thought`, and `Mixed`. This matches the event-type structure in the source data while making no-event and multi-event sentences explicit. |
| `haunss_papea_fgz_forms` | Pass with documented subset | Keeps seven common original PAPEA form labels. This is not the full source label universe, so the report and provenance describe it as a restricted source-label task. |
| `halterman_keith_cmp` | Pass with documented aggregation | Uses a seven-domain aggregation of the CMP/MARPOR coding scheme from the Halterman and Keith setup, not the full Manifesto Project category set. |
| `osnabruegge_cross_domain_topic` | Pass | Uses the source 8-topic target-corpus specification and the adjudicated source topic label. |
| `muller_fujimura_campaign_policy_area` | Pass | Uses the direct `policy_area` labels from the public supervised splits, including `No Policy Area`. |
| `mellon_bes_mii_2024` | Pass | Uses the source MII issue labels and source-like coding rule: assign one issue, using the first issue when multiple are mentioned. |
| `wesleyan_creative_ads_2022` | Pass | Uses a source-derived tone-toward-candidate task. Input includes sponsor, candidate, and ad text, which are necessary for the target. |
| `burnham_polnli_entailment` | Pass | Source codes entailment as `0`; the benchmark maps this to `entails = 1` so the output field has the intuitive direction. This inversion is documented and correct. |
| `toxicity_protests_es` | Pass with agreement restriction | Keeps rows where both human coders agree and uses the agreed toxicity label. This is stricter than using all rows, but faithful to a gold-standard classification target. |
| `brandt_gtd_attack_type` | Pass | Uses direct GTD primary attack-type labels from the source corpus. |
| `haunss_papea_claims` | Pass with documented label filter | Uses direct source claim labels after dropping missing, conflicting, duplicate, and non-kept labels. The provenance notes that unsupported labels are removed. |
| `twitcivility_impoliteness` | Pass with scope caveat | Uses the direct public `impoliteness` label, not the separate `intolerance` label. It is a documented public dataset source rather than a political-science replication archive. |
| `bestvater_wm_stance` | Pass | Uses the direct binary stance label for Women's March tweets. The benchmark names the positive direction as pro-Women's-March. |
| `erlich_ati_topics` | Pass | Uses seven direct non-exclusive request-topic labels from the source data as a multi-binary task. |
| `plover_cameo_event` | Pass with wording clarification | Uses gold-standard CAMEO examples classified into PLOVER event categories. The report now avoids calling the target a generic CAMEO event type. |
| `burnham_polnli_event_entailment` | Pass with dependence caveat | Uses the event-extraction subset of the PolNLI task. It is source-faithful but not independent of `burnham_polnli_entailment`. |
| `burnham_trump_stance` | Pass | Uses direct source stance labels and maps them to `Oppose`, `Neutral`, and `Support`. |
| `burnham_covid_threat_minimization` | Pass | Uses the primary source `threatmin` label. The second source label is retained only as source metadata. |
| `dicocco_manifesto_populism` | Pass | Uses a direct source populism label for manifesto sentences. |
| `bestvater_kavanaugh_stance` | Pass | Uses the direct binary stance label for Kavanaugh tweets. The benchmark names the positive direction as pro-Kavanaugh. |
| `politicause_causal_relation` | Pass | Uses the direct binary source label for causal-relation presence. |
| `cap_party_platform_policy_topic` | Pass with standard-topic restriction | Uses CAP major-topic codes and keeps standard major topics. The task is faithful to the CAP major-topic construct. |
| `cap_crs_policy_topic` | Pass with standard-topic restriction | Uses CAP-coded CRS titles and summaries and keeps standard major topics. |
| `agoraspeech_criticism_agenda` | Pass with translation/human-label caveat | Uses English translations and the human-validated criticism-or-agenda label, not GPT labels or continuous AgoraSpeech dimensions. |

## Fixes Made From This Audit

- Changed public-scope wording in the README and report from "published
  political science replication archives" to "political science papers, public
  replication archives, and documented public datasets."
- Broadened the appendix definition of "Derived" so it covers prompts based on
  source codebooks, public label definitions, and task descriptions.
- Clarified the Brandt relevance task as conflict- or politics-relevance.
- Clarified that PAPEA protest forms use seven common source form labels.
- Clarified the PLOVER task as PLOVER event categories based on CAMEO examples,
  not a generic CAMEO event-type task.

## Remaining Judgment Calls

1. **Ballard incivility prompt.** This is the only task I would consider
   changing before a future rerun. The prompt should probably drop the
   statement that every tweet was pre-filtered as divisive unless the original
   source files document that exact filter.
2. **Prompt footer dates.** The Chae/SemEval prompt footer has a stale year in
   a provenance note. The semantic task is correct. Changing it now would make
   the prompt file differ slightly from the run prompt.
3. **Task independence.** `burnham_polnli_entailment` and
   `burnham_polnli_event_entailment` are source-faithful but related. This is
   acceptable if the benchmark treats them as two task variants, not independent
   source families.

## Second Audit, 2026-09-18

The first audit asked whether each task's gold label comes from its source. It
does, almost everywhere, and that conclusion stands. This second pass asked a
different question: whether the gold label is **recoverable from the text the
model is actually shown**, and whether the harness scores every model on equal
terms. Those are not the same question, which is why none of what follows was
visible in the first pass.

Four of these findings have been fixed under task versioning (see `tasks_v2/`).
Two are recorded here without action, by decision on 2026-09-18.

### Labels that are not recoverable from the item

**What is compared:** for each task, the gold label of an item against the
predictions all ten benchmarked models made for that same item. Where every
model agrees on an answer that gold calls wrong, the likely cause is the task,
not the models.

- `osnabruegge_cross_domain_topic`. **Resolved 2026-09-18, and my first reading
  of it was wrong.** 123 of 500 items (25%) were answered "No Topic" by at least
  eight of the ten models while gold is a substantive domain, on mid-sentence
  fragments such as *"No, not all, but the more difficult and complex parts will
  be put into a Bill."* (gold **Freedom and Democracy**) and *"On behalf of my
  constituent, Mr Kelliher, I thank the committee for its deliberations. Motion
  agreed to."* (gold **Fabric of Society**). I inferred from this that the source
  annotators had the surrounding speech and the benchmark item did not.

  The replication capsule (Harvard Dataverse, doi:10.7910/DVN/CHTWUB) says
  otherwise. `target_corpus.csv` has nine columns and **none of them carry
  context** — no speech, debate, bill, date, speaker or position field. The
  coders saw exactly what the model is shown. Re-running the build script against
  the capsule reproduces `data/osnabruegge_cross_domain_topic.csv` byte for byte,
  so the import was correct all along.

  What the capsule does add is three independent validation coders on 250 of the
  4,165 rows. They establish a human ceiling:

  | Measure | Value |
  |---|---|
  | Coder 1 / 2 / 3 vs the published `topic_8` label | 0.616 / 0.652 / 0.652 |
  | All three coders agree with each other | 0.548 |
  | Pairwise coder agreement | 0.648 - 0.712 |
  | Cohen's kappa between coders | 0.572 - 0.649 |

  So this task is not broken and not mis-imported. It is genuinely
  under-determined by its input: expert coders reading the same excerpt reproduce
  the published label about two times in three. Model accuracy of 0.453 should be
  read against a ceiling near 0.65, not against 1.0. On the 23 sampled items that
  fall in the human-coded subset, model accuracy tracks human agreement
  monotonically (0.171 where no coder matched gold, 0.443 where all three did),
  which is consistent with the same explanation, though 23 items is too few to
  rest anything on.

  **Implication:** keep the task, and report its human ceiling alongside the
  model scores. Treating 1.0 as attainable here overstates how far models are
  from competent human coding, and dropping the task would discard the
  benchmark's only measured example of an irreducibly subjective coding problem.
  The build script now records the source URL and carries the three coder columns
  into `data/osnabruegge_cross_domain_topic.csv` so the ceiling is recomputable.
- `brandt_gtd_attack_type`. 47 of 500 items carry gold `Unknown`, scored at
  3.2% accuracy, with 414 of 470 model-item rows answering "Armed Assault". GTD
  assigns `Unknown` when the underlying source report does not specify the
  method. The benchmark input is the GTD event summary, which usually does
  describe it: *"Assailants attacked an Iraqi Army barracks near Baiji... At
  least four soldiers were killed."* The v1 prompt compounded this by defining
  `Unknown` as "the summary does not provide enough information", which is a
  different construct from the one GTD codes. **Fixed** in
  `tasks_v2/brandt_gtd_attack_type.yaml` via `ground_truth.exclude_labels`.
- `halterman_ccc_protest`. Gold contradicts the codebook printed in the task's
  own prompt. Three of the first five sampled items are gold `PROTEST` while the
  text matches the prompt's `MARCH` definition verbatim ("a 65-mile march from
  Milwaukee to Madison"; "marched from Sample Gates to the Monroe County
  Courthouse"). The dominant error across all models is PROTEST → MARCH / RALLY
  / DEMONSTRATION, which is what a codebook-gold mismatch of this shape
  predicts. **Cause established 2026-09-18; the task still cannot be rebuilt.**

  The upstream Crowd Counting Consortium release (Harvard Dataverse,
  doi:10.7910/DVN/6OPP7H, 72,181 events) confirms the mechanism. CCC's `type`
  field is a free-text, semicolon-separated descriptor, not a mutually exclusive
  category: 4,910 events (6.8%) carry several types at once, such as
  `protest; march`, `rally; march` and `march; rally`. Among the 2,883 events
  whose type includes "march", **1,546 (54%) list `protest` or `rally` first**.
  An import that collapses the field to its first listed value therefore labels a
  march as PROTEST about half the time, which is precisely the error the models
  make and precisely the mismatch against the codebook printed in the prompt.
  `protest` also functions as CCC's generic default, covering 34% of all events
  on its own.

  So the prompt is faithful to the source codebook (its PROTEST definition is
  verbatim from Halterman and Keith) and the gold labels are faithful to CCC. The
  two are simply not the same construct: the prompt supplies crisp, mutually
  exclusive definitions for a field that CCC populates as a loose multi-label
  descriptor. This is the failure mode the source paper is itself about.

  The task cannot be rebuilt from public data. CCC's compiled file carries only
  terse curator `notes` ("White House Peace Vigil continuous since June 3, 1981"),
  not the news-story prose the benchmark uses; none of the benchmark texts appear
  in it. The pairing of CCC events to retrieved news articles is Halterman and
  Keith's own work and is not published with the paper (no code or data URL
  appears on the ACL page or the arXiv HTML). Obtaining their file requires
  contacting the authors.

  **Implication:** treat this task's gold as a first-listed-type collapse of a
  multi-label field, and say so wherever the task is reported, rather than
  presenting it as four mutually exclusive protest forms. Until the authors'
  pairing file is available, do not rebuild it, and do not read its low scores as
  a model capability result.

**Implication for the main result:** none of these move the headline. Removing
GTD `Unknown` and dropping osnabruegge entirely both leave the best-API-minus-
best-local gap at +0.011 and local models ahead on 10 of 34 tasks, because the
defects penalise every model equally. They matter for per-task numbers and for
any claim about absolute task difficulty, not for the local-versus-API
comparison.

### Metric: labels with no gold support counted as zeros

**What is compared:** the published per-task headline F1 against the same
quantity computed over only those labels that actually occur in the sampled
gold column.

The categorical macro average ran over every label declared in the manifest.
A label with no gold support in the sample can only score 0, so it entered the
mean as a zero. `mellon_bes_mii_2024` declares 50 labels and 35 occur, so 30% of
its denominator was structurally zero: it reported accuracy 0.936 alongside
headline F1 0.505. Correcting it gives 0.781. `cap_crs_policy_topic` gains 0.037
and `haunss_papea_claims` 0.036; the other 31 tasks are unaffected.

**Implication:** the published cross-task comparisons were distorted in
proportion to how much unused slack each manifest's label list carried, which is
an artefact of manifest bookkeeping rather than anything about the tasks or the
models. **Fixed** in `code/scoring.py`, which is now the single implementation
for every summary and analysis script; `--legacy-all-labels` reproduces the
published numbers.

### Prompts that contradict themselves

Four prompts carry a verbatim source-codebook header instructing the model to
write the bare label "with no other text", followed by an appended block
instructing it to output only a JSON object and explicitly *not* the bare label:
`halterman_ccc_protest`, `halterman_keith_bfrs`, `halterman_keith_cmp`,
`osnabruegge_cross_domain_topic`. A model that follows the first instruction is
penalised by the parser. **Fixed** for `halterman_keith_bfrs` and
`halterman_keith_cmp` in `prompts_v2/`; the other two wait on the source
questions above.

### Codebook depth is not comparable across tasks

Prompt body length per label ranges from 24 characters (`haunss_papea_claims`,
28 labels) to 1,130 (`halterman_keith_bfrs`, 12 labels). Four tasks give a bare
list of labels with no definitions at all: `haunss_papea_claims`,
`plover_cameo_event`, `cap_crs_policy_topic`, `cap_party_platform_policy_topic`,
with `mellon_bes_mii_2024` borderline at 80. Measured task difficulty therefore
confounds the task with how much of the source codebook happened to be included.
The four bare-list tasks are also among the five largest API-over-local gaps.
**Fixed** for the two CAP tasks in `prompts_v2/`, which now carry short CAP
master-codebook definitions. `plover_cameo_event` and `haunss_papea_claims`
remain bare lists and should either gain definitions or have the omission
recorded as deliberate in `docs/prompts_provenance.md`.

### CAP major topic 5 carried a name that collides with topic 9

`code/cap_topic_labels.py` named topic 5 "Labor and Immigration" — the legacy
name from before immigration was split into its own topic 9 — while also
offering "Immigration". The label list therefore presented two plausible buckets
for the same content. In the v1 run, gold "Labor and Immigration" drew 37
"Social Welfare" and 10 "Civil Rights" predictions, and 5 gold "Immigration"
items were answered "Labor and Immigration". **Fixed** in `tasks_v2/` via
`ground_truth.label_map`, which renames gold values at load time and so leaves
the sampled items identical to v1.

### Recorded, not acted on

**Duplicate items (pseudo-replication).** Decision on 2026-09-18 was to document
and defer. Several tasks sample the same text many times over, because the
underlying corpus repeats it:

| Task | Duplicate share of 500 sampled items | Unique items | Most repeated text |
|---|---|---|---|
| `mellon_bes_mii_2024` | 56% | 222 | "Open-ended response: Covid" ×74 |
| `haunss_papea_fgz_forms` | 50% | 249 | "Protest-related text span: Kundgebung" ×44 |
| `bestvater_kavanaugh_stance` | 31% | 344 | ×14 |
| `gilardi_stance` | 23% | 383 | ×9 |
| `gilardi_relevance` | 10% | 448 | one tweet ×36 |

Two consequences. First, `mellon_bes_mii_2024` and `haunss_papea_fgz_forms` are
closer to string-lookup tasks than to text classification — the PAPEA items are
frequently a single German word ("Kundgebung", "Demonstration", "Warnstreik") —
and their near-ceiling accuracies (0.936 and 0.947) raise the benchmark's
reported level. Second, the paired bootstrap in `code/build_summary.py` resamples
by item, so 74 copies of one string count as 74 independent draws; the published
confidence intervals are narrower than the effective sample supports. Fixing
this means deduplicating, or clustering the bootstrap by text.

**One contradictory gold pair.** `data/gilardi_relevance.csv` rows 242 and 1588
hold identical tweet text with `gt_relevant` 1 and 0. This violates the entry
rule in `TODO.md` that conflicting duplicate labels are removed. It is one pair
out of 2,403 rows.

### Harness asymmetry between local and API models

**What is compared:** how the benchmark runner constrains output for local
models versus API models, holding the task and prompt fixed.

`code/benchmark.py` sends no `format` field to Ollama, so local models must
produce valid JSON and a valid label string from the prompt alone. API models
receive server-side enum enforcement (OpenAI strict `json_schema`, Anthropic
tool `input_schema`). All 95 parse failures in the published run belong to local
models, and they are mostly invalid label strings — "Intelligence Affairs",
"Homeland Security", "Tax-Exempt Organizations" on the CAP tasks — that an enum
schema makes impossible to emit.

The API advantage is concentrated in the same large-label-space tasks where this
bites: mean gap +0.045 across the seven tasks with local parse failures against
+0.002 across the other 27. But label-space size correlates with the gap at
r = 0.39 on its own, and among the nine tasks with eight or more labels those
*without* parse failures still show +0.027. Schema enforcement and task
difficulty are collinear, so the observational split cannot separate them.

**Implication:** this is the one finding that could change the headline, and it
cannot be settled from the existing run. The planned test holds model, GPU,
prompt and items fixed and varies only constrained decoding, by running each
open-weight checkpoint on Hive twice — once with vLLM JSON-schema constrained
decoding, once with plain prompting.

### Reproducibility gaps found along the way

- Eleven of 34 tasks have no `code/build_*_task.py`, so their cleaned CSVs
  cannot be regenerated from source and their import cannot be verified at all:
  `gilardi_relevance`, `ballard_incivility`, `gilardi_stance`,
  `chae_semeval_stance`, `ornstein_scotus_sentiment`, `wesleyan_creative_ads_2022`,
  `halterman_ccc_protest`, `halterman_keith_bfrs`, `halterman_keith_cmp`,
  `mellon_bes_mii_2024`, `rheault_line_of_fire_incivility`. This is why the
  `halterman_ccc_protest` and `osnabruegge_cross_domain_topic` questions above
  are blocked rather than answered.
- Where build scripts exist, only five record the source URL. Every new build
  script should.
- `data/halterman_ccc.csv` (50 rows) is referenced by no manifest.
- 175 `item_id` values are reused across different tasks, so any join over
  predictions must key on `(task, item_id)` rather than `item_id` alone.

### Glossary

- **Gold label**: the human-assigned label the benchmark scores predictions
  against, taken from the source paper or dataset.
- **Headline F1**: the benchmark's per-task metric. Positive-class F1 for binary
  tasks, macro F1 across labels for categorical tasks, mean positive-class F1
  across labels for multi-binary tasks.
- **Item-paired**: two task versions draw exactly the same rows, so their results
  can be compared item by item and bootstrapped in pairs. Versions that drop
  rows are not item-paired.
- **Constrained decoding / schema enforcement**: the inference backend restricts
  the model's output tokens to those permitted by a JSON schema, so an invalid
  label or malformed JSON cannot be produced.
- **Pseudo-replication**: treating repeated copies of the same observation as
  independent, which understates uncertainty.
