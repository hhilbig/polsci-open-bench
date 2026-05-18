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
