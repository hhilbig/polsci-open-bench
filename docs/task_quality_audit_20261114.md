# Task quality audit of the 33 release tasks, 2026-11-14

Third audit of the tasks in release `refresh_20261106` (after the two in
[`task_source_fidelity_audit.md`](task_source_fidelity_audit.md)). Three
reviewers each took 11 tasks and worked from the scored 100-item sample, the
data CSVs, the prompts and the release scores. Nothing in the repository was
changed by the audit.

**Verdict.** Four tasks produce scores that do not measure what the prompt
asks, and should be dropped or rebuilt before the next release:
`burnham_covid_threat_minimization`, `ornstein_scotus_sentiment`,
`bestvater_wm_stance` and `brandt_political_relevance`. A second, larger
problem cuts across tasks: the release scored the v1 prompts for five tasks
whose fixes the September audit records as made (`tasks_v2/`), so those fixes
are not in the published numbers. Thirteen more tasks need fixes (gold that
follows rules the prompt never states, noisy negatives, or one or two classes
with 1 to 2 items that decide the macro-F1). The rest are usable with or
without a caveat.

## How the audit worked

The scored items are in `output/sidecar/refresh_20260910/panel.json`; the
reviewers matched them to the release by per-class support in `classes.csv`
and by the per-task prompt digests. To find items worth reading, each reviewer
flagged items where most of the 8 frontier API models in
`comparison_predictions.csv` agree on an answer different from gold, then read
all of those plus a random 8 to 20 more per task and judged whether the gold
label is defensible under the task's own prompt.

Two cautions for reading the counts below:

1. "Clearly wrong gold" counts are one reviewer's judgments on about 15 to 25
   items per task. They are not error rates: the flagged items were chosen
   because models disagree with gold, so they over-represent errors.
2. Using model disagreement to pick items could favor the models' reading of a
   task. The blocking findings below do not rest on that: each also has
   evidence that does not use model output (a keyword that predicts gold, a
   trivial baseline, gold that contradicts the prompt's definition).

## Cross-cutting findings

| Problem | Evidence | Fix |
|---|---|---|
| Release scored v1 prompts that the September audit lists as fixed | Prompt digests match `tasks/`, not `tasks_v2/`, for `brandt_gtd_attack_type` (12 `Unknown` items still scored), `cap_crs_policy_topic`, `cap_party_platform_policy_topic` (still offer "Labor and Immigration" and "Immigration"), `halterman_keith_bfrs`, `halterman_keith_cmp` (contradictory "no other text" line still present) | Score these five on v2 in the next release, or describe the published numbers as v1 |
| Release manifest misdescribes scoring | `manifest.scoring` says absent-class F1 "remains zero"; the published headline averages present classes only (e.g. `halterman_keith_bfrs` median 0.753 present-only, 0.628 over all classes) | Correct the wording; the numbers are right |
| Classes with 1 to 2 gold items decide macro-F1 | `wesleyan` "unclear" (2 items): rank correlation of 3-class vs 2-class headline 0.60; `muller_fujimura` three classes with 2 items: 0.38; `erlich` `Other` (1 item) lowers the headline about 0.12; `osnabruegge` "No Topic" (2 items, predicted about 22 times); `gilardi_stance` `pro` (5) | Minimum gold support per class in the sample, or drop such classes from the headline |
| Text residue | Literal `\n` / `\"` escapes (`bestvater_wm`, `burnham_trump`, `haunss_papea_claims` 324 rows, `rheault` 567 rows), mojibake in `cap_party_platform` (10.9% of rows), scraped field markers in 97% of `wesleyan` texts | Clean in the builders |
| Duplicate texts in the sample | `bestvater_kavanaugh` 62% of rows duplicate another row's text (87 unique in 100); `mellon_bes_mii` 62 unique in 100; `haunss_papea_fgz_forms` 33 copies | Deduplicate before sampling, or cluster the bootstrap by text |

## Tasks to drop or rebuild

| Task | Problem | Evidence |
|---|---|---|
| `burnham_covid_threat_minimization` | Gold tracks whether the tweet mentions "lockdown", not threat minimization | In the 293-row file, 89% of tweets mentioning "lockdown" are gold 1 against 10% of the rest (rechecked). Pro-public-health texts are gold 1 (`covid_110`, `covid_225`). Nine rows start with a literal `\n`, which looks like a broken multiline parse. 27 of 43 models score below always-1 (F1 0.45). |
| `ornstein_scotus_sentiment` | Gold codes the tweet's general tone; the prompt asks for sentiment toward the ruling | "I SAY GOOD" about the ruling is gold Negative; a tweet praising the dissenters is Positive. Many tweets are about other cases. Best model 0.71 macro-F1; 25 items missed by all 8 frontier models. |
| `bestvater_wm_stance` | 90 of 100 items are gold "support"; no model beats always predicting it | Always-1 gives F1 0.947; best model 0.935 (rechecked). Many gold-1 items are sarcastic or anti-march, or depend on an unseen image. |
| `brandt_political_relevance` | Gold is UK party politics; the prompt calls government economic policy relevant | Gold-0 items include Venezuelan land reform and a Japanese minister's forecast. Recall is 1.0 for all models, so the score measures how narrowly a model reads "political". |

## Tasks to fix before the next release

| Task | Main problem | Suggested fix |
|---|---|---|
| `brandt_gtd_attack_type` | v1 scored; prompt calls any targeted killing Assassination, GTD reserves it for prominent targets and applies a method hierarchy (about 9 items) | Score v2; add GTD's attack-type hierarchy |
| `cap_crs_policy_topic` | v1 scored; 38 of 100 items are title-only with an empty "Summary:"; 27 titles in the file carry conflicting labels | Score v2; drop the empty summary field |
| `cap_party_platform_policy_topic` | v1 scored; CAP conventions not in the prompt (territories as Public Lands); fragments and headers as items; mojibake | Score v2; repair encoding |
| `halterman_keith_bfrs` | Gold codes domestic and honour killings as ASSASSINATION while the prompt's own OTHER example is one (9 of 31 sampled ASSASSINATION items) | State the rule or recode; score v2 |
| `douglass_icbe_sentence_event_type` | Crisis start/end coded Thought without the prompt saying so; unaligned sentences filled as "No Event"; max F1 0.45 | Add the ICBe rule; drop unaligned sentences |
| `dicocco_manifesto_populism` | 7 of 13 positives rejected by all frontier models; gold broader than the derived definition; prompt does not say the text is Italian | Take the definition from the source codebook |
| `erlich_ati_topics` | `Other` has 1 item; `Evaluation` gold means requests for statistics, not audits | Drop `Other` from the headline; redefine `Evaluation` from the source codebook |
| `ballard_incivility` | 12 positives; inconsistent gold (#GOPTaxScam 1, #KochCartel 0); "pre-filtered as divisive" claim false | Remove the claim; report a human ceiling |
| `muller_fujimura_campaign_policy_area` | Gold follows Diet committee jurisdiction (disaster prevention and public safety under Cabinet), which the prompt does not describe | Rewrite class definitions from committee jurisdictions |
| `wesleyan_creative_ads_2022` | 37% of rows name "Down-ballot candidate/governor" instead of a candidate; "unclear" (2 items) decides the ranking | Use sponsor or target; score promote/attack only |
| `plover_cameo_event` | Bare label list; PLOVER conventions (intent counts as AGREE) invisible; item ids contain the gold label | Add PLOVER definitions; hash ids |
| `theocharis_dynamics_incivility` | Noisy negatives (about 10 of 13 flagged items); 222 URL-only rows; ids stored as floats | Drop URL-only rows; relabel flagged items |
| `twitcivility_impoliteness` | Noisy labels both ways (about 14 of 19 flagged items); all 43 models between 0.48 and 0.64 | Relabel flagged items or report a ceiling |
| `burnham_polnli_event_entailment` | All its pairs are a subset of `burnham_polnli_entailment`, so one subtask counts twice; near ceiling (median 0.978); a keyword rule reaches 0.93 | Drop or merge into the parent task |

## Keep with a caveat or as is

Keep with a caveat: `agoraspeech_criticism_agenda` (machine-translated
English; labels may be GPT labels checked by humans, not verified),
`bestvater_kavanaugh_stance` (heavy duplication), `burnham_trump_stance` (13%
of rows do not mention Trump; about 7 clearly wrong among 17 flagged),
`chae_semeval_stance` (known SemEval noise), `gilardi_stance`,
`halterman_keith_cmp` (score v2), `haunss_papea_claims` (10 of 21 classes with
2 items or fewer), `haunss_papea_fgz_forms` (a keyword lookup, near ceiling),
`mellon_bes_mii_2024`, `osnabruegge_cross_domain_topic` (recorded fixes still
not applied), `politicause_causal_relation` (many causal statements coded 0),
`rheault_line_of_fire_incivility`, `toxicity_protests_es` (noisy negatives).

Keep as is: `burnham_polnli_entailment`, `gilardi_relevance`.

## Implication

The headline mean and model ranking rest partly on tasks whose gold labels do
not match their prompts. Before the next release: drop or rebuild the four
tasks above, score the five v2 tasks, and set a minimum class support in the
sample. A rescoring of the November release without the four tasks would show
how much the ranking depends on them; that has not been done.

## Fixes made, 2026-11-14

Every task in the two tables above now has a v2 manifest in `tasks_v2/`; the
sources, rules and counts are in the 2026-11-14 section of
[`prompts_provenance.md`](prompts_provenance.md). No v1 file was changed. What
the fixes found about the four blocking tasks:

1. `burnham_covid_threat_minimization`: v1 scored models against Burnham's NLI
   model predictions (`threatmin`), not the human label (`non_comp`). Rebuilt
   on the human label from the original CSV. The "lockdown" pattern weakens
   (0.71 against 0.14) and is expected under the human definition, which
   includes opposing lockdowns.
2. `ornstein_scotus_sentiment`: the authors coded each tweet's own sentiment,
   and their few-shot examples include tweets about the Court in general. The
   v2 prompt reproduces their pre-registered prompts; no topic filter is
   possible from the source.
3. `bestvater_wm_stance`: rebuilt on the authors' class-balanced training set
   (about 50% opposing).
4. `brandt_political_relevance`: gold is the BBC News "politics" section, not
   conflict or political relevance. No coded political-relevance construct can
   be recovered from the source, so the task is dropped (see below).

## Decisions for the next release (2026-11-14)

Hanno decided these on 2026-11-14:

1. `brandt_political_relevance` is dropped from the next release
   Its v1 manifest stays unchanged, because the frozen panels load it; the
   next release's task list leaves it out, and no v2 manifest is kept.
2. `burnham_polnli_event_entailment` is dropped from the next release
   (v1 manifest unchanged, as above), since all its pairs are already in `burnham_polnli_entailment`.
3. `halterman_keith_bfrs`: ASSASSINATION is to be left out of the headline,
   because gold codes domestic and honour killings as ASSASSINATION while the
   codebook's OTHER example is one. Implemented with the manifest field
   `headline_exclude_labels`: the class stays an answer option and its items
   stay in the sample, so the conflict can still lower OTHER's F1 when a model
   follows the codebook.
4. `ornstein_scotus_sentiment` v2 drops the 12 tweets that the prompt shows as
   worked examples (945 to 933 rows).
5. The stratified draw (`code/release_sampling.py`, at most 5 items per class,
   reserved items capped at 60% of the 100) draws from the 500-item pool the
   loader returns, not the full data file. This keeps the loader and item ids
   unchanged. Known consequence: in the 21-class CAP tasks the floor is 2 items
   per class, and CAP Culture has 1 item in the pool.

Still open: `haunss_papea_claims` and the `osnabruegge` and `halterman_ccc`
prompt issues are unchanged.

## Not checked

Source replication files and codebooks were not re-opened for most tasks, so
statements about what gold "means" are inferred from the items. Item-level
predictions of the 35 non-frontier models were not used. No fuzzy
near-duplicate search. The `panel.json` file hash did not match the release's
`panel_sha256` for one reviewer although per-task digests and class supports
matched; this is unresolved.

## Glossary

- **Gold**: the human label the benchmark scores against.
- **Frontier models**: the 8 API models whose item-level predictions are in
  `comparison_predictions.csv`.
- **Headline F1**: positive-class F1 for binary tasks, macro-F1 over present
  classes for categorical tasks.
- **v1 / v2**: `tasks/` + `prompts/` versus the corrected `tasks_v2/` +
  `prompts_v2/` from the September audit.
