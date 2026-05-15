# Task Inventory

This file is generated from task manifests. Regenerate it after adding, removing, or changing a task:

```bash
python3 code/task_inventory.py --write
```

## Counts

| Category | Directory | Count | Meaning |
| --- | --- | --- | --- |
| Canonical benchmark tasks | `tasks/` | 34 | Public 34-task benchmark manifests. |

## Task Families

| Family | Tasks |
| --- | --- |
| Causal relation detection | 1 |
| Event coding | 8 |
| Hypothesis-conditioned classification | 1 |
| Policy-topic coding | 7 |
| Relevance / Incivility | 7 |
| Rhetoric / Discourse Function | 1 |
| Rhetoric / Populism | 1 |
| Sentiment / Stance / Tone | 8 |

## Output Coverage

| Artifact | Path | Tasks | Models | Cells | Rows | Batch sizes |
| --- | --- | --- | --- | --- | --- | --- |
| Canonical serial predictions | `output/predictions.csv` | 34 | 9 | 306 | 147,825 |  |
| Canonical serial summary | `output/summary.csv` | 34 | 9 | 306 | 306 |  |
| Canonical batched predictions | `output/predictions_batched.csv` | 34 | 7 | 234 | 114,125 | 10, 20 |
| Canonical batched summary | `output/summary_batched.csv` | 34 | 9 | 540 | 540 | 1, 10, 20 |
| Local b=10 summary | `output/summary_batched_local_b10.csv` | 34 | 5 | 170 | 170 | 10 |

## Tasks

| Task | Family | Labels | Manifest |
| --- | --- | --- | --- |
| `gilardi_relevance` | Relevance / Incivility | binary | `tasks/gilardi_relevance.yaml` |
| `ballard_incivility` | Relevance / Incivility | binary | `tasks/ballard_incivility.yaml` |
| `gilardi_stance` | Sentiment / Stance / Tone | 3 labels | `tasks/gilardi_stance.yaml` |
| `chae_semeval_stance` | Sentiment / Stance / Tone | 3 labels | `tasks/chae_semeval_stance.yaml` |
| `ornstein_scotus_sentiment` | Sentiment / Stance / Tone | 3 labels | `tasks/ornstein_scotus_sentiment.yaml` |
| `wesleyan_creative_ads_2022` | Sentiment / Stance / Tone | 3 labels | `tasks/wesleyan_creative_ads_2022.yaml` |
| `halterman_ccc_protest` | Event coding | 4 labels | `tasks/halterman_ccc_protest.yaml` |
| `halterman_keith_bfrs` | Event coding | 12 labels | `tasks/halterman_keith_bfrs.yaml` |
| `halterman_keith_cmp` | Policy-topic coding | 7 labels | `tasks/halterman_keith_cmp.yaml` |
| `mellon_bes_mii_2024` | Policy-topic coding | 50 labels | `tasks/mellon_bes_mii_2024.yaml` |
| `osnabruegge_cross_domain_topic` | Policy-topic coding | 8 labels | `tasks/osnabruegge_cross_domain_topic.yaml` |
| `rheault_line_of_fire_incivility` | Relevance / Incivility | binary | `tasks/rheault_line_of_fire_incivility.yaml` |
| `haunss_papea_fgz_forms` | Event coding | 7 labels | `tasks/haunss_papea_fgz_forms.yaml` |
| `brandt_political_relevance` | Relevance / Incivility | binary | `tasks/brandt_political_relevance.yaml` |
| `douglass_icbe_sentence_event_type` | Event coding | 5 labels | `tasks/douglass_icbe_sentence_event_type.yaml` |
| `muller_fujimura_campaign_policy_area` | Policy-topic coding | 12 labels | `tasks/muller_fujimura_campaign_policy_area.yaml` |
| `burnham_polnli_entailment` | Hypothesis-conditioned classification | binary | `tasks/burnham_polnli_entailment.yaml` |
| `theocharis_dynamics_incivility` | Relevance / Incivility | binary | `tasks/theocharis_dynamics_incivility.yaml` |
| `toxicity_protests_es` | Relevance / Incivility | binary | `tasks/toxicity_protests_es.yaml` |
| `brandt_gtd_attack_type` | Event coding | 9 labels | `tasks/brandt_gtd_attack_type.yaml` |
| `haunss_papea_claims` | Event coding | 28 labels | `tasks/haunss_papea_claims.yaml` |
| `twitcivility_impoliteness` | Relevance / Incivility | binary | `tasks/twitcivility_impoliteness.yaml` |
| `bestvater_wm_stance` | Sentiment / Stance / Tone | binary | `tasks/bestvater_wm_stance.yaml` |
| `erlich_ati_topics` | Policy-topic coding | 7 binary labels | `tasks/erlich_ati_topics.yaml` |
| `plover_cameo_event` | Event coding | 18 labels | `tasks/plover_cameo_event.yaml` |
| `burnham_polnli_event_entailment` | Event coding | binary | `tasks/burnham_polnli_event_entailment.yaml` |
| `burnham_trump_stance` | Sentiment / Stance / Tone | 3 labels | `tasks/burnham_trump_stance.yaml` |
| `burnham_covid_threat_minimization` | Sentiment / Stance / Tone | binary | `tasks/burnham_covid_threat_minimization.yaml` |
| `dicocco_manifesto_populism` | Rhetoric / Populism | binary | `tasks/dicocco_manifesto_populism.yaml` |
| `bestvater_kavanaugh_stance` | Sentiment / Stance / Tone | binary | `tasks/bestvater_kavanaugh_stance.yaml` |
| `politicause_causal_relation` | Causal relation detection | binary | `tasks/politicause_causal_relation.yaml` |
| `cap_party_platform_policy_topic` | Policy-topic coding | 21 labels | `tasks/cap_party_platform_policy_topic.yaml` |
| `cap_crs_policy_topic` | Policy-topic coding | 21 labels | `tasks/cap_crs_policy_topic.yaml` |
| `agoraspeech_criticism_agenda` | Rhetoric / Discourse Function | 2 labels | `tasks/agoraspeech_criticism_agenda.yaml` |
