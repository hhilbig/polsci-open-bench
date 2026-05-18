# Prompt provenance

For each live task in the benchmark, this file records where the prompt in
`prompts/` came from and what modifications were made.

| Task | Prompt file | Source | Modifications |
|---|---|---|---|
| `gilardi_relevance` | `prompts/gilardi_relevance.txt` | Verbatim from Gilardi, Alizadeh, and Kubli (2023) PNAS replication template. | Replaced the final free-form label instruction with a JSON-output instruction returning `relevant` as `0/1`. |
| `gilardi_stance` | `prompts/gilardi_stance.txt` | Verbatim from Gilardi, Alizadeh, and Kubli (2023) SI Task 6 codebook. | Renamed the stance labels to `pro / neutral / contra` to match the replication data, then appended a JSON-output instruction. |
| `ballard_incivility` | `prompts/ballard_incivility.txt` | Derived. Ballard (2022) did not publish an LLM prompt; the task description is built from the paper's incivility definition and adjacent literature. | Fully derived. |
| `brandt_political_relevance` | `prompts/brandt_political_relevance.txt` | Derived from Brandt et al. (2025) and the public binary-classification replication file `raw_bc_data.tab`, whose positive examples are politics-relevant articles and whose negative examples are non-political news articles. | Converted the article-level filtering task into a zero-shot binary prompt with a strict JSON-output instruction. |
| `rheault_line_of_fire_incivility` | `prompts/rheault_line_of_fire_incivility.txt` | Derived from Rheault, Rayment, and Musulan (2019), which explicitly define uncivil tweets as those containing one or more of six elements: swear words, vulgarities, insults, threats, personal attacks on private life, or attacks targeted at groups. | Converted the paper definition into a zero-shot binary prompt and appended a strict JSON-output instruction. |
| `theocharis_dynamics_incivility` | `prompts/theocharis_dynamics_incivility.txt` | Derived from Theocharis et al. (2020) and the public `incivility-sage-open` training data, which expose tweet text plus direct `uncivil` labels. | Converted the source yes/no incivility label into a zero-shot binary prompt with a strict JSON-output instruction. |
| `ornstein_scotus_sentiment` | `prompts/ornstein_scotus_sentiment.txt` | Derived from Ornstein, Blasingame, and Truscott (2025) and the associated `promptr` task framing. | Converted a few-shot task into a zero-shot system prompt with the same three-class label space. |
| `chae_semeval_stance` | `prompts/chae_semeval_stance.txt` | Derived from the SemEval-2016 Task 6 stance definitions used by Chae and Davidson (2026). | Fully derived. Includes the SemEval edge case that criticism of a third party does not necessarily imply stance toward the target. |
| `halterman_ccc_protest` | `prompts/halterman_ccc_protest.txt` | Verbatim subset from Halterman and Keith (2026) CCC codebook. | Restricted to the four labels present in `halterman_ccc_hk2025.csv` and appended a JSON-output instruction. |
| `halterman_keith_bfrs` | `prompts/halterman_keith_bfrs.txt` | Verbatim from Halterman and Keith (2026) BFRS codebook. | Appended a JSON-output instruction only. |
| `douglass_icbe_sentence_event_type` | `prompts/douglass_icbe_sentence_event_type.txt` | Derived from Douglass et al. (2024) and the public ICBe agreed-event and sentence-corpus files, which expose aligned sentence text plus agreed event types. | Converted the public ICBe annotations into a sentence-level five-class task (`No Event`, `Action`, `Speech`, `Thought`, `Mixed`), added crisis-title context, and appended a strict JSON-output instruction. |
| `haunss_papea_fgz_forms` | `prompts/haunss_papea_fgz_forms.txt` | Derived from the seven most common original PAPEA form labels in `AppendixB3_fgz_forms.tab`, which also exposes the source `FORM` codebook strings for each human-coded snippet. | Restricted the task to the seven most common source labels, renamed them into benchmark-stable output strings, and appended a strict JSON-output instruction. |
| `halterman_keith_cmp` | `prompts/halterman_keith_cmp.txt` | Derived from the CMP Aggregation 1 domain structure used in Halterman and Keith's labeled split and the Manifesto Project CMP/MARPOR coding scheme. | Custom seven-domain prompt plus JSON-output instruction. |
| `osnabruegge_cross_domain_topic` | `prompts/osnabruegge_cross_domain_topic.txt` | Derived from the Osnabruegge, Ash, and Morelli (2023) eight-topic specification, using the same broad policy-domain label set as the paper's `target_corpus.csv` and the same CMP-style topic structure already used elsewhere in the benchmark. | Added brief domain definitions, added `No Topic`, and appended a strict JSON-output instruction. |
| `muller_fujimura_campaign_policy_area` | `prompts/muller_fujimura_campaign_policy_area.txt` | Derived from Müller and Fujimura (2025) and the public supervised sentence splits, which expose statement text plus `policy_area` labels directly. | Converted the replication labels into a 12-class statement-level policy-topic prompt, kept `No Policy Area` as an explicit class, and appended a strict JSON-output instruction. |
| `mellon_bes_mii_2024` | `prompts/mellon_bes_mii_2024.txt` | Derived from Mellon et al. (2024) MII issue coding task and the labeled response categories in the replication materials. | Custom 50-class issue prompt plus JSON-output instruction. |
| `wesleyan_creative_ads_2022` | `prompts/wesleyan_creative_ads_2022.txt` | Derived from the Wesleyan Media Project tone-coding construct used in Zhang et al. (2025). | Custom three-class tone prompt plus JSON-output instruction. Ad text is truncated at 4000 chars when necessary in the loader. |
| `burnham_polnli_entailment` | `prompts/burnham_polnli_entailment.txt` | Derived from Burnham, Kahn, Wang, and Peng (2025) and the public PolNLI test split, which exposes premise-hypothesis pairs and entailment labels directly. | Converted the source NLI coding into a binary JSON task where `entails = 1` means the hypothesis is supported by the premise. |

## Additional benchmark prompts

| Task | Prompt file | Source | Modifications |
|---|---|---|---|
| `toxicity_protests_es` | `prompts/toxicity_protests_es.txt` | Derived from González-Bustamante's public `toxicity-protests-ES` gold-standard file, which exposes Spanish-language protest tweets plus two human toxicity coder labels. | Kept only coder-agreement rows, converted the agreed label into a zero-shot binary toxicity prompt, and appended a strict JSON-output instruction. |
| `brandt_gtd_attack_type` | `prompts/brandt_gtd_attack_type.txt` | Derived from Brandt et al. (2025) and the public GTD multi-label file `raw_gtd_multilabel_data.tab`, which exposes GTD event summaries plus primary attack-type labels. | Converted the source GTD attack-type labels into a nine-class event-coding prompt with a strict JSON-output instruction. |
| `haunss_papea_claims` | `prompts/haunss_papea_claims.txt` | Derived from Haunss et al. (2025) and the public `fgz_papea_claims.tab` file, which exposes German protest-related sentence text plus PAPEA claim labels. | Kept the direct source claim labels after removing missing labels, conflicting exact-text labels, and duplicate text-label pairs; converted them into a 28-class JSON-output prompt. |
| `twitcivility_impoliteness` | `prompts/twitcivility_impoliteness.txt` | Derived from Pendzel et al. (2023) and the public TwitCivility Hugging Face release, which exposes political tweet text plus direct binary labels for `impoliteness` and `intolerance`. | Uses the direct `impoliteness` label as a binary JSON-output task; combines train and test splits. |
| `bestvater_wm_stance` | `prompts/bestvater_wm_stance.txt` | Derived from Bestvater and Monroe (2023) and the public `WM_tweets_groundtruth.tab` file, which exposes Women's March tweets plus direct stance and sentiment labels. | Uses the direct stance label as a binary target-aware stance task and appends a strict JSON-output instruction. |
| `erlich_ati_topics` | `prompts/erlich_ati_topics.txt` | Derived from Erlich et al. (2022) and the public `hc_new.tab` file, which exposes Mexican access-to-information request text plus multiple binary topic indicators. | Keeps the seven S8 request-subject labels as a multi-binary JSON task. |
| `plover_cameo_event` | `prompts/plover_cameo_event.txt` | Derived from Halterman et al. (2023) PLOVER gold-standard CAMEO records, which expose sentence text plus high-level event labels. | Removes the documentation record, keeps the 18 observed event types, and appends a strict JSON-output instruction. |
| `burnham_polnli_event_entailment` | `prompts/burnham_polnli_event_entailment.txt` | Derived from the event-extraction rows in the public Political DEBATE / PolNLI test split. | Restricts the existing PolNLI entailment formulation to event-extraction rows only. |
| `burnham_trump_stance` | `prompts/burnham_trump_stance.txt` | Derived from Burnham (2025) and the public Trump-stance data. | Converts the source stance labels into a three-class JSON-output prompt. |
| `burnham_covid_threat_minimization` | `prompts/burnham_covid_threat_minimization.txt` | Derived from Burnham (2025) and the public COVID threat-minimization labeled sample. | Converts the source binary label into a strict JSON-output task. |
| `dicocco_manifesto_populism` | `prompts/dicocco_manifesto_populism.txt` | Derived from Di Cocco and Monechi's public Italian manifesto-sentence annotations. | Converts the source populism label into a binary JSON-output prompt. |
| `bestvater_kavanaugh_stance` | `prompts/bestvater_kavanaugh_stance.txt` | Derived from Bestvater and Monroe's public Brett Kavanaugh stance tweets. | Uses the direct stance label as a binary target-aware stance task. |
| `politicause_causal_relation` | `prompts/politicause_causal_relation.txt` | Derived from the public PolitiCAUSE train, validation, and test splits. | Converts the source causal-relation label into a binary JSON-output task. |
| `cap_party_platform_policy_topic` | `prompts/cap_party_platform_policy_topic.txt` | Derived from public Democratic and Republican party-platform quasi-statements coded with Comparative Agendas Project major topics. | Restricts the label space to the observed CAP major-topic labels and appends a strict JSON-output instruction. |
| `cap_crs_policy_topic` | `prompts/cap_crs_policy_topic.txt` | Derived from public Congressional Research Service report titles and summaries coded with Comparative Agendas Project major topics. | Uses report title and summary text, restricts labels to observed CAP major topics, and appends a strict JSON-output instruction. |
| `agoraspeech_criticism_agenda` | `prompts/agoraspeech_criticism_agenda.txt` | Derived from Sermpezis et al. (2026) and the AgoraSpeech campaign-speech paragraphs with human-validated criticism-or-agenda labels and English translations. | Converts the source label into a two-class JSON-output task. |

## Rating summary

Closest to the original paper/codebook setup:

- `gilardi_relevance`
- `gilardi_stance`
- `halterman_ccc_protest`
- `halterman_keith_bfrs`

Derived rather than verbatim:

- `ballard_incivility`
- `brandt_political_relevance`
- `rheault_line_of_fire_incivility`
- `theocharis_dynamics_incivility`
- `ornstein_scotus_sentiment`
- `chae_semeval_stance`
- `douglass_icbe_sentence_event_type`
- `haunss_papea_fgz_forms`
- `halterman_keith_cmp`
- `osnabruegge_cross_domain_topic`
- `muller_fujimura_campaign_policy_area`
- `mellon_bes_mii_2024`
- `wesleyan_creative_ads_2022`
- `burnham_polnli_entailment`
- `toxicity_protests_es`
- `brandt_gtd_attack_type`
- `haunss_papea_claims`
- `twitcivility_impoliteness`
- `bestvater_wm_stance`
- `erlich_ati_topics`
- `plover_cameo_event`
- `burnham_polnli_event_entailment`
- `burnham_trump_stance`
- `burnham_covid_threat_minimization`
- `dicocco_manifesto_populism`
- `bestvater_kavanaugh_stance`
- `politicause_causal_relation`
- `cap_party_platform_policy_topic`
- `cap_crs_policy_topic`
- `agoraspeech_criticism_agenda`

## Policy for future additions

1. Prefer verbatim prompts from replication archives, supplemental material, or paper text.
2. If you need to modify a verbatim prompt for machine-readable output, document the exact change here.
3. If no prompt exists and you derive one from a codebook or paper description, mark it as derived here and note the source materials.
