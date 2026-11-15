# Prompt provenance

For each live task in the benchmark, this file records where the prompt in
`prompts/` came from and what modifications were made.

For a stricter construct-level audit of whether each task matches the source
label and source objective, see
[`docs/task_source_fidelity_audit.md`](task_source_fidelity_audit.md).

| Task | Prompt file | Source | Modifications |
|---|---|---|---|
| `gilardi_relevance` | `prompts/gilardi_relevance.txt` | Verbatim from Gilardi, Alizadeh, and Kubli (2023) PNAS replication template. | Replaced the final free-form label instruction with a JSON-output instruction returning `relevant` as `0/1`. |
| `gilardi_stance` | `prompts/gilardi_stance.txt` | Verbatim from Gilardi, Alizadeh, and Kubli (2023) SI Task 6 codebook. | Renamed the stance labels to `pro / neutral / contra` to match the replication data, then appended a JSON-output instruction. |
| `ballard_incivility` | `prompts/ballard_incivility.txt` | Derived. Ballard (2022) did not publish an LLM prompt; the task description is built from the paper's incivility definition and adjacent literature. | Fully derived. |
| `brandt_political_relevance` | `prompts/brandt_political_relevance.txt` | Derived from Brandt et al. (2025) and the public binary-classification replication file `raw_bc_data.tab`, whose positive examples are conflict- or politics-relevant articles and whose negative examples are not relevant to that target. | Converted the article-level filtering task into a zero-shot binary prompt with a strict JSON-output instruction. |
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
| `haunss_papea_claims` | `prompts/haunss_papea_claims.txt` | Derived from Haunss et al. (2025) and the public `fgz_papea_claims.tab` file, which exposes German protest-related sentence text plus PAPEA claim labels. | Kept the mapped source claim labels after removing missing labels, unsupported labels, conflicting exact-text labels, and duplicate text-label pairs; converted them into a 28-class JSON-output prompt. |
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
| `benoit_manifesto_economic_position` | `prompts/benoit_manifesto_economic_position.txt` | Verbatim from the Benoit, Conway, Lauderdale, Laver, and Mikhaylov (2016) coder instructions and form (`02 Create a New CF job/setup/instructions.html` and `cml.txt` in github.com/kbenoit/CSTA-APSR); the economic-policy definitions, scale labels, and economic example are unchanged. | The source first asks coders to choose economic, social, or neither; the task keeps only sentences a majority of experts assigned to economic policy, so the prompt states the domain and drops the social-policy definitions and example. The red highlight becomes `>>> <<<` markers around the target sentence, the two context sentences on each side are kept, a typo ("levels of levels") is fixed, and a JSON-output instruction is appended. |
| `benoit_manifesto_social_position` | `prompts/benoit_manifesto_social_position.txt` | Verbatim from the same Benoit et al. (2016) instructions and form; the social-policy definitions, scale labels, and social example are unchanged. | Same changes as the economic task, with the economic-policy definitions and example dropped. |
| `gpd_speech_populism` (long text, `tasks_ext/`) | `prompts/gpd_speech_populism.txt` | The three category descriptions are verbatim from the Global Populism Database v2.1 codebook (Hawkins et al., doi:10.7910/DVN/LFTQEZ), which coders apply to whole speeches. | Added a one-sentence definition of populist discourse and the instruction to judge the speech as a whole. Coders also had a fuller rubric of populist elements and anchor texts, which the prompt does not include. Gold is the mean of the coder grades for each speech file, rounded to the rubric category with the codebook's rule (0.5 rounds to 1, 1.5 to 2); the GPD's own `averagerubric` is not used because it pools two texts per slot for Trump's first term. Research-assistant headers that name the speech type are stripped from the texts. JSON-output instruction appended. |
| `scdb_opinion_issue_area` (long text, `tasks_ext/`) | `prompts/scdb_opinion_issue_area.txt` | Issue-area names and, under each, the descriptions of its issues are verbatim from the Supreme Court Database online codebook (variables `issueArea` and `issue`, scdb.wustl.edu). SCDB coders assign the issue from the case record; LexGLUE pairs the resulting issue area with the opinion text. | Miscellaneous (3 cases) and Private Action (absent from LexGLUE) dropped from the label list. The opinion may or may not begin with the syllabus (about 42% do). JSON-output instruction appended. |
| `kiesel_hyperpartisan_article` (long text, `tasks_ext/`) | `prompts/kiesel_hyperpartisan_article.txt` | The definition of a hyperpartisan article is quoted from Kiesel et al. (2019); the two classes follow the paper's binarization of the crowd's 5-point scale (ratings 4-5 hyperpartisan, 1-2 not). | Crowd workers rated bias on the 5-point scale rather than answering a binary question; the prompt describes the two ends in the scale's own wording. JSON-output instruction appended. |
| `cap_executive_order_policy_topic` (long text, `tasks_ext/`) | `prompts/cap_executive_order_policy_topic.txt` | Identical to the v2 `cap_crs_policy_topic` prompt (verbatim CAP master-codebook descriptions and PAP 2019 coding rules) except that it names an executive order instead of a CRS report, so the two tasks form a short-versus-long pair. Major topic 5 is `Labor`, mapped from the source name via `label_map`. | CAP codes correspond to the dataset's Description column, which is in practice the order's title (fuzzy match >= 90 for 97% of orders); the model sees that title plus the full order body. The CAP codes for EO 13673 and 13674 are swapped in the source and are swapped back by the builder. |
| `lemens_gallego_tweet_position` | `prompts/lemens_gallego_tweet_position.txt` | Verbatim from the Le Mens and Gallego (2025) LLM instruction for single tweets (`config_us_tweets_crowdsourcing_118_CO.py` in doi:10.7910/DVN/YFM0BW); human raters answered the same question on a 0-100 slider with a "Not Applicable" option. | Replaced the paper's "key Score" output line with a JSON-output instruction for the `score` field. |
| `lemens_gallego_tweet_position3` | `prompts/lemens_gallego_tweet_position3.txt` | Derived from the same question. | Asks for `left`, `center`, or `right` instead of a 0-100 score; gold labels are equal thirds of the mean human rating (cutoffs 33.3 and 66.7), a benchmark choice rather than the authors'. |

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

## v2 prompts (2026-09-18)

`prompts_v2/` holds corrected prompts for the tasks that the second source-fidelity
audit found defective. The v1 files above are unchanged, so the published results
stay reproducible; v2 manifests in `tasks_v2/` point at the v2 prompts. The
September v2 prompts first carried a `Source:` footer; since 2026-11-14 no v2
prompt has one, because the whole prompt file is sent to the model. Provenance
is recorded only in this document.

| Task | v2 prompt | Change | Item-paired with v1 |
|---|---|---|---|
| `brandt_gtd_attack_type` | `prompts_v2/brandt_gtd_attack_type.txt` | Removed the `Unknown` label. The v1 prompt defined it as "the summary does not provide enough information", but GTD assigns it when the underlying *source report* did not specify the method, which the model cannot see. Rows are dropped via `ground_truth.exclude_labels`. | No, rows are dropped |
| `cap_crs_policy_topic` | `prompts_v2/cap_crs_policy_topic.txt` | Renamed major topic 5 to `Labor`; added CAP master-codebook definitions; stopped promising a summary unconditionally (6,492 of 16,510 rows have none). | Yes |
| `cap_party_platform_policy_topic` | `prompts_v2/cap_party_platform_policy_topic.txt` | Renamed major topic 5 to `Labor`; added CAP master-codebook definitions. | Yes |
| `halterman_keith_bfrs` | `prompts_v2/halterman_keith_bfrs.txt` | Removed the source "Output Reminder" line, which told the model to write the bare label "with no other text" and contradicted the appended JSON-output block. | Yes |
| `halterman_keith_cmp` | `prompts_v2/halterman_keith_cmp.txt` | Same "Output Reminder" removal. | Yes |

### Open prompt issues

1. **CAP definitions** were checked against the CAP master codebook on
   2026-11-14 and replaced with verbatim text (see the next section). Resolved.
2. **`haunss_papea_claims` is still a bare label list** (28 labels, about 24
   characters of prompt per label). The PAPEA claim codebook is not in the repo.
   `plover_cameo_event` now has verbatim PLOVER definitions.
3. **Two prompts still contradict themselves.** `halterman_ccc_protest` and
   `osnabruegge_cross_domain_topic` carry the "Output Reminder" line removed
   from the Halterman-Keith v2 prompts. They were left alone because both tasks
   have unresolved gold-label questions (see the second audit section of
   [`docs/task_source_fidelity_audit.md`](task_source_fidelity_audit.md)).

## v2 prompts and data (2026-11-14)

Fixes from the third task audit
([`task_quality_audit_20261114.md`](task_quality_audit_20261114.md)). Each
task was fixed against its downloaded source by a separate agent. Rules
followed: v1 files unchanged and every changed builder still writes its v1 CSV
byte for byte; no gold label changed by judgment (gold changes only where the
source pipeline was shown to be wrong); items dropped only by stated mechanical
rules; definitions quoted from the authors' materials where those could be
obtained. "Paired" means v2 samples the same items as v1.

| Task | v2 prompt source | Data and gold changes | Paired |
|---|---|---|---|
| `burnham_covid_threat_minimization` | Non-compliance definition quoted from Block Jr et al. (2022, Soc Sci Med, sect. 4.2) and Burnham (2024, arXiv 2305.01723, sect. 5.1); list of mitigation practices ours, from Burnham's Appendix B. Block et al.'s supplement could not be obtained. | **v1 gold was a model prediction.** v1 used `threatmin`, Burnham's NLI zero-shot prediction; the human label is `non_comp` (Burnham's `sect_4_analysis.R` line 78 computes `mcc(training$non_comp, training$threatmin)`). The Dataverse `.tab` export also split multiline tweets (319 rows against 300 in the original CSV). v2 reads the original CSV, uses `non_comp`, drops 5 exact duplicates: 295 rows (0: 217, 1: 78). | No |
| `burnham_trump_stance` | Category definitions verbatim from Burnham's GPT-4 prompt (`in_context_classification.ipynb`, doi:10.7910/DVN/XS0ULP), typo "the might" corrected; the sentence that the target may be implicit is ours, following Burnham sect. 3.3. | Original CSVs fix 26 column-shifted `.tab` rows (12 had wrong labels) and escape residue in about 1,022 texts. Texts without a Trump reference are kept, as in Burnham's design. 4,773 rows. | No |
| `bestvater_wm_stance` | Annotation question paraphrased from Bestvater and Monroe (2023) appendix sect. 1 (approving vs opposing, no neutral); examples ours, from paper p. 237. | Restricted to the authors' class-balanced training set (`balanced_train == 1`), which holds every opposing tweet; 57 duplicates dropped. 19,516 to 5,253 rows (0: 2,598, 1: 2,655). The 73% drop follows the authors' own subset. | No |
| `bestvater_kavanaugh_stance` | Same appendix question, target "the confirmation of Brett Kavanaugh"; note on negative tone from appendix Table A3. | Exact duplicate texts dropped (one row per text): 3,636 to 1,398 rows (1: 711, 0: 687). Original CSV fixes `\"` in about 9% of texts. | No |
| `brandt_political_relevance` | **Dropped from the next release (decision 2026-11-14); no v2.** Gold is the BBC News "politics" category: all 322 source rows match the BBC corpus folders (Greene and Cunningham 2006), gold 1 the politics folder and gold 0 business, entertainment, sport and tech. The paper's description (expert-coded conflict relevance) does not match the data. | None. | n/a |
| `brandt_gtd_attack_type` | Adds the GTD Codebook (Aug 2021, pp. 24-26, Wayback copy) attack-type paragraph, hierarchy rule and all eight definitions verbatim (Assassination limited to "specific, prominent individuals"). | As in September (`Unknown` excluded). | No |
| `ornstein_scotus_sentiment` | Both pre-registered preambles and the 12 few-shot examples from Ornstein, Blasingame and Truscott (2025, Table A3; promptr `scotus_tweets_examples`) verbatim; the authors coded each tweet's own sentiment. First paragraph ours. | New builder `code/build_ornstein_scotus_sentiment_task.py` reproduces the v1 CSV from promptr's `scotus_tweets.rda`. v2 drops the 12 tweets used as few-shot examples: 945 to 933 rows. | No |
| `cap_crs_policy_topic` | Each label gives the CAP Master Codebook's verbatim general-subtopic description and subtopic names; PAP 2019 codebook guidelines 1 and 4 and the Agricultural Trade rule quoted verbatim. | Source `;` line breaks replaced by spaces (7,895 summaries, 19 titles); 2 duplicate or conflicting texts and 54 rows from 26 titles with more than one gold label dropped: 16,454 rows. v1 cannot be rebuilt from the current CAP release (renumbered ids), so v2 is built from the v1 CSV. | No |
| `cap_party_platform_policy_topic` | As `cap_crs_policy_topic`. | Encoding residue repaired by a mapping table in the builder (0 rows keep mojibake), `[p.N]` page markers removed, integer years, 444 bare section headers dropped (8 words or fewer, no final punctuation, all caps or title case): 36,932 rows. | No |
| `halterman_keith_bfrs` | Removed `""",` residue and the footer. | None to the data. ASSASSINATION is left out of the headline (`headline_exclude_labels`, decision 2026-11-14) because the Halterman and Keith codebook has no rule that puts domestic or honour killings under ASSASSINATION, and its OTHER example is such a killing, but gold codes about 207 of them ASSASSINATION. The original BFRS codebook is not public. | Yes |
| `halterman_keith_cmp` | Footer removed. | None. | Yes |
| `douglass_icbe_sentence_event_type` | Think/Say/Do definitions quoted from Douglass et al. (2024, ICBeLLM) and the Think/Say/Do groups from the ICBe paper's Table 1, including "Alert (Start/End Crisis)"; the start/end-crisis sentence and label lines are ours. No ICBe codebook is public. | No Event kept only where coders answered "No Action"; sentences coders marked "Background", "Covered by Previous Sentence" or "Error" were never coded and are dropped, as are 97 sentences absent from the agreed-event file and 11 fragments. 12,676 to 9,119 rows (28% drop; No Event 217). | No |
| `plover_cameo_event` | PLOVER manual 0.6b1 (commit b4c42af) definitions and selected "Potential ambiguities" verbatim, including that future-tense cooperative actions imply intentions; CAMEO mapping from the 2016 README. FIGHT has no definition in the manual. | Ids are `plover_` plus a text hash, so they no longer contain the label; old id kept as `source_original_id`. Same 312 rows. | No (new ids) |
| `dicocco_manifesto_populism` | Coding rule and ideational definition quoted from Di Cocco and Monechi (2022, Political Analysis, sect. 5.2.4 and p. 313); states Italian, German (SVP rows) and Ladin, and PDF-extraction noise. | None. | Yes |
| `erlich_ati_topics` | Six topic definitions verbatim from the Palmer-Rubin et al. (2025) appendix variable table, which describes classifiers trained on the same hand-coding; the original S8 questionnaire is available only on request. | `Other` dropped from the labels by a rule (multi-binary labels under 2% pool prevalence; `Other` is 1.5%). Items unchanged. | Yes |
| `muller_fujimura_campaign_policy_area` | Classes defined as House of Representatives standing-committee jurisdictions (our translation of Rules of the House Art. 92), because the source's coding instructions (SI sect. G) say each area "corresponds to one Diet committee"; 29 example statements verbatim from sect. G, 4 that are dataset items removed; topic summaries ours. | Escaped quotes cleaned in 15 texts. Same rows and gold. | No (new file) |
| `wesleyan_creative_ads_2022` | WMP's three tone answer options verbatim; coding per referenced candidate; rest ours. The question wording could not be obtained. | Candidate taken from the coder-entered `othercand1_txt` for non-federal candidates (1,094 rows); 14 ads with no identifiable candidate dropped; field markers and literal `\n` cleaned; 4,000-character truncation removed (longest ad 6,970 tokens). 2,877 rows (unclear: 77). | No |
| `mellon_bes_mii_2024` | Category descriptions verbatim from the authors' coding prompt `main_prompt5.txt` (replication repo commit 33c661a). | `constitution` removed from the labels (no description, not in the authors' prompt, no gold item). | Yes |
| `gilardi_stance` | Footer removed (v1 sent it to the model). | None. | Yes |
| `ballard_incivility` | v1 derived definition kept, since the codebook is not public; the unsupported "pre-filtered as divisive" claim and the in-prompt note removed. | New builder `code/build_ballard_incivility_task.py`: HTML entities, mojibake and doubled quotes repaired; 307 truncated retweets, 20 conflicting-label duplicates and 105 repeat copies dropped: 2,682 to 2,250 rows (16%). Coder A vs coder B F1 0.694. | No |
| `theocharis_dynamics_incivility` | Civil and uncivil definitions, examples and coder note verbatim from Theocharis et al. (2020, Sage Open). | `id_str` read as string; 261 rows with no text left after removing URLs, mentions and punctuation dropped: 3,737 rows. | No |
| `twitcivility_impoliteness` | Impoliteness and intolerance definitions verbatim from Pendzel et al. (2023, arXiv 2305.14964, sect. 1); annotator examples from Table 1. Codebook not public. | None. | Yes |
| `rheault_line_of_fire_incivility` | Definition verbatim from Rheault, Rayment and Musulan (2019); conservative-approach passage and civil example from the Online Appendix. | Original CSV fixes escape residue in 576 texts; labels checked row by row. Same rows. | No (new file) |
| `toxicity_protests_es` | The Spanish question coders answered, verbatim from the project's Labelbox export (Perspective API toxicity definition); framing ours. | None. | Yes |
| `politicause_causal_relation` | Definition, step 1, extended guidelines and four examples verbatim from Garcia Corral et al. (2024, LREC-COLING, sect. 3); the causal relation must be explicit within the sentence. | None. | Yes |

`burnham_polnli_event_entailment` has no v2: all 2,854 of its pairs are the
`source_task == "event extraction"` rows of `burnham_polnli_entailment`, with
the same gold. It is dropped from the next release (decision 2026-11-14).


## Extension tasks from the November sourcing pass (2026-11-14)

Eight tasks built from the sources that passed the quality gate in
[`task_expansion_candidates.md`](task_expansion_candidates.md). They live in
`tasks_ext/` and have no model runs yet. Their prompts carry no provenance
footer.

| Task | Prompt source | Data and gold |
|---|---|---|
| `rivlin_angert_pdd_delegitimization` | Coding-book paragraph and five bullets (Rivlin-Angert and Mor-Lan 2025, arXiv 2508.15524, App. A) and the §4 policy-critique exclusion, verbatim; framing and output instruction ours. Worked examples left out because they are dataset items. | github guymorlan/pdd at 80a72d5, CC BY 4.0. Gold `delegitimization`, splits pooled, Hebrew text only (machine translation and sub-labels excluded). The file's News and Facebook source values are swapped back (5,969 of 6,690 "News" rows carry facebook.com URLs or page metadata). 326 rows under 50% Hebrew characters dropped (all 0). 10,084 rows (1,812 positive). Most items single-coded; kappa 0.82 on 170 triple-coded sentences; sampling enriched (Knesset 47% positive) and undocumented. |
| `parlasent_en_sentiment` | Six-level definitions and the 3-way collapse rule verbatim from Mochtak, Rupnik and Ljubešić (arXiv 2309.09783, §3.4); framing, explicit mapping and output instruction ours. | HF classla/ParlaSent at 0587c2b, `ParlaSent_EN.jsonl` only (the single-annotator `_test` file excluded), CC BY-SA 4.0. Gold `label`, the 3-way collapse of the reconciled 6-level code. Seed-word stratified sample, so class shares are not representative. 2,600 rows (1,269 / 680 / 651), no drops. Ids include the date because document and sentence ids repeat. |
| `heseltine_politician_tweet_tone` | SI B.2 (negative) and B.3 (three-category sentiment) verbatim from the OSF preprint cx752 SI, pp. 19-20; value codes from SI C.5; framing and output instruction ours. | Dataverse V2P6YL `Python.rar`, CC0. Expert `neg3` gold (`label` in the US file). US tweets double-coded and reconciled; German, Italian and Chilean tweets single-coded. Mojibake repaired by a mechanical rule (240 texts), HTML entities unescaped; 237 truncated retweets kept. GPT columns and the 200 news articles excluded. 1,995 rows (539 / 975 / 481), no drops. |
| `chalkidis_trump_populism` | Degrees of Populism Coding Manual v3.0 §§1.1-1.2.3, 2.2, 3.2.1-3.2.3 verbatim (Google Drive file linked from arXiv 2507.19303 App. C), including examples C1-C15, none of which are dataset items; framing and label mapping ours. | HF parisaslanidis/populism-trump-2016 at 2ce6927e, CC BY-NC-ND 4.0: rows selected only, codes and text unmodified. Labels are single-student codes assigned with full-speech context (expert re-annotation not released). Dropped sentences under 4 words (1,219), all rows of 53 texts with conflicting codes (160), and same-code repeats (1,217): 15,025 to 12,429 (17%). |
| `horak_czech_propaganda` | Attribute definitions verbatim from the HF MU-NLPC/Propaganda dataset card, with glosses verbatim from Horák et al. 2021 (arXiv 2108.11669, sect. 5.4.4); framing and output instruction ours. | HF MU-NLPC/Propaganda at cde669cb, CC BY-NC-SA 4.0. Gold = the document-level yes/no attributes (2-of-3 majority). Only articles with annotator evidence spans are kept, which are the triple-coded 2017-18 articles; 2016 articles were single-coded (5,551 dropped, 64%). Fabulation and relativization left out; spans never shown. 8 articles over 10,000 tokens dropped. 3,079 articles. Agreement is low (ESWA 2024 Table A.7, kappa): blaming 0.14, opinion 0.23, labeling 0.26, fear-mongering 0.20, demonization 0.16; agreement on `yes` alone 18-33%. ESWA says "at least two annotators" in the second phase, the 2021 report says three. |
| `erhard_popbert_populism` | Translated codebook (Erhard et al. 2025, Cambridge supplement Appendix 2, 2.1-2.2), verbatim from the authors' English translation, long example lists and cross-references omitted; framing, headings and output instruction ours. | Dataverse HZMSUR `labelled_data` (CC0). Gold per label = at least 3 of 5 coders (our rule; the authors train on any coder). Host ideology counts only with a populism majority, per codebook 2.2 (40 host labels set to 0). 25 same-label duplicates dropped. 8,770 sentences: 1,340 / 281 / 360 / 156. Sample enriched for populism. |
| `benson_tucker_unsc_government_stance` | UNCR Codebook (Conflict-Resolution) pp. 9-11 sentiment section verbatim: verb lists, the Mali UNSCR 2164 examples, attribution rules, progov/antigov/mixgov definitions; the `none` definition and framing are ours (the codebook has none). | Labels UNCR (Dataverse EUYIW0, CC0); text CR-UNSC 2024-05-19 `EN_TXT_BEST` (Zenodo 11212056, CC0); government and opposition side names from UCDP/PRIO ACD v4-2016 for the resolution year (nearest active year when missing). Single-conflict resolutions only (umbrella rows dropped: 405 rows, 36%). 2 resolutions over 10,000 tokens dropped. 725 (none 382 / supportive 195 / condemning 93 / mixed 55). Agreement 95% pooled, no chance-corrected figure. |
| `arslan_claimbuster_checkworthy` | §4.1 category definitions and examples and all nine Table 1 training samples with explanations, verbatim (Arslan et al. 2020, arXiv 2004.14425); none of them are task items. Framing ours. | Zenodo 3836810, CC BY 4.0. `crowdsourced.csv` only (the bundled README swaps the file names); `Verdict` -1/0/1; `Sentiment` column hidden. 14 rows of 7 conflicting texts and 86 same-label repeats dropped. 22,401 (14,611 / 2,394 / 5,396). No chance-corrected agreement; annotators could see four preceding sentences. |
