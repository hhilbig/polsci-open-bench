# Task expansion candidates

This note records the first sourcing pass for expanding the benchmark within
the existing four-family structure.

## Current family counts

- Relevance / Incivility: 2 tasks
- Sentiment / Stance / Tone: 4 tasks
- Event coding: 2 tasks
- Policy-topic coding: 2 tasks

The benchmark is currently thinnest in Relevance / Incivility, Event coding,
and Policy-topic coding.

## Repo constraint

There are no spare labeled task files already sitting in `data/` that can be
added just by writing new manifests. The only extra local asset beyond the
live benchmark tasks is `data/wmp2022_raw/fb_2022_adid_text.csv.gz`, which is
raw ad text without benchmark-ready labels.

That means the next expansion step is source acquisition and task design, not
just manifest plumbing.

## Selection rubric

Candidate additions should be screened with the following criteria:

1. Political-science relevance
   - Hard filter. The task should map clearly onto substantive political-science research rather than generic NLP alone.

2. Labeled data and replication access
   - Hard filter. The underlying labeled text and replication materials need to be obtainable in practice.

3. Prompt or codebook directly obtainable
   - Strong preference. Best case is a verbatim prompt, coding instructions, appendix language, or replication codebook that can be reused with minimal change.
   - Derived prompts are still acceptable when necessary, but they should be lower priority than tasks with directly documented instructions.

4. Incremental diversity within family
   - Hard filter. A new task should fit one of the four existing families without being too close to the benchmark's current tasks.
   - Useful margins of differentiation include text source or platform, unit of analysis, country or political setting, label structure, and document length or complexity.

5. Benchmark practicality
   - The task should have enough labeled items, a clean text field, a stable label schema, and a task shape that can be benchmarked without turning into a one-off engineering special case.

## Verification status

- On 2026-04-30, every candidate currently tagged as a `paper` below was
  checked against a publisher or DOI page.
- Verified paper-backed candidates:
  - Bryan T. Gervais, Connor Dye, and Amber Chin 2025, "Incivility or Invalidity? Evaluating Perspective API Scores as a Measure of Political Incivility"
  - Kevin Munger 2021, "Don't @ Me: Experimentally Reducing Partisan Incivility on Twitter"
  - Max Griswold, Michael W. Robbins, and Michael S. Pollard 2025, "Stay Tuned: Improving Sentiment Analysis and Stance Detection Using Large Language Models"
  - Samuel E. Bestvater and Burt L. Monroe 2023, "Sentiment Is Not Stance: Target-Aware Opinion Classification for Political Text Analysis"
  - Moritz Osnabrügge, Elliott Ash, and Massimo Morelli 2023, "Cross-Domain Topic Classification for Political Texts"
  - Yu Wang 2023, "Topic Classification for Political Texts with Pretrained Language Models"
  - Mihai Croicu and Nils B. Weidmann 2015, "Improving the Selection of News Reports for Event Coding Using Ensemble Classification"
  - Aaron Erlich et al. 2022, "Multi-Label Prediction for Political Text-as-Data"
- Non-paper resources retained as possible candidates, but explicitly not
  treated as published-paper replications:
  - TwitCivility
  - MMAD
  - PLOVER
  - Manifesto Project official corpus
- Earlier attribution errors have been corrected here. `Stay Tuned` is not a
  Hilbig-Velez paper, and `Cross-Domain Topic Classification for Political
  Texts` is not a Velez-Hilbig paper.

## Family-balance rule

- Task additions should be distributed roughly uniformly across the four
  existing families.
- Because the current count is `2 / 4 / 2 / 2`, the first expansion priority
  should be Relevance / Incivility, Event coding, and Policy-topic coding.
- Sentiment / Stance / Tone should be treated as a lower priority family until
  the other three have at least one additional task each.
- Operational rule: do not add two new tasks to the same family before adding
  at least one to each currently lower-count family.

## Candidate sources

### Relevance / Incivility

1. Gervais et al. 2025, "Incivility or Invalidity? Evaluating Perspective API Scores as a Measure of Political Incivility"
   - Fit: strong topical fit with the current incivility family and with the repo's political-science framing.
   - Likely task shape: binary or ordinal incivility classification on political tweets.
   - Risk: replication-file availability needs to be confirmed before implementation.
   - Source: https://doi.org/10.1177/1532673X241309627

2. TwitCivility
   - Fit: good task fit for incivility detection, weaker fit with the current "published political-science replication archive" benchmark framing.
   - Likely task shape: binary or multi-class incivility classification on political tweets or replies.
   - Risk: cleaner as an NLP benchmark than as a political-science replication task.
   - Source: https://github.com/Twirtybird/TwitCivility

### Sentiment / Stance / Tone

1. Max Griswold, Michael W. Robbins, and Michael S. Pollard 2025, "Stay Tuned: Improving Sentiment Analysis and Stance Detection Using Large Language Models"
   - Fit: very strong. Direct continuation of the current sentiment / stance family and already discussed in project notes.
   - Likely task shape: short political-text stance or sentiment classification with explicit targets.
   - Risk: this family is already the largest, so it should not be the first expansion priority unless the goal is strict growth in all four families immediately.
   - Source article: https://www.cambridge.org/core/journals/political-analysis/article/stay-tuned-improving-sentiment-analysis-and-stance-detection-using-large-language-models/B95E9A9A1009934A3F1C6BF50A5F4E34
   - Source data: https://doi.org/10.7910/DVN/KHNBZL

### Event coding

1. Mass Mobilization in Autocracies Database (MMAD)
   - Fit: very strong. Political event text, report-level labels, and obvious continuity with the current Halterman / Keith event-coding tasks.
   - Likely task shape: anti-government vs pro-government mobilization, or another report-level event attribute on news text.
   - Risk: some possible task formulations may require careful negative-class construction or label collapsing.
   - Source: https://mmadatabase.org/

2. PLOVER gold-standard records
   - Fit: good event-coding fit, though somewhat less straightforward than MMAD for a first addition.
   - Likely task shape: event or actor coding from event-record text.
   - Risk: implementation may require more preprocessing and task design than MMAD.
   - Source: https://github.com/openeventdata/plover

### Policy-topic coding

1. Moritz Osnabrügge, Elliott Ash, and Massimo Morelli 2023, "Cross-Domain Topic Classification for Political Texts"
   - Fit: very strong. Already oriented around political-text topic classification, with replication materials and multiple domains.
   - Likely task shape: fine-grained or collapsed topic classification on speeches, manifesto text, or adjacent political corpora.
   - Risk: label-space choice matters; the benchmark should decide whether to use the published granularity or a benchmark-specific collapse.
   - Source article: https://www.cambridge.org/core/journals/political-analysis/article/crossdomain-topic-classification-for-political-texts/F074564984969CE168BCBCF5E7D931C8
   - Source data: https://doi.org/10.7910/DVN/CHTWUB

2. Manifesto Project official corpus
   - Fit: strong. Natural extension of the current CMP-derived task, with a richer official label structure.
   - Likely task shape: full or semi-collapsed quasi-sentence policy-domain classification.
   - Risk: larger label spaces are attractive scientifically but may sharply raise malformed-output rates and complicate apples-to-apples comparisons.
   - Source: https://manifesto-project.wzb.eu/

### Additional candidates identified after the first pass

1. Kevin Munger 2021, "Don't @ Me: Experimentally Reducing Partisan Incivility on Twitter"
   - Family: Relevance / Incivility
   - Fit: strong on the published-political-science-paper criterion and on replication access.
   - Risk: needs verification that the replication materials expose a benchmark-clean text-label task, rather than primarily an experimental design artifact.
   - Source article: https://www.cambridge.org/core/journals/journal-of-experimental-political-science/article/dont-me-experimentally-reducing-partisan-incivility-on-twitter/CA72D8773AC00916F5A551F80E6C06D3
   - Source data: https://doi.org/10.7910/DVN/OUYTUP

2. Samuel Bestvater and Burt Monroe 2023, "Sentiment Is Not Stance: Target-Aware Opinion Classification for Political Text Analysis"
   - Family: Sentiment / Stance / Tone
   - Fit: very strong on the published-paper plus Dataverse criterion, and notably diverse within the family because it includes multiple target-aware corpora.
   - Risk: lower immediate priority because the family is already overrepresented.
   - Source article: https://www.cambridge.org/core/journals/political-analysis/article/sentiment-is-not-stance-targetaware-opinion-classification-for-political-text-analysis/743A9DD62DF3F2F448E199BDD1C37C8D
   - Source data: https://doi.org/10.7910/DVN/MUYYG4

3. Yu Wang 2023, "Topic Classification for Political Texts with Pretrained Language Models"
   - Family: Policy-topic coding
   - Fit: strong on the published-paper plus Dataverse criterion and useful as a distinct topic-classification benchmark relative to the current CMP and BES tasks.
   - Risk: needs inspection to confirm which target-corpus labels are easiest to turn into one benchmark task.
   - Source article: https://www.cambridge.org/core/journals/political-analysis/article/topic-classification-for-political-texts-with-pretrained-language-models/9AA6401CAB1FA3D1EADC7A3D155BB265
   - Source data: https://doi.org/10.7910/DVN/FMT8KR

4. Mihai Croicu and Nils Weidmann 2015, "Improving the Selection of News Reports for Event Coding Using Ensemble Classification"
   - Family: Event coding
   - Fit: attractive because it would add an event-report relevance-screening task rather than another event-type classifier.
   - Risk: the key unresolved question is whether the labeled relevant/irrelevant source articles are actually accessible as replication materials.
   - Source article: https://journals.sagepub.com/doi/10.1177/2053168015615596

5. Aaron Erlich et al. 2022, "Multi-Label Prediction for Political Text-as-Data"
   - Family: Policy-topic coding adjacent
   - Fit: potentially valuable for a second-wave expansion because it would diversify the benchmark toward multi-label tasks.
   - Risk: likely to become a special-case benchmark module rather than a clean first-wave addition.
   - Source article: https://www.cambridge.org/core/journals/political-analysis/article/multilabel-prediction-for-political-textasdata/0EC24DBB4E3854EBE0AD247DCB27F828
   - Source data: https://doi.org/10.7910/DVN/SOVPA4

## Candidate summary table

| Candidate | Family | Type | Published political-science paper | Replication or labeled data access | Direct prompt or codebook likely available | Novelty within family | Current read |
|---|---|---|---|---|---|---|---|
| Gervais et al. 2025, `Incivility or Invalidity?` | Relevance / Incivility | paper | Yes | Unclear beyond supplementary materials | Likely yes | Medium | Conditional |
| Munger 2021, `Don't @ Me` | Relevance / Incivility | paper | Yes | Yes, Dataverse | Possibly | High | Strong if labels are benchmark-clean |
| TwitCivility | Relevance / Incivility | dataset / repo | No, mainly NLP resource | Yes | Likely yes | Medium | Deprioritize on political-science-paper criterion |
| Griswold, Robbins, Pollard 2025, `Stay Tuned` | Sentiment / Stance / Tone | paper | Yes | Yes, Dataverse and GitHub | Likely yes | Low to medium | High quality, lower marginal value |
| Bestvater and Monroe 2023, `Sentiment Is Not Stance` | Sentiment / Stance / Tone | paper | Yes | Yes, Dataverse | Likely yes | High | Strong but lower priority family |
| MMAD | Event coding | project / database | No single paper replication package; project plus codebook | Yes | Yes, codebook | High | Substantively strong, slightly weaker on preferred filter |
| Croicu and Weidmann 2015, `Improving the Selection...` | Event coding | paper | Yes | Unclear | Probably yes | High | Very attractive if labeled texts are accessible |
| PLOVER gold-standard records | Event coding | ontology / repo | No clear single paper target | Yes | Unclear | Medium | Second-tier event candidate |
| Osnabrügge, Ash, Morelli 2023, `Cross-Domain Topic Classification` | Policy-topic coding | paper | Yes | Yes, Dataverse and Code Ocean | Yes | High | Top candidate |
| Wang 2023, `Topic Classification for Political Texts with Pretrained Language Models` | Policy-topic coding | paper | Yes | Yes, Dataverse | Likely yes | Medium to high | Strong candidate |
| Erlich et al. 2022, `Multi-Label Prediction for Political Text-as-Data` | Policy-topic coding adjacent | paper | Yes | Yes, Dataverse | Likely yes | High | Good second-wave candidate |
| Manifesto Project official corpus | Policy-topic coding | project / corpus | No single paper replication package | Yes | Yes | Medium | Deprioritize relative to paper-linked candidates |

## Priority shortlist

These are the four candidates to carry into the first verification wave.

1. Osnabrügge, Ash, Morelli 2023, `Cross-Domain Topic Classification`
   - Family: Policy-topic coding
   - Reason: strongest overall paper-backed candidate in an underrepresented family.

2. Munger 2021, `Don't @ Me`
   - Family: Relevance / Incivility
   - Reason: strongest current paper-backed relevance or incivility candidate with accessible replication materials.

3. MMAD
   - Family: Event coding
   - Reason: strongest event-coding candidate on substantive fit and likely benchmark practicality, even though it is a project rather than a single-paper replication package.

4. Croicu and Weidmann 2015, `Improving the Selection of News Reports for Event Coding Using Ensemble Classification`
   - Family: Event coding
   - Reason: strongest paper-backed event-coding candidate, retained alongside MMAD until the actual accessibility of benchmark-ready labeled texts is checked.

## First verification wave

The next sourcing pass should focus only on these four candidates and answer
the following gating questions before any cleaning or manifest work starts.

1. `Cross-Domain Topic Classification`
   - Are the labeled text units directly downloadable from the replication package?
   - Which corpus and label granularity produce the cleanest single benchmark task?
   - Is there an existing codebook or appendix language that can be adapted into a prompt with minimal invention?

2. `Don't @ Me`
   - Do the replication materials expose a clean text-plus-label classification task, rather than only experimental design metadata?
   - Are labels already in a benchmark-friendly form, or would they require nontrivial reconstruction?
   - Is the text field sufficiently direct and self-contained for one-item classification?

3. `MMAD`
   - Which label or label collapse is most defensible for a first event-coding task?
   - Does the downloadable data include enough directly usable report text, or only event records and metadata?
   - What negative class or contrast class would be used if the raw labels are too sparse or too fine-grained?

4. `Croicu and Weidmann`
   - Are the labeled relevant versus irrelevant source texts actually accessible in the replication materials?
   - If yes, is the text field clean enough to benchmark without heavy reconstruction?
   - If no, this candidate drops below MMAD for the first event-coding addition.

## First verification wave findings

Status after the first replication-material inspection on 2026-04-30:

1. `Cross-Domain Topic Classification`
   - Status: full pass
   - What is confirmed:
     - The paper has a public Dataverse archive and public Code Ocean capsule.
     - The article uses labeled manifesto statements as the source corpus and parliamentary speeches as the target corpus.
     - The paper explicitly describes both a 44-topic specification and a collapsed 8-topic specification based on the Manifesto codebook.
     - The replication ZIP includes directly usable public corpus files:
       - `data/corpora/source_corpus.csv` with 115,410 labeled manifesto statements and columns `topic_44`, `topic_8`, `text`
       - `data/corpora/target_corpus.csv` with 4,165 labeled parliamentary-speech excerpts and columns `text`, `topic_44`, `topic_8`, plus additional rater columns
     - The replication README explicitly describes `target_corpus.csv` as the annotated subset of the target corpus.
   - Current read:
     - This is still the strongest immediate implementation candidate.
     - The most natural first task is probably the 8-topic specification, because it is more interpretable and benchmark-friendly than the 44-topic version.
     - I infer that `target_corpus.csv` is the better first benchmark task than `source_corpus.csv`, because the parliamentary-speech domain is more novel within the current repo than another manifesto-style classification task.
   - Remaining judgment call:
     - Whether to benchmark the final adjudicated `topic_8` label only, or use the additional rater columns for an agreement or robustness note. For the main benchmark, the simplest choice is the adjudicated `topic_8` label only.

2. `Don't @ Me`
   - Status: fail for direct benchmark ingestion
   - What is confirmed:
     - The paper has a public Dataverse archive.
     - The public file list includes analysis scripts, anonymized data objects, and a `validation_tweets` file.
     - The public `validation_tweets` file contains coder labels and scores, not raw tweet text.
     - The archive README states that the raw text is not shared because even a single tweet can identify a user.
   - Current read:
     - This is not a benchmark-ready text-classification task for the current repo because the public replication materials do not expose the underlying tweet text.
     - It remains a useful substantive reference for incivility measurement, but not a direct next task.

3. `MMAD`
   - Status: fail for direct benchmark ingestion
   - What is confirmed:
     - The public download page describes the report-level and event-level files in detail.
     - The downloadable report-level data include coded attributes such as `side`, `actors`, `issue`, `scope`, `part_violence`, `sec_engagement`, participant counts, and source metadata.
     - The public description does not indicate that full article text, headlines, or report bodies are included in the downloadable dataset.
   - Current read:
     - MMAD is substantively excellent, but the downloadable public asset is a coded event database rather than a benchmark-ready document-classification corpus.
     - Unless another public text source tied to the same labels can be identified, it does not fit the repo's current task format.

4. `Croicu and Weidmann`
   - Status: fail or strong hold
   - What is confirmed:
     - The article describes a very large hand-coded training set of raw LexisNexis news articles containing headline, dateline, body, and unique ID.
     - A public replication archive with those labeled source texts did not surface in the article page or in follow-up searches.
   - Current read:
     - This would be an excellent event-report relevance task if the labeled texts were public.
     - At the moment the practical answer is negative: accessibility is unresolved and likely poor because the source material comes from proprietary news data.

## Revised shortlist after verification

1. `Cross-Domain Topic Classification`
   - Only candidate in the first verification wave that still looks like a direct near-term ingest target.

2. `Topic Classification for Political Texts with Pretrained Language Models`
   - Not in the first wave, but it remains a strong second policy-topic candidate because it is another paper-backed Political Analysis Dataverse task in the same general topic-classification space.

3. `Incivility or Invalidity?`
   - Now relatively more attractive for Relevance / Incivility because `Don't @ Me` does not expose public raw text.

## Replacement candidates found after infeasibility screening

The initial replacement search for the infeasible tasks produced two verified
public-text candidates and one additional negative finding.

1. `Politicians in the Line of Fire: Incivility and the Treatment of Women on Social Media`
   - Family: Relevance / Incivility
   - Paper: Rheault, Rayment, and Musulan 2019, *Research & Politics*
   - Replication archive: https://doi.org/10.7910/DVN/B97TGX
   - Status: strong replacement for `Don't @ Me`
   - What is confirmed:
     - The Dataverse archive contains human-coded public training files `usa_training.tab` and `canada_training.tab`.
     - The sampled public rows include a binary label column `code` plus public text columns `clean_text` and `text`.
     - The archive also includes much larger final datasets for the USA and Canada.
   - Current read:
     - This is the strongest current replacement for the Relevance / Incivility slot because it satisfies both the political-science-paper criterion and the public-text criterion.
     - The most natural benchmark slice is one of the human-coded training files rather than the much larger final datasets.

2. `PAPEA: A Modular Pipeline for the Automation of Protest Event Analysis`
   - Family: Event coding
   - Paper: Haunss et al. 2025, *Political Science Research and Methods*
   - Replication archive: https://doi.org/10.7910/DVN/KVP7HA
   - Status: strong replacement for the failed event-coding candidates
   - What is confirmed:
     - The Dataverse archive includes public full-text article samples such as `taz2015_sample.tab`.
     - The archive also includes a manual gold-standard file, `AppendixC_manual_goldstandard.csv`, with full article text and event labels including `dominant_form`, `second_form`, `dominant_claim`, and `second_claim`.
   - Current read:
     - This is the strongest current event-coding replacement because it is paper-backed, public, and directly benchmarkable.
     - The most defensible first task from this archive is probably protest-form classification from the manual gold-standard file.

3. `Violent political rhetoric on Twitter`
   - Family: Relevance / Incivility adjacent
   - Paper: Kim 2023, *Political Science Research and Methods*
   - Replication archive: https://doi.org/10.7910/DVN/NEC17Z
   - Status: rejected as a direct replacement
   - What is confirmed:
     - The archive exposes aggregate files and `df_violent_tweet_ids.csv`.
     - The surfaced public files did not expose raw tweet text.
   - Current read:
     - This fails the same basic feasibility test as `Don't @ Me`: the paper exists and the archive exists, but the benchmark-ready text is not public in the inspected files.

## Practical implication

- The first verification wave did eliminate `Don't @ Me`, `MMAD`, and `Croicu-Weidmann` as direct next additions.
- A follow-up replacement search did recover viable public-text replacements for the missing families:
  - `Politicians in the Line of Fire` for Relevance / Incivility
  - `PAPEA` for Event coding
- `Cross-Domain Topic Classification` has now been implemented in the live
  benchmark manifests, using the annotated target corpus and the 8-topic
  specification.
- `Politicians in the Line of Fire` has now been implemented in the live
  benchmark manifests, using the combined public US and Canada human-coded
  training samples with exact-text deduplication.
- `PAPEA` has now been implemented in the live benchmark manifests, using the
  public FGZ sentence-level form annotations, restricted to the seven most
  common source labels and filtered to remove exact text snippets that appear
  with conflicting labels.
- There is no remaining first-wave replacement task pending from the original
  underrepresented-family shortlist.

## Refreshed shortlist after first implementation wave

With `Cross-Domain Topic Classification`, `Politicians in the Line of Fire`,
and `PAPEA` protest-form classification now implemented in the live manifests,
the next sourcing pass focused on the highest-value additions under the
balancing rule. `Sentiment / Stance / Tone` remains lower priority because it
is already the largest family. The strongest new leads are in `Relevance /
Incivility` and `Event coding`.

### Relevance / Incivility

1. `Extractive versus Generative Language Models for Political Conflict Text Classification`
   - Family: `Relevance / Incivility` (relevance side)
   - Paper: Brandt et al. 2025, *Political Analysis*
   - Paper URL: https://www.cambridge.org/core/journals/political-analysis/article/extractive-versus-generative-language-models-for-political-conflict-text-classification/B00077A671DF5656043E4BC45B3864A1
   - DOI: https://doi.org/10.1017/pan.2025.10027
   - Replication archive: https://doi.org/10.7910/DVN/KDO5AM
   - What is confirmed:
     - The Dataverse archive includes `raw_bc_data.tab`, described as the raw source data for the binary conflict-classification task.
     - Direct inspection confirms that `raw_bc_data.tab` contains exactly two benchmark-ready columns: `text` and `bin`.
     - The public file has 322 rows, with label counts `269` negative and `53` positive.
     - The public texts are much longer than the current benchmark median task lengths: the inspected file has a median text length of about `1,927` characters.
   - Current read:
     - This is now the strongest next `Relevance / Incivility` addition.
     - It satisfies the published-paper criterion, the public-text criterion, and it adds a long-document relevance task rather than another short social-media task.

2. `The Silenced Text: Field Experiments on Gendered Experiences of Political Participation`
   - Family: `Relevance / Incivility` adjacent
   - Paper: Yan and Bernhard 2024, *American Political Science Review*
   - Paper URL: https://www.cambridge.org/core/journals/american-political-science-review/article/silenced-text-field-experiments-on-gendered-experiences-of-political-participation/E008099CDD9573F7D8ADB128919EFDC4
   - DOI: https://doi.org/10.1017/S0003055423000217
   - Replication archive: https://doi.org/10.7910/DVN/UYKE7T
   - What is confirmed:
     - The Dataverse archive contains public message files and text-rating files.
     - Direct inspection confirms that `text_ratings_clean.csv` contains `2,989` rows, `2,988` non-missing text strings, and the columns `text`, `offensive_avg`, and `discourage_avg`.
     - The raw experiment message file `01-experiment-1-messages.tab` exposes public message text in `message_body`, but not a direct benchmark label.
   - Current read:
     - This is a viable conditional candidate, but it is weaker than Brandt et al. because the most natural ground truth would be a thresholded or otherwise transformed rating scale rather than a directly exposed classification label.
     - It remains attractive if the benchmark later wants a text-harassment task that is clearly political and public-text based.

### Event coding

1. `Introducing ICBe: an Event Extraction Dataset from Narratives about International Crises`
   - Family: `Event coding`
   - Paper: Douglass et al. 2024, *Political Science Research and Methods*
   - Paper URL: https://www.cambridge.org/core/journals/political-science-research-and-methods/article/introducing-icbe-an-event-extraction-dataset-from-narratives-about-international-crises/8CF9BB8C34354129A20925352FB5483B
   - DOI: https://doi.org/10.1017/psrm.2024.17
   - Replication archive: https://doi.org/10.7910/DVN/MNVUEP
   - Public repo: https://github.com/CenterForPeaceAndSecurityStudies/ICBEdataset
   - What is confirmed:
     - The Dataverse replication package is public.
     - The GitHub README explicitly documents public agreed datasets, public span data in `icb_long_spans.Rds`, and a public crisis-narrative corpus under `replication_corpus`.
   - Current read:
     - This is the strongest next `Event coding` addition.
     - It adds a substantively different international-crisis ontology instead of another protest-only task.
     - The main implementation decision will be how to collapse or subset the label space into a benchmarkable classification task.

2. `Extractive versus Generative Language Models for Political Conflict Text Classification`
   - Family: `Event coding`
   - Paper: Brandt et al. 2025, *Political Analysis*
   - Paper URL: https://www.cambridge.org/core/journals/political-analysis/article/extractive-versus-generative-language-models-for-political-conflict-text-classification/B00077A671DF5656043E4BC45B3864A1
   - DOI: https://doi.org/10.1017/pan.2025.10027
   - Replication archive: https://doi.org/10.7910/DVN/KDO5AM
   - What is confirmed:
     - The Dataverse archive includes `raw_gtd_multilabel_data.tab`, described as the pre-processed data behind the multi-class event-classification results.
     - Direct inspection confirms that the file contains `37,709` rows, public `text`, and direct event labels such as `attacktype1_txt`, `attacktype2_txt`, and `attacktype3_txt`.
     - The primary label space has `9` distinct `attacktype1_txt` values.
   - Current read:
     - This is a strong and highly practical event-task candidate.
     - It is somewhat less attractive than `ICBe` on diversity grounds because it comes from the same archive family as the conflict-relevance candidate above, but it is cleaner and more immediately benchmarkable than many older event-coding papers.

3. `PAPEA: A Modular Pipeline for the Automation of Protest Event Analysis`
   - Family: `Event coding`
   - Paper: Haunss et al. 2025, *Political Science Research and Methods*
   - Paper URL: https://doi.org/10.1017/psrm.2025.10013
   - Replication archive: https://doi.org/10.7910/DVN/KVP7HA
   - Candidate file: `fgz_papea_claims.tab`
   - What is confirmed:
     - Direct inspection confirms that the public file contains the columns `fid`, `claim`, `claim_txt`, `event_id`, and `text`.
     - A permissive parse yields `3,277` rows, `3,267` non-missing text rows, and about `28` distinct `claim_txt` labels.
   - Current read:
     - This is still a strong next event task, especially because it is already in an archive we have used successfully.
     - It is slightly lower priority than `ICBe` and the Brandt GTD task because it is more correlated with the already-implemented PAPEA protest-form task.

4. `PLOVER and POLECAT: A New Political Event Ontology and Dataset`
   - Family: `Event coding`
   - Status: acceptable fallback if the published-paper preference is relaxed
   - Preprint: https://osf.io/preprints/socarxiv/rm5dw/
   - Repo: https://github.com/openeventdata/PLOVER
   - Related Dataverse: https://doi.org/10.7910/DVN/AJGVIT
   - What is confirmed:
     - The public repo README documents a gold-standard sentence file, `gold_standard_records/PLOVER_GSR_CAMEO.txt`, containing example sentences classified by PLOVER categories.
   - Current read:
     - This remains a substantively attractive fallback because it is broad, public, and clearly text based.
     - It remains below the paper-backed candidates because it is not a journal article replication package.

5. `PAPEA` full-article dominant-claim classification
   - Family: `Event coding`
   - Paper and archive: same as PAPEA above
   - Candidate file: `AppendixC_manual_goldstandard.csv`
   - What is confirmed:
     - The public file contains `fulltext`, `dominant_claim`, and `second_claim`.
   - Current read:
     - This is the most attractive small pilot for moving toward longer documents within event coding.
     - It is not a strong core-benchmark addition yet because the file only contains `100` labeled articles.

## Recommended order

1. `Relevance / Incivility`
   - Best next target: Brandt et al. binary conflict classification from `raw_bc_data.tab`.
   - Conditional backup: `The Silenced Text` if a derived label from offensiveness ratings is acceptable.

2. `Event coding`
   - Best next target: `ICBe`.
   - Best easier fallback: Brandt et al. GTD attack-type classification.
   - Best same-archive fallback: PAPEA protest-claim classification.

3. `Sentiment / Stance / Tone`
   - Still lower priority because this family already has 4 tasks and no immediate balancing need.

## Proposed next module

1. Implement the strongest refreshed event-side addition
   - `brandt_political_relevance` is now implemented in the live manifests, using the public binary-classification corpus from Brandt et al. (2025) after exact-text deduplication.
   - The next best expansion target is now `ICBe`, with the main remaining design choice being a defensible label collapse or subset.
