# Long-text task scoping

Scoped 2026-11-14. Nothing is built or run. This note asks which public
labeled datasets could add tasks in which each item is a whole document
(roughly 400 to 5,000 words), and what running them would cost.

**Verdict.** Three sources pass every requirement: populism-graded leader
speeches (Global Populism Database), Supreme Court opinions coded by issue area
(Supreme Court Database labels, via LexGLUE), and news articles labeled as
hyperpartisan or not (SemEval-2019). A fourth, Policy Agendas topic codes on
full executive orders, is feasible but its labels were assigned from one-line
summaries. Two legal datasets are conditional. That is short of
the 8 to 10 long-text tasks needed for a length comparison with reasonable
power. Document-level human labels with public full text are rare. Most
well-known corpora either withhold the text (LexisNexis, copyright) or label
sentences rather than documents. Running ten long-text tasks on the 22 API
models would cost roughly $70 at 1,500 words per document and $130 at 3,000
words. The open models run on Hive at no cash cost.

## Why

The release items are short. Across the 34 task definitions, the median item
is 27 words, and only three tasks have a median above 200 words
([`output/task_length_audit.csv`](../output/task_length_audit.csv)). Critics of
the benchmark have pointed this out. Research applications often code whole
speeches, articles, bills or opinions. The benchmark currently says little
about how models do on those, or whether the open-model gap grows with length.
The May length analysis found a correlation of 0.29 between task median length
and the local-minus-API gap, but it rests on a handful of long tasks.

## What was checked

Two searches covered political science and adjacent fields (law, economics,
communication, sociology). Each source was checked against the data files
themselves (Dataverse, Zenodo, GitHub, Hugging Face), not only the paper. The
requirements were:

1. A published coding scheme that social scientists use.
2. Human labels at the document level.
3. Full text publicly downloadable, not only IDs or URLs.
4. At least about 300 labeled documents, so 100 can be sampled with all classes
   present.

Word counts are measured from the files unless the table says otherwise.

## Usable

| Source | Labels | N documents | Words per document | License | Notes |
|---|---|---|---|---|---|
| Global Populism Database speeches (Team Populism, Hawkins et al.; Dataverse doi:10.7910/DVN/LFTQEZ) | Holistic populism grade 0 to 2, mean of 4 or more trained coders per speech | About 1,050 graded speeches; 1,310 text files | Median 1,750, p90 4,460; 4% above 7,000 | CC0 | 77 countries, original languages; about 320 speeches look English. Skewed: 79% average below 0.5, 4% above 1.5. Three bins (not / somewhat / very populist) with a stratified sample would work. |
| Supreme Court opinions, issue area (Chalkidis et al. 2022, LexGLUE `scotus`; labels from the Supreme Court Database) | 14 issue areas | 7,800 | Median about 2,600, p90 about 9,400 (50-document sample only) | CC-BY-4.0 per dataset card; underlying texts and SCDB are public | 15 to 20% of opinions exceed the context budget and need truncation. Labels and opinions have been public for decades, so training-data exposure is likely. |
| Hyperpartisan news, by-article set (Kiesel et al. 2019, SemEval-2019 Task 4; Zenodo 5776081) | Binary hyperpartisan | 1,273 (645 train, 628 test) | Median about 420, p90 about 1,170 | CC-BY-4.0 | Labels are crowd consensus from a computational-linguistics shared task, not trained political-science coders. Sits at the short end of the target range. |

## Conditional

| Source | What is missing |
|---|---|
| Policy Agendas Project executive orders, 1994 on (comparativeagendas.net; text from the Federal Register API) | Checked 2026-11-14. 1,462 orders from 1994 on match exactly one Federal Register document; the API needs no key. A 40-order sample has median 890 words (p10 365, p90 2,180), so about 1,100 orders fall in range. Same 20-topic scheme as the two existing CAP tasks, which gives a within-scheme short-versus-long comparison. The catch: the codebook says the code corresponds to the Description column, a summary of about 13 words, so coders labeled the summary, not the order. The task would test whether a reading of the full text agrees with a label assigned from a one-line description. Topic labels are CC BY-NC-SA 4.0, the same terms as the existing CAP tasks. |
| ECtHR judgments, violated articles (LexGLUE `ecthr_a`) | Usable data (11,000 documents, median about 890 words, CC-BY-4.0), but a legal-prediction task with little political-science content, and the text is the facts section only. |
| EUR-LEX documents, EuroVoc concepts (LexGLUE `eurlex`) | Labels are library subject headings assigned by indexers, not research codes. |

## Not usable

Media Frames Corpus (text only via LexisNexis); Trillion Dollar Words FOMC
stance (sentence-level labels; the document-level score is model output);
Grimmer press releases (no raw text, model-based topics); Count Love and Crowd
Counting Consortium (URLs only); PAPEA full-article gold standard (100
documents, German, median 376 words); Baly et al. article ideology (label is
the outlet's AllSides rating, not a judgment about the article); Manifesto
Project and CAP State of the Union (labels at the quasi-sentence level);
Framing Across Borders (140 articles); Hobbs and Green open-ended answers
(short, no archive found). SemEval-2023 Task 3 article framing (checked
2026-11-14: 590 English articles of about 850 words, but data come only after
a vetted registration, and the usage agreement forbids redistribution "in part
or full"). Policy Agendas public laws (labels come from a CQ Almanac blurb;
a 30-law sample from 1995 on has median 383 words, and only a third fall
between 400 and 5,000). Earlier rejections in
[`task_expansion_candidates.csv`](task_expansion_candidates.csv) still hold,
including ParlSpeech-linked speeches with withheld text.

## Cost

Estimated by scaling each API model's November release cost by the ratio of
new to old input tokens. The release used about 520 input tokens per item and
cost $73.90 for all 22 API models. A long-text item is assumed to be about 400
prompt tokens plus 1.35 tokens per word. Output length per item does not grow
with document length, so this scaling overstates cost slightly.

| Tasks × items | Words per document | Tokens per item | Cost, 22 API models | Costliest single model |
|---|---|---|---|---|
| 10 × 100 | 500 | about 1,100 | about $30 | about $7 |
| 10 × 100 | 1,500 | about 2,400 | about $70 | about $17 |
| 10 × 100 | 3,000 | about 4,500 | about $130 | about $32 |

Three usable tasks at their measured lengths (about 1,750, 2,600 and 420 words)
come to roughly $35. Local models need no cash, but prefill time on Hive grows
with input length.

## Constraints

1. **Context length.** Releases run with a 16,384-token context. Speeches and
   articles fit. About a fifth of Supreme Court opinions do not. The choice is
   between truncating (and saying so) or filtering to opinions under a word
   cap, which changes the sample.
2. **New release.** Adding tasks changes the frozen sample and its hash. Every
   model must run the new tasks for the release to stay comparable, including
   the two small models now in their pilots.
3. **Language.** The populism speeches are mostly not English. Keeping all
   languages tests multilingual coding of long text; filtering to English
   leaves about 320 speeches.
4. **Exposure.** Supreme Court opinions and their issue codes are among the
   most widely published legal texts. The timing check in
   [`contamination_timing_check.md`](contamination_timing_check.md) cannot
   address this, because the labels predate every model.

## Implication

A length comparison needs more long tasks than public data currently supply.
Realistic options are (i) add the three usable tasks plus the two built Benoit
manifesto tasks in `tasks_ext/` as task-variety additions, and report long-text
results descriptively, or (ii) also build the CAP executive-order task, which
brings the long-text set to four and gives one same-scheme short-versus-long
pair, at the cost of labels that were assigned from summaries. SemEval-2023
framing is ruled out by its redistribution terms.

## Glossary

- **CAP / Policy Agendas Project**: the Comparative Agendas Project's 20-topic
  policy coding scheme, used in two existing tasks.
- **LexGLUE**: a 2022 collection of legal text-classification datasets on
  Hugging Face.
- **Holistic grading**: coders read the whole speech and assign one populism
  score, rather than coding sentences.
- **Prefill**: the part of local inference that reads the input; its time grows
  with input length.

## Built, 2026-11-14

The four tasks are built in `tasks_ext/` (not `tasks/`, so the frozen 33-task
panels are unchanged) and have no model runs yet. Builders, prompts and data:
`code/build_<task>_task.py`, `prompts/<task>.txt`, `data/<task>.csv`; sources
and prompt changes are in [`prompts_provenance.md`](prompts_provenance.md).

| Task | Documents | Median words | Median tokens | Classes | Smallest class |
|---|---|---|---|---|---|
| `gpd_speech_populism` | 1,131 | 1,771 | 3,404 | 3 (ordinal) | 6.2% (very populist) |
| `scdb_opinion_issue_area` | 698 | 4,097 | 5,838 | 12 | 1.1% (Privacy) |
| `kiesel_hyperpartisan_article` | 1,184 | 440 | 584 | 2 | 44.8% |
| `cap_executive_order_policy_topic` | 1,428 | 867 | 1,130 | 21 | 1 item (Culture) |

Decisions made while building:

1. **No truncation.** Documents above 10,000 tokens (the larger of the
   o200k_base and cl100k_base counts; `code/long_text_budget.py`) are dropped,
   so every model reads whole documents within the 16,384-token Hive context.
   This keeps only the shorter half of Supreme Court opinions (median 6,900
   words in the source split) and tilts that task away from First Amendment
   and Due Process cases. For the speeches it removes 29% of very populist,
   19% of somewhat populist and 10% of not populist speeches, and most
   speeches in non-Latin scripts (Central Asia, Israel, Georgia, Armenia).
2. **Populism gold** is the mean of each speech file's own coder grades, not
   the GPD's `averagerubric`, which pools two texts per slot for Trump's first
   term. Research-assistant headers that name the speech type are stripped.
3. **Executive orders** keep the title, which is what CAP coded; two swapped
   CAP labels (EO 13673, 13674) are corrected in the builder.
4. **Supreme Court prompt** lists, under each issue area, the SCDB codebook's
   descriptions of its issues (about 3,300 tokens).

Each task was reviewed against its raw source by a separate agent; the
problems found (wrong-speech labels, a name collision between two Argentine
presidents, heading regex errors, swapped labels, label-revealing headers)
were fixed before this note.

**Open decision before any run.** The benchmark draws a random 100 items per
task. In such a draw the Supreme Court task has no Privacy item about 30% of
the time and no Interstate Relations, Attorneys or Unions item 13 to 25% of the
time; the speech task averages about 6 very populist items; the executive-order
task averages about 6 classes with 2 or fewer items. Macro-F1 will move with
single items. Options are a stratified draw with a minimum per class, a larger
draw for these tasks, or reporting accuracy alongside macro-F1.
