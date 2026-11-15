# Scoping memo: a paper on differential error in LLM annotation of political text

Written 2026-11-14. Status: proposal for discussion. No new analysis was run for this memo; every number comes from an existing result file linked below. The cue experiments in the section "What we already have" are already public in the `hhilbig/llm-party-names` repository.

## Verdict

A Political Analysis paper looks feasible, and the benchmark itself should not be its contribution. The proposed claim: when a large language model (LLM) labels political text, its errors are correlated with the variables political scientists put on the right-hand side of their regressions (party, government status, language, country), and part of that correlation is caused by cues in the text such as party names.

Three recent papers already cover the neighbouring questions (whether LLM labels change statistical conclusions, and whether disagreement with human coders is model error). One of them, Yang et al., is conditionally accepted at AJPS and replicates the original analyses of 14 political science articles, so a paper that counts how often LLM labels change published conclusions would not be new. Neither measures the direction of LLM error relative to covariates, and neither manipulates cues. We already have randomized cue experiments on three position tasks and eleven models. What is missing is the same question asked observationally across the roughly 20 tasks that carry covariates.

**HH:**

## The question and why it is open

"Differential error" means measurement error whose size or direction depends on a covariate. If a model codes Conservative sentences too far right and Labour sentences too far left, a regression of the coded position on party is biased even when overall accuracy is high. Random error would only add noise to an outcome variable.

What the literature has established (read status in brackets):

| Paper | What it shows | What it leaves open |
| --- | --- | --- |
| Baumann et al. 2025, "Large Language Model Hacking", arXiv 2509.08825 [main text read] | 37 tasks from 21 datasets, 18 models, 2,361 group-difference hypotheses. LLM labels give a different significance decision than human labels in about 31% of hypotheses for the largest models and about 50% for the smallest. F1 explains about 8% of the variation in that risk. | States that LLM errors do worse than a random-error baseline at equal F1 and "may not be uniformly distributed but rather systematically biased", but does not measure direction, correlates or causes. No cue experiments. 52% of the group splits are keyword splits, not real metadata. Drops all items on which human coders disagreed. Models are Llama 3, Qwen 2.5/3, Gemma and GPT-4o. |
| Nakamura, Tan and Yean 2026, "Observational Equivalence of LLM and Human Annotation", arXiv 2609.22133 [main text read] | 14 binary tasks from AJPS/APSR/JOP/ISQ articles, 100 texts each, 13 models, three PhD-student coders, 165 crowd coders. LLM-human agreement is about as high as human-human agreement, and disagreement falls on the same texts. | Agreement only. Does not ask whether errors line up with any covariate. Argues that disagreement often reflects ambiguity in the text or codebook. |
| Yang, Wang, Zhou and Xu 2026, "Data Annotation with Large Language Models: Lessons from a Large Empirical Evaluation", conditionally accepted at AJPS per the first author's website [introduction, study selection and results passages read; not the appendices] | Re-annotates the data of 14 articles from APSR, AJPS, JOP, BJPS and PSRM with 16 models and re-runs the original analyses. LLM labels have low reliability against the original labels; 35 to 42% of LLM-derived estimates lead to a different conclusion than the original study; prompt variation matters little; bias correction works but needs a sizable gold sample. | Says the "sources and direction of annotation errors" of LLMs are hard to anticipate and "far less understood" than those of human coders, citing the human party-cue literature, and does not study them. Reports no consistent pattern in which model gives larger or smaller coefficients. Discrete labels only. |
| Bisbee and Spirling 2026, "What to Do When Humans Are No Longer the Gold Standard", working paper dated 2026-02-10 [introduction and measurement-error sections read] | A sensitivity framework for a misclassified binary treatment or outcome when no human gold standard is trusted. Argues that as models improve, estimates that reject a null mostly get stronger. | Differential error (misclassification correlated with the outcome) enters as an assumed parameter, shown at zero and at an "extreme" value of 0.7; the paper notes it "can produce bias in either direction". It gives no empirical values for that parameter. |
| Vallejo Vera and Driggers 2024, arXiv 2408.15895 [read in full] | Replicates Ennser-Jedenastik and Meyer (2018): 200 Austrian manifesto statements on immigration, each shown with no party label and with each of four party labels, to GPT-3.5 Turbo, GPT-4o, Llama 3 70B and Llama 3.1 70B. Left labels make the models code statements as more pro-immigration and right labels as less. Unlike the human coders, the models also react to the two centrist parties, and the effects are larger than for humans. | One task, one country, one issue, four models from 2023-24, labels attached to statements by the experimenter (not cues occurring in the text). No downstream estimate. The conclusion asks for tests with other cues and tasks. |
| Le Mens and Gallego 2025; Ashwin et al. 2025; Stuhler et al. 2025; Kostikova et al. 2026; TeBlunthuis et al. 2024; Knox, Lucas and Cho 2022 [abstract or partial reads by search agents] | Each raises non-random LLM or classifier error as a concern for downstream estimates. | Evidence is one application each, or simulation. |

Bottom line: the correction methods in this literature (design-based supervised learning, prediction-powered inference; see Glossary) are motivated by differential error, but no paper measures how large and how directional it is across many real political science tasks, and only one small study manipulates a cue.

Baumann et al.'s Appendix E.10 was checked for a breakdown by group. It contains none. It tests whether the LLM-based and human-based coefficients differ (they do in 35 to 42% of hypotheses) and adjusts for sampling noise in the human-label test.

Two consequences for positioning. First, our vary-the-title result differs from Vallejo Vera and Driggers in one finding that matters: they report cue effects larger than for human coders in every model, while in our data two of the API models (Gemini 3.8 Flash, GPT-6.1 Sol) react less than Benoit et al.'s experts and others react several times more. Second, Bisbee and Spirling's framework needs a value for differential error and has none; measuring it across tasks would supply that input.

Not checked: Yang and Wang 2026, "Benchmark Illusion" (arXiv 2602.11898), abstract only. It reports that switching the annotation model changes estimated effects in re-analyses of education and political science studies.

**HH:**

## What we already have

### 1. Masking party and politician names

*Why:* if names in a passage push the model's code toward the named party's reputation, removing the names should move Conservative and Labour sentences toward each other.

*What is compared:* for each sentence that names a party or politician, the model's code with the names replaced by `[PARTY]` / `[POLITICIAN]` minus its code for the original sentence; that change for Conservative sentences minus the change for Labour sentences. Sentences with no name are identical in both runs and measure run-to-run noise (the placebo). Benoit et al. (2016) UK manifesto sentences on a -2 to 2 scale; Le Mens and Gallego (2025) tweets by US members of Congress on a 0 to 100 scale, Republican minus Democratic.

*Result:* masking moves the two parties' codes toward each other in every open-weight model and in the tweet task for every model. The placebo changes are near zero.

- Manifestos, economic, four open-weight models: Conservative-minus-Labour change between -0.15 and -0.44 scale points on named sentences (192 sentences); placebo between -0.01 and -0.06.
- Manifestos, economic, four API models: -0.06 to -0.20; only DeepSeek V4 Pro's interval excludes zero.
- Tweets, named (144 tweets): Republican-minus-Democratic change of -10 to -23 points for the open-weight models and -6 to -14 for the API models; placebo within about 1 point of zero.
- The party gap in error (model minus human mean) on named tweets is 9 to 13 points for the API models before masking and -2 to 3 after.
- Masking raises mean absolute error on named tweets for every model (for example 7.9 to 9.9 for Claude Sonnet 5).

*Implication:* names cause a party-dependent shift. They also carry valid information, since accuracy falls when they are removed. The paper has to treat cues as both signal and source of differential error, not as pure contamination.

Files: [`analyze_masking_experiment.py`](../code/analyze_masking_experiment.py), [open-weight summary](../output/sidecar/task_ext_20261010_masked/analysis/summary.txt), [API summary](../output/sidecar/task_ext_20261011_masked_api/analysis_stdout.txt).

### 2. Showing or hiding the manifesto title

*Why:* Benoit et al. recorded whether each human coder saw the manifesto title, so the same manipulation can be applied to models and compared with people.

*What is compared:* the model's code for a sentence with one added line ("Source: Conservative Party manifesto, 1992") minus its code without that line; the Conservative shift minus the Labour shift. Human benchmarks from the same data: experts who coded each sentence once with the title and about a year later without it, and semi-experts who were randomly shown the title or not.

*Result:* every model shifts in the same direction as the experts, and the size varies by more than a factor of twenty across models. Two models react less than the experts, six about as much, three more.

On the 1,033 economic sentences (12 manifestos) that all eleven models coded:

| Coder | Conservative-minus-Labour shift, scale points (95% interval) | Compared with experts |
| --- | --- | --- |
| Experts, same coder a year apart | 0.22 (0.12, 0.35) | |
| Gemini 3.8 Flash | 0.04 (-0.01, 0.07) | less |
| GPT-6.1 Sol | 0.04 (0.01, 0.08) | less |
| Claude Sonnet 5 | 0.14 (0.08, 0.24) | not distinguishable |
| GPT-5.6 Terra | 0.16 (0.07, 0.22) | not distinguishable |
| Qwen3.8 27B | 0.18 (0.11, 0.23) | not distinguishable |
| Gemma 4 31B | 0.24 (0.17, 0.32) | not distinguishable |
| DeepSeek V4 Pro | 0.25 (0.19, 0.31) | not distinguishable |
| Qwen3.6 27B | 0.27 (0.18, 0.38) | not distinguishable |
| Qwen3.6 35B-A3B | 0.39 (0.30, 0.48) | more |
| Nemotron 3.5 Lightning | 0.50 (0.38, 0.64) | more |
| Qwen3 30B | 0.90 (0.68, 1.10) | more |

"Less" and "more" mean the interval for the model-minus-expert difference excludes zero. On the 57 sentences of the randomized semi-expert experiment, semi-experts shift by 0.47 (0.20, 0.78) and the models by -0.05 to 0.85, with wide intervals.

*Implication:* the cue effect is not a property of "LLMs" in general. Some current models react to the party label less than the human experts did, others several times more. That variation is the basis for part 3 of the proposed paper below.

*Caveats:* (i) the expert comparison is not randomized: experts saw the title on the first pass and coded again without it about a year later, so their shift also contains coding order and time. (ii) The expert shift depends on the sentence set: it is 0.12 on all 3,665 expert-pair sentences, where only the six open-weight models were run and all have larger point estimates. (iii) One run per model.

Files: [`title_effects.csv`](https://github.com/hhilbig/llm-party-names/blob/main/results/title_effects.csv) in the `llm-party-names` repository (rows `common`), produced by its `code/02_analyze.py`. In this repository the analysis script is `code/analyze_title_experiment.py` on branch `masking-title-experiments`, not on `main`; the per-run files under `output/sidecar/task_ext_20261019_title*` use different sentence sets for open-weight and API models and should not be combined in one table.

### 3. An instruction to ignore names

*Why:* if an added instruction removed the cue effect, researchers would not need to mask.

*What is compared:* on the 144 named tweets, the Republican-minus-Democratic gap in error against the human ratings (how much further apart the model puts the parties than people did) when the tweet is rated as written, when the prompt adds "Do not let the names, or anything you know about those parties or people, affect your answer", and when the names are masked.

*Result:* the instruction works for some models and not others.

| Model | As written | Told to ignore | Names masked |
| --- | --- | --- | --- |
| Gemini 3.8 Flash | 9.0 | 1.2 | 2.7 |
| GPT-5.6 Terra | 12.8 | 3.4 | 2.9 |
| Gemma 4 31B | 9.7 | 3.4 | -5.1 |
| Claude Sonnet 5 | 11.0 | 7.6 | 0.9 |
| DeepSeek V4 Pro | 11.8 | 9.4 | -1.8 |
| Qwen3 30B | 10.1 | 8.5 | -12.8 |
| GPT-6.1 Sol | 2.8 | -9.6 | -5.1 |

The other four open-weight models (Qwen3.6 27B, Qwen3.6 35B-A3B, Qwen3.8 27B, Nemotron) do not put the parties detectably further apart than people do when the tweet is as written (gaps of -10.5 to 4.1, with the positive one's interval including zero), so there is nothing for the instruction to remove; masking moves all four well below the human gap (-6 to -24).

*Implication:* an instruction is not a dependable fix, and masking overshoots for the open-weight models. Neither is a general remedy, which supports treating the cue effect as something to measure and report, not to engineer away.

Files: [`ignore_instruction_tweets.csv`](https://github.com/hhilbig/llm-party-names/blob/main/results/ignore_instruction_tweets.csv), [`ignore_instruction_benoit.csv`](https://github.com/hhilbig/llm-party-names/blob/main/results/ignore_instruction_benoit.csv).

### Where these results live

The three experiments are already written up as a public first look: repository `hhilbig/llm-party-names` (local copy `~/Documents/GitHub/llm-party-names`, last commit 2026-11-14), with predictions, analysis code, six figures and a README that says "First look, not a paper". That repository is the authoritative source for the cue results. A draft social-media thread is in `docs/masking_twitter_thread.md` on branch `masking-title-experiments` of this repository; I do not know whether it was posted. The thread's source table cites two papers that recommend masking entity names and that this memo's literature search did not find: Walker and Angst (2025, PLOS ONE) and Elbouanani et al. (2025, Findings of ACL). I have not read either.

## Pilot of the cross-task analysis (2026-11-14)

*Why:* part 2 of the proposed paper asks whether model error differs across the groups a researcher would compare, outside the cue experiments. This pilot checks whether existing predictions show such differences at all. No model was run. ParlaSent, named as a pilot task in an earlier draft of this memo, has no predictions on disk, so the pilot uses the two Benoit tasks and AgoraSpeech.

Script: [`code/pilot_differential_error.py`](../code/pilot_differential_error.py). Results: [`benoit_party_contrast.csv`](../output/sidecar/pilot_differential_error_20261114/benoit_party_contrast.csv), [`agoraspeech_government_contrast.csv`](../output/sidecar/pilot_differential_error_20261114/agoraspeech_government_contrast.csv).

### UK manifesto positions

*What is compared:* the mean code of Conservative sentences minus the mean code of Labour sentences, computed from the model's codes and from the experts' mean codes, on the 500-sentence sample of each domain, for six open-weight models rating sentences as written. "Contrast bias" is the model's difference minus the experts' difference. Intervals come from a bootstrap over the 18 manifestos.

*Result (against the expert means used as benchmark gold, which the next subsection shows are not a like-for-like reference):* the party contrast differs from the experts' in both domains, in opposite directions.

| | Experts' Conservative-minus-Labour difference | Contrast bias, range across models | Models whose interval excludes zero |
| --- | --- | --- | --- |
| Economic, all 500 sentences | 1.11 | 0.09 to 0.22 (five models); -0.27 (Nemotron) | 3 of 5; Nemotron |
| Economic, 192 sentences naming a party or politician | 1.27 | 0.29 to 0.54 (five models) | 5 of 5 |
| Economic, 308 sentences naming no one | 1.02 | -0.05 to 0.08 (five models) | 0 of 5 |
| Social, all 500 sentences | 0.53 | -0.15 to -0.23 | 3 of 6 |
| Social, 146 sentences naming a party or politician | 0.75 | -0.04 to 0.23 | 1 of 6 |
| Social, 354 sentences naming no one | 0.43 | -0.28 to -0.39 | 6 of 6 |

Nemotron 3.5 Lightning is listed separately on the economic task because it compresses the scale: its codes rise by 0.63 points per point of the expert mean, against 0.93 to 1.02 for the other five.

Against this reference, on the economic task the models overstate the party difference by 8 to 19 percent, and all of it comes from sentences that name a party or politician. On the social task the models understate the party difference by about 30 to 40 percent overall and by 65 to 90 percent on sentences that name no one. Comparing Conservative and Labour sentences that experts placed at the same position (the "conditional gap" in the results file) gives the same signs: 0.41 to 0.78 on named economic sentences, with all six intervals excluding zero, and -0.18 to -0.36 on unnamed social sentences, where only one interval excludes zero.

*Implication:* superseded by the next subsection. The social understatement is produced by the reference codes, not by the models.

### The same comparison against experts who did not see the party

*Why:* the expert means used above come mostly from codings made with the manifesto title shown (about 80 percent of expert codings in both domains), while the models rated the sentences without the title. Benoit et al. recorded `title_shown` for every coding, so the experts can be split.

*What is compared:* for the sample sentences that experts coded in the domain both with and without the title (388 economic, 398 social), the Conservative-minus-Labour difference from the mean of title-hidden expert codes, from the mean of title-shown expert codes, and from each model's codes.

*Result:* on social policy the models agree with the experts who did not see the party. The gap found above is the experts' own reaction to the party label.

| | Experts, title hidden | Experts, title shown | Models (range) | Model bias against title-hidden experts |
| --- | --- | --- | --- | --- |
| Social, all 398 | 0.28 | 0.61 | 0.34 to 0.44 | 0.06 to 0.16; 1 of 6 intervals excludes zero |
| Social, 278 naming no one | 0.11 | 0.53 | 0.09 to 0.24 | -0.03 to 0.13; 0 of 6 |
| Economic, all 388 | 1.20 | 1.16 | 1.23 to 1.38 (five models) | 0.03 to 0.18; 1 of 5 |
| Economic, 152 naming a party or politician | 1.47 | 1.32 | 1.54 to 1.74 (five models) | 0.07 to 0.27; 2 of 5 |
| Economic, 236 naming no one | 1.03 | 1.06 | 0.99 to 1.19 (five models) | -0.03 to 0.17; 0 of 5 |

On social sentences the experts' Conservative-minus-Labour difference is 0.33 larger with the title than without it (interval 0.05 to 0.54), and 0.42 larger on sentences that name no one (0.10 to 0.60). On economic sentences the two expert references do not differ (-0.04; -0.20 to 0.07). Nemotron is again left out of the economic ranges because it compresses the scale (bias -0.34 against title-hidden experts).

*Implication:* three points for the paper.

1. The "second source" of party-related error suggested by the first comparison does not exist in these data. Against experts who did not know the party, the models show no detectable bias on sentences that name no one, in either domain.
2. The name effect on economic sentences survives but is smaller and less certain against the title-hidden reference (0.07 to 0.27, two of five intervals excluding zero) than against the title-shown one (0.22 to 0.42).
3. Measured differential error depends on what the human coders knew. A reference coded with the party visible contains the coders' own cue effect, and a model that does not see the cue then appears to be biased in the opposite direction. Any cross-task analysis has to establish, task by task, what the human coders could see. This is a concrete case of the gold-standard concern in Nakamura et al. and Bisbee and Spirling.

*Caveats:* the title-hidden reference rests on about 1.25 expert codings per sentence, against about 4 for title-shown, so it is noisier. Title-hidden codings were made about a year later and in random order, so the expert difference also contains order and time. The expert difference on social sentences here (0.33) is larger than the within-coder estimate from the title experiment (0.13); the two use different sentence sets and this one compares across coders.

Result file: [`benoit_party_contrast_title_reference.csv`](../output/sidecar/pilot_differential_error_20261114/benoit_party_contrast_title_reference.csv). The script needs `--codings` pointing to `coding_all_long_2014-03-14.csv` from `kbenoit/CSTA-APSR`, which is not stored in this repository.

### Greek election speeches

*What is compared:* the share of paragraphs coded "criticism" (as opposed to "political agenda") for the five opposition leaders minus the share for Mitsotakis, who was prime minister, computed from model labels and from the gold labels, on the 500-paragraph sample (133 by Mitsotakis), for the ten models in the serial panel. The prompt shows the speaker's name. Intervals come from a bootstrap over speeches.

*Result:* every model codes opposition paragraphs as criticism more readily than Mitsotakis's. Among paragraphs whose gold label is "political agenda", the models code 8 to 26 percent as criticism for opposition leaders and 0 to 5 percent for Mitsotakis. That gap is 7 to 21 percentage points and its interval excludes zero for all ten models. The effect on the group contrast is small: the gold opposition-minus-government difference is 36 points and the models overstate it by 0 to 9 points, with the interval excluding zero for four models. Accuracy is 88 to 92 percent for all ten.

*Implication:* error rates differ by speaker in every model even on a task with about 90 percent accuracy. The pilot cannot say why. The speaker's name in the prompt may act as a cue, or opposition paragraphs labeled "agenda" may contain more critical content, which is the ambiguity reading in Nakamura et al. Hiding the speaker line would separate the two, and the task is a candidate for extending the cue experiments.

### Caveats for both

1. Samples are 500 items, clustered in 18 manifestos or in speeches, and one run per model.
2. The models differ between the two tasks (six open-weight models from October 2026; ten models from the older serial panel), so the tasks are not compared on the same models.
3. The expert reference for the manifesto tasks was coded mostly with the party known; the subsection above redoes the comparison without it. For AgoraSpeech I have not checked what the annotators saw.
4. Intervals are not adjusted for the number of models and subsets examined.

Verdict: the pilot supports going ahead with part 2, with a change of design. Error rates differ by speaker on AgoraSpeech in all ten models. On the manifesto tasks the only model bias that survives a like-for-like reference is the name effect on economic sentences. The apparent bias on social sentences came from the human reference. Part 2 therefore needs, for each task, a record of what the human coders could see, and should report differential error relative to that.

## Proposed paper

1. **Cue experiments (main contribution).** Masking, title and ignore-instruction results on the three position tasks, compared with Benoit et al.'s experts and semi-experts. Extend to two or three more tasks where a speaker or party label can be shown or hidden without changing the text (candidates: PopBERT faction, ParlaSent party, AgoraSpeech speaker, which the current prompt already shows).
2. **Observational differential error across tasks.** For each task with a covariate, compare the model's error rate and signed error across groups, and compare the group difference estimated from model labels with the one from human labels. Report the direction and size of the error gap between groups. Baumann et al. and Yang et al. already report how often conclusions change, so that count should be a secondary output here.
3. **Does model capability reduce it?** Relate cue effects and group error gaps to overall accuracy and model release date across the 40+ models.
4. **Does a gold-labeled subsample fix it?** Apply design-based supervised learning with realistic gold-sample sizes and report the bias removed and the power lost.

Assumptions this rests on: (i) the human labels are an acceptable reference for group differences, which Nakamura et al. dispute for ambiguous items; (ii) the covariates in our task files are the ones the source papers used, which is unverified for most tasks; (iii) full datasets, not the 500-item benchmark samples, will be labeled for part 2.

**HH:**

## Which tasks can be used for part 2

From the columns in each task's data file. "Source estimate" is what I recall the source paper estimating and is unverified.

| Tier | Tasks | Covariate in the data |
| --- | --- | --- |
| Strong: covariate present and the source paper's main estimate is a contrast on it | Benoit economic and social positions; Di Cocco and Monechi populism; PopBERT populism; Le Mens and Gallego tweet position; ParlaSent sentiment; Global Populism Database; CAP executive orders | party, faction, ruling party, leader, country, president, year |
| Usable: a natural contrast exists but is descriptive or needs a merge | Wesleyan ad tone (candidate; party needs a merge); AgoraSpeech (politician); ClaimBuster (speaker party); CAP CRS, CAP party platforms, SCDB (year); UNSC stance (conflict, year); GTD attack type (country); ICBe (crisis); Spanish protest toxicity (country); Heseltine tone (country); PDD delegitimization (page); Theocharis incivility (date) | |
| None in the data | Gilardi (2), Ballard, Burnham (4), Chae, Halterman (3), Mellon, Osnabrügge, PolitiCause, TwitCivility, PLOVER, Erlich, Müller and Fujimura, Kiesel, Bestvater (2), Rheault, Ornstein, PAPEA (2), Yan and Bernhard, Horák; Chalkidis (one speaker) | |

Most of the strong tasks are in `tasks_ext/`, not in the 34 public tasks. Each benchmark task has at most 500 labeled items per model, which is too few for most group contrasts; labeling the full files with a handful of open-weight models on Hive costs nothing but queue time.

## Risks

- **Overlap with Baumann et al. and Yang et al.** Baumann et al. use the Benoit manifesto data and their running example is the Conservative-versus-Labour contrast. Yang et al. replicate published political science analyses and will appear in AJPS. The paper must state clearly that it explains and tests a cause of what they count.
- **The ambiguity argument.** Nakamura et al. would read group error gaps as ambiguity. The randomized experiments answer this for cues, because the text is held fixed. The observational part 2 does not have that protection.
- **Small numbers of clusters.** The manifesto intervals come from a bootstrap over manifestos, and there are 18 or fewer per domain. Tasks with one country or one speaker cannot support a group contrast.
- **Model turnover.** Results on specific models date quickly. The paper should lead with the pattern across models and the comparison with human coders.

## Next steps, in order

1. Done 2026-11-14: Yang et al., Bisbee and Spirling, Vallejo Vera and Driggers, and Baumann et al.'s Appendix E.10 read; see the table above. Still to read in full: Yang et al.'s appendices and Yang and Wang's "Benchmark Illusion".
2. Done 2026-11-14: the title-experiment script and the ignore-instruction analysis already exist (branch `masking-title-experiments` and the `llm-party-names` repository). Nothing was re-run.
3. Read Walker and Angst (2025) and Elbouanani et al. (2025), which recommend masking names.
4. For the eight strong tasks, check the source papers' estimates and confirm the covariates match.
5. Done 2026-11-14: pilot of part 2 on the Benoit tasks and AgoraSpeech; see the pilot section. Manifesto comparison redone against title-hidden expert codes the same day. Next: establish what the AgoraSpeech annotators saw, then run AgoraSpeech with the speaker line hidden and label the full Benoit and ParlaSent files.

## Glossary

- **Differential error:** measurement error whose distribution depends on a covariate in the analysis.
- **Cue:** information in the text or prompt about who produced it (party name, politician name, manifesto title).
- **Placebo (here):** items whose text is identical in the masked and unmasked runs; any change in their codes is run-to-run noise.
- **Gold labels:** the human codes a model is compared with.
- **Design-based supervised learning (DSL):** Egami et al. (2023). Uses model labels for all documents and human labels for a random subsample with known sampling probabilities, and corrects the regression so that it is unbiased even if the model labels are biased.
- **Prediction-powered inference (PPI):** Angelopoulos et al. (2023). Same idea for means and regression coefficients: a small human-labeled sample is used to estimate and subtract the bias of estimates computed from model predictions.
- **Open-weight / API models:** models run locally on Hive or mac2 from downloaded weights, versus models called through a provider.
- **Type I / Type II error:** finding a difference that the human-label analysis does not find, and missing one that it does.
