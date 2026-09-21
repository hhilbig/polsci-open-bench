# External human-vote task audit

## Verdict

Human disagreement does not provide a general explanation for the model
regressions. Across five datasets, a one-standard-deviation increase in human
agreement predicts only a 0.84-point increase in Qwen3.6's advantage over
Llama 3.1 (95% bootstrap interval: -0.20 to 1.83 points). The direction is
consistent with the hypothesis, but the interval crosses zero and the pattern
varies across datasets. Measuring Hate Speech supports the hypothesis clearly;
HateXplain and Moral Foundations Reddit do not.

The strongest defensible conclusion is therefore narrower: ambiguity may
explain some task-specific regressions, especially stance, but it does not
explain the broad pattern by itself. The evidence remains more consistent with
heterogeneous capability shifts across task types than with one universal
ambiguity mechanism.

This analysis asks whether external text datasets, absent from the repository's
34 canonical tasks, contain the original item-level judgments needed to test
whether newer-model gains shrink as human disagreement rises. The initial audit
identified eligible sources. The completed stress test then compares immutable
Llama 3.1 70B and Qwen3.6 27B predictions on three new 1,000-item tasks and two
existing 500-item human-vote tasks.

## Requirements and procedure

A candidate passes the data gate when it supplies (i) the original text, (ii)
separate human judgments rather than only an aggregate label, (iii) at least
three judgments for at least 200 items, and (iv) a label that can be converted
into a reproducible classification task. The script downloads the current
public source files to a temporary directory and writes summary statistics
only. It does not retain source text in the repository.

The audit treats majority share as the fraction of an item's annotators who
selected its most common label. For the Moral Foundations Reddit Corpus, an
annotator's complete set of moral labels is one vote because the source permits
multi-label judgments. For Dagstuhl, each item-by-quality-dimension rating is an
agreement unit because the 15 ordinal dimensions are separate judgments.

## Results

| Dataset | Usable items | Median votes | Mean majority share | Unanimous | Decision |
|---|---:|---:|---:|---:|---|
| CrowdTruth Twitter events | 3,001 | 6 | 0.899 | 63.2% | Conditional on license clarification |
| HateXplain | 20,148 | 3 | 0.814 | 48.9% | Include |
| Measuring Hate Speech | 17,352 | 2 overall | 0.835 | 54.5% | Include the subset with at least three votes |
| Moral Foundations Reddit Corpus | 17,751 | 3 | 0.645 | 20.4% | Include |
| Dagstuhl Argument Quality | 304 | 3 | 0.758 | 34.4% | Conditional on license and dimension choice |
| Political Enthymemes | Not available | — | — | — | Exclude for now |

The sources provide substantial variation in observed agreement. CrowdTruth is
the clearest relatively explicit event/topic task, while moral, hate, and
argument judgments supply more interpretive comparisons. This makes a
within-dataset disagreement test feasible without relying only on stance.

The current CrowdTruth release contains judgments for 3,020 units but text for
3,019. Unit 23,980 has no released text. Eighteen additional units have fewer
than three non-spam judgments, leaving 3,001 usable items. The script excludes
these cases explicitly.

The current Moral Foundations Reddit Corpus snapshot contains 17,886 unique
texts, although its dataset card describes 16,123 comments. Of the current
snapshot's texts, 17,751 have at least three distinct annotators. Any benchmark
task should pin the exact file revision rather than rely on the card's count.

The Political Enthymemes landing page describes a dataset of roughly 1,482
tweets, but its sample, first-release, and final-release entries are disabled
plain text rather than download links. The page also does not state a license.
The task therefore fails both the availability and provenance gates.

## Completed model stress test

The licensed frozen roster contains HateXplain, Measuring Hate Speech, and the
Moral Foundations Reddit Corpus, each sampled deterministically to 1,000 items.
The analysis adds 500 SemEval stance items and 500 ALIA civic-stance items from
the prior human-vote audit. CrowdTruth and Dagstuhl remain excluded because
their licenses were not sufficiently clear; Political Enthymemes remains
unavailable.

Both models completed all 3,000 new classifications with no malformed output on
one 97,887 MiB Hive GPU. The runs used immutable model revisions, the same task
inputs and generation controls, and the frozen vLLM 0.26/CUDA 13 environment.
The item-level analysis standardizes agreement within each dataset, estimates
the change in the newer model's score for a one-standard-deviation agreement
increase, and averages the five dataset-specific slopes equally. Its percentile
interval uses 10,000 within-dataset bootstrap samples with seed 20260822.

The equal-dataset average score difference is -2.42 points: Qwen3.6 is not
better overall on this deliberately subjective task set. Dataset-specific
agreement slopes range from -1.05 points for HateXplain to +3.09 points for
Measuring Hate Speech. Only the latter has a clearly positive interval. This
heterogeneity prevents a stronger causal or general predictive claim.

## Implication

Do not summarize the result as “higher human disagreement predicts smaller
improvements over time.” A defensible short-form claim is: “On five datasets
with original annotation votes, newer-model gains were somewhat larger where
humans agreed more, but the relationship was weak and inconsistent across
datasets.” The comparison involves two open models and a non-exhaustive task
roster, so it does not identify a general time trend or the global open-model
frontier.

## Reproduction

Run:

```bash
python3 code/audit_external_human_vote_tasks.py
python3 code/build_external_human_vote_tasks.py \
  --output data/human_votes/external \
  --cache /tmp/external-human-votes-cache
python3 code/analyze_external_disagreement.py \
  --external-root output/sidecar/frontier_2026/human_votes/external_runs \
  --human-votes-root output/sidecar/frontier_2026/human_votes \
  --output-dir output/sidecar/frontier_2026/human_votes/external_analysis
python3 -m unittest tests.test_external_human_vote_audit
```

The authoritative outputs are
[`candidate_audit.csv`](../output/sidecar/frontier_2026/human_votes/external_audit/candidate_audit.csv)
and
[`agreement_summary.csv`](../output/sidecar/frontier_2026/human_votes/external_audit/agreement_summary.csv).
The completed model comparison is in
[`disagreement_effect_summary.csv`](../output/sidecar/frontier_2026/human_votes/external_analysis/disagreement_effect_summary.csv),
with the diagnostic
[`human_agreement_improvement.pdf`](../output/sidecar/frontier_2026/human_votes/external_analysis/human_agreement_improvement.pdf).

## Definitions

- **Majority share:** the largest label count divided by the number of votes for
  an item.
- **Raw vote:** one separately identifiable annotator judgment.
- **Usable item:** an item with released text and at least three raw votes.
