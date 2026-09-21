# Event-coding confirmation test

## Verdict

The apparent event-coding gain in the repository benchmark did not generalize to four outcome-blind external tasks. Qwen3.6 exceeded Llama 3.1 on all seven repository event tasks, averaging 4.3 F1 points higher, but averaged 4.4 points lower across the external tasks. The evidence therefore does not support a general claim that newer open models improved event coding.

## Why this test was run

The 34-task comparison suggested that gains were concentrated in event and policy coding. Because the five task categories were constructed after the initial results, small groups or unusual repository tasks could create that pattern. This analysis froze a narrower event-coding hypothesis and external inclusion rules before generating any external model outcomes.

## What is compared

The analysis compares Qwen3.6 27B with Llama 3.1 70B on identical items, prompts, schemas, generation controls, and task-specific macro-F1 definitions. Both checkpoints ran on one 97,887 MiB Blackwell GPU using immutable revisions and the pinned vLLM 0.26 runtime.

## Existing evidence

All seven repository event tasks favor Qwen3.6. The mean difference is +4.3 F1 points. The pattern occurs in both the frozen Broad18 subset (3/3 tasks, mean +1.9) and the 16 tasks excluded beforehand (4/4, mean +6.1). Across 21 historical checkpoints, event scores increase 2.4 points per year faster than other task scores under equal source-family weighting (source-bootstrap interval +0.8 to +4.1). This longitudinal contrast is positive in all three active-parameter classes and both Qwen dense lineages, but approximately zero in the Llama 70B lineage. Model family and date therefore remain confounded.

## External confirmation

Four tasks were selected using the frozen access and outcome rules in [the hypothesis file](../experiments/event_progress_hypothesis_20260823.yaml): MAVEN event presence, RAMS event type, and Arabic GSR assault and protest presence. [MAVEN](https://github.com/THU-KEG/MAVEN-dataset) and [RAMS](https://nlp.jhu.edu/rams/) are separate English event-extraction sources. The two [Arabic Event GSR](https://github.com/openeventdata/arabic_event_gsr) tasks form one source family.

Qwen3.6 loses 14.0 points on Arabic assault detection and 4.8 on Arabic protest detection; both paired-item intervals exclude zero. It loses 1.9 on MAVEN event presence, with an interval spanning zero. It gains 3.1 on RAMS event typing, although absolute macro-F1 is only 5.7% for Qwen and 2.6% for Llama. The equal-task difference is −4.4 points and the equal-source-family difference is −2.7.

The external result changes the conclusion. Event-task improvement is a stable description of this repository, not a portable capability trend. Differences in language, label ontology, task formulation, and source distribution could explain the reversal, but the current experiment does not separate them.

## Audit

The external evaluation contains 2,000 unique items across four tasks. Qwen produced 2,000 valid responses. Llama produced 1,998 valid responses; its two malformed MAVEN responses were retained and scored wrong. The runs completed as Slurm jobs 21149935 and 21149936 with peaks of 88,408 and 89,146 MiB on separate 97,887 MiB GPUs. No paid API was used.

Authoritative results are [external_confirmation.csv](../output/sidecar/frontier_2026/event_progress/external_confirmation.csv), [event_progress_sensitivity.csv](../output/sidecar/frontier_2026/event_progress/event_progress_sensitivity.csv), and the [diagnostic figure](../output/sidecar/frontier_2026/event_progress/external_event_confirmation.png).
