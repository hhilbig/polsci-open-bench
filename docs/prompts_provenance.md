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
| `ornstein_scotus_sentiment` | `prompts/ornstein_scotus_sentiment.txt` | Derived from Ornstein, Blasingame, and Truscott (2025) and the associated `promptr` task framing. | Converted a few-shot task into a zero-shot system prompt with the same three-class label space. |
| `chae_semeval_stance` | `prompts/chae_semeval_stance.txt` | Derived from the SemEval-2016 Task 6 stance definitions used by Chae and Davidson (2025). | Fully derived. Includes the SemEval edge case that criticism of a third party does not necessarily imply stance toward the target. |
| `halterman_ccc_protest` | `prompts/halterman_ccc_protest.txt` | Verbatim subset from Halterman and Keith (2025) CCC codebook. | Restricted to the four labels present in `halterman_ccc_hk2025.csv` and appended a JSON-output instruction. |
| `halterman_keith_bfrs` | `prompts/halterman_keith_bfrs.txt` | Verbatim from Halterman and Keith (2025) BFRS codebook. | Appended a JSON-output instruction only. |
| `haunss_papea_fgz_forms` | `prompts/haunss_papea_fgz_forms.txt` | Derived from the seven most common original PAPEA form labels in `AppendixB3_fgz_forms.tab`, which also exposes the source `FORM` codebook strings for each human-coded snippet. | Restricted the task to the seven most common source labels, renamed them into benchmark-stable output strings, and appended a strict JSON-output instruction. |
| `halterman_keith_cmp` | `prompts/halterman_keith_cmp.txt` | Derived from the CMP Aggregation 1 domain structure used in Halterman and Keith's labeled split. | Custom seven-domain prompt plus JSON-output instruction. |
| `osnabruegge_cross_domain_topic` | `prompts/osnabruegge_cross_domain_topic.txt` | Derived from the Osnabruegge, Ash, and Morelli (2023) eight-topic specification, using the same broad policy-domain label set as the paper's `target_corpus.csv` and the same CMP-style topic structure already used elsewhere in the benchmark. | Added brief domain definitions, added `No Topic`, and appended a strict JSON-output instruction. |
| `mellon_bes_mii_2024` | `prompts/mellon_bes_mii_2024.txt` | Derived from Mellon et al. (2024) MII issue coding task and the labeled response categories in the replication materials. | Custom 50-class issue prompt plus JSON-output instruction. |
| `wesleyan_creative_ads_2022` | `prompts/wesleyan_creative_ads_2022.txt` | Derived from the Wesleyan Media Project tone-coding construct used in Zhang et al. (2025). | Custom three-class tone prompt plus JSON-output instruction. Ad text is truncated at 4000 chars when necessary in the loader. |

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
- `ornstein_scotus_sentiment`
- `chae_semeval_stance`
- `haunss_papea_fgz_forms`
- `halterman_keith_cmp`
- `osnabruegge_cross_domain_topic`
- `mellon_bes_mii_2024`
- `wesleyan_creative_ads_2022`

## Policy for future additions

1. Prefer verbatim prompts from replication archives, supplemental material, or paper text.
2. If you need to modify a verbatim prompt for machine-readable output, document the exact change here.
3. If no prompt exists and you derive one from a codebook or paper description, mark it as derived here and note the source materials.
