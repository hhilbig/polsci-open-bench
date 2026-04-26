# Prompt provenance

For each task, this records how the prompt in `prompts/` was obtained and what (if any) modifications were made.

| Task | Prompt file | Source | Modifications |
|---|---|---|---|
| `state_adaptation` | `prompts/state_adaptation.txt` | Production few-shot prompt tuned on this corpus (25 examples, multiple iterations documented in the upstream `state_adaptation` repo's CLAUDE.md). Not from a published paper — from the user's own working pipeline. | None. |
| `gilardi_relevance` | `prompts/gilardi_relevance.txt` | **Verbatim** from Gilardi, Alizadeh & Kubli (2023) PNAS replication: `dataverse_repo/src/03-01-chatgpt-Zeroshot-Task-template.py` lines 37 / 81 (same prompt duplicated across two calls). The last sentence instructs "just label 'relevant' or 'irrelevant' without any more explanation" — we replaced this with a JSON-output instruction for machine-readable parsing. | Replaced final single-sentence "just label 'relevant' or 'irrelevant'" with "Return JSON only, with a single field `relevant` equal to 0 or 1." |
| `gilardi_stance` | `prompts/gilardi_stance.txt` | **Verbatim** from Gilardi, Alizadeh & Kubli (2023) PNAS SI §S1 Annotation Codebooks: "Background on content moderation" paragraph + "Task 6: Stance Detection" body. SI PDF archived locally at `docs/pnas.2305016120.sapp.pdf`. | Two documented modifications: (1) The SI codebook calls the three labels "positive / negative / neutral"; this prompt substitutes "pro / contra / neutral" to match the `gt_stance` values used in the replication archive's data + Python task template. (2) Appended a JSON-output instruction for machine-readable parsing (the SI is a human-coder codebook). |
| `ballard_incivility` | `prompts/ballard_incivility.txt` | **Derived**. Ballard (2022, LSQ) used fine-tuned BERT; no LLM prompt was published. The prompt draws on Ballard's paper description of uncivil rhetoric + standard political-incivility literature (Brooks & Geer 2007; Mutz 2015). | Full derivation. Not apples-to-apples with the paper. |
| `ornstein_scotus_sentiment` | `prompts/ornstein_scotus_sentiment.txt` | **Derived**. Ornstein, Blasingame & Truscott (2025, PSRM) used few-shot sentiment prompts in the `promptr` R package. The package uses example-based prompting with 12 seed examples. We did not replicate the few-shot structure — the current prompt is a zero-shot system-message version covering the same SCOTUS-sentiment task with three-class labels (Positive/Negative/Neutral). | Zero-shot derivation from promptr's codebook. Not matching the paper's few-shot format. |
| `halterman_ccc_protest` | `prompts/halterman_ccc_protest.txt` | **Verbatim (subset)** from Halterman & Keith (2025, PA) replication archive: `data/codebooks/ccc_codebook_new_format.txt`. Restricted to the 4 classes present in H&K's `ccc_test.tab` labeled split (PROTEST, RALLY, DEMONSTRATION, MARCH). The original codebook also defined CARAVAN, BICYCLE_RIDE, DIRECT_ACTION, COUNTER_PROTEST; we drop those because H&K's labeled split has no gold items for them. | Subsetted label space + appended JSON-output instruction (same JSON wrapper rationale as before: some models emitted bare labels, which the case-insensitive fallback parser also catches). 2026-04-24: upgraded from the previous 50-row 8-class CSV to H&K's 1,010-row 4-class test split; previous N=50 predictions archived at `output/archive/predictions_halterman_ccc_v1_n50.csv`. |
| `halterman_keith_bfrs` | `prompts/halterman_keith_bfrs.txt` | **Verbatim** from Halterman & Keith (2025, PA) replication archive: `data/codebooks/bfrs_codebook_new_format.txt`. Full 12-class Pakistani political-violence taxonomy with definitions, clarifications, and positive/negative examples per class. | Appended JSON-output instruction at the end. No other changes. |
| `halterman_keith_cmp` | `prompts/halterman_keith_cmp.txt` | **Derived** from the CMP (Manifesto Project) "Aggregation1" major-domain structure used in H&K's `data/train_dev_test/manifestos_test.tab` `label_category` column. H&K ship a 142-label fine codebook (`data/codebooks/manifesto_codebook_new_hand.txt`); we target the 7 `Aggregation1` domains instead because their `label_category` column is the aggregation that produces the gold label. | Custom 7-domain prompt drawn from CMP documentation, not verbatim from H&K's fine codebook. Listing only the 7 Aggregation1 values (External Relations, Freedom and Democracy, Political System, Economy, Welfare and Quality of Life, Fabric of Society, Social Groups) with concise definitions + JSON-output instruction. |
| `wesleyan_creative_ads_2022` | `prompts/wesleyan_creative_ads_2022.txt` | **Derived**. Zhang et al. (2025, *Sci Data*) / Wesleyan Media Project CREATIVE ad data. Gold labels are the WMP project's hand-coded TONE1 field (tone of the ad toward the first mentioned candidate). No LLM prompt published by WMP; ad tone classification is traditionally done by trained human coders using a Qualtrics survey (the raw fields are in `fb_2022_train.xlsx`). Our prompt paraphrases the WMP tone-coding construct (promote / attack / unclear) into a self-contained LLM prompt. | Derived from the WMP tone-coding construct with three compact labels (promote/attack/unclear). Ad text is capped at 4000 chars to keep Ollama prefill tractable; >90% of ads fit without truncation. |
| `chae_semeval_stance` | `prompts/chae_semeval_stance.txt` | **Derived**. Chae & Davidson (2025, SMR) used prompts embedded in their notebooks but not in a standalone form. The SemEval-2016 Task 6 task (Trump / Clinton stance) is the underlying data. Our prompt is derived from the SemEval codebook's three-class stance definition (FAVOR/AGAINST/NONE) and covers the "negative sentiment toward someone else doesn't mean the target is opposed" edge case that SemEval's guidelines call out. | Full derivation. |

## Rating summary

Prompts we can reasonably call "apples-to-apples" with the original paper:
- `gilardi_relevance` — verbatim structure + intent; only the output-format tail differs.
- `gilardi_stance` — verbatim SI §S1 Task 6 codebook; labels renamed pro/contra/neutral to match the replication archive's data + JSON wrapper appended.
- `halterman_ccc_protest` — verbatim codebook entries for 4 classes (subset of H&K's 8-class codebook; the other 4 classes have no gold items in H&K's test split) + added JSON wrapper.
- `halterman_keith_bfrs` — verbatim codebook from H&K replication archive + added JSON wrapper.

Prompts we call "derived" (not apples-to-apples):
- `ballard_incivility` (paper didn't use an LLM)
- `ornstein_scotus_sentiment` (paper used few-shot, we did zero-shot)
- `chae_semeval_stance` (no standalone prompt in the paper)
- `halterman_keith_cmp` (H&K ship a 142-label fine codebook; we target the 7 Aggregation1 domains the gold labels use, with a custom 7-domain prompt)
- `wesleyan_creative_ads_2022` (WMP uses a Qualtrics coder survey, not an LLM prompt; the tone construct is paraphrased into a self-contained LLM prompt)

`state_adaptation` is "production" — tuned for this corpus, not drawn from a published paper.

## Policy for future additions

When adding a new task to the benchmark:

1. Prefer **verbatim** prompts from the paper (replication archive > SI > paper body). If modifications are needed (e.g., JSON output instruction), document them here.
2. When the paper didn't use an LLM, mark the prompt as **derived** and describe its source (codebook, paper prose, established literature).
3. Note in the report's caveats which tasks use derived vs. verbatim prompts — cross-task F1 comparisons should weight derived-prompt tasks with a grain of salt.
