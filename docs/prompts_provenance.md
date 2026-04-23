# Prompt provenance

For each task, this records how the prompt in `prompts/` was obtained and what (if any) modifications were made.

| Task | Prompt file | Source | Modifications |
|---|---|---|---|
| `state_adaptation` | `prompts/state_adaptation.txt` | Production few-shot prompt tuned on this corpus (25 examples, multiple iterations documented in the upstream `state_adaptation` repo's CLAUDE.md). Not from a published paper — from the user's own working pipeline. | None. |
| `gilardi_relevance` | `prompts/gilardi_relevance.txt` | **Verbatim** from Gilardi, Alizadeh & Kubli (2023) PNAS replication: `dataverse_repo/src/03-01-chatgpt-Zeroshot-Task-template.py` lines 37 / 81 (same prompt duplicated across two calls). The last sentence instructs "just label 'relevant' or 'irrelevant' without any more explanation" — we replaced this with a JSON-output instruction for machine-readable parsing. | Replaced final single-sentence "just label 'relevant' or 'irrelevant'" with "Return JSON only, with a single field `relevant` equal to 0 or 1." |
| `gilardi_stance` | `prompts/gilardi_stance.txt` | **Derived**. Gilardi et al.'s stance prompt is in PNAS SI §S1 but is paywalled; we have not yet fetched it. The current prompt mirrors the structure of Gilardi's published relevance prompt (context paragraph + per-class definitions + JSON output instruction) and covers Section 230, content-moderation practice, and stance categorization. | Full derivation, documented in the prompt's footnote. **Should be replaced with verbatim SI text once retrieved.** |
| `ballard_incivility` | `prompts/ballard_incivility.txt` | **Derived**. Ballard (2022, LSQ) used fine-tuned BERT; no LLM prompt was published. The prompt draws on Ballard's paper description of uncivil rhetoric + standard political-incivility literature (Brooks & Geer 2007; Mutz 2015). | Full derivation. Not apples-to-apples with the paper. |
| `ornstein_scotus_sentiment` | `prompts/ornstein_scotus_sentiment.txt` | **Derived**. Ornstein, Blasingame & Truscott (2025, PSRM) used few-shot sentiment prompts in the `promptr` R package. The package uses example-based prompting with 12 seed examples. We did not replicate the few-shot structure — the current prompt is a zero-shot system-message version covering the same SCOTUS-sentiment task with three-class labels (Positive/Negative/Neutral). | Zero-shot derivation from promptr's codebook. Not matching the paper's few-shot format. |
| `halterman_ccc_protest` | `prompts/halterman_ccc_protest.txt` | **Verbatim** codebook from Halterman & Keith (2025, PA) replication archive: `data/codebooks/ccc_codebook_new_format.txt`. Full 8-class CCC taxonomy definitions. | Appended a strict JSON-output instruction at the end to supersede the codebook's "write just the label" instruction. This change was necessary because some models (notably Qwen3-30B-A3B) followed the codebook literally and emitted bare labels, bypassing our JSON parser. A case-insensitive bare-label fallback parser is also implemented (`code/benchmark.py:parse_content`) so either format works. |
| `chae_semeval_stance` | `prompts/chae_semeval_stance.txt` | **Derived**. Chae & Davidson (2025, SMR) used prompts embedded in their notebooks but not in a standalone form. The SemEval-2016 Task 6 task (Trump / Clinton stance) is the underlying data. Our prompt is derived from the SemEval codebook's three-class stance definition (FAVOR/AGAINST/NONE) and covers the "negative sentiment toward someone else doesn't mean the target is opposed" edge case that SemEval's guidelines call out. | Full derivation. |

## Rating summary

Prompts we can reasonably call "apples-to-apples" with the original paper:
- `gilardi_relevance` — verbatim structure + intent; only the output-format tail differs.
- `halterman_ccc_protest` — verbatim codebook + added JSON wrapper.

Prompts we call "derived" (not apples-to-apples):
- `gilardi_stance` (should upgrade when SI retrieved)
- `ballard_incivility` (paper didn't use an LLM)
- `ornstein_scotus_sentiment` (paper used few-shot, we did zero-shot)
- `chae_semeval_stance` (no standalone prompt in the paper)

`state_adaptation` is "production" — tuned for this corpus, not drawn from a published paper.

## Policy for future additions

When adding a new task to the benchmark:

1. Prefer **verbatim** prompts from the paper (replication archive > SI > paper body). If modifications are needed (e.g., JSON output instruction), document them here.
2. When the paper didn't use an LLM, mark the prompt as **derived** and describe its source (codebook, paper prose, established literature).
3. Note in the report's caveats which tasks use derived vs. verbatim prompts — cross-task F1 comparisons should weight derived-prompt tasks with a grain of salt.
