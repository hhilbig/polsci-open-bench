# Benchmark Candidates Inventory

Structured inventory of political-science text classification tasks eligible for local-LLM vs cloud-API benchmarking.

## Files

- **`candidates.yaml`** — source of truth. One entry per paper/task. Hand-maintained.
- **`candidates.md`** — auto-generated human-readable summary (regenerate with the script below).
- **`README.md`** — this file.

## Schema

See the header comment in `candidates.yaml`. Key fields for triage:

| Field | Values | Used for |
|---|---|---|
| `has_original_llm_prompt` | YES / NO / PARTIAL / NEEDS_VERIFICATION | Apples-to-apples with paper's own LLM run? |
| `benchmark_status` | already_run / ready / blocked / partial / deprioritized | Pipeline stage |
| `suitability_score` | HIGH / MEDIUM / LOW | Priority for next round |

## What "has_original_llm_prompt" means

- **YES**: The paper's original LLM prompt is verbatim available in the replication archive, paper SI, or paper appendix. We can use the paper's exact prompt for apples-to-apples comparison.
- **PARTIAL**: Some subtasks have prompts public, others don't.
- **NEEDS_VERIFICATION**: Paper likely has the prompt somewhere but we haven't verified the exact location yet.
- **NO**: Paper didn't use an LLM (used BERT / traditional ML / human coders only). We'd derive a prompt from the codebook — still usable, but not apples-to-apples.

## Triage workflow

1. Filter to `benchmark_status: ready` + `suitability_score: HIGH`.
2. Among those, prefer `has_original_llm_prompt: YES`.
3. Verify `replication_url` still live.
4. Check `accessibility_notes` for data-access gotchas (tweet rehydration, IRB, encryption).

## To regenerate `candidates.md` from YAML

```bash
cd /Users/hanno/Desktop/state_adaptation/output/benchmark_inventory
python3 generate_md.py
```

(Script to be added.)

## Currently run in benchmarks

| Round | Task IDs | Date | Models | N |
|---|---|---|---|---|
| 1 | `state_adaptation_bills` | 2026-04-22 | 5 (Gemma 4, Qwen 2.5/3.5/3.6, gpt-5.4-nano) | 50 |
| 2 | `state_adaptation_bills`, `gilardi_relevance`, `ballard_incivility`, `gilardi_stance` | 2026-04-22 | 6 (+ Qwen3-14B, Qwen3-30B-A3B, Mistral Small 24B, + gpt-5.4-mini late) | 50 each |

## Next additions (pending agent reports)

Three agents are currently surveying:
1. Recent (2023–2026) PolSci journals for new LLM-classification papers with prompts.
2. Verifying prompt accessibility for candidates already identified.
3. Sub-field-specific hunts (comparative, IR, public opinion, legal, communication).

Expected additions: 20–30 more entries.
