#!/usr/bin/env python3
"""Count every prompt of a run with each model's own tokenizer, without a GPU.

hive_vllm_benchmark.py fails a whole task when one prompt exceeds
max_model_len - max_tokens - prompt_margin_tokens. Long documents were capped
at 10,000 o200k tokens, but other tokenizers count Czech, Hebrew or Estonian
text very differently, so this checks the real budget per model before any GPU
time is spent. It uses the runner's own build_conversations and
chat_token_count, and Hugging Face tokenizers from the local HF cache
(HF_HOME, offline). For models the runner loads with tokenizer_mode=mistral the
Hugging Face tokenizer is used instead, so their counts are approximate.

Run in a short CPU job on Hive, e.g.
  python3 code/preflight_prompt_budget.py --config experiments/task_ext_20261114/vllm026.yaml \
      --out output/sidecar/task_ext_20261114/preflight_vllm026.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from hive_vllm_benchmark import build_conversations, chat_token_count  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", help="model keys (default: all in the config)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    cfg = yaml.safe_load((REPO / args.config).read_text())
    budget = (int(cfg["runtime"]["max_model_len"]) - int(cfg["generation"]["max_tokens"])
              - int(cfg["generation"]["prompt_margin_tokens"]))
    tasks = load_task_definitions(tasks_dir=REPO / cfg["task_scope"]["tasks_dir"])
    convs = {}
    for t in tasks:
        convs[t["name"]] = build_conversations(Path(t["prompt_path"]).read_text(), t["loader"]())
    report = {"config": args.config, "budget": budget, "models": {}}
    for key in args.models or list(cfg["models"]):
        m = cfg["models"][key]
        kwargs = m.get("chat_template_kwargs") or {"enable_thinking": False}
        try:
            tok = AutoTokenizer.from_pretrained(m["model_id"], revision=m["revision"],
                                                trust_remote_code=bool(m.get("reviewed_remote_code")))
        except Exception as exc:  # report, do not stop the other models
            report["models"][key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            print(key, "TOKENIZER ERROR", exc, flush=True)
            continue
        per_task, over, near, worst = {}, [], 0, 0
        for name, cs in convs.items():
            lens = [chat_token_count(tok, c, kwargs) for c in cs]
            per_task[name] = max(lens)
            worst = max(worst, max(lens))
            near += sum(1 for x in lens if x > 0.9 * budget)
            over += [(name, i, x) for i, x in enumerate(lens) if x > budget]
        report["models"][key] = {"max_prompt_tokens": worst, "n_over_budget": len(over),
                                 "over_budget_first": over[:20], "n_above_90pct": near,
                                 "approximate": m.get("llm_kwargs", {}).get("tokenizer_mode") == "mistral",
                                 "max_by_task": per_task}
        print(f"{key}: max {worst} of {budget}, over {len(over)}, above 90% {near}", flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
