#!/usr/bin/env python3
"""Write the Hive configs for the 2026-11-14 open-model run on new and v2 tasks.

Scope: every manifest in tasks_ext_v2_20261114/ (symlinks to the 23 tasks_ext
and 24 tasks_v2 manifests), each task's full loader pool (500 items, fewer for
QEvasion, 1,000 for yan_bernhard_offensive). Any release draw is a subset.

Each model's entry is copied unchanged from the config of its panel run, so
model id, revision and kwargs match the release. Two settings differ from most
panel runs and are the same for every model here:
  - max_model_len 24576 (long documents: with 16384 the Gemma tokenizers put
    one Supreme Court opinion over the input budget and the Qwen3.x ones came
    within one token of it; 18 of the panel runs used 8192),
  - one runtime lock per group: vLLM 0.26.0 matched-v2 (sha 4f46101f) for the
    single-GPU models, vLLM 0.32.0 for MiMo. The three models from the
    2026-08-04 bakeoff ran on an older lock that no longer exists on Hive, so
    they use matched-v2 here, as the October task_ext reruns did.
Not covered: qwen3_8_flash_next_fp8 (its apptainer runner is not in the repo),
nemotron3_5_super_vl_bf16 (panel remainder still running), intern_s2.

Usage: python3 experiments/task_ext_20261114/make_configs.py COMMIT
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "code"))
import task_registry  # noqa: E402

TASKS_DIR = "tasks_ext_v2_20261114"

SOURCES = {
    "deepseek_r1_distill_qwen_32b_bf16": "output/sidecar/refresh_20260910_release/backfills/deepseek_r1_distill_qwen_32b_bf16/remainder.yaml",
    "gemma3_27b_it_fp8_dynamic": "output/sidecar/refresh_20260910_release/backfills/gemma3_27b_it_fp8_dynamic/remainder.yaml",
    "gemma4_31b_it_qat_w4a16": "experiments/hive_model_bakeoff_20260804.yaml",
    "glm4_7_flash": "experiments/hive_model_bakeoff_extension_cuda130_20260805.yaml",
    "llama3_1_70b_instruct_fp8_dynamic_full34": "experiments/frontier_full34_llama31_holdout_20260821.yaml",
    "llama3_3_70b_instruct_fp8_dynamic": "output/sidecar/refresh_20260910_release/backfills/llama3_3_70b_instruct_fp8_dynamic/remainder.yaml",
    "mistral_small_3_1_24b_bf16": "output/sidecar/refresh_20260910_release/backfills/mistral_small_3_1_24b_bf16/remainder.yaml",
    "mistral_small_4_119b_nvfp4": "experiments/hive_model_bakeoff_extension_cuda130_20260805.yaml",
    "qwen1_5_32b_chat": "output/sidecar/refresh_20260910_release/backfills/qwen1_5_32b_chat/remainder.yaml",
    "qwen2_5_32b_instruct_bf16": "output/sidecar/refresh_20260910_release/backfills/qwen2_5_32b_instruct_bf16/remainder.yaml",
    "qwen2_5_72b_instruct_fp8_dynamic": "output/sidecar/refresh_20260910_release/backfills/qwen2_5_72b_instruct_fp8_dynamic/remainder.yaml",
    "qwen3_30b_a3b_bf16": "output/sidecar/refresh_20260910_release/backfills/qwen3_30b_a3b_bf16/remainder.yaml",
    "qwen3_30b_a3b_instruct_2507_fp8": "experiments/hive_model_bakeoff_20260804.yaml",
    "qwen3_32b_bf16": "output/sidecar/refresh_20260910_release/backfills/qwen3_32b_bf16/remainder.yaml",
    "qwen3_5_35b_a3b_fp8": "output/sidecar/refresh_20260910_release/backfills/qwen3_5_35b_a3b_fp8/remainder.yaml",
    "qwen3_6_27b_fp8": "experiments/hive_model_bakeoff_extension_cuda130_20260805.yaml",
    "qwen3_6_35b_a3b_fp8": "experiments/hive_model_bakeoff_20260804.yaml",
    "qwen3_8_27b_fp8": "experiments/refresh_hive_20260914_remainder.yaml",
    "qwen3_next_80b_a3b_fp8": "output/sidecar/refresh_20260910_release/backfills/qwen3_next_80b_a3b_fp8/remainder.yaml",
    "qwen3_5_9b_bf16": "experiments/refresh_november_20261114/qwen3_5_9b_bf16/remainder.yaml",
    "gemma3_12b_it_fp8_dynamic": "experiments/refresh_november_20261114/gemma3_12b_it_fp8_dynamic/remainder.yaml",
    "mimo_v2_6_flash_mopd_fp8": "experiments/refresh_october_20261029/mimo_v2_6_flash_mopd_fp8/remainder.yaml",
}
MULTI_GPU = {"mimo_v2_6_flash_mopd_fp8"}

CUDA_COMPONENTS = {
    "nvidia-cuda-cccl": "13.0.85", "nvidia-cuda-crt": "13.0.88", "nvidia-cuda-nvcc": "13.0.88",
    "nvidia-cuda-nvrtc": "13.0.88", "nvidia-cuda-runtime": "13.0.96", "nvidia-nvvm": "13.0.88",
}
RUNTIME_026 = {
    "vllm_version": "0.26.0", "python_version": "3.12",
    "runtime_lock_sha256": "4f46101fa493e38b312af034b8d740c612bd0001bab4cb0f624f4a9e1c49f5b3",
    "runtime_lock_path": "/nfs/hive/scratch/hhilbig/polsci-bakeoff-vllm-0.26.0-py312-cuda130-matched-v2.freeze.txt",
    "max_model_len": 24576, "gpu_memory_utilization": 0.9,
    "cuda_toolkit_meta_version": "13.0.2", "cuda_component_versions": CUDA_COMPONENTS,
}
GENERATION = {
    "temperature": 0.0, "max_tokens": 256, "seed": 20261114, "enable_thinking": False,
    "structured_outputs": "json_schema", "prompt_margin_tokens": 128,
}


def main(commit: str) -> None:
    tasks = task_registry.load_task_definitions(tasks_dir=REPO / TASKS_DIR)
    n_items = sum(len(t["loader"]()) for t in tasks)
    scope = {"tasks_dir": TASKS_DIR, "expected_tasks": len(tasks), "expected_items": n_items,
             "sampling_note": "Full loader pool of every tasks_ext and tasks_v2 manifest; release draws are subsets."}
    models_026, models_032 = {}, {}
    for key, src in SOURCES.items():
        entry = yaml.safe_load((REPO / src).read_text())["models"][key]
        (models_032 if key in MULTI_GPU else models_026)[key] = entry
    gate = {"baseline_model_key": "qwen3_6_27b_fp8", "minimum_mean_f1_gain": 0.01,
            "equivalence_tolerance": 0.005, "maximum_parse_error_increase": 1.0,
            "material_throughput_ratio": 1.25}
    cfg = {"version": 1, "run_id": "task_ext_20261114_vllm026", "benchmark_commit": commit,
           "task_scope": scope, "runtime": RUNTIME_026, "generation": GENERATION,
           "promotion_gate": gate, "models": models_026}
    (OUT / "vllm026.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    mimo = yaml.safe_load((REPO / SOURCES["mimo_v2_6_flash_mopd_fp8"]).read_text())
    rt = dict(mimo["runtime"])
    rt["max_model_len"] = 24576
    cfg32 = {"version": 1, "run_id": "task_ext_20261114_vllm032", "benchmark_commit": commit,
             "task_scope": scope, "runtime": rt, "generation": GENERATION,
             "promotion_gate": dict(gate, baseline_model_key="mimo_v2_6_flash_mopd_fp8"),
             "models": models_032}
    for k in ("hardware",):
        if k in mimo:
            cfg32[k] = mimo[k]
    (OUT / "vllm032.yaml").write_text(yaml.safe_dump(cfg32, sort_keys=False))
    print(f"{len(tasks)} tasks, {n_items} items; {len(models_026)} + {len(models_032)} models")


if __name__ == "__main__":
    main(sys.argv[1])
