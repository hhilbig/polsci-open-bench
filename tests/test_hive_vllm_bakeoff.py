import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from hive_vllm_benchmark import (  # noqa: E402
    BakeoffError,
    build_conversations,
    assert_compact_hardware,
    assert_compact_offline_provenance,
    assert_compact_prediction_coverage,
    assert_model_launch_eligible,
    assert_runtime_versions,
    compact_failure_category,
    evaluate_compact_pilot,
    evaluate_fit_parser_pilot,
    file_sha256,
    harmony_generation_error_to_malformed,
    load_config,
    load_task_items,
    malformed_rate_audit,
    next_evidence_stage,
    prediction_rows,
    preflight_prompt_lengths,
    require_sidecar_output_dir,
    refuse_uncommitted_checkpoint,
    responses_output_text,
    responses_request_payload,
    resolved_chat_template_kwargs,
    resume_identity_matches,
    run,
    selected_tasks,
    task_metrics_frame,
    task_checkpoint_complete,
    task_done_path,
    validate_scope,
    write_csv_atomic,
    write_json_atomic,
)
from panel_manifest import (  # noqa: E402
    benchmark_requests,
    load_panel_manifest,
    split_pilot_remainder,
)
from summarize_hive_bakeoff import (  # noqa: E402
    markdown_table,
    normalized_prediction_frame,
    ordered_blinded_predictions,
    random_blinding_aliases,
    rare_class_recall,
    render_report,
    validate_comparable_runtime,
    validate_task_generation_seconds,
)


CONFIG = REPO / "experiments" / "hive_model_bakeoff_20260804.yaml"
WATCHLIST_CONFIG = (
    REPO / "experiments" / "hive_model_bakeoff_extension_cuda130_20260805.yaml"
)
FRONTIER_PANEL = REPO / "experiments" / "frontier_panel_18.yaml"
FRONTIER_PENDING_CONFIG = (
    REPO / "experiments" / "frontier_hive_pending_20260820.yaml"
)
COMPACT_PANEL = REPO / "experiments" / "frontier_compact_8.yaml"
BROAD_PANEL = REPO / "experiments" / "frontier_broad_18.yaml"
COMPACT_HIVE_CONFIG = (
    REPO / "experiments" / "frontier_compact8_hive_20260820.yaml"
)
TARGETED_ABLATION_CONFIG = (
    REPO / "experiments" / "frontier_broad18_targeted_ablation_20260821.yaml"
)
HISTORICAL_EXTENSION_CONFIG = (
    REPO / "experiments" / "frontier_broad18_historical_extension_20260823.yaml"
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        self.last_enable_thinking = enable_thinking
        length = sum(len(message["content"].split()) for message in conversation) + 3
        return list(range(length))


class FakeMappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": super().apply_chat_template(*args, **kwargs), "attention_mask": [1]}


class CapturingTokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        self.last_kwargs = kwargs
        return [1, 2, 3]


class HiveBakeoffConfigTests(unittest.TestCase):
    def test_blackwell_wrapper_preflights_flashinfer_cublas_linkage(self):
        wrapper = (REPO / "experiments/hive_model_bakeoff_20260804.sbatch").read_text()
        self.assertIn("libcublas.so", wrapper)
        self.assertIn("libcublasLt.so", wrapper)
        self.assertIn("-lcublas -lcublasLt", wrapper)

    def test_broad18_historical_extension_is_exact_and_chronological(self):
        config = load_config(HISTORICAL_EXTENSION_CONFIG)
        self.assertEqual(config["execution"]["panel_id"], "frontier_broad_18")
        self.assertEqual(config["task_scope"]["expected_tasks"], 18)
        self.assertEqual(config["execution"]["pilot_items"], 16)
        self.assertEqual(config["execution"]["remainder_items"], 3584)
        self.assertEqual(
            {key: model["revision"] for key, model in config["models"].items()},
            {
                "qwen1_5_32b_chat": "0997b012af6ddd5465d40465a8415535b2f06cfc",
                "qwen1_5_72b_chat_awq": "6909dcb756186bbc97bf009ab958d640f9defd3e",
                "qwen2_72b_instruct_awq": "cdbb90d7a1fda3a1f6d7d280b7c92f39e1129d3f",
                "llama3_70b_instruct_fp8": "87b86c042b8d131484c691754b3e49202c19175a",
                "llama3_1_nemotron_70b_fp8_dynamic": "855e3b87e62e74773d3618ed0a740f2e359a2a40",
            },
        )
        dates = [m["artifact_publication_date"] for m in config["models"].values()]
        self.assertEqual(dates, sorted(dates))
        anchor = config["models"]["qwen1_5_32b_chat"]
        self.assertEqual(anchor["role"], "matched_size_historical_anchor")
        self.assertEqual(anchor["quantization"], "BF16")

    def test_frozen_config_and_full_scope(self):
        config = load_config(CONFIG)
        self.assertEqual(len(config["models"]), 4)
        self.assertEqual(config["generation"]["temperature"], 0.0)
        self.assertFalse(config["generation"]["enable_thinking"])
        self.assertEqual(
            config["promotion_gate"]["baseline_model_key"],
            "qwen3_30b_a3b_instruct_2507_fp8",
        )
        self.assertEqual(
            config["benchmark_commit"], "3c7ad0756d447b1d57ed4daf26bbb85f5f296042"
        )
        expected_revisions = {
            "qwen3_30b_a3b_instruct_2507_fp8": "5a5a776300a41aaa681dd7ff0106608ef2bc90db",
            "qwen3_6_35b_a3b_fp8": "95a723d08a9490559dae23d0cff1d9466213d989",
            "qwen3_6_27b_fp8": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
            "gemma4_31b_it_qat_w4a16": "52f3f65bc7a02d555763bc923bd1d9094898219d",
        }
        self.assertEqual(
            {key: model["revision"] for key, model in config["models"].items()},
            expected_revisions,
        )

        tasks = selected_tasks(config)
        scope = validate_scope(config, tasks)
        self.assertEqual(scope["total_tasks"], 34)
        self.assertEqual(scope["total_items"], 16425)

    def test_watchlist_config_pins_models_and_model_specific_runtime(self):
        config = load_config(WATCHLIST_CONFIG)
        self.assertEqual(
            set(config["models"]),
            {
                "qwen3_6_27b_fp8",
                "glm4_7_flash",
                "mistral_small_4_119b_nvfp4",
            },
        )
        self.assertEqual(
            config["promotion_gate"]["baseline_model_key"],
            "qwen3_6_27b_fp8",
        )
        self.assertEqual(
            config["models"]["glm4_7_flash"]["revision"],
            "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        )
        self.assertEqual(
            config["models"]["glm4_7_flash"]["llm_kwargs"]["moe_backend"],
            "triton",
        )
        mistral = config["models"]["mistral_small_4_119b_nvfp4"]
        self.assertEqual(
            mistral["revision"],
            "b1a9048590131d38491bd23a7c9f6ed0962f0358",
        )
        self.assertEqual(mistral["llm_kwargs"]["tensor_parallel_size"], 1)
        self.assertEqual(mistral["llm_kwargs"]["tokenizer_mode"], "mistral")
        self.assertEqual(mistral["llm_kwargs"]["attention_backend"], "TRITON_MLA")
        self.assertEqual(mistral["llm_kwargs"]["moe_backend"], "cutlass")
        self.assertEqual(mistral["llm_kwargs"]["linear_backend"], "cutlass")
        self.assertEqual(
            resolved_chat_template_kwargs(mistral),
            {"enable_thinking": False, "reasoning_effort": "none"},
        )
        self.assertEqual(config["runtime"]["cuda_toolkit_meta_version"], "13.0.2")
        self.assertEqual(
            config["runtime"]["runtime_lock_sha256"],
            "4f46101fa493e38b312af034b8d740c612bd0001bab4cb0f624f4a9e1c49f5b3",
        )
        self.assertEqual(
            config["runtime"]["cuda_component_versions"]["nvidia-cuda-nvcc"],
            "13.0.88",
        )

    def test_frontier_pending_config_has_five_chronological_exact_pins(self):
        config = load_config(FRONTIER_PENDING_CONFIG)
        expected = {
            "gpt_oss_20b_mxfp4": "6cee5e81ee83917806bbde320786a8fb61efebee",
            "gpt_oss_120b_mxfp4": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "qwen3_next_80b_a3b_fp8": "c5f5f263bdd5cc134092897864e8905d8fe7b928",
            "ministral_3_14b_bf16": "29439f81c2be264d8d393273f99e7db9c0961120",
            "qwen3_5_35b_a3b_fp8": "9d1823d2dee688a6b25e77009dc727688c44936e",
        }
        self.assertEqual(list(config["models"]), list(expected))
        self.assertEqual(
            {key: value["revision"] for key, value in config["models"].items()},
            expected,
        )
        self.assertEqual(
            resolved_chat_template_kwargs(config["models"]["gpt_oss_20b_mxfp4"])[
                "reasoning_effort"
            ],
            "medium",
        )
        with self.assertRaisesRegex(BakeoffError, "ineligible for launch"):
            assert_model_launch_eligible(
                "ministral_3_14b_bf16",
                config["models"]["ministral_3_14b_bf16"],
            )
        self.assertEqual(
            config["models"]["qwen3_next_80b_a3b_fp8"]["llm_kwargs"]["max_num_seqs"],
            16,
        )

    def test_compact_hive_config_has_seventeen_chronological_exact_pins(self):
        config = load_config(COMPACT_HIVE_CONFIG)
        expected = {
            "llama3_1_70b_instruct_fp8_dynamic": "019d944e8e566c43939ea83775a27197ebb9b559",
            "qwen2_5_32b_instruct_bf16": "70e8dfb9ad18a7d499f765fe206ff065ed8ca197",
            "qwen2_5_72b_instruct_fp8_dynamic": "4d9910ef10cf92b072dad8ce7c2a2929fac4fe0f",
            "llama3_3_70b_instruct_fp8_dynamic": "9069d043f499121b771d7df1c86cb414e66880c6",
            "deepseek_r1_distill_qwen_32b_bf16": "ca24ee48c0532a014ddea4c32437a6f4be981ab2",
            "deepseek_r1_distill_llama_70b_fp8_dynamic": "e5626ac0aad8ed4233206041bb041138dcca3f8d",
            "mistral_small_3_1_24b_bf16": "4b8dd8aae705887db5295fcbff4aedbb92d682eb",
            "qwen3_32b_bf16": "c36b7534bc7164e32375e869644e3f5fee30218f",
            "qwen3_30b_a3b_bf16": "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
            "gemma3_27b_it_fp8_dynamic": "306078afe860ef87821018f28078ecf762f31455",
            "gpt_oss_120b_mxfp4": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "qwen3_next_80b_a3b_fp8": "c5f5f263bdd5cc134092897864e8905d8fe7b928",
            "glm4_7_flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
            "qwen3_5_35b_a3b_fp8": "9d1823d2dee688a6b25e77009dc727688c44936e",
            "mistral_small_4_119b_nvfp4": "b1a9048590131d38491bd23a7c9f6ed0962f0358",
            "qwen3_6_27b_fp8": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
            "gemma4_31b_it_qat_w4a16": "52f3f65bc7a02d555763bc923bd1d9094898219d",
        }
        self.assertEqual(list(config["models"]), list(expected))
        self.assertEqual(
            {key: model["revision"] for key, model in config["models"].items()},
            expected,
        )
        self.assertNotIn("promotion_gate", config)
        self.assertEqual(config["execution"]["panel_id"], "frontier_compact_8")
        self.assertEqual(config["execution"]["pilot_items"], 16)
        self.assertEqual(config["execution"]["remainder_items"], 3984)
        self.assertEqual(config["execution"]["expected_models"], 17)
        self.assertEqual(
            config["execution"]["output_root"],
            "output/sidecar/frontier_2026/compact8/hive",
        )
        self.assertEqual(config["execution"]["required_gpu_gres"], "6000_blackwell:1")
        self.assertEqual(config["execution"]["partition_selection"], "live_sbatch_test")
        self.assertFalse(
            any("preferred_partition" in model for model in config["models"].values())
        )
        self.assertEqual(config["execution"]["expected_gpu_capacity_mib"], 97887)
        self.assertEqual(
            config["runtime"]["runtime_lock_sha256"],
            "4f46101fa493e38b312af034b8d740c612bd0001bab4cb0f624f4a9e1c49f5b3",
        )
        self.assertEqual(
            str(config["models"]["qwen3_30b_a3b_bf16"]["artifact_publication_date"]),
            "2025-04-30",
        )
        self.assertEqual(
            config["models"]["gpt_oss_120b_mxfp4"]["inference_interface"],
            "vllm_responses_harmony",
        )
        self.assertEqual(
            resolved_chat_template_kwargs(config["models"]["gpt_oss_120b_mxfp4"])[
                "reasoning_effort"
            ],
            "low",
        )

    def test_gptoss_responses_payload_constrains_only_the_final_channel(self):
        schema = {
            "type": "object",
            "properties": {"relevant": {"type": "boolean"}},
            "required": ["relevant"],
            "additionalProperties": False,
        }
        conversation = [
            {"role": "system", "content": "Classify."},
            {"role": "user", "content": "Text"},
        ]
        payload = responses_request_payload(
            model_id="openai/gpt-oss-120b",
            conversation=conversation,
            schema=schema,
            max_output_tokens=256,
            reasoning_effort="low",
        )
        self.assertEqual(payload["input"], conversation)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["text"]["format"]["schema"], schema)
        self.assertTrue(payload["text"]["format"]["strict"])
        response = {
            "output": [
                {"type": "reasoning", "content": [{"type": "text", "text": "ignore"}]},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"relevant":true}'}],
                },
            ]
        }
        self.assertEqual(responses_output_text(response), '{"relevant":true}')

    def test_only_invalid_generated_harmony_header_becomes_malformed(self):
        malformed = harmony_generation_error_to_malformed(
            'HarmonyError: unexpected tokens remaining in message header: Some("<|constrain|>")'
        )
        self.assertEqual(malformed["_benchmark_parse_error"], "harmony_invalid_message_header")
        self.assertEqual(malformed["_benchmark_token_usage_status"], "unavailable")
        unknown_channel = harmony_generation_error_to_malformed(
            '{"error":{"message":"Unknown channel: finalEntities",'
            '"type":"BadRequestError","param":null,"code":400}}'
        )
        self.assertEqual(
            unknown_channel["_benchmark_parse_error"], "harmony_unknown_generated_channel"
        )
        self.assertEqual(unknown_channel["output"], [])
        self.assertEqual(unknown_channel["_benchmark_token_usage_status"], "unavailable")
        self.assertIsNone(
            harmony_generation_error_to_malformed(
                "error downloading or loading vocab file"
            )
        )
        self.assertIsNone(harmony_generation_error_to_malformed(
            '{"error":{"message":"Unknown channel: finalEntities",'
            '"type":"ConfigurationError","code":400}}'
        ))

    def test_compact_manifest_partitions_pilot_and_remainder_without_overlap(self):
        panel = load_panel_manifest(COMPACT_PANEL, tasks_dir=REPO / "tasks")
        requests = benchmark_requests(panel)
        pilot, remainder = split_pilot_remainder(panel)
        keys = [(request.task_name, request.item_id) for request in requests]
        pilot_keys = {(request.task_name, request.item_id) for request in pilot}
        remainder_keys = {(request.task_name, request.item_id) for request in remainder}
        self.assertEqual(panel.panel_id, "frontier_compact_8")
        self.assertEqual((panel.expected_tasks, panel.expected_items), (8, 4000))
        self.assertEqual((len(pilot), len(remainder)), (16, 3984))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(pilot_keys & remainder_keys)
        self.assertEqual(pilot_keys | remainder_keys, set(keys))

        rows = []
        for request in requests:
            rows.append(
                {
                    "task": request.task_name,
                    "item_id": request.item_id,
                    "model": "candidate",
                    "panel_sha256": panel.panel_sha256,
                    "panel_task_fingerprint": panel.task_fingerprints[
                        request.task_name
                    ],
                }
            )
        predictions = pd.DataFrame(rows)
        assert_compact_prediction_coverage(
            predictions,
            panel,
            model_key="candidate",
        )
        duplicated = pd.concat(
            [predictions.iloc[:-1], predictions.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(BakeoffError, "duplicate|manifest order"):
            assert_compact_prediction_coverage(
                duplicated,
                panel,
                model_key="candidate",
            )

    def test_compact_hardware_and_offline_snapshot_gates(self):
        execution = load_config(COMPACT_HIVE_CONFIG)["execution"]
        gpu = {
            "gpus": [
                {
                    "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    "memory_total_mib": 97887,
                }
            ]
        }
        assert_compact_hardware(execution, gpu)
        wrong_capacity = json.loads(json.dumps(gpu))
        wrong_capacity["gpus"][0]["memory_total_mib"] = 97888
        with self.assertRaisesRegex(BakeoffError, "capacity mismatch"):
            assert_compact_hardware(execution, wrong_capacity)

        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / revision
            snapshot.mkdir()
            resolved = assert_compact_offline_provenance(
                execution,
                {"revision": revision},
                environ={
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "BAKEOFF_MODEL_SNAPSHOT_PATH": str(snapshot),
                },
            )
            self.assertEqual(resolved, snapshot.resolve())
            with self.assertRaisesRegex(BakeoffError, "HF_HUB_OFFLINE"):
                assert_compact_offline_provenance(
                    execution,
                    {"revision": revision},
                    environ={
                        "HF_HUB_OFFLINE": "0",
                        "TRANSFORMERS_OFFLINE": "1",
                        "BAKEOFF_MODEL_SNAPSHOT_PATH": str(snapshot),
                    },
                )

    def test_compact_malformed_gate_is_inclusive_at_five_percent(self):
        clean = pd.DataFrame({"parse_error": [None] * 20})
        at_boundary = clean.copy()
        at_boundary.loc[0, "parse_error"] = "bad JSON"
        above_boundary = at_boundary.copy()
        above_boundary.loc[1, "parse_error"] = "bad JSON"
        self.assertTrue(
            malformed_rate_audit(clean, max_malformed_rate=0.05)["passed"]
        )
        boundary = malformed_rate_audit(at_boundary, max_malformed_rate=0.05)
        self.assertTrue(boundary["passed"])
        self.assertEqual(boundary["malformed_rate"], 0.05)
        self.assertFalse(
            malformed_rate_audit(above_boundary, max_malformed_rate=0.05)["passed"]
        )

    def test_compact_failure_categories_distinguish_terminal_gates(self):
        self.assertEqual(compact_failure_category(RuntimeError("CUDA out of memory")), "oom")
        self.assertEqual(
            compact_failure_category(BakeoffError("snapshot revision mismatch")),
            "provenance",
        )
        self.assertEqual(
            compact_failure_category(BakeoffError("GPU capacity mismatch")),
            "hardware",
        )

    def test_pilot_gate_and_stage_transitions(self):
        metadata = {
            "status": "completed",
            "model_id": "provider/model",
            "revision": "a" * 40,
            "gpu_before_load": {
                "gpus": [
                    {
                        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "memory_total_mib": 97887,
                    }
                ]
            },
            "observed_peak_gpu_memory_used_mib": 97887,
        }
        predictions = pd.DataFrame(
            {"parse_error": [None] * 16, "item_id": [str(i) for i in range(16)]}
        )
        audit = evaluate_fit_parser_pilot(
            metadata,
            predictions,
            expected_model_id="provider/model",
            expected_revision="a" * 40,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["next_stage"], "panel18")
        self.assertEqual(
            next_evidence_stage(
                "panel18", gate_passed=True, promotion_triggered=True
            ),
            "full34",
        )
        self.assertEqual(
            next_evidence_stage(
                "panel18", gate_passed=True, promotion_triggered=False
            ),
            "terminal_screened",
        )

        over_capacity = dict(metadata)
        over_capacity["observed_peak_gpu_memory_used_mib"] = 97888
        audit = evaluate_fit_parser_pilot(
            over_capacity,
            predictions,
            expected_model_id="provider/model",
            expected_revision="a" * 40,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["next_stage"], "terminal_ineligible")

        malformed = predictions.copy()
        malformed.loc[0, "parse_error"] = "bad JSON"
        audit = evaluate_fit_parser_pilot(
            metadata,
            malformed,
            expected_model_id="provider/model",
            expected_revision="a" * 40,
        )
        self.assertFalse(audit["passed"])
        self.assertAlmostEqual(audit["malformed_rate"], 1 / 16)

        compact_audit = evaluate_compact_pilot(
            metadata,
            predictions,
            expected_model_id="provider/model",
            expected_revision="a" * 40,
        )
        self.assertTrue(compact_audit["passed"])
        self.assertEqual(compact_audit["next_stage"], "remainder")
        self.assertNotIn("panel18", json.dumps(compact_audit))

    def test_runtime_validation_checks_cuda_toolchain_components(self):
        runtime = {
            "vllm_version": "0.26.0",
            "python_version": "3.12",
            "cuda_toolkit_meta_version": "13.0.2",
            "cuda_component_versions": {
                "nvidia-cuda-nvcc": "13.0.88",
                "nvidia-cuda-runtime": "13.0.96",
            },
        }
        details = {
            "python": "3.12.13",
            "packages": {
                "vllm": "0.26.0",
                "cuda-toolkit": "13.0.2",
                "nvidia-cuda-nvcc": "13.0.88",
                "nvidia-cuda-runtime": "13.0.96",
            },
        }
        assert_runtime_versions(runtime, details)
        details["packages"]["nvidia-cuda-nvcc"] = "13.3.73"
        with self.assertRaises(BakeoffError):
            assert_runtime_versions(runtime, details)

    def test_runtime_validation_can_pin_exact_runtime_lock(self):
        expected_sha = "a" * 64
        runtime = {
            "vllm_version": "0.26.0",
            "python_version": "3.12",
            "runtime_lock_sha256": expected_sha,
        }
        details = {
            "python": "3.12.13",
            "packages": {"vllm": "0.26.0"},
            "runtime_lock": {"path": "/scratch/runtime.freeze.txt", "sha256": expected_sha},
        }
        assert_runtime_versions(runtime, details)
        details["runtime_lock"]["sha256"] = "b" * 64
        with self.assertRaises(BakeoffError):
            assert_runtime_versions(runtime, details)
        details["runtime_lock"] = None
        with self.assertRaises(BakeoffError):
            assert_runtime_versions(runtime, details)

    def test_sbatch_can_build_a_separate_matched_cuda_runtime_lock(self):
        sbatch = (REPO / "experiments" / "hive_model_bakeoff_20260804.sbatch").read_text()
        self.assertIn('BAKEOFF_RUNTIME_LOCK="${BAKEOFF_RUNTIME_LOCK:-', sbatch)
        self.assertIn("BAKEOFF_RUNTIME_BASE_LOCK", sbatch)
        self.assertIn(
            "cuda-toolkit[cccl,crt,cudart,nvcc,nvrtc,nvvm]", sbatch
        )
        for requirement in [
            "nvidia-cuda-cccl==13.0.85",
            "nvidia-cuda-crt==13.0.88",
            "nvidia-cuda-nvcc==13.0.88",
            "nvidia-cuda-nvrtc==13.0.88",
            "nvidia-cuda-runtime==13.0.96",
            "nvidia-nvvm==13.0.88",
        ]:
            self.assertIn(requirement, sbatch)
        self.assertIn('MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"', sbatch)
        self.assertIn('FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"', sbatch)
        self.assertIn("export HF_HUB_OFFLINE=1", sbatch)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", sbatch)
        self.assertIn("snapshot_download", sbatch)
        self.assertIn("local_files_only=local_files_only", sbatch)
        self.assertIn("export BAKEOFF_MODEL_SNAPSHOT_PATH", sbatch)
        self.assertIn("immutable snapshot mismatch", sbatch)
        self.assertIn("#SBATCH --mem=96G", sbatch)
        self.assertIn('CUDA_RUNTIME_SONAME="$CUDA_RUNTIME_LIB/libcudart.so.13"', sbatch)
        self.assertIn('CUDA_NVRTC_SONAME="$CUDA_RUNTIME_LIB/libnvrtc.so.13"', sbatch)
        self.assertIn('ln -s lib "$CUDA_HOME/lib64"', sbatch)
        self.assertIn('ln -s "$CUDA_RUNTIME_SONAME" "$CUDA_HOME/lib64/libcudart.so"', sbatch)
        self.assertIn('ln -s "$CUDA_NVRTC_SONAME" "$CUDA_HOME/lib64/libnvrtc.so"', sbatch)
        self.assertIn('-lcudart -lcuda -lnvrtc -lcublas -lcublasLt -o "$CUDA_LINK_TEST"', sbatch)
        self.assertIn('readelf -d "$CUDA_LINK_TEST"', sbatch)
        self.assertIn("grep -q 'libnvrtc.so.13'", sbatch)
        self.assertIn('EFFECTIVE_RUNTIME_LOCK="$SCRATCH_ROOT/effective-runtime.freeze.txt"', sbatch)
        self.assertIn('cmp -s "$EFFECTIVE_RUNTIME_LOCK" "$BAKEOFF_RUNTIME_LOCK"', sbatch)
        self.assertIn("uv pip check", sbatch)

    def test_all_frozen_items_have_unique_ids_and_gold(self):
        config = load_config(CONFIG)
        for task in selected_tasks(config):
            items = load_task_items(task)
            ids = [str(item["item_id"]) for item in items]
            self.assertEqual(len(ids), len(set(ids)), task["name"])
            for item in items:
                self.assertTrue(item["gt"], task["name"])
                self.assertFalse(any(pd.isna(value) for value in item["gt"].values()))

    def test_frontier_panel_replaces_full_scope_without_changing_config(self):
        config = load_config(CONFIG)
        panel = load_panel_manifest(
            FRONTIER_PANEL,
            tasks_dir=REPO / config["task_scope"]["tasks_dir"],
        )
        tasks = selected_tasks(config, panel=panel)
        scope = validate_scope(
            config,
            tasks,
            expected_scope=(panel.expected_tasks, panel.expected_items),
        )
        self.assertEqual(scope["total_tasks"], 18)
        self.assertEqual(scope["total_items"], 8793)

    def test_explicit_direct_manifest_override_derives_broad_scope(self):
        args = SimpleNamespace(
            config=COMPACT_HIVE_CONFIG,
            model_key="qwen3_6_27b_fp8",
            output_dir=REPO / "output" / "sidecar" / "frontier_2026" / "broad18" / "test",
            panel_manifest=BROAD_PANEL,
            only_task=None,
            limit_items=None,
            force=False,
            plan_only=True,
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            run(args)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["panel_id"], "frontier_broad_18")
        self.assertEqual((plan["total_tasks"], plan["total_items"]), (18, 3600))
        self.assertEqual((plan["pilot_items"], plan["remainder_items"]), (16, 3584))


class HiveBakeoffRunnerTests(unittest.TestCase):
    def test_targeted_ablation_config_and_conversation_layouts(self):
        config = load_config(TARGETED_ABLATION_CONFIG)
        self.assertEqual(
            config["execution"]["protocol"], "broad18_targeted_ablation"
        )
        self.assertEqual(set(config["models"]), {
            "qwen3_32b_bf16_thinking_off_1024",
            "qwen3_32b_bf16_thinking_on_1024",
            "qwen3_6_27b_fp8_single_user",
        })
        self.assertEqual(config["generation"]["max_tokens"], 1024)
        self.assertFalse(
            config["models"]["qwen3_32b_bf16_thinking_off_1024"]
            ["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertTrue(
            config["models"]["qwen3_32b_bf16_thinking_on_1024"]
            ["chat_template_kwargs"]["enable_thinking"]
        )
        conversations = build_conversations(
            "Exact instructions", [{"user_content": "Exact item"}], "single_user"
        )
        self.assertEqual(conversations, [[{
            "role": "user", "content": "Exact instructions\n\nExact item"
        }]])

        panel = load_panel_manifest(
            BROAD_PANEL,
            tasks_dir=REPO / config["task_scope"]["tasks_dir"],
        )
        tasks = selected_tasks(config, panel=panel)
        scope = validate_scope(config, tasks, expected_scope=(18, 3600))
        self.assertEqual((scope["total_tasks"], scope["total_items"]), (18, 3600))

    def test_compact_run_generates_pilot_then_exact_remainder_without_retries(self):
        model_key = "qwen3_6_27b_fp8"
        config = load_config(COMPACT_HIVE_CONFIG)
        model = config["models"][model_key]
        runtime = config["runtime"]
        runtime_details = {
            "python": "3.12.13",
            "packages": {
                "vllm": runtime["vllm_version"],
                "cuda-toolkit": runtime["cuda_toolkit_meta_version"],
                **runtime["cuda_component_versions"],
            },
            "runtime_lock": {
                "path": runtime["runtime_lock_path"],
                "sha256": runtime["runtime_lock_sha256"],
            },
        }
        gpu_snapshot = {
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    "memory_total_mib": 97887,
                    "memory_used_mib": 90000,
                }
            ],
            "total_used_mib": 90000,
        }
        generated_keys = []
        generated_batch_sizes = []

        def fake_generate_task_batch(**kwargs):
            task = kwargs["task"]
            items = kwargs["items"]
            generated_batch_sizes.append(len(items))
            rows = []
            for item in items:
                generated_keys.append((task["name"], str(item["item_id"])))
                row = {
                    "task": task["name"],
                    "model": kwargs["model_alias"],
                    "model_id": kwargs["model"]["model_id"],
                    "model_revision": kwargs["model"]["revision"],
                    "item_id": str(item["item_id"]),
                    "latency_s": 0.01,
                    "eval_count": 1,
                    "prompt_tokens": 1,
                    "parse_error": None,
                    "finish_reason": "stop",
                    "structured_output_backend": "fake",
                    "raw_content_preview": "{}",
                    **(kwargs.get("checkpoint_metadata") or {}),
                }
                for key, value in item["gt"].items():
                    row[f"pred_{key}"] = value
                    row[f"gt_{key}"] = value
                rows.append(row)
            frame = pd.DataFrame(rows)
            seconds = max(len(items) / 100.0, 0.01)
            return frame, {
                "generation_seconds": seconds,
                "items_per_second": len(items) / seconds,
                "max_prompt_tokens": 1,
                "mean_output_tokens": 1.0,
                "parse_ok": len(items),
            }

        fake_llm = SimpleNamespace(get_tokenizer=lambda: object())
        sidecar_root = REPO / "output" / "sidecar"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=sidecar_root) as output_temp:
            with tempfile.TemporaryDirectory() as snapshot_temp:
                snapshot = Path(snapshot_temp) / model["revision"]
                snapshot.mkdir()
                args = SimpleNamespace(
                    config=COMPACT_HIVE_CONFIG,
                    model_key=model_key,
                    output_dir=Path(output_temp) / model_key,
                    panel_manifest=None,
                    only_task=None,
                    limit_items=None,
                    force=False,
                    plan_only=False,
                )
                with patch.dict(
                    "os.environ",
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "BAKEOFF_MODEL_SNAPSHOT_PATH": str(snapshot),
                    },
                    clear=False,
                ), patch(
                    "hive_vllm_benchmark.runtime_versions",
                    return_value=runtime_details,
                ), patch(
                    "hive_vllm_benchmark.query_gpu",
                    return_value=gpu_snapshot,
                ), patch(
                    "hive_vllm_benchmark.create_llm",
                    return_value=(fake_llm, {"model": model["model_id"]}),
                ), patch(
                    "hive_vllm_benchmark.generate_task_batch",
                    side_effect=fake_generate_task_batch,
                ):
                    run(args)

                output_dir = Path(output_temp) / model_key
                predictions = pd.read_csv(output_dir / "predictions.csv")
                pilot = pd.read_csv(output_dir / "pilot" / "predictions.csv")
                metadata = json.loads((output_dir / "run_metadata.json").read_text())

        self.assertEqual(generated_batch_sizes, [16, 484] + [500] * 7)
        self.assertEqual(sum(generated_batch_sizes), 4000)
        self.assertEqual(len(generated_keys), len(set(generated_keys)))
        self.assertEqual((len(pilot), len(predictions)), (16, 4000))
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["qualification"]["passed"])
        self.assertEqual(metadata["qualification"]["pilot_rows"], 16)
        self.assertEqual(metadata["qualification"]["remainder_rows"], 3984)

    def test_task_metrics_index_retains_malformed_rows(self):
        task = {
            "name": "binary_task",
            "label_kind": "binary",
            "label_key": "relevant",
            "labels": [0, 1],
        }
        predictions = pd.DataFrame(
            {
                "task": ["binary_task"] * 4,
                "model": ["candidate"] * 4,
                "gt_relevant": [1, 1, 0, 0],
                "pred_relevant": [pd.NA, 1, 0, 0],
                "parse_error": ["malformed", pd.NA, pd.NA, pd.NA],
                "latency_s": [0.1] * 4,
            }
        )
        metrics = task_metrics_frame(predictions, [task], "candidate")
        self.assertEqual(metrics.iloc[0]["n"], 4)
        self.assertEqual(metrics.iloc[0]["parse_ok"], 3)
        self.assertEqual(metrics.iloc[0]["parse_err_rate"], 0.25)
        self.assertLess(metrics.iloc[0]["headline_f1"], 1.0)

    def test_prompt_preflight_disables_thinking_and_never_truncates(self):
        tokenizer = FakeTokenizer()
        conversations = [
            [
                {"role": "system", "content": "classify this"},
                {"role": "user", "content": "one two three"},
            ]
        ]
        lengths = preflight_prompt_lengths(
            tokenizer,
            conversations,
            max_model_len=20,
            max_tokens=5,
            margin_tokens=2,
        )
        self.assertEqual(lengths, [8])
        self.assertFalse(tokenizer.last_enable_thinking)

        mapping_lengths = preflight_prompt_lengths(
            FakeMappingTokenizer(),
            conversations,
            max_model_len=20,
            max_tokens=5,
            margin_tokens=2,
        )
        self.assertEqual(mapping_lengths, [8])

        with self.assertRaises(BakeoffError):
            preflight_prompt_lengths(
                tokenizer,
                conversations,
                max_model_len=12,
                max_tokens=3,
                margin_tokens=2,
            )

    def test_preflight_uses_model_specific_chat_template_kwargs(self):
        tokenizer = CapturingTokenizer()
        lengths = preflight_prompt_lengths(
            tokenizer,
            [[{"role": "user", "content": "classify"}]],
            max_model_len=20,
            max_tokens=5,
            margin_tokens=2,
            chat_template_kwargs={
                "enable_thinking": False,
                "reasoning_effort": "none",
            },
        )
        self.assertEqual(lengths, [3])
        self.assertEqual(
            tokenizer.last_kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "reasoning_effort": "none",
            },
        )

    def test_checkpoint_requires_exact_coverage(self):
        rows = pd.DataFrame(
            [
                {"task": "task_a", "model": "model_a", "item_id": "1", "parse_error": None},
                {"task": "task_a", "model": "model_a", "item_id": "2", "parse_error": None},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.csv"
            write_csv_atomic(path, rows)
            self.assertTrue(task_checkpoint_complete(path, "task_a", "model_a", ["1", "2"]))
            write_json_atomic(
                task_done_path(path),
                {
                    "task": "task_a",
                    "model_key": "model_a",
                    "rows": 2,
                    "task_fingerprint": "fingerprint-a",
                    "checkpoint_sha256": file_sha256(path),
                },
            )
            self.assertTrue(
                task_checkpoint_complete(
                    path,
                    "task_a",
                    "model_a",
                    ["1", "2"],
                    expected_fingerprint="fingerprint-a",
                )
            )
            tampered = rows.copy()
            tampered.loc[0, "parse_error"] = "tampered"
            write_csv_atomic(path, tampered)
            self.assertFalse(
                task_checkpoint_complete(
                    path,
                    "task_a",
                    "model_a",
                    ["1", "2"],
                    expected_fingerprint="fingerprint-a",
                )
            )
            write_csv_atomic(path, rows)
            self.assertFalse(
                task_checkpoint_complete(
                    path,
                    "task_a",
                    "model_a",
                    ["1", "2"],
                    expected_fingerprint="stale-fingerprint",
                )
            )
            self.assertFalse(task_checkpoint_complete(path, "task_a", "model_a", ["1", "2", "3"]))
            self.assertFalse(task_checkpoint_complete(path, "task_a", "model_b", ["1", "2"]))

            panel_rows = rows.assign(
                panel_sha256="panel-a",
                panel_task_fingerprint="input-a",
            )
            write_csv_atomic(path, panel_rows)
            write_json_atomic(
                task_done_path(path),
                {
                    "task": "task_a",
                    "model_key": "model_a",
                    "rows": 2,
                    "task_fingerprint": "checkpoint-a",
                    "panel_sha256": "panel-a",
                    "panel_task_fingerprint": "input-a",
                    "checkpoint_sha256": file_sha256(path),
                },
            )
            self.assertTrue(
                task_checkpoint_complete(
                    path,
                    "task_a",
                    "model_a",
                    ["1", "2"],
                    expected_fingerprint="checkpoint-a",
                    expected_panel_sha256="panel-a",
                    expected_panel_task_fingerprint="input-a",
                )
            )
            self.assertFalse(
                task_checkpoint_complete(
                    path,
                    "task_a",
                    "model_a",
                    ["1", "2"],
                    expected_fingerprint="checkpoint-a",
                    expected_panel_sha256="panel-b",
                    expected_panel_task_fingerprint="input-a",
                )
            )

    def test_uncommitted_task_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "saved.csv"
            refuse_uncommitted_checkpoint(path, "task_a")
            saved = pd.DataFrame([{"task": "task_a", "model": "model_a",
                                   "item_id": "1", "parse_error": "schema_invalid"}])
            write_csv_atomic(path, saved)
            before = file_sha256(path)
            with self.assertRaisesRegex(BakeoffError, "refusing to overwrite"):
                refuse_uncommitted_checkpoint(path, "task_a")
            self.assertEqual(file_sha256(path), before)

            marker_only = Path(temp_dir) / "marker-only.csv"
            write_json_atomic(task_done_path(marker_only), {"task": "task_a"})
            with self.assertRaisesRegex(BakeoffError, "completion marker"):
                refuse_uncommitted_checkpoint(marker_only, "task_a")

    def test_resume_identity_separates_pilot_and_full_scopes(self):
        full = {
            "config_sha256": "a",
            "model_key": "model_a",
            "revision": "b",
            "only_task": None,
            "limit_items": None,
            "task_counts": {"task_a": 100, "task_b": 200},
        }
        pilot = {
            **full,
            "only_task": "task_a",
            "limit_items": 16,
            "task_counts": {"task_a": 16},
        }
        self.assertTrue(resume_identity_matches(full, dict(full)))
        self.assertFalse(resume_identity_matches(full, pilot))

    def test_resume_identity_separates_panel_hashes(self):
        prior = {
            "config_sha256": "a",
            "model_key": "model_a",
            "revision": "b",
            "only_task": None,
            "limit_items": None,
            "task_counts": {"task_a": 100},
            "panel_sha256": "panel-a",
        }
        changed = {**prior, "panel_sha256": "panel-b"}
        self.assertTrue(resume_identity_matches(prior, dict(prior)))
        self.assertFalse(resume_identity_matches(prior, changed))

    def test_output_guard_rejects_canonical_paths(self):
        with self.assertRaises(BakeoffError):
            require_sidecar_output_dir(REPO / "output")
        allowed = REPO / "output" / "sidecar" / "test-run" / "model"
        self.assertEqual(require_sidecar_output_dir(allowed), allowed.resolve())

    def test_prediction_rows_match_existing_summary_schema(self):
        config = load_config(CONFIG)
        task = selected_tasks(config, "brandt_political_relevance")[0]
        item = load_task_items(task, limit_items=1)[0]
        completion = SimpleNamespace(
            text=json.dumps(item["gt"]),
            token_ids=[1, 2, 3],
            finish_reason="stop",
        )
        output = SimpleNamespace(outputs=[completion])
        model_key = "qwen3_30b_a3b_instruct_2507_fp8"
        rows = prediction_rows(
            task=task,
            items=[item],
            outputs=[output],
            prompt_lengths=[100],
            model_alias=model_key,
            model=config["models"][model_key],
            effective_latency_s=0.25,
            structured_backend="StructuredOutputsParams",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsNone(row["parse_error"])
        self.assertEqual(row["task"], task["name"])
        self.assertEqual(row["model"], model_key)
        self.assertEqual(row["item_id"], str(item["item_id"]))
        self.assertIn(f"pred_{task['label_key']}", row)
        self.assertIn(f"gt_{task['label_key']}", row)


class HiveBakeoffSummaryTests(unittest.TestCase):
    def test_report_uses_dynamic_model_count_and_config_path(self):
        config = load_config(WATCHLIST_CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.md"
            render_report(
                path,
                model_frame=pd.DataFrame([{"model": "qwen3_6_27b_fp8"}]),
                contrast_frame=pd.DataFrame(
                    [{"model": "glm4_7_flash", "promotion_gate": False}]
                ),
                support_frame=pd.DataFrame([{"support": 30}]),
                config=config,
                config_path=WATCHLIST_CONFIG,
            )
            report = path.read_text()
        self.assertIn("compares 3 revision-pinned", report)
        self.assertIn("hive_model_bakeoff_extension_cuda130_20260805.yaml", report)

    def test_blinding_uses_random_permutation_and_alias_column_order(self):
        with patch("summarize_hive_bakeoff.secrets.SystemRandom") as system_random:
            system_random.return_value.shuffle.side_effect = lambda values: values.reverse()
            aliases = random_blinding_aliases(["baseline", "candidate_a", "candidate_b"])
        self.assertEqual(
            aliases,
            {"candidate_b": "model_1", "candidate_a": "model_2", "baseline": "model_3"},
        )
        blinded = ordered_blinded_predictions(
            {"baseline": "b", "candidate_a": "a", "candidate_b": "c"},
            aliases,
        )
        self.assertEqual(list(blinded), ["model_1", "model_2", "model_3"])
        self.assertEqual(list(blinded.values()), ["c", "a", "b"])

    def test_generation_timing_must_match_checkpoint_rows(self):
        frame = pd.DataFrame({"latency_s": [0.25, 0.25]})
        self.assertEqual(
            validate_task_generation_seconds(
                frame,
                {"generation_seconds": 0.5},
                "model_a",
                "task_a",
            ),
            0.5,
        )
        with self.assertRaises(BakeoffError):
            validate_task_generation_seconds(
                frame,
                {"generation_seconds": 0.6},
                "model_a",
                "task_a",
            )

    def test_runtime_comparison_requires_same_stack_and_gpu_type(self):
        def metadata(gpu_name="GPU A", memory=100, vllm="0.26.0", uuid="one"):
            return {
                "runtime_versions": {
                    "python": "3.12.13",
                    "packages": {"vllm": vllm},
                    "runtime_lock": {"sha256": "abc"},
                },
                "gpu_after_load": {
                    "gpus": [
                        {
                            "name": gpu_name,
                            "memory_total_mib": memory,
                            "uuid": uuid,
                        }
                    ]
                },
            }

        validate_comparable_runtime(
            {"model_a": metadata(uuid="one"), "model_b": metadata(uuid="two")}
        )
        with self.assertRaises(BakeoffError):
            validate_comparable_runtime(
                {"model_a": metadata(), "model_b": metadata(memory=32)}
            )
        with self.assertRaises(BakeoffError):
            validate_comparable_runtime(
                {"model_a": metadata(), "model_b": metadata(vllm="0.27.0")}
            )

    def test_rare_class_recall_uses_observed_rarest_label(self):
        y_true = pd.Series([0, 0, 0, 1])
        self.assertEqual(rare_class_recall(y_true, pd.Series([0, 0, 0, 1]), [0, 1]), 1.0)
        self.assertEqual(rare_class_recall(y_true, pd.Series([0, 0, 0, 0]), [0, 1]), 0.0)

    def test_rare_class_recall_averages_minimum_support_ties(self):
        y_true = pd.Series(["a", "a", "b", "c"])
        y_pred = pd.Series(["a", "a", "b", "b"])
        self.assertEqual(rare_class_recall(y_true, y_pred, ["a", "b", "c"]), 0.5)

    def test_rare_class_recall_counts_unusable_outputs_as_misses(self):
        y_true = pd.Series([0, 0, 0, 1])
        y_pred = pd.Series([0, 0, 0, pd.NA])
        usable = pd.Series([True, True, True, False])
        self.assertEqual(rare_class_recall(y_true, y_pred, [0, 1], usable), 0.0)

    def test_markdown_table_has_no_optional_tabulate_dependency(self):
        rendered = markdown_table(pd.DataFrame([{"model": "a|b", "score": 0.5}]))
        self.assertIn("a\\|b", rendered)
        self.assertIn("0.5000", rendered)

    def test_prediction_comparison_normalizes_mixed_item_id_types(self):
        per_task = pd.DataFrame(
            [
                {"task": "a", "model": "m", "item_id": 2, "eval_count": 20},
                {"task": "a", "model": "m", "item_id": 10, "eval_count": 100},
            ]
        )
        merged = pd.DataFrame(
            [
                {"task": "a", "model": "m", "item_id": "10", "eval_count": 100},
                {"task": "a", "model": "m", "item_id": "2", "eval_count": 20},
            ]
        )
        pd.testing.assert_frame_equal(
            normalized_prediction_frame(per_task),
            normalized_prediction_frame(merged),
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
