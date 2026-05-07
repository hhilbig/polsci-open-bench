import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import benchmark  # noqa: E402
import batch_benchmark  # noqa: E402
import build_coverage_matrix  # noqa: E402
import model_registry  # noqa: E402
import task_registry  # noqa: E402
from cap_topic_labels import CAP_MAJOR_TOPIC_LABELS_IN_ORDER  # noqa: E402


class LoaderIntegrityTests(unittest.TestCase):
    def test_all_loaders_return_expected_unique_item_ids(self):
        for task in benchmark.TASKS:
            with self.subTest(task=task["name"]):
                items = task["loader"]()
                manifest = yaml.safe_load((REPO / "tasks" / f"{task['name']}.yaml").read_text())
                data_path = (REPO / "tasks" / manifest["data_file"]).resolve()
                df = pd.read_csv(data_path, low_memory=False)
                sampling = manifest.get("sampling", {}) or {}
                n_v1 = int(sampling.get("n_v1", task_registry.DEFAULT_SAMPLE_N_V1))
                n_v2_new = int(sampling.get("n_v2_new", task_registry.DEFAULT_SAMPLE_N_V2_NEW))
                expected_n = min(len(df), n_v1 + n_v2_new)
                self.assertEqual(len(items), expected_n)
                item_ids = [item["item_id"] for item in items]
                self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_gilardi_relevance_duplicate_source_ids_are_suffixed(self):
        task = next(t for t in benchmark.TASKS if t["name"] == "gilardi_relevance")
        item_ids = [item["item_id"] for item in task["loader"]()]
        dup_ids = [item_id for item_id in item_ids if "__dup" in item_id]
        self.assertCountEqual(
            dup_ids,
            [
                "1259252076959412224__dup1",
                "1259438138633617408__dup1",
            ],
        )

    def test_example_custom_task_dir_loads(self):
        tasks = task_registry.load_task_definitions(task_dir=REPO / "examples" / "minimal_custom_task")
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["name"], "minimal_custom_task")
        items = task["loader"]()
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["item_id"], "1")

    def test_cross_domain_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "osnabruegge_cross_domain_topic.csv")
        self.assertEqual(len(df), 4165)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["gt_policy_domain"].isna().sum(), 0)
        self.assertEqual(
            set(df["gt_policy_domain"]),
            {
                "Economy",
                "External Relations",
                "Fabric of Society",
                "Freedom and Democracy",
                "No Topic",
                "Political System",
                "Social Groups",
                "Welfare and Quality of Life",
            },
        )

        task = next(t for t in benchmark.TASKS if t["name"] == "osnabruegge_cross_domain_topic")
        items = task["loader"]()
        self.assertEqual(task["family"], "Policy-topic coding")
        self.assertEqual(task["label_key"], "policy_domain")
        self.assertEqual(len(task["labels"]), 8)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["policy_domain"] in task["labels"] for item in items))

    def test_line_of_fire_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "rheault_line_of_fire_incivility.csv")
        self.assertEqual(len(df), 9923)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["gt_uncivil"].isna().sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["gt_uncivil"]), {0, 1})

        task = next(t for t in benchmark.TASKS if t["name"] == "rheault_line_of_fire_incivility")
        items = task["loader"]()
        self.assertEqual(task["family"], "Relevance / Incivility")
        self.assertEqual(task["label_key"], "uncivil")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["uncivil"] in [0, 1] for item in items))

    def test_theocharis_dynamics_incivility_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "theocharis_dynamics_incivility.csv")
        self.assertEqual(len(df), 3997)
        self.assertEqual(df["source_tweet_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["gt_uncivil"].isna().sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["source_uncivil_raw"]), {"no", "yes"})
        self.assertEqual(set(df["gt_uncivil"]), {0, 1})

        task = next(t for t in benchmark.TASKS if t["name"] == "theocharis_dynamics_incivility")
        items = task["loader"]()
        self.assertEqual(task["family"], "Relevance / Incivility")
        self.assertEqual(task["label_key"], "uncivil")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["uncivil"] in [0, 1] for item in items))

    def test_papea_fgz_forms_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "haunss_papea_fgz_forms.csv")
        self.assertEqual(len(df), 4816)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["gt_protest_form"].isna().sum(), 0)
        labels = {
            "Demonstration / Assembly",
            "Petition",
            "Strike",
            "Non-verbal Protest / Cultural Event",
            "Leaflet / Resolution / Open Letter",
            "Attack with Damage to Property",
            "Blockade / Sit-in",
        }
        self.assertEqual(set(df["gt_protest_form"]), labels)

        task = next(t for t in benchmark.TASKS if t["name"] == "haunss_papea_fgz_forms")
        items = task["loader"]()
        self.assertEqual(task["family"], "Event coding")
        self.assertEqual(task["label_key"], "protest_form")
        self.assertEqual(len(task["labels"]), 7)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["protest_form"] in labels for item in items))

    def test_brandt_political_relevance_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "brandt_political_relevance.csv")
        self.assertEqual(len(df), 320)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["gt_relevant"].isna().sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["gt_relevant"]), {0, 1})

        task = next(t for t in benchmark.TASKS if t["name"] == "brandt_political_relevance")
        items = task["loader"]()
        self.assertEqual(task["family"], "Relevance / Incivility")
        self.assertEqual(task["label_key"], "relevant")
        self.assertEqual(len(items), 320)
        self.assertTrue(all(item["gt"]["relevant"] in [0, 1] for item in items))

    def test_icbe_sentence_event_type_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "douglass_icbe_sentence_event_type.csv")
        self.assertEqual(len(df), 12676)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["crisis_title"].isna().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["gt_event_type"].isna().sum(), 0)
        labels = {"No Event", "Action", "Speech", "Thought", "Mixed"}
        self.assertEqual(set(df["gt_event_type"]), labels)

        task = next(t for t in benchmark.TASKS if t["name"] == "douglass_icbe_sentence_event_type")
        items = task["loader"]()
        self.assertEqual(task["family"], "Event coding")
        self.assertEqual(task["label_key"], "event_type")
        self.assertEqual(len(task["labels"]), 5)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["event_type"] in labels for item in items))

    def test_muller_fujimura_campaign_policy_area_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "muller_fujimura_campaign_policy_area.csv")
        self.assertEqual(len(df), 2915)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["gt_policy_area"].isna().sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        labels = {
            "Agriculture, Forestry, and Fisheries",
            "Committees on Cabinet",
            "Economy, Trade and Industry",
            "Education, Culture, Sports, Science, and Technology",
            "Environment",
            "Financial Affairs",
            "Foreign Affairs",
            "Health, Labour, and Welfare",
            "Internal Affairs and Communications",
            "Land, Infrastructure, Transport, and Tourism",
            "No Policy Area",
            "Security",
        }
        self.assertEqual(set(df["gt_policy_area"]), labels)

        task = next(t for t in benchmark.TASKS if t["name"] == "muller_fujimura_campaign_policy_area")
        items = task["loader"]()
        self.assertEqual(task["family"], "Policy-topic coding")
        self.assertEqual(task["label_key"], "policy_area")
        self.assertEqual(len(task["labels"]), 12)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["policy_area"] in labels for item in items))

    def test_burnham_polnli_entailment_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "burnham_polnli_entailment.csv")
        self.assertEqual(len(df), 15314)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["premise"].isna().sum(), 0)
        self.assertEqual(df["premise"].eq("").sum(), 0)
        self.assertEqual(df["hypothesis"].isna().sum(), 0)
        self.assertEqual(df["hypothesis"].eq("").sum(), 0)
        self.assertEqual(df.duplicated(["premise", "hypothesis"]).sum(), 0)
        self.assertEqual(set(df["gt_entails"]), {0, 1})
        self.assertEqual(set(df["source_entailment"]), {0, 1})

        task = next(t for t in benchmark.TASKS if t["name"] == "burnham_polnli_entailment")
        items = task["loader"]()
        self.assertEqual(task["family"], "Hypothesis-conditioned classification")
        self.assertEqual(task["label_key"], "entails")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["entails"] in [0, 1] for item in items))


class StagedNextTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks_next = {
            task["name"]: task
            for task in task_registry.load_task_definitions(tasks_dir=REPO / "tasks_next")
        }

    def test_all_staged_next_loaders_return_unique_item_ids(self):
        for name, task in self.tasks_next.items():
            with self.subTest(task=name):
                items = task["loader"]()
                self.assertEqual(len(items), len({item["item_id"] for item in items}))

    def test_burnham_trump_stance_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "burnham_trump_stance.csv")
        self.assertEqual(len(df), 4776)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        labels = {"Oppose", "Neutral", "Support"}
        self.assertEqual(set(df["gt_stance_toward_trump"]), labels)

        task = self.tasks_next["burnham_trump_stance"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Sentiment / Stance / Tone")
        self.assertEqual(task["label_key"], "stance_toward_trump")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["stance_toward_trump"] in labels for item in items))

    def test_burnham_covid_threat_minimization_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "burnham_covid_threat_minimization.csv")
        self.assertEqual(len(df), 293)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["gt_threat_minimizing"]), {0, 1})

        task = self.tasks_next["burnham_covid_threat_minimization"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Sentiment / Stance / Tone")
        self.assertEqual(task["label_key"], "threat_minimizing")
        self.assertEqual(len(items), 293)
        self.assertTrue(all(item["gt"]["threat_minimizing"] in [0, 1] for item in items))

    def test_dicocco_manifesto_populism_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "dicocco_manifesto_populism.csv")
        self.assertEqual(len(df), 7084)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["gt_populist"]), {0, 1})

        task = self.tasks_next["dicocco_manifesto_populism"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Rhetoric / Populism")
        self.assertEqual(task["label_key"], "populist")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["populist"] in [0, 1] for item in items))

    def test_bestvater_kavanaugh_stance_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "bestvater_kavanaugh_stance.csv")
        self.assertEqual(len(df), 3636)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertGreater(df["text"].duplicated().sum(), 2000)
        self.assertEqual(df.groupby("text")["gt_pro_kavanaugh"].nunique().max(), 1)
        self.assertEqual(set(df["gt_pro_kavanaugh"]), {0, 1})

        task = self.tasks_next["bestvater_kavanaugh_stance"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Sentiment / Stance / Tone")
        self.assertEqual(task["label_key"], "pro_kavanaugh")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["pro_kavanaugh"] in [0, 1] for item in items))

    def test_politicause_causal_relation_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "politicause_causal_relation.csv")
        self.assertEqual(len(df), 17771)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        self.assertEqual(set(df["gt_causal_relation"]), {0, 1})

        task = self.tasks_next["politicause_causal_relation"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Causal relation detection")
        self.assertEqual(task["label_key"], "causal_relation")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["causal_relation"] in [0, 1] for item in items))

    def test_cap_party_platform_policy_topic_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "cap_party_platform_policy_topic.csv")
        self.assertEqual(len(df), 37338)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        labels = set(CAP_MAJOR_TOPIC_LABELS_IN_ORDER)
        self.assertEqual(set(df["gt_policy_topic"]), labels)

        task = self.tasks_next["cap_party_platform_policy_topic"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Policy-topic coding")
        self.assertEqual(task["label_key"], "policy_topic")
        self.assertEqual(task["labels"], CAP_MAJOR_TOPIC_LABELS_IN_ORDER)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["policy_topic"] in labels for item in items))

    def test_cap_crs_policy_topic_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "cap_crs_policy_topic.csv")
        self.assertEqual(len(df), 16510)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["title"].isna().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        labels = set(CAP_MAJOR_TOPIC_LABELS_IN_ORDER)
        self.assertEqual(set(df["gt_policy_topic"]), labels)

        task = self.tasks_next["cap_crs_policy_topic"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Policy-topic coding")
        self.assertEqual(task["label_key"], "policy_topic")
        self.assertEqual(task["labels"], CAP_MAJOR_TOPIC_LABELS_IN_ORDER)
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["policy_topic"] in labels for item in items))

    def test_agoraspeech_criticism_agenda_task_file_and_labels(self):
        df = pd.read_csv(REPO / "data" / "agoraspeech_criticism_agenda.csv")
        self.assertEqual(len(df), 5279)
        self.assertEqual(df["source_id"].duplicated().sum(), 0)
        self.assertEqual(df["text"].isna().sum(), 0)
        self.assertEqual(df["text"].eq("").sum(), 0)
        self.assertEqual(df["text"].duplicated().sum(), 0)
        labels = {"criticism", "political agenda"}
        self.assertEqual(set(df["gt_criticism_or_agenda"]), labels)

        task = self.tasks_next["agoraspeech_criticism_agenda"]
        items = task["loader"]()
        self.assertEqual(task["family"], "Rhetoric / Discourse Function")
        self.assertEqual(task["label_key"], "criticism_or_agenda")
        self.assertEqual(len(items), 500)
        self.assertTrue(all(item["gt"]["criticism_or_agenda"] in labels for item in items))


class ModelRegistryTests(unittest.TestCase):
    def test_example_custom_model_manifest_loads(self):
        models = model_registry.load_model_definitions(
            model_manifest=REPO / "examples" / "minimal_custom_models" / "my_ollama_model.yaml"
        )
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["name"], "my-local-ollama-model")
        self.assertEqual(model["backend"], "ollama")
        self.assertEqual(model["compute_class"], "local")
        self.assertFalse(model["think"])

    def test_remote_openai_compatible_model_without_key_returns_no_client(self):
        with mock.patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""},
            clear=False,
        ):
            client = benchmark.make_openai_client(
                {
                    "backend": "openai",
                    "base_url": "https://api.deepseek.com",
                    "api_key_env": "DEEPSEEK_API_KEY",
                }
            )
        self.assertIsNone(client)

    def test_local_openai_compatible_model_without_key_uses_dummy_client(self):
        with mock.patch.dict(
            "os.environ",
            {"LOCAL_OPENAI_KEY": "", "OPENAI_API_KEY": ""},
            clear=False,
        ):
            client = benchmark.make_openai_client(
                {
                    "backend": "openai",
                    "base_url": "http://localhost:8000/v1",
                    "api_key_env": "LOCAL_OPENAI_KEY",
                }
            )
        self.assertIsNotNone(client)

    def test_deepseek_manifest_loads_with_provider_specific_fields(self):
        models = model_registry.load_model_definitions(
            model_manifest=REPO / "models" / "deepseek_v4_pro.yaml"
        )
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["provider"], "deepseek")
        self.assertEqual(model["response_format_type"], "json_object")
        self.assertEqual(model["thinking_mode"], "disabled")

    def test_gemma_26b_manifest_loads_as_local_ollama_model(self):
        models = model_registry.load_model_definitions(
            model_manifest=REPO / "models" / "gemma4_26b_a4b.yaml"
        )
        self.assertEqual(len(models), 1)
        model = models[0]
        self.assertEqual(model["name"], "gemma4:26b")
        self.assertEqual(model["display_name"], "Gemma 4 26B A4B")
        self.assertEqual(model["backend"], "ollama")
        self.assertEqual(model["compute_class"], "local")
        self.assertFalse(model["think"])

    def test_deepseek_serial_calls_use_json_object_and_disable_thinking(self):
        task = next(t for t in benchmark.TASKS if t["name"] == "ballard_incivility")
        client = mock.Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"uncivil": 0}'))],
            usage=SimpleNamespace(completion_tokens=7),
        )
        model = {
            "name": "deepseek-v4-pro",
            "provider": "deepseek",
            "response_format_type": "json_object",
            "thinking_mode": "disabled",
            "reasoning_effort": None,
        }

        result = benchmark.classify_openai(
            client=client,
            model_def=model,
            task=task,
            system_prompt="Classify the tweet.",
            user_content="Tweet: hello world",
        )

        self.assertEqual(result["content"], '{"uncivil": 0}')
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertIn("return JSON only", kwargs["messages"][0]["content"])

    def test_openai_serial_calls_keep_strict_json_schema(self):
        task = next(t for t in benchmark.TASKS if t["name"] == "ballard_incivility")
        client = mock.Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"uncivil": 1}'))],
            usage=SimpleNamespace(completion_tokens=8),
        )
        model = {
            "name": "gpt-5.5",
            "provider": "openai",
            "response_format_type": None,
            "thinking_mode": None,
            "reasoning_effort": "medium",
        }

        benchmark.classify_openai(
            client=client,
            model_def=model,
            task=task,
            system_prompt="Classify the tweet.",
            user_content="Tweet: hello world",
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertNotIn("extra_body", kwargs)

    def test_deepseek_batched_calls_use_json_object_results_wrapper(self):
        task = next(t for t in benchmark.TASKS if t["name"] == "ballard_incivility")
        items = task["loader"]()[:2]
        client = mock.Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"results": [{"uncivil": 0}, {"uncivil": 1}]}'))],
            usage=SimpleNamespace(completion_tokens=14),
        )
        model = {
            "name": "deepseek-v4-pro",
            "provider": "deepseek",
            "response_format_type": "json_object",
            "thinking_mode": "disabled",
            "reasoning_effort": None,
        }

        batch_benchmark.classify_openai_batched(
            client=client,
            model_def=model,
            system_prompt="Classify the tweets.",
            batch=items,
            task_def=task,
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertIn('"results"', kwargs["messages"][1]["content"])

    def test_build_summary_respects_custom_model_manifest(self):
        task = task_registry.load_task_definitions(
            task_dir=REPO / "examples" / "minimal_custom_task"
        )[0]
        items = task["loader"]()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            predictions_path = tmpdir / "predictions.csv"
            summary_path = tmpdir / "summary.csv"

            rows = []
            for item in items:
                rows.append(
                    {
                        "task": task["name"],
                        "model": "my-local-ollama-model",
                        "item_id": item["item_id"],
                        "latency_s": 0.5,
                        "eval_count": 8,
                        "parse_error": None,
                        "raw_content_preview": "",
                        "pred_relevant": item["gt"]["relevant"],
                        "gt_relevant": item["gt"]["relevant"],
                    }
                )
            pd.DataFrame(rows).to_csv(predictions_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "code" / "build_summary.py"),
                    "--task-dir",
                    str(REPO / "examples" / "minimal_custom_task"),
                    "--model-manifest",
                    str(REPO / "examples" / "minimal_custom_models" / "my_ollama_model.yaml"),
                    "--predictions",
                    str(predictions_path),
                    "--output",
                    str(summary_path),
                ],
                check=True,
                cwd=REPO,
            )

            summary = pd.read_csv(summary_path)
            self.assertEqual(summary.loc[0, "model"], "my-local-ollama-model")
            self.assertGreater(summary.loc[0, "gpu_hours_per_1000"], 0)
            self.assertTrue(pd.isna(summary.loc[0, "usd_per_1000"]))


class MergeIntoTests(unittest.TestCase):
    def test_merge_into_replaces_matching_task_model_item_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            old = pd.DataFrame(
                [
                    {"task": "t1", "model": "m1", "item_id": "a", "pred_relevant": 0},
                    {"task": "t1", "model": "m1", "item_id": "b", "pred_relevant": 1},
                    {"task": "t2", "model": "m2", "item_id": "c", "pred_relevant": 0},
                ]
            )
            old.to_csv(path, index=False)

            new_rows = [
                {"task": "t1", "model": "m1", "item_id": "b", "pred_relevant": 0},
                {"task": "t1", "model": "m1", "item_id": "d", "pred_relevant": 1},
            ]
            benchmark.merge_into(path, new_rows)

            merged = pd.read_csv(path)
            merged = merged.sort_values(["task", "model", "item_id"]).reset_index(drop=True)

            expected = pd.DataFrame(
                [
                    {"task": "t1", "model": "m1", "item_id": "a", "pred_relevant": 0},
                    {"task": "t1", "model": "m1", "item_id": "b", "pred_relevant": 0},
                    {"task": "t1", "model": "m1", "item_id": "d", "pred_relevant": 1},
                    {"task": "t2", "model": "m2", "item_id": "c", "pred_relevant": 0},
                ]
            )
            pd.testing.assert_frame_equal(merged, expected)


class CoverageMatrixTests(unittest.TestCase):
    def test_find_predictions_files_includes_live_sidecar_and_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "predictions.csv"
            sidecar = root / "sidecar"
            archive = root / "archive"
            archived_run = archive / "archived_run"
            sidecar.mkdir()
            archived_run.mkdir(parents=True)

            for path in [
                canonical,
                sidecar / "api_v2pt2_predictions.csv",
                archived_run / "archived_predictions.csv",
                sidecar / "not_predictions.txt",
            ]:
                path.write_text("task,model,item_id\n")

            files = build_coverage_matrix.find_predictions_files(
                canonical_path=canonical,
                sidecar_root=sidecar,
                archive_root=archive,
            )

            labels = [label for label, _ in files]
            paths = [path.name for _, path in files]
            self.assertEqual(
                labels,
                ["canonical", "live_sidecar_api_v2pt2", "archived_run"],
            )
            self.assertEqual(
                paths,
                ["predictions.csv", "api_v2pt2_predictions.csv", "archived_predictions.csv"],
            )


class ArtifactIntegrityTests(unittest.TestCase):
    def _allowed_task_columns(self):
        allowed = set()
        for task in benchmark.TASKS:
            if task["label_kind"] == "multi_binary":
                labels = task["labels"]
            else:
                labels = [task["label_key"]]
            for label in labels:
                allowed.add(f"pred_{label}")
                allowed.add(f"gt_{label}")
        return allowed

    def test_serial_predictions_have_unique_keys(self):
        preds = pd.read_csv(REPO / "output" / "predictions.csv", low_memory=False)
        key_cols = ["task", "model", "item_id"]
        self.assertEqual(len(preds), preds[key_cols].drop_duplicates().shape[0])

    def test_serial_predictions_have_latency_for_clean_rows(self):
        preds = pd.read_csv(REPO / "output" / "predictions.csv", low_memory=False)
        clean = preds[preds["parse_error"].isna()]
        self.assertEqual(clean["latency_s"].isna().sum(), 0)
        self.assertTrue((clean["latency_s"] > 0).all())

    def test_batched_predictions_have_unique_keys_and_no_stale_task_columns(self):
        preds = pd.read_csv(REPO / "output" / "predictions_batched.csv", low_memory=False)
        key_cols = ["task", "model", "batch_size", "item_id"]
        self.assertEqual(len(preds), preds[key_cols].drop_duplicates().shape[0])

        observed_task_cols = {
            c for c in preds.columns if c.startswith("pred_") or c.startswith("gt_")
        }
        self.assertEqual(observed_task_cols - self._allowed_task_columns(), set())

    def test_batched_predictions_have_latency_for_clean_rows(self):
        preds = pd.read_csv(REPO / "output" / "predictions_batched.csv", low_memory=False)
        clean = preds[preds["parse_error"].isna()]
        self.assertEqual(clean["latency_s"].isna().sum(), 0)
        self.assertTrue((clean["latency_s"] > 0).all())
        self.assertEqual(clean["batch_latency_s"].isna().sum(), 0)
        self.assertTrue((clean["batch_latency_s"] > 0).all())


class TaskAuditTests(unittest.TestCase):
    def test_length_audit_builds_for_custom_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "audit.csv"
            out_md = Path(tmpdir) / "audit.md"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "code" / "build_task_length_audit.py"),
                    "--task-dir",
                    str(REPO / "examples" / "minimal_custom_task"),
                    "--summary",
                    str(Path(tmpdir) / "missing_summary.csv"),
                    "--output-csv",
                    str(out_csv),
                    "--output-md",
                    str(out_md),
                ],
                check=True,
                cwd=REPO,
            )
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())
            audit = pd.read_csv(out_csv)
            self.assertEqual(audit.loc[0, "task"], "minimal_custom_task")
            self.assertEqual(int(audit.loc[0, "sample_n"]), 4)


if __name__ == "__main__":
    unittest.main()
