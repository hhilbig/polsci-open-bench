import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import anthropic_batch_collect  # noqa: E402
import anthropic_batch_submit  # noqa: E402
from task_registry import load_task_definitions  # noqa: E402


class AnthropicBatchSchemaTests(unittest.TestCase):
    def test_multi_binary_schema_keys_are_anthropic_safe(self):
        task = next(
            t for t in load_task_definitions(tasks_dir=REPO / "tasks")
            if t["name"] == "erlich_ati_topics"
        )
        schema = anthropic_batch_submit.anthropic_input_schema(task)
        self.assertIn("External_Contracts", schema["properties"])
        self.assertIn("Institutional_Structure", schema["properties"])
        self.assertNotIn("External Contracts", schema["properties"])
        self.assertNotIn("Institutional Structure", schema["properties"])

    def test_categorical_prompt_restates_allowed_labels(self):
        task = next(
            t for t in load_task_definitions(tasks_dir=REPO / "tasks")
            if t["name"] == "cap_crs_policy_topic"
        )
        prompt = anthropic_batch_submit.anthropic_system_prompt(task, "Classify this.")
        self.assertIn("must be exactly one of", prompt)
        self.assertIn("International Affairs", prompt)
        self.assertIn("Do not invent or paraphrase labels", prompt)

    def test_collector_restores_anthropic_safe_keys(self):
        task = next(
            t for t in load_task_definitions(tasks_dir=REPO / "tasks")
            if t["name"] == "erlich_ati_topics"
        )
        result = {
            "custom_id": "req_000001",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "classify",
                            "input": {
                                "Activities": 0,
                                "Budget": 1,
                                "Evaluation": 0,
                                "External_Contracts": 1,
                                "Institutional_Structure": 0,
                                "Other": 0,
                                "Regulatory": 1,
                            },
                        }
                    ],
                    "usage": {"output_tokens": 20},
                },
            },
        }
        content, eval_count, parse_err = anthropic_batch_collect.response_content(result, task)
        parsed = json.loads(content)
        self.assertIsNone(parse_err)
        self.assertEqual(eval_count, 20)
        self.assertEqual(parsed["External Contracts"], 1)
        self.assertEqual(parsed["Institutional Structure"], 0)


if __name__ == "__main__":
    unittest.main()
