import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import refresh_jev
from task_registry import load_task_definitions


class JevRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {task["name"]: task for task in load_task_definitions()}

    def test_every_task_maps_to_supported_choice_questions(self):
        panel = json.loads(refresh_jev.PANEL.read_text())
        refresh_jev.validate_frozen_panel(panel)
        seen = set()
        for row in panel["rows"]:
            task = self.tasks[row["task"]]
            payload = refresh_jev.build_payload(task, row["item"])
            expected = task["labels"] if task["label_kind"] == "multi_binary" else [task["label_key"]]
            self.assertEqual(list(payload["questions"]), list(expected))
            self.assertEqual(payload["model"], refresh_jev.REQUEST_MODEL)
            self.assertEqual(payload["state"]["text_to_code"], row["item"]["user_content"])
            self.assertEqual(
                payload["state"]["benchmark_instructions"],
                Path(task["prompt_path"]).read_text(),
            )
            for field in expected:
                question = payload["questions"][field]
                self.assertEqual(question["type"], "choice")
                self.assertLessEqual(len(question["criteria"]), 255)
            seen.add(task["name"])
        self.assertEqual(seen, set(self.tasks))

    def test_decode_preserves_categorical_and_binary_values(self):
        for name in ["gilardi_relevance", "gilardi_stance"]:
            task = self.tasks[name]
            field = task["labels"][0] if task["label_kind"] == "multi_binary" else task["label_key"]
            values = task["json_schema"]["properties"][field]["enum"]
            selected = values[-1]
            row = {"item_id": "1", "gold": {field: values[0]}}
            response = {
                "model": refresh_jev.MODEL,
                "answers": {
                    field: {
                        "type": "choice",
                        "choice": str(selected),
                        "probabilities": {str(value): 1.0 if value == selected else 0.0 for value in values},
                        "confidence": 1.0,
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 4},
            }
            decoded = refresh_jev.decode(row, task, response)
            self.assertEqual(decoded[f"pred_{field}"], selected)
            self.assertEqual(decoded["parse_error"], "")
            self.assertAlmostEqual(decoded["cost_usd_upper"], 0.0000042)

    def test_decode_marks_invalid_typed_response_malformed(self):
        task = self.tasks["gilardi_relevance"]
        row = {"item_id": "1", "gold": {"relevant": 1}}
        response = {
            "model": refresh_jev.MODEL,
            "answers": {
                "relevant": {
                    "type": "choice",
                    "choice": "7",
                    "probabilities": {"0": 0.1, "1": 0.9},
                    "confidence": 0.8,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        decoded = refresh_jev.decode(row, task, response)
        self.assertIn("unknown choice", decoded["parse_error"])

    def test_prepare_is_deterministic_ignoring_creation_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(refresh_jev, "ROOT", root):
                first = refresh_jev.prepare()
                second = refresh_jev.prepare()
                self.assertEqual(first["requests"], 3400)
                self.assertEqual(second["pilot_requests"], 68)
                prepared = json.loads((root / "requests.json").read_text())
                self.assertEqual(len(prepared["requests"]), 3400)


if __name__ == "__main__":
    unittest.main()
