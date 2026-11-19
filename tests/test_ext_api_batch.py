import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import ext_api_batch as ext  # noqa: E402

DRAW = ROOT / "output" / "sidecar" / "task_ext_20261114_api" / "draw.json"


@unittest.skipUnless(DRAW.exists(), "API draw not built")
class ExtApiBatchTests(unittest.TestCase):
    def setUp(self):
        ext.install()

    def test_pilot_and_remainder_partition_the_draw(self):
        rows = json.loads(DRAW.read_text())["rows"]
        pilot, _ = ext.load_requests("gpt-6-luna", "pilot")
        rest, _ = ext.load_requests("gpt-6-luna", "remainder")
        self.assertEqual(len(pilot), 40)
        self.assertEqual(len(pilot) + len(rest), len(rows))
        ids = [r["custom_id"] for r in pilot + rest]
        self.assertEqual(len(ids), len(set(ids)))

    def test_instructions_come_first_and_anthropic_gets_a_cache_marker(self):
        anthropic, _ = ext.load_requests("claude-haiku-5.5", "pilot")
        openai, _ = ext.load_requests("gpt-6-luna", "pilot")
        system = anthropic[0]["body"]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertEqual(system["content"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(system["content"][0]["text"], openai[0]["body"]["messages"][0]["content"])
        self.assertEqual(anthropic[0]["body"]["messages"][1]["role"], "user")
        self.assertNotIn("cache_control", json.dumps(openai[0]["body"]))

    def test_score_task_answers_are_not_marked_schema_invalid(self):
        requests, tasks = ext.load_requests("gpt-6-luna", "remainder")
        req = next(r for r in requests if tasks[r["task"]]["label_kind"] == "score")
        task = tasks[req["task"]]
        for text, error in [('{"score":20}', None), ('{"score":"NA"}', "not_applicable")]:
            entry = {"response": {"body": {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}}}
            row = ext.rob.decode("gpt-6-luna", req, entry, task)
            self.assertEqual(row["parse_error"] or None, error)


if __name__ == "__main__":
    unittest.main()
