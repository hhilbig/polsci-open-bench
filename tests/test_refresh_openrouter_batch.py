import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import refresh_openrouter_batch as r

TASK = {"name": "t", "label_kind": "categorical", "label_key": "label", "labels": ["a", "b"],
        "json_schema": {"type": "object"}}
REQUEST = {"custom_id": "r0001", "task": "t", "item": {"item_id": "x", "user_content": "hi", "gt": {"label": "a"}}}


def entry(content, finish="stop", **usage):
    return {"custom_id": "r0001", "error": None, "response": {"status_code": 200, "body": {
        "model": "m", "provider": "p", "usage": usage,
        "choices": [{"finish_reason": finish, "message": {"content": content}}]}}}


class BatchRunnerTests(unittest.TestCase):
    def test_requests_field_comes_last(self):
        payload = r.batch_payload(r.MODELS["gpt-6-luna"], [{"custom_id": "r1", "body": {"x": 1}}])
        self.assertEqual(list(payload)[-1], "requests")
        self.assertEqual(payload["model"], "openai/gpt-6-luna")
        self.assertNotIn(":batch", json.dumps(payload))

    def test_only_gemini_splits_batches_by_task(self):
        requests = [{"task": "a"}, {"task": "b"}, {"task": "a"}]
        self.assertEqual(len(r.batch_groups(r.MODELS["gemini-4-argon"], requests)), 2)
        self.assertEqual(len(r.batch_groups(r.MODELS["gpt-6.1-sol"], requests)), 1)

    def test_poll_merges_batches_and_waits_for_all_of_them(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, "ROOT", Path(folder)):
            r.save(r.state_path("gemini-4-argon", "pilot"),
                   {"status": "submitted", "batch_ids": ["a", "b"], "refused": []})
            done = {"status": "completed", "results": [{"custom_id": "r1"}], "usage": {"cost": 0.5},
                    "request_counts": {"total": 1, "completed": 1, "failed": 0}, "provider": "Google"}
            with patch.object(r, "http", side_effect=[done, {"status": "in_progress"}]):
                self.assertFalse(r.poll_batch("k", "gemini-4-argon", "pilot"))
            with patch.object(r, "http", side_effect=[done, done]):
                self.assertTrue(r.poll_batch("k", "gemini-4-argon", "pilot"))
            state = r.read_state("gemini-4-argon", "pilot")
            self.assertEqual(state["status"], "completed")
            self.assertAlmostEqual(state["cost_usd"], 1.0)
            merged = r.read(r.state_path("gemini-4-argon", "pilot").parent / "results.json")
            self.assertEqual(len(merged["results"]), 2)

    def test_poll_marks_a_failed_batch_as_ended(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, "ROOT", Path(folder)):
            r.save(r.state_path("gpt-6-luna", "pilot"), {"status": "submitted", "batch_id": "a"})
            failed = {"status": "failed", "error": {"message": "no"}, "usage": None}
            with patch.object(r, "http", return_value=failed):
                self.assertTrue(r.poll_batch("k", "gpt-6-luna", "pilot"))
            self.assertEqual(r.read_state("gpt-6-luna", "pilot")["status"], "ended")

    def test_body_uses_schema_and_model_settings(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write("system")
            f.flush()
            task = {**TASK, "prompt_path": f.name}
            body = r.build_body(r.MODELS["gemini-4-argon"], task, REQUEST["item"])
            self.assertEqual(body["messages"][0]["content"], "system")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["reasoning"], {"effort": "low"})
            self.assertTrue(body["response_format"]["json_schema"]["strict"])
            # No temperature setting is offered for these models.
            self.assertNotIn("temperature", r.build_body(r.MODELS["gpt-6.1-sol"], task, REQUEST["item"]))

    def test_worst_case_assumes_every_request_hits_the_limit(self):
        spec = r.MODELS["claude-opus-5.5"]
        usd = r.worst_case_usd(spec, [{"body": {}}])
        self.assertGreaterEqual(usd, 2048 * 10.0 / 1e6)

    def test_cap_counts_actual_cost_and_open_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r.save(root / "a/pilot/state.json", {"status": "collected", "cost_usd": 10.0, "reserved_usd": 99.0})
            r.save(root / "b/pilot/state.json", {"status": "submitted", "reserved_usd": 30.0})
            self.assertAlmostEqual(r.committed_usd(root), 40.0)
            self.assertAlmostEqual(r.check_cap(10.0, 50.0, root), 40.0)
            with self.assertRaisesRegex(SystemExit, "exceeds"):
                r.check_cap(10.01, 50.0, root)

    def test_remainder_needs_a_collected_pilot_and_scales_its_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(SystemExit, "pilot not collected"):
                r.reservation("gpt-6-luna", "remainder", [{}] * 10, root)
            r.save(root / "gpt-6-luna/pilot/state.json",
                   {"status": "collected", "cost_usd": 1.0, "n_requests": 10})
            self.assertAlmostEqual(r.reservation("gpt-6-luna", "remainder", [{}] * 100, root), 15.0)

    def test_decode_valid_answer_and_usage(self):
        with patch.object(r.benchmark, "parse_content", return_value=({"label": "a"}, None)):
            row = r.decode("gpt-6-luna", REQUEST, entry('{"label": "a"}', prompt_tokens=5, completion_tokens=2,
                                                        completion_tokens_details={"reasoning_tokens": 1}), TASK)
        self.assertIsNone(row["parse_error"])
        self.assertEqual(row["pred_label"], "a")
        self.assertEqual(row["reasoning_tokens"], 1)
        self.assertFalse(row["truncated"])

    def test_decode_flags_truncation_and_bad_labels(self):
        with patch.object(r.benchmark, "parse_content", return_value=({}, None)):
            row = r.decode("gpt-6-luna", REQUEST, entry('{"label": "c"}'), TASK)
            self.assertIn("schema_invalid", row["parse_error"])
            row = r.decode("gpt-6-luna", REQUEST, entry("", finish="length"), TASK)
            self.assertTrue(row["truncated"])

    def test_decode_keeps_failed_and_missing_results_as_failures(self):
        row = r.decode("gpt-6-luna", REQUEST, {"custom_id": "r0001", "response": None,
                                               "error": {"message": "overloaded"}}, TASK)
        self.assertIn("call_failed", row["parse_error"])
        self.assertIn("overloaded", row["parse_error"])
        self.assertIn("no result", r.decode("gpt-6-luna", REQUEST, None, TASK)["parse_error"])

    def test_decode_reads_sync_responses_too(self):
        sync = {"custom_id": "r0001", "response": entry('{"label": "a"}')["response"]["body"]}
        with patch.object(r.benchmark, "parse_content", return_value=({"label": "a"}, None)):
            self.assertEqual(r.decode("beam-1.0", REQUEST, sync, TASK)["pred_label"], "a")


    def test_sync_retries_transient_errors_but_not_client_errors(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, "ROOT", Path(folder)), \
                patch.object(r.time, "sleep"), patch.object(r, "committed_usd", return_value=0.0):
            ok = {"usage": {"cost": 0.01}}
            with patch.object(r, "http", side_effect=[urllib.error.URLError("down"), ok]) as call:
                r.run_sync("k", "beam-1.0", "pilot", [{"custom_id": "a", "body": {}}], 0.0, 50.0)
            self.assertEqual(call.call_count, 2)
            bad = urllib.error.HTTPError("u", 400, "no", {}, io.BytesIO(b"bad"))
            with patch.object(r, "http", side_effect=bad) as call:
                r.run_sync("k", "beam-1.0", "remainder", [{"custom_id": "b", "body": {}}], 0.0, 50.0)
            self.assertEqual(call.call_count, 1)
            rows = r.read(r.state_path("beam-1.0", "remainder").parent / "results.json")["results"]
            self.assertIn("HTTP 400", rows[0]["error"])


    def test_sync_stops_without_writing_rows_when_balance_runs_out(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, "ROOT", Path(folder)), \
                patch.object(r.time, "sleep"), patch.object(r, "committed_usd", return_value=0.0):
            broke = urllib.error.HTTPError("u", 402, "no", {}, io.BytesIO(b"balance"))
            with patch.object(r, "http", side_effect=broke), self.assertRaises(SystemExit) as stop:
                r.run_sync("k", "beam-1.0", "pilot", [{"custom_id": "a", "body": {}}], 0.0, 50.0)
            self.assertEqual(stop.exception.code, r.NO_BALANCE_EXIT)
            calls = r.state_path("beam-1.0", "pilot").parent / "calls.jsonl"
            self.assertEqual(calls.read_text(), "")

    def test_retry_resubmits_only_requests_without_an_answer(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r, "ROOT", Path(folder)):
            path = r.state_path("gpt-6-luna", "remainder")
            r.save(path, {"status": "collected", "mode": "batch", "cost_usd": 1.0, "batch_ids": ["a"],
                          "refused": ["HTTP 402"]})
            good = entry('{"label": "a"}')
            r.save(path.parent / "results.json", {"results": [good, {"custom_id": "r0002", "error": "x"}]})
            reqs = [{"custom_id": c, "task": "t", "body": {}} for c in ("r0001", "r0002", "r0003")]
            posted = []
            with patch.object(r, "load_requests", return_value=(reqs, {})), \
                    patch.object(r, "reservation", return_value=0.1), \
                    patch.object(r, "post_batch", side_effect=lambda k, p: posted.append(p) or {"id": "b"}):
                r.retry_missing("k", "gpt-6-luna", "remainder", 50.0)
            self.assertEqual([q["custom_id"] for q in posted[0]["requests"]], ["r0002", "r0003"])
            state = r.read(path)
            self.assertEqual((state["status"], state["batch_ids"], state["refused"]), ("submitted", ["a", "b"], []))


if __name__ == "__main__":
    unittest.main()
