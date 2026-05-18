#!/usr/bin/env python3
"""Tiny OpenAI-compatible local server for smoke-testing custom benchmarks.

This is not a model. It is a deterministic test double that implements just
enough of `/v1/chat/completions` for the bundled minimal custom task.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


POLITICAL_TERMS = {
    "bill",
    "candidate",
    "election",
    "government",
    "policy",
    "politician",
    "referendum",
    "voter",
    "voters",
}


def classify_relevance(messages: list[dict]) -> int:
    user_text = ""
    for message in messages:
        if message.get("role") == "user":
            user_text = str(message.get("content", ""))
    words = user_text.lower().replace(".", " ").replace(",", " ").split()
    return int(any(word in POLITICAL_TERMS for word in words))


class StubHandler(BaseHTTPRequestHandler):
    server_version = "PolSciOpenBenchStub/0.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json({"error": f"unsupported path: {self.path}"}, status=404)
            return

        length = int(self.headers.get("content-length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self.send_json({"error": f"invalid json: {exc}"}, status=400)
            return

        relevant = classify_relevance(payload.get("messages", []))
        response = {
            "id": "chatcmpl-local-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "local-openai-stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant": relevant}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        self.send_json(response)
        self.server.request_count += 1
        max_requests = self.server.max_requests
        if max_requests and self.server.request_count >= max_requests:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib name
        return

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Exit after serving this many chat-completion requests. 0 means run until interrupted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), StubHandler)
    server.request_count = 0
    server.max_requests = args.max_requests
    print(f"Serving OpenAI-compatible stub on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
