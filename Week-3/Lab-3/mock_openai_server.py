#!/usr/bin/env python3
"""Deterministic local OpenAI-compatible server for the smoke test only."""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


GOLD = {
    "nli-001": "entailment",
    "nli-002": "contradiction",
    "nli-003": "neutral",
    "nli-004": "entailment",
    "nli-005": "contradiction",
    "nli-006": "neutral",
    "nli-007": "entailment",
    "nli-008": "contradiction",
    "nli-009": "neutral",
    "nli-010": "entailment",
    "nli-011": "contradiction",
    "nli-012": "neutral",
}
LABELS = ("entailment", "neutral", "contradiction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--model", default="task")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--first-token-delay", type=float, default=0.003)
    parser.add_argument("--token-delay", type=float, default=0.0005)
    return parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Lab3Mock/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        return

    @property
    def settings(self) -> argparse.Namespace:
        return self.server.settings  # type: ignore[attr-defined]

    def send_json(self, status: int, value: Any) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.settings.model,
                            "object": "model",
                            "owned_by": "lab3-mock",
                        }
                    ],
                },
            )
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length))
            text = body["messages"][-1]["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.send_json(400, {"error": {"message": "invalid request"}})
            return
        if body.get("model") != self.settings.model:
            self.send_json(404, {"error": {"message": "unknown model"}})
            return
        output, token_parts = self.generate(text, body)
        completion_tokens = len(token_parts)
        usage = {
            "prompt_tokens": max(1, len(text.split())),
            "completion_tokens": completion_tokens,
            "total_tokens": max(1, len(text.split())) + completion_tokens,
        }
        if body.get("stream"):
            self.stream(output, token_parts, usage)
        else:
            self.send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "model": self.settings.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )

    def generate(self, prompt: str, body: dict[str, Any]) -> tuple[str, list[str]]:
        if "Benchmark request:" in prompt:
            count = int(body.get("max_tokens", 16))
            parts = [f"token{i} " for i in range(count)]
            return "".join(parts), parts

        match = re.search(r"Example ID:\s*(nli-\d+)", prompt)
        example_id = match.group(1) if match else "nli-001"
        label = GOLD[example_id]
        number = int(example_id.rsplit("-", 1)[1])
        if self.settings.precision == "fp8" and number % 5 == 0:
            label = LABELS[(LABELS.index(label) + 1) % len(LABELS)]

        wants_json = "Return only JSON" in prompt
        constrained = "guided_json" in body or "structured_outputs" in body
        if wants_json and not constrained and number % 4 == 0:
            output = f"The label is {label}."  # Deliberately invalid JSON.
        elif wants_json:
            output = json.dumps({"label": label, "confidence": 0.9})
        else:
            output = label
        return output, [output]

    def stream(
        self, output: str, token_parts: list[str], usage: dict[str, int]
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        def emit(event: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.flush()

        emit(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            }
        )
        time.sleep(self.settings.first_token_delay)
        for index, part in enumerate(token_parts):
            if index:
                time.sleep(self.settings.token_delay)
            emit(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": part}}],
                }
            )
        emit(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class MockServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def main() -> None:
    args = parse_args()
    server = MockServer((args.host, args.port), Handler)
    server.settings = args  # type: ignore[attr-defined]
    print(
        f"Mock server listening on http://{args.host}:{args.port}/v1 "
        f"(precision={args.precision})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
