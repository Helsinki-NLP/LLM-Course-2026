#!/usr/bin/env python3
"""Deterministic local OpenAI-compatible server for the smoke test only."""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


from lab3lib import DEFAULT_TASK_CONFIG, Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
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
                            "id": "task",
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
            text = "\n".join(message["content"] for message in body["messages"])
        except (json.JSONDecodeError, KeyError, TypeError):
            self.send_json(400, {"error": {"message": "invalid request"}})
            return
        if body.get("model", "").split(":")[0] != "task":
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
                    "model": body["model"],
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
        task: Task = self.server.task  # type: ignore[attr-defined]
        index, example = next(
            (
                (number, row)
                for number, row in enumerate(self.server.examples)  # type: ignore[attr-defined]
                if task.prompting.task_example(row, include_answer=False) in prompt
            ),
            (0, self.server.examples[0]),  # type: ignore[attr-defined]
        )
        label = example["label"]
        # Deliberate, deterministic differences so the smoke test can check them.
        if self.settings.precision == "fp8" and index % 3 == 1:
            label = task.labels[(task.labels.index(label) + 1) % len(task.labels)]
        if "Answer with JSON only" in prompt:
            if "response_format" not in body and index % 2 == 0:
                output = f"The label is {label}."  # invalid JSON without a schema
            else:
                output = json.dumps({"label": label})
        else:
            output = f"Label: {label}"
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
    server.task = Task(args.task_config)  # type: ignore[attr-defined]
    server.examples = server.task.test_examples()  # type: ignore[attr-defined]
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
