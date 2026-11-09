#!/usr/bin/env python3
"""Ask for a JSON answer twice: once with the prompt only, once with a JSON schema.

The schema request uses the standard OpenAI field
response_format={"type": "json_schema", ...}, which SGLang supports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab3lib import (
    DEFAULT_TASK_CONFIG,
    Task,
    chat_completion,
    load_server_config,
    utc_now,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = load_server_config(args.server_config)
    base_url = f"http://127.0.0.1:{server['port']}/v1"
    task = Task(args.task_config)
    examples = task.test_examples()
    rows = []
    modes = []
    for mode in ("free", "schema"):
        mode_rows = []
        for example in examples:
            payload = {
                "model": server["request_model"],
                "messages": task.json_messages(example),
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            }
            if mode == "schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": task.schema()},
                }
            response = chat_completion(base_url, payload, args.timeout)
            text = response["choices"][0]["message"]["content"] or ""
            valid, reason, label = task.check_json(text)
            row = {
                "id": example["id"],
                "configuration": server["label"],
                "mode": mode,
                "gold": example["label"],
                "prediction": label,
                "correct": label == example["label"],
                "valid": valid,
                "invalid_reason": reason,
                "raw_output": text,
                "observed_at": utc_now(),
            }
            rows.append(row)
            mode_rows.append(row)
        count = len(mode_rows)
        invalid = sum(not row["valid"] for row in mode_rows)
        modes.append(
            {
                "mode": mode,
                "example_count": count,
                "invalid_count": invalid,
                "invalid_rate": invalid / count if count else None,
                "task_score": sum(row["correct"] for row in mode_rows) / count if count else None,
            }
        )
    summary = {
        "configuration": server["label"],
        "invalid_definition": 'not JSON, not exactly {"label": ...}, or a label outside the task labels',
        "scoring": "an invalid answer counts as wrong",
        "schema": task.schema(),
        "model": server["model"],
        "model_revision": server["revision"],
        "adapter": server.get("adapter"),
        "seed": args.seed,
        "modes": modes,
    }
    write_jsonl(args.output_dir / "structured-observations.jsonl", rows)
    write_json(args.output_dir / "structured-summary.json", summary)
    for item in modes:
        print(
            f"{item['mode']}: invalid={item['invalid_rate']:.1%}, "
            f"accuracy={item['task_score']:.3f} (n={item['example_count']})"
        )


if __name__ == "__main__":
    main()
