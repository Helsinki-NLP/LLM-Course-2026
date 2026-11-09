#!/usr/bin/env python3
"""Score the Lab 2 test set through the running server (same prompt and scorer as Lab 2)."""

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
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = load_server_config(args.server_config)
    base_url = f"http://127.0.0.1:{server['port']}/v1"
    task = Task(args.task_config)
    rows = []
    for example in task.test_examples():
        payload = {
            "model": server["request_model"],
            "messages": task.messages(example),
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }
        response = chat_completion(base_url, payload, args.timeout)
        text = response["choices"][0]["message"]["content"] or ""
        prediction = task.extract(text)
        rows.append(
            {
                "id": example["id"],
                "configuration": server["label"],
                "gold": example["label"],
                "prediction": prediction,
                "correct": prediction == example["label"],
                "raw_output": text,
                "observed_at": utc_now(),
            }
        )
    correct = sum(row["correct"] for row in rows)
    summary = {
        "configuration": server["label"],
        "metric": "accuracy",
        "score": correct / len(rows) if rows else None,
        "correct": correct,
        "example_count": len(rows),
        "no_label": sum(row["prediction"] is None for row in rows),
        "task": task.name,
        "task_config": str(task.config_path),
        "test_data": str(task.test_path),
        "model": server["model"],
        "model_revision": server["revision"],
        "adapter": server.get("adapter"),
        "seed": args.seed,
    }
    write_jsonl(args.output_dir / "task-observations.jsonl", rows)
    write_json(args.output_dir / "task-summary.json", summary)
    print(f"{server['label']}: accuracy={summary['score']:.3f} (n={len(rows)})")


if __name__ == "__main__":
    main()
