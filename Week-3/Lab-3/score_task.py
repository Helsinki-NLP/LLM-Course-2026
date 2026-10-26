#!/usr/bin/env python3
"""Score a fixed held-out classification set through the serving endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from lab3lib import (
    chat_completion,
    extract_label,
    load_jsonl,
    task_prompt,
    utc_now,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="task")
    parser.add_argument("--data", type=Path, default=Path("data/heldout-nli.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter-revision", default="none")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data)
    observations = []
    for example in examples:
        prompt = task_prompt(example)
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }
        response = chat_completion(args.base_url, payload, args.timeout)
        text = response["choices"][0]["message"]["content"]
        prediction = extract_label(text)
        observations.append(
            {
                "example_id": example["id"],
                "configuration": args.config_label,
                "gold": example["label"],
                "prediction": prediction,
                "correct": prediction == example["label"],
                "raw_output": text,
                "seed": args.seed,
                "model": args.model,
                "model_revision": args.model_revision,
                "adapter_revision": args.adapter_revision,
                "observed_at": utc_now(),
            }
        )

    correct = sum(row["correct"] for row in observations)
    summary = {
        "configuration": args.config_label,
        "metric": "accuracy",
        "score": correct / len(observations) if observations else None,
        "correct": correct,
        "example_count": len(observations),
        "seed": args.seed,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_revision": args.adapter_revision,
        "data": str(args.data),
    }
    write_jsonl(args.output_dir / "task-observations.jsonl", observations)
    write_json(args.output_dir / "task-summary.json", summary)
    print(f"{args.config_label}: accuracy={summary['score']:.3f} (n={len(observations)})")


if __name__ == "__main__":
    main()
