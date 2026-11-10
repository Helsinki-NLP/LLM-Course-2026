#!/usr/bin/env python3
"""Evaluation: score the Lab 2 test set, with the Lab 2 prompt and scorer.

Usage:
    python3 score_task.py --backend hf     --task-config ../../Week-2/Lab-2/configs/my-task.json --output-dir results/hf
    python3 score_task.py --backend server --task-config ../../Week-2/Lab-2/configs/my-task.json --output-dir results/vllm

--limit sets how many test examples are used (default 100; 0 means all).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab3lib import (
    DEFAULT_TASK_CONFIG,
    Task,
    load_model_config,
    make_backend,
    utc_now,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["hf", "server"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16, help="batch size (HF) or parallel requests (server)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    task = Task(args.task_config)
    backend = make_backend(args.backend, config, task)
    examples = task.test_examples()
    if args.limit:
        examples = examples[: args.limit]
    messages = [task.messages(example) for example in examples]

    started = time.perf_counter()
    if args.backend == "server":
        with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            outputs = list(pool.map(lambda m: backend.answer(m, args.max_tokens), messages))
    else:
        outputs = []
        for first in range(0, len(messages), args.batch_size):
            outputs += backend.answer_batch(messages[first : first + args.batch_size], args.max_tokens)
    seconds = time.perf_counter() - started

    rows = []
    for example, text in zip(examples, outputs):
        prediction = task.extract(text)
        rows.append(
            {
                "id": example["id"],
                "gold": example["label"],
                "prediction": prediction,
                "correct": prediction == example["label"],
                "raw_output": text,
            }
        )
    correct = sum(row["correct"] for row in rows)
    summary = {
        "backend": args.backend,
        "model": config["model"],
        "revision": config["revision"],
        "quantization": config.get("quantization"),
        "task": task.name,
        "task_config": str(task.config_path),
        "examples": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "no_label": sum(row["prediction"] is None for row in rows),
        "seconds": seconds,
        "created_at": utc_now(),
    }
    write_jsonl(args.output_dir / "eval-answers.jsonl", rows)
    write_json(args.output_dir / "eval-summary.json", summary)
    print(
        f"accuracy {100 * summary['accuracy']:.1f}% on {len(rows)} examples "
        f"({summary['no_label']} without a label), {seconds:.1f} s"
    )
    print(f"Saved to {args.output_dir}/eval-summary.json")


if __name__ == "__main__":
    main()
