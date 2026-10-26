#!/usr/bin/env python3
"""Compare free JSON generation with JSON-schema constrained generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lab3lib import (
    TASK_SCHEMA,
    chat_completion,
    extract_label,
    load_jsonl,
    task_prompt,
    utc_now,
    validate_task_json,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="task")
    parser.add_argument("--data", type=Path, default=Path("data/heldout-nli.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="bf16")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter-revision", default="none")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--schema-field",
        choices=["guided_json", "structured_outputs"],
        default="guided_json",
        help="vLLM request spelling supported by the installed version",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def schema_body(field: str) -> dict[str, Any]:
    if field == "guided_json":
        return {"guided_json": TASK_SCHEMA}
    return {"structured_outputs": {"json": TASK_SCHEMA}}


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data)
    observations = []
    summaries = []
    for mode in ("free", "schema"):
        mode_rows = []
        for example in examples:
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": task_prompt(example, json_output=True)}
                ],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            }
            if mode == "schema":
                payload.update(schema_body(args.schema_field))
            response = chat_completion(args.base_url, payload, args.timeout)
            text = response["choices"][0]["message"]["content"]
            valid, invalid_reason = validate_task_json(text)
            prediction = extract_label(text)
            row = {
                "example_id": example["id"],
                "configuration": args.config_label,
                "mode": mode,
                "gold": example["label"],
                "prediction": prediction,
                "correct": prediction == example["label"],
                "valid": valid,
                "invalid_reason": invalid_reason,
                "raw_output": text,
                "seed": args.seed,
                "model": args.model,
                "model_revision": args.model_revision,
                "adapter_revision": args.adapter_revision,
                "schema_field": args.schema_field if mode == "schema" else None,
                "observed_at": utc_now(),
            }
            observations.append(row)
            mode_rows.append(row)
        valid_count = sum(row["valid"] for row in mode_rows)
        correct = sum(row["correct"] for row in mode_rows)
        summaries.append(
            {
                "mode": mode,
                "example_count": len(mode_rows),
                "invalid_count": len(mode_rows) - valid_count,
                "invalid_rate": (
                    (len(mode_rows) - valid_count) / len(mode_rows) if mode_rows else None
                ),
                "task_score": correct / len(mode_rows) if mode_rows else None,
            }
        )

    summary = {
        "configuration": args.config_label,
        "metric": "accuracy",
        "invalid_definition": (
            "not parseable as JSON, not an object with exactly label and confidence, "
            "label outside the enum, or confidence outside [0, 1]"
        ),
        "seed": args.seed,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_revision": args.adapter_revision,
        "schema": TASK_SCHEMA,
        "schema_field": args.schema_field,
        "modes": summaries,
    }
    write_jsonl(args.output_dir / "structured-observations.jsonl", observations)
    write_json(args.output_dir / "structured-summary.json", summary)
    for item in summaries:
        print(
            f"{item['mode']}: invalid={item['invalid_rate']:.1%}, "
            f"accuracy={item['task_score']:.3f} (n={item['example_count']})"
        )


if __name__ == "__main__":
    main()
