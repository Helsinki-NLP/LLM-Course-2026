#!/usr/bin/env python3
"""Print the Lab 3 results as Markdown tables and write speed-quality.csv.

Usage:
    python3 summarize_results.py results/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def num(value: Any, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir

    labels = [name for name in ("bf16", "fp8") if (root / name / "benchmark-summary.json").exists()]
    if not labels:
        raise SystemExit(f"No results found in {root}")

    first = read(root / labels[0] / "benchmark-summary.json")
    print(f"## Lab 3 results: {root}\n")
    print(
        f"Model: {first.get('model_id', first['model'])} (revision {first['model_revision']}). "
        f"{first['requests_per_level']} requests per level, {first['max_tokens']} output tokens each, "
        f"{first['warmup_requests_discarded']} warm-up requests discarded.\n"
    )

    print("### Speed\n")
    print("| Precision | Concurrency | Time to first token (ms, median) | Time per output token (ms, median) | Output tokens/s (all requests) | Failed |")
    print("|---|---:|---:|---:|---:|---:|")
    csv_rows = []
    for label in labels:
        benchmark = read(root / label / "benchmark-summary.json")
        task = read(root / label / "task-summary.json") or {}
        for level in benchmark["levels"]:
            print(
                f"| {label} | {level['concurrency']} | {num(level['ttft_ms_p50'])} | "
                f"{num(level['tpot_ms_p50'], 2)} | {num(level['output_tokens_per_second'], 0)} | "
                f"{level['requests_failed']} |"
            )
            csv_rows.append(
                {
                    "precision": label,
                    "concurrency": level["concurrency"],
                    "ttft_ms_p50": level["ttft_ms_p50"],
                    "ttft_ms_p95": level["ttft_ms_p95"],
                    "tpot_ms_p50": level["tpot_ms_p50"],
                    "tpot_ms_p95": level["tpot_ms_p95"],
                    "output_tokens_per_second": level["output_tokens_per_second"],
                    "requests_failed": level["requests_failed"],
                    "task_accuracy": task.get("score"),
                }
            )

    print("\n### Quality on the test set\n")
    print("| Precision | Accuracy (%) | Examples | No label found |")
    print("|---|---:|---:|---:|")
    for label in labels:
        task = read(root / label / "task-summary.json")
        if task:
            print(
                f"| {label} | {num(100 * task['score'])} | {task['example_count']} | {task['no_label']} |"
            )

    structured = read(root / "bf16" / "structured-summary.json")
    if structured:
        print("\n### JSON output (bf16)\n")
        print("| Request | Invalid outputs (%) | Accuracy (%) |")
        print("|---|---:|---:|")
        names = {"free": "prompt only", "schema": "with JSON schema"}
        for mode in structured["modes"]:
            print(
                f"| {names[mode['mode']]} | {num(100 * mode['invalid_rate'])} | "
                f"{num(100 * mode['task_score'])} |"
            )

    with (root / "speed-quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {root / 'speed-quality.csv'}")
    if "mock" in str(first.get("model_id")):
        print("Note: mock server. These numbers test the pipeline and are not real measurements.")


if __name__ == "__main__":
    main()
