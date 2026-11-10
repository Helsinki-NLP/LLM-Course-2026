#!/usr/bin/env python3
"""Print all Lab 4 results as Markdown tables.

Usage:
    python3 summarize_results.py results

Every folder inside results/ (for example hf/ and vllm/) becomes rows of the
tables. Paste the output into your report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def num(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    args = parser.parse_args()
    runs = sorted(p for p in args.results_dir.iterdir() if p.is_dir())

    speed_rows = []
    print("### Speed\n")
    print("| Run | Model | Concurrency | First token (ms) | Per token (ms) | Tokens/s (all) |")
    print("|---|---|---:|---:|---:|---:|")
    for run in runs:
        summary = read(run / "speed-summary.json")
        if not summary:
            continue
        model = summary["model"] + (" (FP8)" if summary.get("quantization") == "fp8" else "")
        for level in summary["levels"]:
            print(
                f"| {run.name} | {model} | {level['concurrency']} | {num(level['ttft_ms_p50'], 0)} | "
                f"{num(level['tpot_ms_p50'], 2)} | {num(level['tokens_per_second'], 0)} |"
            )
            speed_rows.append({"run": run.name, "model": model, **level})
    print("\nMedians over the requests; every request produced the same number of tokens.")

    print("\n### Evaluation\n")
    print("| Run | Accuracy (%) | Examples | No label | Time (s) |")
    print("|---|---:|---:|---:|---:|")
    for run in runs:
        summary = read(run / "eval-summary.json")
        if summary:
            print(
                f"| {run.name} | {num(100 * summary['accuracy'])} | {summary['examples']} | "
                f"{summary['no_label']} | {num(summary['seconds'])} |"
            )

    for run in runs:
        structured = read(run / "structured-summary.json")
        if structured:
            print(f"\n### JSON output ({run.name})\n")
            print("| Request | Invalid (%) | Accuracy (%) |")
            print("|---|---:|---:|")
            for mode in structured["modes"]:
                name = {"free": "prompt only", "schema": "with JSON schema", "schema-reasoning": "schema, reasoning first"}[mode["mode"]]
                print(f"| {name} | {num(100 * mode['invalid_rate'])} | {num(100 * mode['task_score'])} |")

    if speed_rows:
        with (args.results_dir / "speed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(speed_rows[0]))
            writer.writeheader()
            writer.writerows(speed_rows)
        print(f"\nSpeed numbers also saved to {args.results_dir / 'speed.csv'}")


if __name__ == "__main__":
    main()
