#!/usr/bin/env python3
"""Build the compact speed-versus-quality table from experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from lab3lib import utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--fp8-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def speed_row(benchmark: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    levels = {int(level["concurrency"]): level for level in benchmark["levels"]}
    c1 = levels[1]
    return {
        "configuration": benchmark["configuration"],
        "ttft_ms_p50_c1": c1["ttft_ms_p50"],
        "ttft_ms_p95_c1": c1["ttft_ms_p95"],
        "tpot_ms_p50_c1": c1["tpot_ms_p50"],
        "tpot_ms_p95_c1": c1["tpot_ms_p95"],
        "output_tok_s_c1": levels[1]["output_tokens_per_second"],
        "output_tok_s_c8": levels[8]["output_tokens_per_second"],
        "output_tok_s_c32": levels[32]["output_tokens_per_second"],
        "task_score": task["score"],
        "invalid_rate": "",
        "requests_per_level": benchmark["requests_per_level"],
        "warmup_requests": benchmark["warmup_requests_discarded"],
        "seed": benchmark["seed"],
        "model": benchmark["model"],
        "model_revision": benchmark["model_revision"],
        "adapter_revision": benchmark["adapter_revision"],
    }


def main() -> None:
    args = parse_args()
    bf16_benchmark = read(args.bf16_dir / "benchmark-summary.json")
    fp8_benchmark = read(args.fp8_dir / "benchmark-summary.json")
    bf16_task = read(args.bf16_dir / "task-summary.json")
    fp8_task = read(args.fp8_dir / "task-summary.json")
    structured = read(args.structured_dir / "structured-summary.json")
    structured_modes = {item["mode"]: item for item in structured["modes"]}

    bf16 = speed_row(bf16_benchmark, bf16_task)
    fp8 = speed_row(fp8_benchmark, fp8_task)
    bf16["invalid_rate"] = structured_modes["free"]["invalid_rate"]
    constrained = {
        key: "" for key in bf16
    }
    constrained.update(
        {
            "configuration": f"{structured['configuration']} + JSON schema",
            "task_score": structured_modes["schema"]["task_score"],
            "invalid_rate": structured_modes["schema"]["invalid_rate"],
            "seed": structured["seed"],
            "model": structured["model"],
            "model_revision": structured["model_revision"],
            "adapter_revision": structured["adapter_revision"],
        }
    )
    rows = [bf16, fp8, constrained]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "speed-quality.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bf16))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output_dir / "experiment-summary.json",
        {
            "created_at": utc_now(),
            "compact_table": rows,
            "notes": [
                "Compact TTFT/TPOT columns use concurrency 1.",
                "Per-concurrency latency percentiles remain in each benchmark-summary.json.",
                "Schema decoding speed is blank because this protocol prioritizes syntax and score.",
            ],
            "source_summaries": {
                "bf16_benchmark": str(args.bf16_dir / "benchmark-summary.json"),
                "fp8_benchmark": str(args.fp8_dir / "benchmark-summary.json"),
                "bf16_task": str(args.bf16_dir / "task-summary.json"),
                "fp8_task": str(args.fp8_dir / "task-summary.json"),
                "structured": str(args.structured_dir / "structured-summary.json"),
            },
        },
    )
    print(f"Wrote {args.output_dir / 'speed-quality.csv'}")


if __name__ == "__main__":
    main()
