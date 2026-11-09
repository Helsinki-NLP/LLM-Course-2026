#!/usr/bin/env python3
"""List the test items where bf16 and FP8 give different answers.

Usage:
    python3 show_differences.py results/<run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab3lib import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--n", type=int, default=10, help="how many to print")
    args = parser.parse_args()

    bf16 = {row["id"]: row for row in load_jsonl(args.results_dir / "bf16" / "task-observations.jsonl")}
    fp8 = {row["id"]: row for row in load_jsonl(args.results_dir / "fp8" / "task-observations.jsonl")}
    different = [key for key in bf16 if key in fp8 and bf16[key]["prediction"] != fp8[key]["prediction"]]

    fp8_better = sum(fp8[key]["correct"] and not bf16[key]["correct"] for key in different)
    bf16_better = sum(bf16[key]["correct"] and not fp8[key]["correct"] for key in different)
    print(f"{len(different)} of {len(bf16)} answers differ: bf16 right and FP8 wrong in {bf16_better}, "
          f"FP8 right and bf16 wrong in {fp8_better}.\n")
    for key in different[: args.n]:
        print(f"{key}  gold={bf16[key]['gold']}")
        print(f"  bf16: {bf16[key]['prediction']}   output: {' '.join(bf16[key]['raw_output'].split())[:80]}")
        print(f"  fp8:  {fp8[key]['prediction']}   output: {' '.join(fp8[key]['raw_output'].split())[:80]}")


if __name__ == "__main__":
    main()
