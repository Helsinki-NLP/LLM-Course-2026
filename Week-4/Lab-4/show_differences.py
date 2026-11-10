#!/usr/bin/env python3
"""List the test examples where two runs give different answers.

Usage:
    python3 show_differences.py results/hf results/vllm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab4lib import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--n", type=int, default=10, help="how many to print")
    args = parser.parse_args()

    a = {row["id"]: row for row in load_jsonl(args.first / "eval-answers.jsonl")}
    b = {row["id"]: row for row in load_jsonl(args.second / "eval-answers.jsonl")}
    shared = [key for key in a if key in b]
    different = [key for key in shared if a[key]["prediction"] != b[key]["prediction"]]
    only_a = sum(a[k]["correct"] and not b[k]["correct"] for k in different)
    only_b = sum(b[k]["correct"] and not a[k]["correct"] for k in different)
    print(
        f"{len(different)} of {len(shared)} answers differ. Right only in {args.first.name}: {only_a}; "
        f"right only in {args.second.name}: {only_b}.\n"
    )
    for key in different[: args.n]:
        print(f"{key}  gold={a[key]['gold']}")
        for name, rows in ((args.first.name, a), (args.second.name, b)):
            output = " ".join(rows[key]["raw_output"].split())[:80]
            print(f"  {name:<10} {rows[key]['prediction']!s:<14} {output}")


if __name__ == "__main__":
    main()
