#!/usr/bin/env python3
"""Print raw model outputs next to the extracted prediction and the gold label.

Usage:
    python3 show_examples.py --config configs/my-task.json
    python3 show_examples.py --config configs/my-task.json --run base-zero --n 20
    python3 show_examples.py --config configs/my-task.json --run lora-sft --wrong

Without --run, the script lists the available runs. Read these outputs before
you trust any score: a parser that misreads the model's answer gives a wrong
score without any error message.
"""
from __future__ import annotations

import argparse

from lab2lib.common import load_config, output_dir, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Show raw generations for one run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", help="run name, e.g. instruct-zero or lora-sft")
    parser.add_argument("--n", type=int, default=10, help="number of examples")
    parser.add_argument("--wrong", action="store_true", help="show only errors")
    parser.add_argument("--prompt", action="store_true", help="also print the prompt")
    args = parser.parse_args()

    folder = output_dir(load_config(args.config)) / "predictions"
    available = sorted(path.stem for path in folder.glob("*.jsonl"))
    if not args.run:
        print("Available runs:\n  " + "\n  ".join(available))
        print("\nChoose one with --run NAME")
        return
    if args.run not in available:
        raise SystemExit(f"Unknown run '{args.run}'. Available: {', '.join(available)}")

    rows = read_jsonl(folder / f"{args.run}.jsonl")
    if args.wrong:
        rows = [row for row in rows if not row["correct"]]
    for row in rows[: args.n]:
        mark = "correct" if row["correct"] else "WRONG"
        print("=" * 72)
        print(f"{row['id']}  gold={row['gold']}  prediction={row['prediction']}  [{mark}]")
        if args.prompt:
            print("--- prompt ---")
            print(row["prompt"])
        print("--- raw output ---")
        print(row["raw_output"])
    print("=" * 72)
    print(f"Shown {min(args.n, len(rows))} of {len(rows)} rows from {args.run}.")


if __name__ == "__main__":
    main()
