#!/usr/bin/env python3
"""Create train, dev and test JSONL files from one labelled data file.

Usage (one file, split randomly with a fixed seed):
    python3 make_splits.py --input mydata.csv --fields text --label-field sentiment \
        --train 2000 --dev 300 --test 500 --out data/my-task

Usage (keep an official test file separate):
    python3 make_splits.py --input train.csv --test-input test.csv \
        --fields premise hypothesis --label-field label \
        --train 2000 --dev 300 --test 500 --out data/my-task

Input files can be .csv, .tsv or .jsonl. Each output record contains an "id",
the chosen input fields and a "label" field. Exact duplicate inputs are removed
before splitting, so the same item cannot end up in two splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8", newline="") as handle:
        if suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise SystemExit(f"{path}: use a .csv, .tsv or .jsonl file")


def convert(rows: list[dict], fields: list[str], label_field: str, source: Path) -> list[dict]:
    converted = []
    for number, row in enumerate(rows, start=1):
        missing = [name for name in (*fields, label_field) if name not in row]
        if missing:
            raise SystemExit(
                f"{source} row {number}: missing column(s) {', '.join(missing)}. "
                f"Available columns: {', '.join(row)}"
            )
        record = {field: str(row[field]).strip() for field in fields}
        record["label"] = str(row[label_field]).strip()
        if all(record[field] for field in fields) and record["label"]:
            converted.append(record)
    return converted


def deduplicate(records: list[dict], fields: list[str], seen: set) -> list[dict]:
    unique = []
    for record in records:
        key = tuple(record[field].lower() for field in fields)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def write(path: Path, prefix: str, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            row = {"id": f"{prefix}-{index:05d}", **record}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Lab 2 data splits.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--test-input", type=Path, help="optional official test file")
    parser.add_argument("--fields", required=True, nargs="+", help="input column(s)")
    parser.add_argument("--label-field", required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set = set()

    test: list[dict] = []
    if args.test_input:
        test = convert(read_rows(args.test_input), args.fields, args.label_field, args.test_input)
        test = deduplicate(test, args.fields, seen)
        rng.shuffle(test)
        test = test[: args.test]

    pool = convert(read_rows(args.input), args.fields, args.label_field, args.input)
    pool = deduplicate(pool, args.fields, seen)
    rng.shuffle(pool)
    if not args.test_input:
        test, pool = pool[: args.test], pool[args.test :]
    dev, pool = pool[: args.dev], pool[args.dev :]
    train = pool[: args.train]

    if len(train) < args.train or len(dev) < args.dev or len(test) < args.test:
        print(
            "WARNING: not enough unique examples for the requested sizes; "
            "the splits are smaller than requested."
        )
    write(args.out / "train.jsonl", "train", train)
    write(args.out / "dev.jsonl", "dev", dev)
    write(args.out / "test.jsonl", "test", test)
    labels = sorted({record["label"] for record in train + dev + test})
    print(f"Wrote {len(train)} train, {len(dev)} dev and {len(test)} test examples to {args.out}/")
    print(f"Seed: {args.seed}. Labels found: {json.dumps(labels, ensure_ascii=False)}")
    print("Copy the labels into task.labels in your config, then run check_data.py.")


if __name__ == "__main__":
    main()
