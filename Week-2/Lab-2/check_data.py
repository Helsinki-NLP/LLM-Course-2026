#!/usr/bin/env python3
"""Check the train, dev and test files before running any model.

Usage:
    python3 check_data.py --config configs/my-task.json

The script reports problems that would otherwise appear only as strange
scores: missing fields, unknown labels, repeated IDs, and items that occur in
more than one split. It uses only the Python standard library, so it can run on
a login node.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from lab2lib.common import load_config, load_knowledge, load_split
from lab2lib.prompting import TASK_FIELDS


def item_text(record: dict) -> str:
    return " ||| ".join(str(record.get(field, "")).strip().lower() for field in TASK_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Lab 2 data splits.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    labels = config["task"]["labels"]
    few_shot = int(config["prompting"]["few_shot_count"])

    errors: list[str] = []
    warnings: list[str] = []
    splits = {name: load_split(config, name) for name in ("train", "dev", "test")}
    seen_ids: dict[str, str] = {}
    texts: dict[str, set[str]] = {}

    print(f"Input fields (TASK_FIELDS in lab2lib/prompting.py): {', '.join(TASK_FIELDS)}")
    print(f"Labels (task.labels in {args.config}): {', '.join(labels)}\n")

    for name, records in splits.items():
        label_counts: Counter[str] = Counter()
        texts[name] = set()
        for number, record in enumerate(records, start=1):
            where = f"{name} line {number}"
            for field in ("id", "label", *TASK_FIELDS):
                if field not in record or str(record[field]).strip() == "":
                    errors.append(f"{where}: missing or empty field '{field}'")
            if "label" in record and record["label"] not in labels:
                errors.append(
                    f"{where}: label '{record['label']}' is not in task.labels"
                )
            if "id" in record:
                identifier = str(record["id"])
                if identifier in seen_ids:
                    errors.append(
                        f"{where}: id '{identifier}' also appears in {seen_ids[identifier]}"
                    )
                seen_ids.setdefault(identifier, where)
            text = item_text(record)
            if text in texts[name]:
                warnings.append(f"{where}: the same input appears twice in {name}")
            texts[name].add(text)
            label_counts[record.get("label")] += 1

        print(f"{name}: {len(records)} examples")
        for label in labels:
            count = label_counts[label]
            share = count / len(records) if records else 0.0
            print(f"  {label:<20} {count:>6}  ({share:.0%})")
            if records and count == 0:
                warnings.append(f"{name}: no examples with label '{label}'")
        print()

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = texts[first] & texts[second]
        if overlap:
            errors.append(
                f"{len(overlap)} input(s) occur in both {first} and {second}"
            )

    if len(splits["train"]) < few_shot:
        errors.append(
            f"train has {len(splits['train'])} examples but few_shot_count is {few_shot}"
        )
    if len(splits["train"]) < 500:
        warnings.append(
            f"train has only {len(splits['train'])} examples; aim for 1,000-5,000 "
            "for a meaningful fine-tuning run"
        )
    if len(splits["test"]) < 200:
        warnings.append(
            f"test has only {len(splits['test'])} examples; scores will be noisy"
        )

    knowledge = load_knowledge(config)
    print(f"general-knowledge questions: {len(knowledge)}\n")

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        print(f"\n{len(errors)} error(s) found. Fix them before running the experiment.")
        sys.exit(1)
    print("No errors found." + (" Read the warnings above." if warnings else ""))


if __name__ == "__main__":
    main()
