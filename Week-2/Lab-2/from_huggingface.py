#!/usr/bin/env python3
"""Download a labelled dataset from Hugging Face and save it as one JSONL file.

Usage:
    python3 from_huggingface.py --dataset stanfordnlp/sst2 --out sst2.jsonl

Numeric labels are replaced by their names (0 -> "negative"), because the
model has to write the label as a word. The output file can then be split with
make_splits.py. Run this after the module load line from README step 3.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Hugging Face dataset to JSONL.")
    parser.add_argument("--dataset", required=True, help="e.g. stanfordnlp/sst2")
    parser.add_argument("--config", default=None, help="dataset configuration, if any")
    parser.add_argument("--split", default="train", help="which split to download")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from datasets import ClassLabel, load_dataset

    data = load_dataset(args.dataset, args.config, split=args.split)
    feature = data.features.get(args.label_field)
    names = feature.names if isinstance(feature, ClassLabel) else None

    written = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in data:
            value = row[args.label_field]
            if names is not None:
                if value is None or value < 0:
                    continue  # unlabelled example, e.g. a hidden test label
                row[args.label_field] = names[value]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {args.out}")
    print(f"Columns: {', '.join(data.column_names)}")
    if names is not None:
        print(f"Labels: {', '.join(names)}")
    print("Next: make_splits.py with --fields set to the text column(s).")


if __name__ == "__main__":
    main()
