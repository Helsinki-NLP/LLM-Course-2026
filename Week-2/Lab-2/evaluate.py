#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lab2lib.common import load_config
from lab2lib.evaluation import (
    evaluate_baselines,
    evaluate_forgetting,
    evaluate_sft,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Lab 2 models.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target",
        required=True,
        choices=("baselines", "sft", "forgetting-before", "forgetting-after"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target == "baselines":
        result = evaluate_baselines(config)
    elif args.target == "sft":
        result = evaluate_sft(config)
    else:
        result = evaluate_forgetting(config, after=args.target.endswith("after"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
