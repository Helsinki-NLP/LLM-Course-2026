#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lab2lib.common import load_config
from lab2lib.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bf16 LoRA adapter with TRL.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = train(load_config(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
