#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil

from lab2lib.common import (
    initialize_results,
    load_config,
    output_dir,
    read_results,
    set_seed,
)
from lab2lib.evaluation import (
    evaluate_baselines,
    evaluate_forgetting,
    evaluate_sft,
)
from lab2lib.training import train
from plot_losses import plot


def run(config_path: str, fresh: bool = False) -> dict:
    config = load_config(config_path)
    destination = output_dir(config)
    if fresh and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    initialize_results(config)

    print("1/6 Scoring four prompting baselines")
    evaluate_baselines(config)
    print("2/6 Scoring general knowledge before SFT")
    evaluate_forgetting(config, after=False)
    print("3/6 Training LoRA adapter")
    train(config)
    print("4/6 Scoring LoRA adapter")
    evaluate_sft(config)
    print("5/6 Scoring general knowledge after SFT")
    evaluate_forgetting(config, after=True)
    print("6/6 Plotting train and evaluation loss")
    plot(config)
    return read_results(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Lab 2 experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove this config's output directory before running.",
    )
    args = parser.parse_args()
    result = run(args.config, fresh=args.fresh)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
