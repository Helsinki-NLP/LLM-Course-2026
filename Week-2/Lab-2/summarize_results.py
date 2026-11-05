#!/usr/bin/env python3
"""Print the Lab 2 results table in Markdown.

Usage:
    python3 summarize_results.py --config configs/my-task.json

Paste the output into your notes or report and add a caption with the
dataset, split sizes, metric and model IDs.
"""
from __future__ import annotations

import argparse

from lab2lib.common import load_config, read_results


def pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise Lab 2 results.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = read_results(load_config(args.config))

    splits = result["data"]["splits"]
    models = result["reproducibility"]["requested_models"]
    best = result.get("best_prompt", {}).get("name")

    print(f"## {result['experiment_name']}\n")
    print(
        f"Splits: train {splits['train']}, dev {splits['dev']}, test {splits['test']}, "
        f"general knowledge {splits['knowledge']}. Metric: {result['data']['metric']} (%).\n"
    )
    print(f"Base model: {models['base']['id']} ({models['base']['revision']})  ")
    print(f"Instruct model: {models['instruct']['id']} ({models['instruct']['revision']})\n")

    print("| Setting | Dev | Test |")
    print("|---|---:|---:|")
    for run in result.get("baselines", []):
        marker = " **(selected on dev)**" if run["name"] == best else ""
        print(f"| {run['name']}{marker} | {pct(run.get('dev_score'))} | {pct(run['score'])} |")
    sft = result.get("sft_evaluation")
    if sft:
        print(f"| LoRA SFT (instruct + adapter, zero-shot) | – | {pct(sft['score'])} |")

    forgetting = result.get("forgetting", {})
    if forgetting:
        before = forgetting.get("before", {}).get("score")
        after = forgetting.get("after", {}).get("score")
        print("\n| General knowledge | Accuracy |")
        print("|---|---:|")
        print(f"| before SFT (instruct) | {pct(before)} |")
        print(f"| after SFT | {pct(after)} |")
        if "delta" in forgetting:
            print(f"| change | {100 * forgetting['delta']:+.1f} points |")

    training = result.get("training")
    if training:
        settings = result["config"].get("training", {})
        print(
            "\nTraining: "
            f"rank {settings.get('lora_r')}, alpha {settings.get('lora_alpha')}, "
            f"learning rate {settings.get('learning_rate')}, "
            f"epochs {settings.get('num_train_epochs')}, "
            f"trainable parameters {training.get('trainable_parameters')}, "
            f"final train loss {training.get('final_train_loss')}, "
            f"best eval loss {training.get('best_eval_loss')}."
        )
    if result["reproducibility"]["backend"] == "mock":
        print("\nNote: mock backend. These numbers test the pipeline and are not model results.")


if __name__ == "__main__":
    main()
