"""Plot training and validation loss from one Lab 1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Run directory containing metrics.jsonl")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.run / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No metrics found in {metrics_path}")

    train_records = [
        record
        for record in records
        if "tokens" in record and "train_loss" in record
    ]
    valid_records = [
        record
        for record in records
        if "tokens" in record and "valid_loss" in record
    ]
    if not train_records or not valid_records:
        raise ValueError(
            f"{metrics_path} must contain both training and validation losses"
        )

    output = args.output or Path("results") / f"{args.run.name}-loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [record["tokens"] / 1e6 for record in train_records],
        [record["train_loss"] for record in train_records],
        label="training",
    )
    axis.plot(
        [record["tokens"] / 1e6 for record in valid_records],
        [record["valid_loss"] for record in valid_records],
        marker="o",
        label="validation",
    )
    axis.set_xlabel("Training tokens (millions)")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(f"Training and validation loss: {args.run.name}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"Wrote {output}")
    print(f"Final training loss: {train_records[-1]['train_loss']:.4f}")
    print(f"Final validation loss: {valid_records[-1]['valid_loss']:.4f}")


if __name__ == "__main__":
    main()
