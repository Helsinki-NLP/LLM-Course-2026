"""Plot final validation loss against estimated training compute."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def final_validation_loss(metrics_path: Path) -> float:
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values = [record["valid_loss"] for record in records if "valid_loss" in record]
    if not values:
        raise ValueError(f"No validation result in {metrics_path}")
    return float(values[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/scaling.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run.name,
                "parameters": summary["parameters"],
                "tokens": summary["actual_train_tokens"],
                "estimated_flops": summary["estimated_training_flops"],
                "valid_loss": final_validation_loss(run / "metrics.jsonl"),
            }
        )
    rows.sort(key=lambda row: row["estimated_flops"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [row["estimated_flops"] for row in rows],
        [row["valid_loss"] for row in rows],
        marker="o",
    )
    for row in rows:
        axis.annotate(
            f"{row['run']} ({row['parameters'] / 1e6:.1f}M)",
            (row["estimated_flops"], row["valid_loss"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Estimated training compute, 6ND (FLOPs)")
    axis.set_ylabel("Final validation loss")
    axis.set_title("Lab 1 scaling experiment")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
