#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from lab2lib.common import (
    load_config,
    output_dir,
    read_jsonl,
    update_results,
)


def _points(rows: list[dict], key: str) -> list[tuple[float, float]]:
    return [
        (float(row["step"]), float(row[key]))
        for row in rows
        if key in row and "step" in row
    ]


def _polyline(
    points: Iterable[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    coordinates = []
    for x, y in points:
        px = 55 + 700 * (x - x_min) / max(x_max - x_min, 1e-9)
        py = 25 + 360 * (y_max - y) / max(y_max - y_min, 1e-9)
        coordinates.append(f"{px:.1f},{py:.1f}")
    return " ".join(coordinates)


def write_svg(
    destination: Path,
    train: list[tuple[float, float]],
    evaluation: list[tuple[float, float]],
) -> None:
    all_points = train + evaluation
    if not all_points:
        raise ValueError("No train or evaluation losses found")
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    margin = max((y_max - y_min) * 0.1, 0.05)
    y_min, y_max = y_min - margin, y_max + margin
    train_line = _polyline(train, x_min, x_max, y_min, y_max)
    eval_line = _polyline(evaluation, x_min, x_max, y_min, y_max)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
<rect width="800" height="450" fill="white"/>
<text x="400" y="18" text-anchor="middle" font-family="sans-serif" font-size="16">LoRA SFT loss</text>
<line x1="55" y1="25" x2="55" y2="385" stroke="black"/>
<line x1="55" y1="385" x2="755" y2="385" stroke="black"/>
<text x="400" y="430" text-anchor="middle" font-family="sans-serif">step</text>
<text x="16" y="205" text-anchor="middle" transform="rotate(-90 16 205)" font-family="sans-serif">loss</text>
<polyline points="{train_line}" fill="none" stroke="#0066cc" stroke-width="3"/>
<polyline points="{eval_line}" fill="none" stroke="#d94801" stroke-width="3"/>
<line x1="575" y1="35" x2="610" y2="35" stroke="#0066cc" stroke-width="3"/>
<text x="618" y="40" font-family="sans-serif">train</text>
<line x1="575" y1="58" x2="610" y2="58" stroke="#d94801" stroke-width="3"/>
<text x="618" y="63" font-family="sans-serif">evaluation</text>
<text x="55" y="404" font-family="sans-serif" font-size="11">{x_min:g}</text>
<text x="755" y="404" text-anchor="end" font-family="sans-serif" font-size="11">{x_max:g}</text>
<text x="48" y="385" text-anchor="end" font-family="sans-serif" font-size="11">{y_min:.2f}</text>
<text x="48" y="33" text-anchor="end" font-family="sans-serif" font-size="11">{y_max:.2f}</text>
</svg>
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


def plot(config: dict) -> Path:
    metrics_path = output_dir(config) / "training_metrics.jsonl"
    rows = read_jsonl(metrics_path)
    train = _points(rows, "loss")
    evaluation = _points(rows, "eval_loss")
    if not train or not evaluation:
        raise ValueError("Both training and evaluation losses are required")
    extension = config["plot"].get("format", "png").lower()
    destination = output_dir(config) / f"loss.{extension}"
    if extension == "svg":
        write_svg(destination, train, evaluation)
    else:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(*zip(*train), marker=".", label="train")
        axis.plot(*zip(*evaluation), marker="o", label="evaluation")
        axis.set(xlabel="step", ylabel="loss", title="LoRA SFT loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
    update_results(
        config,
        {"artifacts": {"loss_plot": str(destination.relative_to(output_dir(config)))}},
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train and evaluation loss.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    destination = plot(load_config(args.config))
    print(destination)


if __name__ == "__main__":
    main()
