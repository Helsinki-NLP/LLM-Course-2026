"""Check the outputs produced by run_smoke_test.sh."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "smoke"


def main() -> None:
    required = [
        RUN_DIR / "model.pt",
        RUN_DIR / "summary.json",
        RUN_DIR / "metrics.jsonl",
        ROOT / "results" / "smoke-scaling.csv",
        ROOT / "results" / "smoke-scaling.png",
        ROOT / "results" / "smoke-fertility.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Missing smoke-test outputs: {missing}"

    metrics = [
        json.loads(line)
        for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_losses = [row["train_loss"] for row in metrics if "train_loss" in row]
    valid_losses = [row["valid_loss"] for row in metrics if "valid_loss" in row]

    assert len(train_losses) >= 2, "Need at least two logged training losses"
    assert valid_losses, "No validation loss was logged"
    assert all(math.isfinite(loss) for loss in train_losses + valid_losses)
    assert train_losses[-1] < train_losses[0], (
        f"Training loss did not fall: {train_losses[0]:.4f} -> "
        f"{train_losses[-1]:.4f}"
    )
    print(
        "Smoke checks passed: "
        f"train loss {train_losses[0]:.4f} -> {train_losses[-1]:.4f}; "
        f"{len(required)} expected outputs found."
    )


if __name__ == "__main__":
    main()
