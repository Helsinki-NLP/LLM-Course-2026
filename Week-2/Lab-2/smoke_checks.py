#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local smoke-test artifacts.")
    parser.add_argument("--results", default="results/smoke/results.json")
    args = parser.parse_args()
    path = Path(args.results)
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["reproducibility"]["seed"] == 2026
    assert result["reproducibility"]["backend"] == "mock"
    assert result["reproducibility"]["decoding"]["temperature"] == 0
    assert len(result["baselines"]) == 4
    assert {(run["model_kind"], run["shots"]) for run in result["baselines"]} == {
        ("base", 0),
        ("base", 3),
        ("instruct", 0),
        ("instruct", 3),
    }
    for run in result["baselines"] + [result["sft_evaluation"]]:
        assert 0.0 <= run["score"] <= 1.0
        assert run["total"] == result["data"]["splits"]["test"]
        assert (path.parent / run["predictions"]).is_file()
    for run in result["baselines"]:
        assert 0.0 <= run["dev_score"] <= 1.0
        assert (path.parent / run["dev_predictions"]).is_file()
    assert result["best_prompt"]["selection_rule"].startswith("highest dev")
    assert result["training"]["backend"] == "mock"
    assert result["training"]["assistant_only_loss"] is True
    assert result["training"]["precision"] == "bf16"
    assert set(result["forgetting"]) == {"before", "after", "delta"}
    assert result["forgetting"]["before"]["total"] == 8
    assert result["forgetting"]["before"]["score"] == result["forgetting"]["after"]["score"]
    assert (path.parent / result["artifacts"]["loss_plot"]).is_file()
    print(
        "Smoke checks passed: four baselines, LoRA path, shared scorer, "
        "forgetting check, result JSON, and loss plot."
    )


if __name__ == "__main__":
    main()
