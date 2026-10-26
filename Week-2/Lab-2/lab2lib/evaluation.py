from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .backends import make_backend
from .common import (
    load_knowledge,
    load_split,
    output_dir,
    update_results,
    write_jsonl,
)
from .prompting import (
    extract_choice,
    extract_task_label,
    knowledge_prompt,
    task_prompt,
)


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(row["correct"] for row in rows) / len(rows) if rows else 0.0


def select_demos(
    train: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count > len(train):
        raise ValueError(f"Requested {count} demonstrations from {len(train)} records")
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    return [train[index] for index in indices[:count]]


def score_task(
    config: dict[str, Any],
    model_key: str,
    shots: int,
    adapter: bool,
    artifact_name: str,
    split: str = "test",
) -> dict[str, Any]:
    train = load_split(config, "train")
    examples = load_split(config, split)
    demos = select_demos(train, shots, int(config["seed"])) if shots else []
    backend = make_backend(config, model_key, adapter=adapter)
    rows: list[dict[str, Any]] = []
    try:
        for record in examples:
            prompt = task_prompt(record, demos)
            raw = backend.generate_task(record, demos, prompt)
            prediction = extract_task_label(raw, config["task"]["labels"])
            rows.append(
                {
                    "id": record["id"],
                    "gold": record["label"],
                    "prediction": prediction,
                    "correct": prediction == record["label"],
                    "raw_output": raw,
                    "prompt": prompt,
                }
            )
        metadata = backend.metadata()
    finally:
        backend.close()
    path = output_dir(config) / "predictions" / f"{artifact_name}.jsonl"
    write_jsonl(path, rows)
    return {
        "name": artifact_name,
        "model_kind": model_key,
        "shots": shots,
        "split": split,
        "score": accuracy(rows),
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "metric": "exact-match accuracy",
        "model": metadata,
        "predictions": str(path.relative_to(output_dir(config))),
    }


def evaluate_baselines(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for model_key in ("base", "instruct"):
        for shots in (0, int(config["prompting"]["few_shot_count"])):
            name = f"{model_key}-{'zero' if shots == 0 else f'{shots}-shot'}"
            candidates.append(
                score_task(
                    config,
                    model_key,
                    shots,
                    False,
                    f"dev-{name}",
                    split="dev",
                )
            )
    best = max(candidates, key=lambda run: (run["score"], run["name"]))
    runs = []
    for candidate in candidates:
        name = candidate["name"].removeprefix("dev-")
        run = score_task(
            config,
            candidate["model_kind"],
            candidate["shots"],
            False,
            name,
            split="test",
        )
        run["dev_score"] = candidate["score"]
        run["dev_predictions"] = candidate["predictions"]
        runs.append(run)
    selected_test = next(
        run
        for run in runs
        if run["model_kind"] == best["model_kind"] and run["shots"] == best["shots"]
    )
    update_results(
        config,
        {
            "baselines": runs,
            "best_prompt": {
                "name": selected_test["name"],
                "dev_score": best["score"],
                "test_score": selected_test["score"],
                "selection_rule": "highest dev exact-match accuracy; name breaks ties",
            },
        },
    )
    return runs


def evaluate_sft(config: dict[str, Any]) -> dict[str, Any]:
    result = score_task(config, "sft", 0, True, "lora-sft")
    current_update = {"sft_evaluation": result}
    update_results(config, current_update)
    return result


def evaluate_forgetting(config: dict[str, Any], after: bool) -> dict[str, Any]:
    records = load_knowledge(config)
    backend = make_backend(config, "sft" if after else "instruct", adapter=after)
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            prompt = knowledge_prompt(record)
            raw = backend.generate_knowledge(record, prompt)
            prediction = extract_choice(raw)
            rows.append(
                {
                    "id": record["id"],
                    "gold": record["answer"],
                    "prediction": prediction,
                    "correct": prediction == record["answer"],
                    "raw_output": raw,
                    "prompt": prompt,
                }
            )
        metadata = backend.metadata()
    finally:
        backend.close()
    phase = "after" if after else "before"
    path = output_dir(config) / "predictions" / f"knowledge-{phase}.jsonl"
    write_jsonl(path, rows)
    result = {
        "score": accuracy(rows),
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "metric": "exact-match accuracy",
        "model": metadata,
        "predictions": str(path.relative_to(output_dir(config))),
    }
    from .common import read_results

    existing = read_results(config).get("forgetting", {})
    existing[phase] = result
    if "before" in existing and "after" in existing:
        existing["delta"] = existing["after"]["score"] - existing["before"]["score"]
    update_results(config, {"forgetting": existing})
    return result
