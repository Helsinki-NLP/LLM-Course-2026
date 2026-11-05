#!/usr/bin/env python3
"""Try your prompt on part of the dev set, interactively, before the full job.

Usage (in an interactive GPU session, README step 8):
    python3 try_prompt.py --config configs/my-task.json
    python3 try_prompt.py --config configs/my-task.json --all
    python3 try_prompt.py --config configs/my-task.json --model base --shots 0

By default this scores the instruct model with few-shot examples on the first
50 dev examples and shows some wrong answers. Edit TASK_SYSTEM in
lab2lib/prompting.py and run it again. The test set is never used.

Every try is logged in <output_dir>/prompt-trials/log.jsonl with the
instruction, the condition and the dev score, so you can report which prompts
you tried. Add --mock to check the script without a GPU.
"""
from __future__ import annotations

import argparse
import datetime
import json

from lab2lib import prompting
from lab2lib.backends import make_backend
from lab2lib.common import load_config, load_split, output_dir, write_jsonl
from lab2lib.evaluation import select_demos


def run_condition(config, backend, model_key, shots, examples, demos):
    rows = []
    for record in examples:
        prompt = prompting.task_prompt(record, demos if shots else [])
        raw = backend.generate_task(record, demos if shots else [], prompt)
        prediction = prompting.extract_task_label(raw, config["task"]["labels"])
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Try a prompt on dev examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("instruct", "base"), default="instruct")
    parser.add_argument("--shots", type=int, help="few-shot examples (default: config)")
    parser.add_argument("--all", action="store_true", help="try all four conditions")
    parser.add_argument("--limit", type=int, default=50, help="dev examples to use")
    parser.add_argument("--show", type=int, default=5, help="wrong answers to print")
    parser.add_argument("--mock", action="store_true", help="fake model, no GPU")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mock:
        config["backend"] = "mock"
    few_shot = int(config["prompting"]["few_shot_count"])
    shots = few_shot if args.shots is None else args.shots

    examples = load_split(config, "dev")[: args.limit]
    demos = select_demos(load_split(config, "train"), max(shots, few_shot), int(config["seed"]))

    if args.all:
        plan = {"base": [0, few_shot], "instruct": [0, few_shot]}
    else:
        plan = {args.model: [shots]}

    trials_dir = output_dir(config) / "prompt-trials"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Instruction (TASK_SYSTEM):\n  {prompting.TASK_SYSTEM}\n")
    print(f"Scoring {len(examples)} dev examples. The test set is not used.\n")

    summary = []
    for model_key, shot_list in plan.items():
        print(f"Loading the {model_key} model ...")
        backend = make_backend(config, model_key)
        try:
            for count in shot_list:
                condition_demos = demos[:count]
                rows = run_condition(config, backend, model_key, count, examples, condition_demos)
                name = f"{model_key}-{'zero' if count == 0 else f'{count}-shot'}"
                score = sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
                missing = sum(row["prediction"] is None for row in rows)
                summary.append((name, score, missing, rows))
                write_jsonl(trials_dir / f"{stamp}-{name}.jsonl", rows)
                with (trials_dir / "log.jsonl").open("a", encoding="utf-8") as log:
                    log.write(
                        json.dumps(
                            {
                                "time": stamp,
                                "condition": name,
                                "dev_examples": len(rows),
                                "dev_accuracy": score,
                                "no_label": missing,
                                "instruction": prompting.TASK_SYSTEM,
                                "backend": config["backend"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            backend.close()

    print("\n| Condition | Dev accuracy | No label found |")
    print("|---|---:|---:|")
    for name, score, missing, _ in summary:
        print(f"| {name} | {100 * score:.1f} | {missing} |")

    name, _, _, rows = summary[-1]
    wrong = [row for row in rows if not row["correct"]][: args.show]
    if wrong:
        print(f"\nSome wrong answers from {name}:")
    for row in wrong:
        output = " ".join(row["raw_output"].split())[:200]
        print(f"- {row['id']}: gold={row['gold']} prediction={row['prediction']}")
        print(f"  output: {output}")
    print(f"\nLogged in {trials_dir / 'log.jsonl'}")
    if config["backend"] == "mock":
        print("Mock model: these scores only check the script.")


if __name__ == "__main__":
    main()
