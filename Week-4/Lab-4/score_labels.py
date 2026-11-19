#!/usr/bin/env python3
"""Option F: classify by scoring each label instead of generating an answer.

For every test example and every label, the model reads the prompt followed
by the label and we take the label's average log-probability per token. The
highest score wins. This needs only prefill (no decode steps), and every
answer is a valid label.

Runs with Hugging Face, in the same session as step 3 of the README:

    python3 score_labels.py --task-config $TASK --output-dir results/hf-scoring

The result is saved like score_task.py's, so summarize_results.py shows it as
an extra row next to generation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lab4lib import DEFAULT_TASK_CONFIG, Task, load_model_config, utc_now, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--answer-prefix",
        default="Label: ",
        help="how your Lab 2 prompt makes the model start its answer (default: 'Label: ')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_model_config(args.config)
    task = Task(args.task_config)
    examples = task.test_examples()[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], dtype=torch.bfloat16
    ).to(device).eval()

    def label_scores(example: dict) -> dict[str, float]:
        prompt = tokenizer.apply_chat_template(task.messages(example), tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt + args.answer_prefix, add_special_tokens=False)["input_ids"]
        scores = {}
        for label in task.labels:
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + label_ids], device=device)
            with torch.inference_mode():
                logprobs = model(ids).logits[0].float().log_softmax(-1)  # one forward pass
            # The token at position i is predicted by the logits at position i - 1.
            total = sum(logprobs[len(prompt_ids) + k - 1, t].item() for k, t in enumerate(label_ids))
            scores[label] = total / len(label_ids)  # average, so long labels are not penalised
        return scores

    started = time.perf_counter()
    rows = []
    for example in examples:
        scores = label_scores(example)
        prediction = max(scores, key=scores.get)
        rows.append(
            {
                "id": example["id"],
                "gold": example["label"],
                "prediction": prediction,
                "correct": prediction == example["label"],
                "scores": scores,
            }
        )
    seconds = time.perf_counter() - started
    summary = {
        "backend": "hf-scoring",
        "model": config["model"],
        "revision": config["revision"],
        "task": task.name,
        "task_config": str(task.config_path),
        "answer_prefix": args.answer_prefix,
        "examples": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "no_label": 0,
        "seconds": seconds,
        "created_at": utc_now(),
    }
    write_jsonl(args.output_dir / "eval-answers.jsonl", rows)
    write_json(args.output_dir / "eval-summary.json", summary)
    print(f"scoring: accuracy {100 * summary['accuracy']:.1f}% on {len(rows)} examples, {seconds:.1f} s")
    print(f"Saved to {args.output_dir}/eval-summary.json")


if __name__ == "__main__":
    main()
