#!/usr/bin/env python3
"""Option E: how many tokens per word does your text need?

Counts tokens per word for your Lab 2 test set, and optionally for a second
file with one text per line (for example the same examples translated into
another language). If results/vllm exists, it also turns tokens per second
into words per second.

    python3 tokens_per_word.py --task-config $TASK
    python3 tokens_per_word.py --task-config $TASK --compare my-translations.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab4lib import DEFAULT_TASK_CONFIG, Task, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--compare", type=Path, help="a text file, one text per line")
    parser.add_argument("--speed", type=Path, default=Path("results/vllm/speed-summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    config = load_model_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    task = Task(args.task_config)
    fields = task.prompting.TASK_FIELDS
    texts = {"test set": [" ".join(str(r[f]) for f in fields) for r in task.test_examples()]}
    if args.compare:
        texts[args.compare.name] = [line.strip() for line in args.compare.open(encoding="utf-8") if line.strip()]

    speed = json.loads(args.speed.read_text()) if args.speed.exists() else None
    for name, items in texts.items():
        tokens = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in items)
        words = sum(len(t.split()) for t in items)
        per_word = tokens / words
        print(f"{name}: {len(items)} texts, {words} words, {tokens} tokens, {per_word:.2f} tokens per word")
        if speed:
            for level in speed["levels"]:
                tps = level["tokens_per_second"]
                print(f"  vLLM, {level['concurrency']} at once: {tps:.0f} tokens/s = about {tps / per_word:.0f} words/s")
    print("Example:", tokenizer.tokenize(texts["test set"][0])[:20])


if __name__ == "__main__":
    main()
