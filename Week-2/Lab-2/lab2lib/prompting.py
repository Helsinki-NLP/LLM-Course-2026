from __future__ import annotations

import re
from typing import Any


TASK_SYSTEM = (
    "Classify the relationship between the premise and hypothesis. "
    "Use exactly one of: entailment, neutral, contradiction. "
    "End with 'Label: <label>'."
)

KNOWLEDGE_SYSTEM = (
    "Answer the multiple-choice question. Choose exactly A, B, C, or D. "
    "End with 'Answer: <letter>'."
)


def task_example(record: dict[str, Any], include_answer: bool) -> str:
    text = f"Premise: {record['premise']}\nHypothesis: {record['hypothesis']}"
    if include_answer:
        text += f"\nLabel: {record['label']}"
    return text


def task_prompt(record: dict[str, Any], demos: list[dict[str, Any]]) -> str:
    sections = ["Examples:"]
    for index, demo in enumerate(demos, start=1):
        sections.append(f"Example {index}\n{task_example(demo, include_answer=True)}")
    if not demos:
        sections = []
    sections.append("Classify this item:\n" + task_example(record, include_answer=False))
    sections.append("Label:")
    return "\n\n".join(sections)


def knowledge_prompt(record: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip("ABCD", record["choices"])
    )
    return f"Question: {record['question']}\n{choices}\nAnswer:"


def extract_task_label(text: str, labels: list[str]) -> str | None:
    lowered = text.lower()
    explicit = re.search(r"label\s*:\s*([a-z-]+)", lowered)
    if explicit and explicit.group(1) in labels:
        return explicit.group(1)
    matches = [
        (lowered.find(label), label)
        for label in labels
        if re.search(rf"\b{re.escape(label)}\b", lowered)
    ]
    return min(matches)[1] if matches else None


def extract_choice(text: str) -> str | None:
    explicit = re.search(r"(?i)(?:answer|choice)\s*:\s*([A-D])\b", text)
    if explicit:
        return explicit.group(1).upper()
    bare = re.search(r"(?i)\b([A-D])\b", text.strip())
    return bare.group(1).upper() if bare else None


def messages_for_sft(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TASK_SYSTEM},
        {"role": "user", "content": task_prompt(record, [])},
        {"role": "assistant", "content": f"Label: {record['label']}"},
    ]
