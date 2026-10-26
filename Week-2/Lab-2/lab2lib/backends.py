from __future__ import annotations

import gc
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import project_path


TOKEN_PATTERN = re.compile(r"[a-z]+")


def tokens(record: dict[str, Any]) -> list[str]:
    text = f"{record.get('premise', '')} {record.get('hypothesis', '')}".lower()
    return TOKEN_PATTERN.findall(text)


class MockBackend:
    """Deterministic backend used only to exercise the local pipeline."""

    def __init__(
        self,
        labels: list[str],
        model_name: str,
        adapter_path: Path | None = None,
    ) -> None:
        self.labels = labels
        self.model_name = model_name
        self.requested_revision = "bundled-v1"
        self.resolved_revision = "bundled-v1"
        self.adapter: dict[str, Any] | None = None
        if adapter_path is not None:
            self.adapter = json.loads((adapter_path / "mock_adapter.json").read_text())

    def _nearest_demo(
        self, record: dict[str, Any], demos: list[dict[str, Any]]
    ) -> str | None:
        query = set(tokens(record))
        scored = [
            (len(query & set(tokens(demo))), -index, demo["label"])
            for index, demo in enumerate(demos)
        ]
        if scored and max(scored)[0] > 0:
            return max(scored)[2]
        return None

    def _adapter_prediction(self, record: dict[str, Any]) -> str:
        assert self.adapter is not None
        vocabulary = self.adapter["token_counts"]
        totals = self.adapter["label_token_totals"]
        vocab_size = len(self.adapter["vocabulary"])
        scores: dict[str, float] = {}
        for label in self.labels:
            score = math.log(self.adapter["label_counts"][label] + 1)
            denominator = totals[label] + vocab_size
            for token in tokens(record):
                score += math.log(vocabulary.get(token, {}).get(label, 0) + 1)
                score -= math.log(denominator)
            scores[label] = score
        return max(self.labels, key=lambda label: (scores[label], -self.labels.index(label)))

    def generate_task(
        self, record: dict[str, Any], demos: list[dict[str, Any]], prompt: str
    ) -> str:
        del prompt
        if self.adapter is not None:
            label = self._adapter_prediction(record)
        elif demos and (nearest := self._nearest_demo(record, demos)):
            label = nearest
        else:
            hypothesis = record["hypothesis"].lower()
            premise = record["premise"].lower()
            if any(word in hypothesis.split() for word in ("not", "never", "nobody")):
                label = "contradiction"
            elif any(word in hypothesis for word in ("might", "could", "maybe", "outside")):
                label = "neutral"
            elif set(TOKEN_PATTERN.findall(hypothesis)) & set(TOKEN_PATTERN.findall(premise)):
                label = "entailment"
            else:
                label = "neutral"
        return f"Reasoning omitted by mock backend. Label: {label}"

    def generate_knowledge(self, record: dict[str, Any], prompt: str) -> str:
        del prompt
        # A stable pseudo-prediction; the mock does not inspect the gold answer.
        checksum = sum(record["question"].encode("utf-8"))
        choice = "ABCD"[checksum % 4]
        return f"Answer: {choice}"

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
        }

    def close(self) -> None:
        return None


class HuggingFaceBackend:
    def __init__(
        self,
        spec: dict[str, Any],
        style: str,
        decoding: dict[str, Any],
        adapter_path: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.style = style
        self.decoding = decoding
        self.requested_revision = spec["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            trust_remote_code=spec.get("trust_remote_code", False),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=spec.get("trust_remote_code", False),
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model = model.eval()
        base_config = getattr(model, "config", None)
        self.resolved_revision = getattr(base_config, "_commit_hash", None)
        self.model_name = spec["id"]

    def _generate(self, user_prompt: str, system_prompt: str) -> str:
        if self.style == "instruct":
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"{system_prompt}\n\n{user_prompt}"
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(self.decoding["max_new_tokens"]),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_task(
        self, record: dict[str, Any], demos: list[dict[str, Any]], prompt: str
    ) -> str:
        del record, demos
        from .prompting import TASK_SYSTEM

        return self._generate(prompt, TASK_SYSTEM)

    def generate_knowledge(self, record: dict[str, Any], prompt: str) -> str:
        del record
        from .prompting import KNOWLEDGE_SYSTEM

        return self._generate(prompt, KNOWLEDGE_SYSTEM)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def make_backend(
    config: dict[str, Any],
    model_key: str,
    adapter: bool = False,
) -> MockBackend | HuggingFaceBackend:
    adapter_path = project_path(config["training"]["adapter_dir"], config) if adapter else None
    if config["backend"] == "mock":
        return MockBackend(
            config["task"]["labels"],
            model_name=f"mock-{model_key}",
            adapter_path=adapter_path,
        )
    if config["backend"] != "huggingface":
        raise ValueError(f"Unsupported backend: {config['backend']}")
    style = "instruct" if model_key in {"instruct", "sft"} else "base"
    return HuggingFaceBackend(
        config["models"][model_key],
        style=style,
        decoding=config["decoding"],
        adapter_path=adapter_path,
    )


def fit_mock_adapter(
    records: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    label_counts = Counter(record["label"] for record in records)
    token_counts: dict[str, Counter[str]] = {}
    label_token_totals = Counter()
    vocabulary: set[str] = set()
    for record in records:
        label = record["label"]
        for token in tokens(record):
            vocabulary.add(token)
            label_token_totals[label] += 1
            token_counts.setdefault(token, Counter())[label] += 1
    return {
        "labels": labels,
        "label_counts": {label: label_counts[label] for label in labels},
        "label_token_totals": {label: label_token_totals[label] for label in labels},
        "vocabulary": sorted(vocabulary),
        "token_counts": {
            token: {label: counts[label] for label in labels}
            for token, counts in sorted(token_counts.items())
        },
    }
