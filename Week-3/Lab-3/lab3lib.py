"""Small, dependency-free helpers shared by the Lab 3 programs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LAB3_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = LAB3_DIR.parent.parent / "Week-2" / "Lab-2" / "configs" / "roihu.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no observations."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _request(url: str, payload: dict[str, Any], timeout: float):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from server: {detail}") from exc


def chat_completion(
    base_url: str, payload: dict[str, Any], timeout: float = 120.0
) -> dict[str, Any]:
    payload = {**payload, "stream": False}
    with _request(f"{base_url.rstrip('/')}/chat/completions", payload, timeout) as response:
        return json.loads(response.read())


def streaming_completion(
    base_url: str, payload: dict[str, Any], timeout: float = 120.0
) -> dict[str, Any]:
    """Call an OpenAI-compatible SSE endpoint and measure token timings."""
    payload = {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started_wall = utc_now()
    started = time.perf_counter()
    first_content_at: float | None = None
    last_content_at: float | None = None
    content_parts: list[str] = []
    content_events = 0
    usage: dict[str, Any] = {}

    with _request(f"{base_url.rstrip('/')}/chat/completions", payload, timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content:
                continue
            now = time.perf_counter()
            if first_content_at is None:
                first_content_at = now
            last_content_at = now
            content_events += 1
            content_parts.append(content)

    finished = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("stream completed without any content")
    output_tokens = int(usage.get("completion_tokens") or content_events)
    token_source = "usage.completion_tokens" if usage.get("completion_tokens") else "content_events"
    decode_span = max(0.0, (last_content_at or first_content_at) - first_content_at)
    tpot = decode_span / max(output_tokens - 1, 1)
    content = "".join(content_parts)
    return {
        "started_at": started_wall,
        "ttft_ms": (first_content_at - started) * 1000,
        "tpot_ms": tpot * 1000,
        "latency_ms": (finished - started) * 1000,
        "output_tokens": output_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "token_count_source": token_source,
        "content": content,
        "response_sha256": sha256_text(content),
    }


def load_server_config(path: str | Path) -> dict[str, Any]:
    """Read a server config and add the model name that requests should use."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    adapter = config.get("adapter")
    # SGLang selects a LoRA adapter with "<served model>:<adapter name>".
    config["request_model"] = f"task:{adapter['name']}" if adapter else "task"
    return config


class Task:
    """The Lab 2 task: labels, test data, prompt and scorer.

    Lab 3 reuses the Lab 2 config and lab2lib/prompting.py unchanged, so the
    quality comparison uses exactly the same prompt and scorer as Lab 2.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.lab2_dir = self.config_path.parent.parent
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.name = config["task"].get("name", "task")
        self.labels: list[str] = list(config["task"]["labels"])
        self.test_path = self.lab2_dir / config["data"]["test"]
        self.prompting = self._load_prompting()

    def _load_prompting(self):
        path = self.lab2_dir / "lab2lib" / "prompting.py"
        spec = importlib.util.spec_from_file_location("lab2_prompting", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Cannot load the Lab 2 prompt from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_examples(self) -> list[dict[str, Any]]:
        return load_jsonl(self.test_path)

    def messages(self, record: dict[str, Any]) -> list[dict[str, str]]:
        """The zero-shot Lab 2 prompt, as used for the fine-tuned model."""
        return [
            {"role": "system", "content": self.prompting.TASK_SYSTEM},
            {"role": "user", "content": self.prompting.task_prompt(record, [])},
        ]

    def json_messages(self, record: dict[str, Any]) -> list[dict[str, str]]:
        """The same item, but asking for a JSON answer."""
        item = self.prompting.task_example(record, include_answer=False)
        instruction = (
            f"Answer with JSON only, in the form {{\"label\": \"<label>\"}}, "
            f"where <label> is one of: {', '.join(self.labels)}."
        )
        return [
            {"role": "system", "content": self.prompting.TASK_SYSTEM.split(" End with")[0]},
            {"role": "user", "content": f"{item}\n\n{instruction}"},
        ]

    def extract(self, text: str) -> str | None:
        return self.prompting.extract_task_label(text, self.labels)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": self.labels}},
            "required": ["label"],
            "additionalProperties": False,
        }

    def check_json(self, text: str) -> tuple[bool, str | None, str | None]:
        """Return (valid, reason if invalid, label)."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False, "not_json", None
        if not isinstance(value, dict):
            return False, "not_object", None
        if set(value) != {"label"}:
            return False, "wrong_fields", None
        if value["label"] not in self.labels:
            return False, "invalid_label", None
        return True, None, value["label"]
