"""Small, dependency-free helpers shared by the Lab 3 programs."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LABELS = ("entailment", "neutral", "contradiction")


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


def task_prompt(row: dict[str, Any], json_output: bool = False) -> str:
    ending = (
        'Return only JSON with keys "label" and "confidence". '
        f'"label" must be one of {list(LABELS)}.'
        if json_output
        else f"Reply with exactly one label: {', '.join(LABELS)}."
    )
    return (
        "Classify the relation between the premise and hypothesis.\n"
        f"Example ID: {row['id']}\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        f"{ending}"
    )


def extract_label(text: str) -> str | None:
    lowered = text.strip().lower()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            lowered = str(value.get("label", "")).lower()
    except json.JSONDecodeError:
        pass
    matches = [label for label in LABELS if label in lowered]
    return matches[0] if len(matches) == 1 else None


def validate_task_json(text: str) -> tuple[bool, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False, "not_json"
    if not isinstance(value, dict):
        return False, "not_object"
    if set(value) != {"label", "confidence"}:
        return False, "wrong_fields"
    if value["label"] not in LABELS:
        return False, "invalid_label"
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, "invalid_confidence"
    if not 0 <= confidence <= 1:
        return False, "confidence_out_of_range"
    return True, None


TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}
