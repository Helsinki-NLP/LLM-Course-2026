#!/usr/bin/env python3
"""Speed benchmark: time to first token, time per output token and throughput.

Usage:
    python3 benchmark.py --backend hf     --output-dir results/hf
    python3 benchmark.py --backend server --output-dir results/vllm
    python3 benchmark.py --backend server --output-dir results/vllm-c --concurrency 1 2 4 8 16 32

--concurrency is the number of requests handled at the same time. For the
server they are sent in parallel; for Hugging Face they form one padded batch
for model.generate(). Every request produces exactly --max-tokens tokens, so
all requests do the same amount of work.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from lab4lib import (
    load_model_config,
    make_backend,
    percentile,
    utc_now,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["hf", "server"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("configs/benchmark-prompts.txt"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=3, help="requests run first and not counted")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    random.Random(2026).shuffle(prompts)
    return prompts


def as_messages(prompt: str, index: int) -> list[dict[str, str]]:
    # The request number makes every prompt unique, so nothing is reused.
    return [{"role": "user", "content": f"{prompt}\nRequest {index}."}]


def run_server_level(backend, prompts, concurrency, count, max_tokens):
    gate = Event()

    def one(index: int) -> dict[str, Any]:
        gate.wait()
        try:
            timing = backend.timed(as_messages(prompts[index % len(prompts)], index), max_tokens)
            timing.pop("content", None)
            return {"status": "ok", **timing}
        except Exception as error:  # keep failures in the raw results
            return {"status": "error", "error": f"{type(error).__name__}: {error}"}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, index) for index in range(count)]
        gate.set()
        rows = [future.result() for future in futures]
    return rows, time.perf_counter() - started


def run_hf_level(backend, prompts, concurrency, count, max_tokens):
    rows = []
    started = time.perf_counter()
    for first in range(0, count, concurrency):
        indices = range(first, min(first + concurrency, count))
        batch = [as_messages(prompts[i % len(prompts)], i) for i in indices]
        timing = backend.timed_batch(batch, max_tokens)
        tpot = (timing["latency_ms"] - timing["ttft_ms"]) / max(max_tokens - 1, 1)
        for _ in indices:
            rows.append(
                {
                    "status": "ok",
                    "ttft_ms": timing["ttft_ms"],
                    "tpot_ms": tpot,
                    "latency_ms": timing["latency_ms"],
                    "output_tokens": max_tokens,
                }
            )
    return rows, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    backend = make_backend(args.backend, config)
    prompts = load_prompts(args.prompts)
    run_level = run_server_level if args.backend == "server" else run_hf_level

    print(f"Warm-up: {args.warmup} requests (not counted)")
    run_level(backend, prompts, 1, args.warmup, args.max_tokens)

    all_rows, levels = [], []
    for concurrency in args.concurrency:
        count = max(args.requests, concurrency)
        rows, elapsed = run_level(backend, prompts, concurrency, count, args.max_tokens)
        good = [row for row in rows if row["status"] == "ok"]
        tokens = sum(int(row["output_tokens"]) for row in good)
        level = {
            "concurrency": concurrency,
            "requests": count,
            "failed": count - len(good),
            "ttft_ms_p50": percentile([r["ttft_ms"] for r in good], 0.5),
            "ttft_ms_p95": percentile([r["ttft_ms"] for r in good], 0.95),
            "tpot_ms_p50": percentile([r["tpot_ms"] for r in good], 0.5),
            "tpot_ms_p95": percentile([r["tpot_ms"] for r in good], 0.95),
            "tokens_per_second": tokens / elapsed if elapsed else None,
            "seconds": elapsed,
        }
        levels.append(level)
        all_rows += [{"concurrency": concurrency, **row} for row in rows]
        print(
            f"concurrency {concurrency:>3}: first token {level['ttft_ms_p50']:.0f} ms, "
            f"per token {level['tpot_ms_p50']:.1f} ms, {level['tokens_per_second']:.0f} tokens/s"
            + (f", {level['failed']} failed" if level["failed"] else "")
        )

    write_jsonl(args.output_dir / "speed-requests.jsonl", all_rows)
    write_json(
        args.output_dir / "speed-summary.json",
        {
            "backend": "vLLM" if args.backend == "server" and config["model"] != "mock" else args.backend,
            "model": config["model"],
            "revision": config["revision"],
            "quantization": config.get("quantization"),
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "prompts": str(args.prompts),
            "created_at": utc_now(),
            "levels": levels,
        },
    )
    print(f"Saved to {args.output_dir}/speed-summary.json")


if __name__ == "__main__":
    main()
