#!/usr/bin/env python3
"""Benchmark streaming TTFT, TPOT, and aggregate output throughput."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event
from typing import Any

from lab3lib import percentile, streaming_completion, utc_now, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="task")
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter-revision", default="none")
    parser.add_argument("--prompts", type=Path, default=Path("configs/benchmark-prompts.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--requests-per-level", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def make_payload(args: argparse.Namespace, prompt: str, request_id: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\nBenchmark request: {request_id}",
            }
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "seed": args.seed,
    }


def run_one(
    args: argparse.Namespace,
    prompt: str,
    request_id: str,
    concurrency: int,
    gate: Event | None = None,
) -> dict[str, Any]:
    if gate is not None:
        gate.wait()
    try:
        measured = streaming_completion(
            args.base_url,
            make_payload(args, prompt, request_id),
            timeout=args.timeout,
        )
        measured.pop("content")
        return {
            "request_id": request_id,
            "configuration": args.config_label,
            "concurrency": concurrency,
            "seed": args.seed,
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter_revision": args.adapter_revision,
            "status": "ok",
            **measured,
        }
    except Exception as exc:  # Preserve failures in the raw observations.
        return {
            "request_id": request_id,
            "configuration": args.config_label,
            "concurrency": concurrency,
            "seed": args.seed,
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter_revision": args.adapter_revision,
            "started_at": utc_now(),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(
    rows: list[dict[str, Any]], concurrency: int, elapsed: float, requested: int
) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "ok"]
    ttft = [float(row["ttft_ms"]) for row in good]
    tpot = [float(row["tpot_ms"]) for row in good]
    output_tokens = sum(int(row["output_tokens"]) for row in good)
    return {
        "concurrency": concurrency,
        "requests_requested": requested,
        "requests_completed": len(good),
        "requests_failed": requested - len(good),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "tpot_ms_p50": percentile(tpot, 0.50),
        "tpot_ms_p95": percentile(tpot, 0.95),
        "output_tokens": output_tokens,
        "batch_elapsed_s": elapsed,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
    }


def main() -> None:
    args = parse_args()
    if args.requests_per_level < max(args.concurrency):
        raise SystemExit("--requests-per-level must be at least the largest concurrency")
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")
    random.Random(args.seed).shuffle(prompts)
    server_config = json.loads(args.server_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.warmup_requests):
        row = run_one(args, prompts[index % len(prompts)], f"warmup-{index:04d}", 1)
        if row["status"] != "ok":
            raise SystemExit(f"warmup failed: {row['error']}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        gate = Event()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for index in range(args.requests_per_level):
                request_id = f"c{concurrency}-{index:04d}"
                prompt = prompts[index % len(prompts)]
                futures.append(
                    executor.submit(run_one, args, prompt, request_id, concurrency, gate)
                )
            gate.set()
            rows = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - started
        rows.sort(key=lambda row: row["request_id"])
        all_rows.extend(rows)
        summaries.append(
            summarize(rows, concurrency, elapsed, args.requests_per_level)
        )

    write_jsonl(args.output_dir / "requests.jsonl", all_rows)
    summary = {
        "configuration": args.config_label,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_revision": args.adapter_revision,
        "seed": args.seed,
        "warmup_requests_discarded": args.warmup_requests,
        "requests_per_level": args.requests_per_level,
        "max_tokens": args.max_tokens,
        "created_at": utc_now(),
        "levels": summaries,
    }
    write_json(args.output_dir / "benchmark-summary.json", summary)
    with (args.output_dir / "benchmark-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(
        args.output_dir / "benchmark-manifest.json",
        {
            "command_arguments": vars(args)
            | {
                "output_dir": str(args.output_dir),
                "prompts": str(args.prompts),
                "server_config": str(args.server_config),
            },
            "server_config_snapshot": server_config,
            "raw_observations": "requests.jsonl",
            "summary_json": "benchmark-summary.json",
            "summary_csv": "benchmark-summary.csv",
        },
    )
    failures = sum(row["requests_failed"] for row in summaries)
    print(f"Wrote {len(all_rows)} raw observations; failures={failures}")


if __name__ == "__main__":
    main()
