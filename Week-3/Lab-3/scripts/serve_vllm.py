#!/usr/bin/env python3
"""Translate a recorded server config into a vLLM command and execute it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    served_name = config["served_model_name"]
    adapter = config.get("adapter") or {}
    adapter_path = adapter.get("path")
    adapter_name = adapter.get("name", "task")

    command = [
        "vllm",
        "serve",
        model,
        "--revision",
        config["model_revision"],
        "--tokenizer-revision",
        config.get("tokenizer_revision", config["model_revision"]),
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--port",
        str(config["port"]),
    ]
    if adapter_path:
        command += [
            "--served-model-name",
            config.get("base_served_model_name", "base"),
            "--enable-lora",
            "--lora-modules",
            f"{adapter_name}={adapter_path}",
        ]
    else:
        command += ["--served-model-name", served_name]
    if config.get("dtype"):
        command += ["--dtype", config["dtype"]]
    if config.get("quantization"):
        command += ["--quantization", config["quantization"]]
    command += [str(value) for value in config.get("extra_args", [])]

    print("Executing:", " ".join(command), flush=True)
    if args.print_command:
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
