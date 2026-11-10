#!/usr/bin/env python3
"""Print the vLLM server command for a model config.

Usage:
    python3 scripts/vllm_command.py configs/model.json
"""

from __future__ import annotations

import json
import shlex
import sys


def command(config: dict) -> list[str]:
    args = [
        "vllm", "serve", config["model"],
        "--revision", config["revision"],
        "--served-model-name", "task",
        "--host", "127.0.0.1",
        "--port", str(config["port"]),
        "--dtype", "bfloat16",
        "--max-model-len", str(config["max_model_len"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--seed", "2026",
    ]
    if config.get("quantization"):
        args += ["--quantization", config["quantization"]]
    adapter = config.get("adapter")
    if adapter:
        args += ["--enable-lora", "--lora-modules", f"{adapter['name']}={adapter['path']}"]
    args += [str(value) for value in config.get("extra_args", [])]
    return args


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(shlex.join(command(json.load(handle))))
