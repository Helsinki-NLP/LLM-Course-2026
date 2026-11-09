#!/usr/bin/env python3
"""Print the SGLang server command for one server config (bf16 or fp8).

Usage:
    python3 scripts/sglang_command.py configs/bf16.json
"""

from __future__ import annotations

import json
import shlex
import sys


def command(config: dict) -> list[str]:
    args = [
        "python", "-m", "sglang.launch_server",
        "--model-path", config["model"],
        "--revision", config["revision"],
        "--served-model-name", "task",
        "--host", "127.0.0.1",
        "--port", str(config["port"]),
        "--dtype", "bfloat16",
        "--context-length", str(config["context_length"]),
        "--mem-fraction-static", str(config["mem_fraction_static"]),
        "--random-seed", "2026",
    ]
    if config.get("quantization"):
        args += ["--quantization", config["quantization"]]
    adapter = config.get("adapter")
    if adapter:
        args += ["--enable-lora", "--lora-paths", f"{adapter['name']}={adapter['path']}"]
    args += [str(value) for value in config.get("extra_args", [])]
    return args


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(shlex.join(command(json.load(handle))))
