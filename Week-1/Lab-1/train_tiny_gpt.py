"""Train the decoder-only transformer used in Lab 1.

The implementation is intentionally single-GPU and explicit. It logs JSONL so
that every number in the report can be traced back to a run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from tiny_gpt import GPTConfig, TinyGPT


class TokenBatches:
    def __init__(self, path: Path, dtype: str, sequence_length: int, seed: int) -> None:
        self.tokens = np.memmap(path, mode="r", dtype=dtype)
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) <= sequence_length + 1:
            raise ValueError(f"{path} is too short for sequence length {sequence_length}")

    def random(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(
            0,
            len(self.tokens) - self.sequence_length - 1,
            size=batch_size,
        )
        offsets = np.arange(self.sequence_length + 1)
        batch = np.asarray(self.tokens[starts[:, None] + offsets[None, :]], dtype=np.int64)
        tensor = torch.from_numpy(batch)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        tensor = tensor.to(device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:]


def cosine_lr(step: int, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + coefficient * (peak_lr - min_lr)


@torch.inference_mode()
def evaluate(
    model: TinyGPT,
    batches: TokenBatches,
    batch_size: int,
    count: int,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for _ in range(count):
        inputs, targets = batches.random(batch_size, device)
        with autocast_context():
            _, loss = model(inputs, targets)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, help="override for a quick smoke test")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    data_dir = Path(config["data_dir"])
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    model_config = GPTConfig(vocab_size=metadata["vocab_size"], **config["model"])
    raw_model = TinyGPT(model_config).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model

    train_config = config["training"]
    batch_size = int(train_config["batch_size"])
    grad_accum = int(train_config.get("gradient_accumulation", 1))
    tokens_per_step = batch_size * model_config.max_seq_len * grad_accum
    requested_tokens = int(float(train_config["train_tokens"]))
    max_steps = math.ceil(requested_tokens / tokens_per_step)
    if args.max_steps is not None:
        max_steps = args.max_steps

    train_batches = TokenBatches(
        data_dir / "train.bin",
        metadata["dtype"],
        model_config.max_seq_len,
        seed,
    )
    valid_batches = TokenBatches(
        data_dir / "valid.bin",
        metadata["dtype"],
        model_config.max_seq_len,
        seed + 1,
    )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=tuple(train_config.get("betas", [0.9, 0.95])),
        weight_decay=float(train_config.get("weight_decay", 0.1)),
        fused=device.type == "cuda",
    )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_bf16
        else nullcontext
    )
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    peak_lr = float(train_config["learning_rate"])
    min_lr = float(train_config.get("min_learning_rate", peak_lr * 0.1))
    warmup_steps = int(train_config.get("warmup_steps", min(100, max_steps // 10)))
    log_interval = int(train_config.get("log_interval", 10))
    eval_interval = int(train_config.get("eval_interval", 100))
    eval_batches = int(train_config.get("eval_batches", 20))
    eval_batch_size = int(train_config.get("eval_batch_size", batch_size))
    grad_clip = float(train_config.get("gradient_clip", 1.0))

    print(
        f"{raw_model.parameter_count() / 1e6:.2f}M parameters | "
        f"{tokens_per_step:,} tokens/step | {max_steps:,} steps | {device}"
    )
    started = time.perf_counter()
    interval_started = started
    optimizer.zero_grad(set_to_none=True)

    for step in range(max_steps):
        lr = cosine_lr(step, max_steps, warmup_steps, peak_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        accumulated_loss = 0.0
        for _ in range(grad_accum):
            inputs, targets = train_batches.random(batch_size, device)
            with autocast_context():
                _, loss = model(inputs, targets)
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            accumulated_loss += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step + 1
        should_log = completed_step == 1 or completed_step % log_interval == 0
        should_eval = completed_step % eval_interval == 0 or completed_step == max_steps
        if should_log or should_eval:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = now - started
            interval_seconds = now - interval_started
            interval_started = now
            record = {
                "step": completed_step,
                "tokens": completed_step * tokens_per_step,
                "train_loss": accumulated_loss / grad_accum,
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    tokens_per_step * (log_interval if completed_step > 1 else 1)
                    / max(interval_seconds, 1e-9)
                ),
            }
            if should_eval:
                record["valid_loss"] = evaluate(
                    model,
                    valid_batches,
                    eval_batch_size,
                    eval_batches,
                    device,
                    autocast_context,
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))

    summary = {
        "config_file": str(args.config),
        "seed": seed,
        "model": model_config.__dict__,
        "training": train_config,
        "parameters": raw_model.parameter_count(),
        "requested_train_tokens": requested_tokens,
        "actual_train_tokens": max_steps * tokens_per_step,
        "tokens_per_step": tokens_per_step,
        "steps": max_steps,
        "estimated_training_flops": 6
        * raw_model.parameter_count()
        * max_steps
        * tokens_per_step,
        "device": str(device),
        "bf16": use_bf16,
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "data_metadata": metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(raw_model.checkpoint(), output_dir / "model.pt")
    print(f"Saved run to {output_dir}")


if __name__ == "__main__":
    main()
