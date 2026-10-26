"""Encode plain text as raw token-id files for the training script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def encode_file(
    tokenizer: Tokenizer,
    source: Path,
    destination: Path,
    eos_id: int,
    dtype: np.dtype,
    batch_size: int = 1_000,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    lines: list[str] = []

    def flush(handle) -> int:
        if not lines:
            return 0
        encoded = tokenizer.encode_batch(lines)
        ids = [token_id for item in encoded for token_id in (*item.ids, eos_id)]
        np.asarray(ids, dtype=dtype).tofile(handle)
        lines.clear()
        return len(ids)

    with source.open(encoding="utf-8") as input_handle, destination.open("wb") as output_handle:
        for line in input_handle:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) == batch_size:
                count += flush(output_handle)
        count += flush(output_handle)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokens"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.dtype("uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer has no <eos> token; train it with train_tokenizer.py")

    train_tokens = encode_file(
        tokenizer,
        args.train,
        args.output_dir / "train.bin",
        eos_id,
        dtype,
    )
    valid_tokens = encode_file(
        tokenizer,
        args.valid,
        args.output_dir / "valid.bin",
        eos_id,
        dtype,
    )
    metadata = {
        "tokenizer": str(args.tokenizer),
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {train_tokens:,} training and {valid_tokens:,} validation "
        f"tokens to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
