"""Train one or more byte-level BPE tokenizers on plain-text files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def text_iterator(paths: list[Path], batch_size: int = 1_000):
    batch: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def train(paths: list[Path], vocab_size: int, output_dir: Path) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(paths), trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    metadata = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "training_files": [str(path) for path in paths],
        "special_tokens": SPECIAL_TOKENS,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{output_dir}: {metadata['actual_vocab_size']:,} tokens "
        f"(requested {vocab_size:,})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=[8_000, 16_000, 32_000],
    )
    parser.add_argument("--output-root", type=Path, default=Path("tokenizers"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in args.files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    for size in args.vocab_sizes:
        train(args.files, size, args.output_root / f"bpe-{size}")


if __name__ == "__main__":
    main()
