"""Compare tokens per whitespace-delimited word across languages."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from tokenizers import Tokenizer


@dataclass
class Result:
    tokenizer: str
    language: str
    sentences: int
    words: int
    tokens: int
    tokens_per_word: float
    median_sentence_fertility: float


def read_sentences(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measure(
    name: str,
    encode: Callable[[str], list[int]],
    language: str,
    sentences: list[str],
) -> Result:
    sentence_counts = [
        (len(sentence.split()), len(encode(sentence))) for sentence in sentences
    ]
    words = sum(word_count for word_count, _ in sentence_counts)
    tokens = sum(token_count for _, token_count in sentence_counts)
    if words == 0:
        raise ValueError(f"No whitespace-delimited words found for {language}")
    per_sentence = [
        token_count / word_count
        for word_count, token_count in sentence_counts
        if word_count
    ]
    return Result(
        name,
        language,
        len(sentences),
        words,
        tokens,
        tokens / words,
        statistics.median(per_sentence),
    )


def parse_mapping(items: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} entries must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        result[name] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each language",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="NAME=TOKENIZER_JSON",
        help="a tokenizer trained in Part A; may be repeated",
    )
    parser.add_argument(
        "--hf-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="a Hugging Face decoder-only model whose tokenizer should be used",
    )
    parser.add_argument("--output", type=Path, default=Path("results/fertility.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language_files = parse_mapping(args.language, "--language")
    local_tokenizers = parse_mapping(args.tokenizer, "--tokenizer")
    corpora = {name: read_sentences(path) for name, path in language_files.items()}

    encoders: list[tuple[str, Callable[[str], list[int]]]] = []
    for name, path in local_tokenizers.items():
        tokenizer = Tokenizer.from_file(str(path))
        encoders.append((name, lambda text, tok=tokenizer: tok.encode(text).ids))

    if args.hf_model:
        from transformers import AutoTokenizer

        for model_id in args.hf_model:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            encoders.append(
                (
                    model_id,
                    lambda text, tok=tokenizer: tok.encode(text, add_special_tokens=False),
                )
            )

    if not encoders:
        raise ValueError("Provide at least one --tokenizer or --hf-model")

    results = [
        measure(tokenizer_name, encode, language, sentences)
        for tokenizer_name, encode in encoders
        for language, sentences in corpora.items()
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    args.output.with_suffix(".json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{'tokenizer':<34} {'language':<12} {'tokens/word':>11}")
    for result in results:
        print(f"{result.tokenizer:<34} {result.language:<12} {result.tokens_per_word:>11.3f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
