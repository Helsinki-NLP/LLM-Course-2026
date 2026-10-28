"""Download the small text collections used in Lab 1.

The script streams Finnish Wikipedia so that students can choose the corpus
size without first downloading the full dump. FLORES+ is gated on Hugging
Face; run ``huggingface-cli login`` and accept its terms before using
``--flores``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_wikipedia(
    output_dir: Path,
    max_articles: int,
    valid_fraction: float,
    seed: int,
    revision: str,
) -> None:
    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fi",
        split="train",
        streaming=True,
        revision=revision,
    )
    articles: list[str] = []
    for row in stream:
        text = " ".join(row["text"].split())
        if len(text) >= 200:
            articles.append(text)
        if len(articles) >= max_articles:
            break

    if len(articles) < 2:
        raise RuntimeError("Wikipedia stream returned too little usable text")

    rng = random.Random(seed)
    rng.shuffle(articles)
    n_valid = max(1, round(len(articles) * valid_fraction))
    write_lines(output_dir / "wikipedia-valid.txt", articles[:n_valid])
    write_lines(output_dir / "wikipedia-train.txt", articles[n_valid:])
    print(
        f"Wrote {len(articles) - n_valid:,} training and {n_valid:,} "
        f"validation articles to {output_dir}"
    )


def download_flores(output_dir: Path, languages: list[str], revision: str) -> None:
    for language in languages:
        dataset = load_dataset(
            "openlanguagedata/flores_plus",
            language,
            split="devtest",
            revision=revision,
        )
        sentences = [row["text"].strip() for row in dataset if row["text"].strip()]
        output = output_dir / f"flores-{language}.txt"
        write_lines(output, sentences)
        print(f"Wrote {len(sentences):,} {language} sentences to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-articles", type=int, default=50_000)
    parser.add_argument("--valid-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--wikipedia-revision",
        default="90f4255398cf28a623e5843079bd99bce484e8ba",
    )
    parser.add_argument("--flores-revision", default="main")
    parser.add_argument(
        "--flores",
        action="store_true",
        help="also download gated FLORES+ devtest data",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["eng_Latn", "fin_Latn", "deu_Latn"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.valid_fraction < 1:
        raise ValueError("--valid-fraction must be between 0 and 1")
    download_wikipedia(
        args.output_dir,
        args.max_articles,
        args.valid_fraction,
        args.seed,
        args.wikipedia_revision,
    )
    if args.flores:
        download_flores(args.output_dir, args.languages, args.flores_revision)
    manifest = {
        "wikipedia": {
            "dataset": "wikimedia/wikipedia",
            "config": "20231101.fi",
            "revision": args.wikipedia_revision,
            "max_articles": args.max_articles,
            "valid_fraction": args.valid_fraction,
            "seed": args.seed,
        },
        "flores": {
            "downloaded": args.flores,
            "dataset": "openlanguagedata/flores_plus",
            "revision": args.flores_revision,
            "split": "devtest",
            "languages": args.languages,
        },
    }
    (args.output_dir / "data-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
