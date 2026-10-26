"""Run one few-shot prompt on the tiny model and a larger base model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tiny_gpt import GPTConfig, TinyGPT


DEFAULT_PROMPT = """Review: The food was cold and the waiter ignored us. Sentiment: negative
Review: Beautiful room, spotless, and the staff were kind. Sentiment: positive
Review: It arrived late and the box was crushed. Sentiment: negative
Review: Exactly what I wanted, and cheaper than the shop. Sentiment:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/generation-comparison.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else DEFAULT_PROMPT
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.run / "model.pt", map_location=device, weights_only=False)
    tiny = TinyGPT(GPTConfig(**checkpoint["config"])).to(device)
    tiny.load_state_dict(checkpoint["state_dict"])
    tiny.eval()
    tiny_tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tiny_ids = torch.tensor([tiny_tokenizer.encode(prompt).ids], device=device)
    tiny_output_ids = tiny.generate(
        tiny_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=0,
    )[0, tiny_ids.size(1) :].tolist()
    tiny_output = tiny_tokenizer.decode(tiny_output_ids)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        use_fast=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        torch_dtype="auto",
    ).to(device)
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        hf_ids = hf_model.generate(
            **hf_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=hf_tokenizer.eos_token_id,
        )[0]
    hf_output = hf_tokenizer.decode(
        hf_ids[hf_inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    result = {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "decoding": {"method": "greedy", "max_new_tokens": args.max_new_tokens},
        "tiny_model": {"run": str(args.run), "output": tiny_output},
        "base_model": {
            "model_id": args.hf_model,
            "revision": args.hf_revision,
            "output": hf_output,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
