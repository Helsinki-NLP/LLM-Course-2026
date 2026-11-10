# Week 3

## Lecture overview

The lecture follows a request from the prompt to the generated tokens. It
covers how the next token is chosen, separates reading the prompt (prefill)
from writing the answer (decode), and explains why the KV cache and batching
decide how many users a GPU can serve. It then covers prefix caching,
quantization and constrained decoding (labels, JSON, grammars), and ends with
three topics for language technology: tokens per word across languages,
classifying by scoring the labels, and why greedy decoding is not always
reproducible. FP8 and INT4 details, speculative decoding, PagedAttention and
the cost of serving are in the appendix.

## Lab 3 assignment

Run the same model with plain Hugging Face `transformers` and with the vLLM
inference engine, interactively in one GPU session:

1. Run a speed benchmark (time to first token, time per output token,
   throughput with 1 and 8 requests at a time) for both.
2. Evaluate both on your Lab 2 test set with the Lab 2 prompt and scorer.
3. Compare the two and explain the difference.
4. Design and run one small experiment of your own.

Follow the step-by-step commands in [`Lab-3/README.md`](Lab-3/README.md) or
the notebook `Lab-3/lab3.ipynb`, which shows all the code in its cells. FP8 weights and JSON output with a schema
are optional extensions.

Keep the result tables, your experiment and a short discussion (when is an
inference engine worth it?) for the inference section of the final report.
