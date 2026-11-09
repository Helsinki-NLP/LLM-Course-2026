# Week 3

## Lecture overview

The lecture follows a request from the prompt to the generated tokens. It
separates reading the prompt (prefill) from writing the answer (decode),
explains why the KV cache and batching decide how many users a GPU can serve,
and introduces FP8 quantization, prefix caching, speculative decoding and
constrained (JSON) output. The examples use the SGLang inference engine.

## Lab 3 assignment

Serve your Lab 2 model with SGLang and measure it:

1. Measure time to first token, time per output token and throughput with 1,
   8 and 32 parallel requests.
2. Repeat the measurements with FP8 weights.
3. Score bf16 and FP8 on your Lab 2 test set with the Lab 2 scorer.
4. Compare JSON answers with and without a JSON schema.

Follow the step-by-step instructions in [`Lab-3/README.md`](Lab-3/README.md).
One GPU job runs the whole experiment. The README lists the minimum needed for
the report and optional extensions.

Keep the three result tables, the raw results and a short recommendation
(bf16 or FP8, with or without a schema) for the inference section of the
final report.
