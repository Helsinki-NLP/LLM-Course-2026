# Week 3

## Lecture overview

The lecture follows a model from a prompt to generated tokens. It separates
prefill from decode, explains KV-cache memory and continuous batching, and
introduces prefix caching, FP8 and weight-only quantization, speculative
decoding, constrained output and the trade-off between latency, throughput
and quality.

## Lab 3 assignment

Serve a model and measure a controlled inference workload:

1. Start an OpenAI-compatible vLLM server in bf16.
2. Measure time to first token, time per output token and aggregate throughput
   at concurrency 1, 8 and 32.
3. Repeat the same measurements in FP8.
4. Score bf16 and FP8 with the same held-out task and metric.
5. Compare prompt-only JSON generation with schema-constrained generation,
   reporting both invalid-output rate and task score.

Compiled slides are in `PDF/`.

The runnable lab reference is in `Lab-3/`. Start with:

```bash
cd Lab-3
bash run_smoke_test.sh
```

The Lab 2 adapter is optional. You may serve another suitable model, provided
the bf16 and FP8 measurements use the same model, task and evaluation protocol.

Keep the raw request logs, summaries, server configurations and a
speed-versus-quality recommendation for the inference section of the final
report.
