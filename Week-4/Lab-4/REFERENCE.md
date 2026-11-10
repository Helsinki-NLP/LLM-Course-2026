# Lab 4 reference

Background for [README.md](README.md). You do not need to read this to
complete the lab.

## Terms

| Term | Meaning here |
|---|---|
| inference engine | software that serves a model to many users efficiently; here vLLM |
| server | the engine running as a web service with an OpenAI-compatible API |
| concurrency | how many requests are handled at the same time |
| TTFT (time to first token) | time until the first output token: mostly *prefill*, reading the prompt |
| TPOT (time per output token) | average time between later tokens: *decode* |
| throughput | output tokens per second for all requests together |
| static batching | a fixed batch runs until its last request finishes (Hugging Face `generate()`) |
| continuous batching | requests join and leave the batch at every step (vLLM) |
| p50 / p95 | the median / the value 95% of requests stay below |

## The two backends

Both scripts, `benchmark.py` and `score_task.py`, take `--backend hf` or
`--backend server`, and read the model from `configs/model.json`.

**`--backend hf`** loads the model with `transformers` in the same Python
process (`lab4lib.HFBackend`) and calls `model.generate()` on a padded batch.

- With concurrency *n*, the benchmark runs batches of *n* prompts.
- TTFT is measured by generating one token for the batch; the full answer is
  then generated again, and TPOT is the remaining time divided by the
  remaining tokens. So every request in a batch gets the same timings.
- This is **static batching**: a batch finishes only when its longest answer
  is done.

**`--backend server`** sends HTTP requests to the vLLM server
(`lab4lib.ServerBackend`).

- With concurrency *n*, *n* requests are in flight at the same time.
- Answers are streamed: TTFT is the time until the first piece of text
  arrives; TPOT is the time from the first to the last piece divided by the
  number of tokens minus one.
- vLLM uses **continuous batching** and a paged **KV cache**.

In both cases every benchmark request produces exactly `--max-tokens` tokens
(128 by default), so all requests do the same work, and 3 warm-up requests are
run first and not counted. Defaults: concurrency 1 and 8, 16 requests per
level. All benchmark options: `python3 benchmark.py --help`.

## The evaluation

`score_task.py` uses your Lab 2 config (labels and test file) and
`lab2lib/prompting.py` (instruction, prompt and answer extraction), with
greedy decoding (`temperature=0`, up to 32 new tokens). It scores the first
100 test examples by default; `--limit 0` scores all of them. Hugging Face
answers in batches of 16; the server gets 16 requests at a time.

Greedy answers from the two backends are usually identical, but not always:
the numbers are computed with different kernels and in a different order, and
a tiny difference can change which token is most likely.
`show_differences.py` shows where they differ.

## The server

`scripts/start_server.sh [config]` loads the `python-vllm` module, starts
`vllm serve` in the background, writes the log to `.server/vllm.log` and waits
until the server answers. `scripts/stop_server.sh` stops it. The command it
runs is printed by `python3 scripts/vllm_command.py configs/model.json`.

Settings in `configs/model.json`:

| Entry | Meaning |
|---|---|
| `model`, `revision` | the Hugging Face model and its version (use a commit hash for final runs) |
| `max_model_len` | longest prompt plus answer, in tokens |
| `gpu_memory_utilization` | share of GPU memory vLLM may use for weights and KV cache |
| `quantization` | `null` for bf16, `"fp8"` for FP8 weights |
| `extra_args` | `--no-enable-prefix-caching`: prefix caching off, so repeated prompt starts do not speed up the benchmark |

You can talk to the server directly while it runs:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "task", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## Output files

Each `results/<run>/` folder contains:

- `speed-summary.json`: median and p95 TTFT and TPOT and throughput per level;
- `speed-requests.jsonl`: one line per benchmark request;
- `eval-summary.json`: accuracy, number of examples, time;
- `eval-answers.jsonl`: every test answer, the prediction and the gold label.

`summarize_results.py results` prints all runs as tables and writes
`results/speed.csv`.

## Serving your Lab 2 adapter

Add the adapter to `configs/model.json`, with its absolute path:

```json
"adapter": {"name": "lab2", "path": "/users/USERNAME/LLM-Course-2026/Week-2/Lab-2/results/my-task/adapter"}
```

Both backends then use it: vLLM serves it under the name `lab2`, and Hugging
Face loads it with PEFT. LoRA support depends on the vLLM version and the
model architecture; if it fails, report that and use the plain model.

## Running everything as a batch job

The core steps (3 and 4) also run unattended:

```bash
mkdir -p logs
sbatch slurm/lab4.slurm ../../Week-2/Lab-2/configs/my-task.json
```

The results go to `results/job-JOBID/`.

## Changing the model

Change `model` in `configs/model.json` (and `configs/model-fp8.json`), for
example to `Qwen/Qwen3.5-4B`, and use the same file for both backends. A larger
model shows bigger differences between the backends and for FP8. Qwen3.5 needs
vLLM 0.17 or newer.

## Limits of the measurements

- Client and server run on the same node, so there is no real network.
- 16 requests per level give a rough picture, not a capacity test; repeated
  runs give somewhat different numbers.
- The speed test forces a fixed answer length, which is not how the model
  answers normally.
- The Hugging Face timings are per batch, so they are coarser than the
  streamed server timings.
