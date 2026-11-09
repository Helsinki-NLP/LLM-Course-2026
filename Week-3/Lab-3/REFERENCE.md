# Lab 3 reference

Background for [README.md](README.md). You do not need to read this to
complete the lab.

## Terms

| Term | Meaning here |
|---|---|
| inference engine | software that loads a model and answers requests efficiently; here SGLang |
| server | the engine running as a web service with an OpenAI-compatible API |
| concurrency | how many requests are being answered at the same time |
| TTFT (time to first token) | time from sending a request to receiving the first piece of the answer; mostly *prefill*, reading the prompt |
| TPOT (time per output token) | average time between later tokens; this is *decode*, writing the answer |
| throughput | output tokens per second for all requests together |
| bf16 | 16-bit numbers for the weights: the normal precision |
| FP8 | 8-bit numbers for the weights: half the memory, possibly faster, possibly slightly different answers |
| JSON schema | a description of the allowed JSON; the server can force the output to follow it (*constrained decoding*) |
| p50 / p95 | the median / the value that 95% of requests stay below |

## What the job does

`scripts/run_experiment.sh` runs these steps for bf16 and then for FP8:

1. start the SGLang server (`scripts/start_server.sh`) and wait until it answers;
2. `benchmark.py`: send 20 warm-up requests (not recorded), then 64 requests
   at each concurrency level 1, 8 and 32;
3. `score_task.py`: send every example of your Lab 2 test set and score the
   answers with the Lab 2 scorer;
4. for bf16 only, `evaluate_structured.py`: ask for a JSON answer for every
   test example, once without and once with a JSON schema;
5. stop the server.

Finally, `summarize_results.py` prints the tables and writes
`speed-quality.csv`.

The bf16 and FP8 configs (`configs/bf16.json`, `configs/fp8.json`) are
identical except for `"quantization": "fp8"`. FP8 here means the weights are
converted to 8 bits when the model is loaded. Keep everything else identical
when you change them.

## How speed is measured

- Every benchmark request asks for exactly 128 output tokens
  (`max_tokens=128`, `ignore_eos=true`), so all requests do the same amount of
  work. This is only for speed tests; scoring uses normal generation.
- Answers are streamed. **TTFT** is the time until the first piece of text
  arrives. **TPOT** is the time from the first to the last piece, divided by
  the number of output tokens minus one.
- **Throughput** is the total number of output tokens divided by the time the
  whole batch of 64 requests took.
- Requests at concurrency 8 or 32 are really sent in parallel.
- The server is started with `--disable-radix-cache`, which switches off
  prefix caching, so repeated prompt beginnings do not make the benchmark look
  faster than normal use.
- `configs/benchmark-prompts.txt` holds varied prompts. You can replace them
  with prompts typical of your task.

## How quality and JSON validity are measured

`score_task.py` uses your Lab 2 config (labels and test file) and
`lab2lib/prompting.py` (instruction, prompt and answer extraction). Every
answer is generated greedily (`temperature=0`). This is the zero-shot prompt
that the fine-tuned model was trained on in Lab 2.

`evaluate_structured.py` asks for an answer of the form `{"label": "..."}`.
An answer is invalid if it is not JSON, has other fields, or uses a label that
is not in your task. Invalid answers count as wrong. With a schema, the server
receives the standard OpenAI field
`response_format={"type": "json_schema", ...}` and only allows output that
matches the schema.

## Output files

`results/lab3-JOBID/` contains, for `bf16/` and `fp8/`:

- `requests.jsonl`: one line per benchmark request, with TTFT, TPOT,
  latency, output tokens and any error;
- `benchmark-summary.json`, `benchmark-summary.csv`: medians, p95 and
  throughput per concurrency level;
- `benchmark-manifest.json`: the settings and server config used;
- `task-observations.jsonl`, `task-summary.json`: every test answer and the accuracy;

and for `bf16/` also `structured-observations.jsonl` and
`structured-summary.json`. `speed-quality.csv` collects the speed and
accuracy numbers in one file.

## Using the server interactively

```bash
srun --account=project_2021893 --partition=gpuinteractive --gres=gpu:gh200:1 --time=00:15:00 --pty bash
bash scripts/start_server.sh configs/bf16.json
curl http://127.0.0.1:30000/v1/models
curl http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "task", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
bash scripts/stop_server.sh
exit
```

The server log is in `.server/sglang-bf16.log`.

## Serving your Lab 2 adapter

Add the adapter to **both** configs, with its absolute path:

```json
"adapter": {"name": "lab2", "path": "/users/USERNAME/LLM-Course-2026/Week-2/Lab-2/results/my-task/adapter"}
```

Requests then use the model name `task:lab2`. The prompt is the zero-shot
Lab 2 prompt, which is what the adapter was trained on. SGLang's LoRA support
depends on the model architecture, and LoRA together with FP8 may not work. If
it fails, report that and use the plain model.

## Changing the model

Change `model` in both configs, for example to `Qwen/Qwen3.5-4B`, and record
its revision: replace `"revision": "main"` with the commit hash from the
model's Hugging Face page. An adapter only works with the model it was trained
on.

## Limits of the measurements

- The client and the server run on the same node, so there is no real network
  between them.
- 64 requests per level give a rough estimate of p95, not a production
  capacity test. Repeat runs give somewhat different numbers.
- With a small model such as Qwen3.5-0.8B, FP8 may not be faster: the GPU is
  not the bottleneck. That is a valid finding.
- FP8 can change some greedy answers even with the same seed.
- The speed tests force 128 output tokens, which is not how the model answers
  normally.

## Installing SGLang

`scripts/install_sglang.sh` creates a separate Python environment with SGLang
(it pins its own PyTorch and Transformers versions, so it cannot share the
Lab 2 environment). It must run on the GPU side of Roihu. The scripts look for
the environment in `SGLANG_ENV`, by default
`/projappl/project_2021893/sglang-env`.
