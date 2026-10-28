# Lab 3 reference: serve and measure

This directory is a complete, small reference implementation for serving one
model with vLLM and measuring the trade-off between speed, quality and output
validity. The programs deliberately use plain Python rather than a notebook so
that every result can be reproduced from a command line.

The workflow answers four questions:

1. How do streaming TTFT, TPOT and output throughput change at concurrency
   1, 8 and 32?
2. What speed does FP8 gain, and what held-out score does it cost?
3. How often is free JSON generation invalid?
4. Does JSON-schema constrained decoding remove invalid outputs without
   changing the task score?

The Lab 2 adapter is optional. You can serve the instruct model directly or
point both server configurations at the same adapter.

Lab 3 is not an intermediate submission. Keep the code, measurements and
analysis for the inference section of the final report.

## Use the held-out data from your selected task

The bundled NLI file only makes the example runnable. For the real lab, replace
it with the held-out test data and scorer for your selected project task.
Lab 3 does not retrain the model, so it normally needs the test split rather
than the Week 2 training and development files.

Keep the quality comparison controlled:

1. use exactly the same test examples for bf16 and FP8;
2. reuse the task instruction, output contract, extraction function and metric
   from the locked Week 2 evaluation;
3. do not change prompts or parsing after seeing which precision scores better;
4. record IDs in every output row so predictions can be compared item by item;
   and
5. inspect disagreements between bf16 and FP8, not only the aggregate scores.

Adapt `data/heldout-nli.jsonl`, `task_prompt()` and the scoring logic in
`score_task.py`. If your task does not naturally return JSON, define a small
JSON schema that represents its answer for the constrained-output comparison.

The speed benchmark uses a separate prompt file. Replace
`configs/benchmark-prompts.txt` with varied, task-representative prompts whose
lengths resemble the intended workload. Do not use one repeated prompt, since
that can turn the experiment into a prefix-cache benchmark.

You may change the task or model between weeks when there is a good reason.
If you do, state the reason and treat Lab 3 as a new controlled comparison
rather than claiming direct continuity with the Week 2 results.

## Start with the local smoke test

The smoke test uses the bundled deterministic OpenAI-compatible server. It
does not use a GPU, download a model or require third-party Python packages.
It starts mock bf16 and FP8 servers, runs the three concurrency levels, scores
the held-out examples, checks both structured-output modes, and builds the
final table.

```bash
bash run_smoke_test.sh
```

Its results go to `results/smoke/`. They prove that the plumbing works; they
are not measurements of vLLM or a language model. This test is safe to run on
a login node because it does not use a GPU or download a model.

## Files to configure

The two controlled server configurations are:

- `configs/bf16.json`
- `configs/fp8.json`

Edit both together. Keep the model, revision, adapter, context length, memory
reservation and extra arguments identical. The intended difference is
`quantization: null` versus `quantization: "fp8"`.

Before a real run:

1. Replace `model_revision: "main"` and `tokenizer_revision: "main"` with the
   exact Hugging Face commit hash used for both runs.
2. If reusing Lab 2, set `adapter.path` to its absolute directory and record
   its revision or checkpoint identifier in `adapter.revision`. Make the same
   edit in both files.
3. If not using an adapter, leave `adapter.path` as `null`. The base model is
   then exposed under the API name `task`.
4. Replace `data/heldout-nli.jsonl`, `task_prompt()` and the accuracy scorer
   with the unchanged held-out data and scorer from your own task. The bundled
   NLI data only makes this example runnable.

`configs/benchmark-prompts.txt` contains varied prompts to reduce prefix-cache
effects. Replace or extend them with task-realistic prompts of comparable
length. The server configs also disable prefix caching while benchmarking.

## Python environment

All client and analysis programs use the Python standard library. The
requirements files are intentionally empty apart from comments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Roihu, vLLM comes from the `python-vllm` module. Do not install another
vLLM, CUDA build of PyTorch, or `torch` into the module environment.

## Run on Roihu

Request an interactive GH200 from the login node when you want to inspect the
server manually:

```bash
srun --account=project_2021893 --partition=gpuinteractive \
  --gres=gpu:gh200:1 --time=00:15:00 --pty bash
```

The shell opened by this command is on a compute node. Set the course project
there so model weights are cached under scratch:

```bash
export PROJECT=project_2021893
```

If the allocation is long enough, the complete workflow is:

```bash
bash scripts/run_roihu_experiment.sh results/roihu
```

It performs these steps:

1. start bf16 and wait for `/v1/models`;
2. discard 20 warm-up requests;
3. record 64 requests at each concurrency level 1, 8 and 32;
4. score the held-out set and compare free versus constrained JSON;
5. stop bf16, start FP8 with the same configuration, and repeat speed and
   held-out scoring;
6. stop the server and write `speed-quality.csv`.

The complete run can take longer than the interactive test allocation because
it generates a fixed 128 output tokens for every benchmark request. Use the
interactive shell for a quick server check:

```bash
bash scripts/restart.sh bf16
curl http://127.0.0.1:8000/v1/models
bash scripts/stop.sh
```

To run each phase manually:

```bash
bash scripts/restart.sh bf16
bash scripts/measure.sh bf16 results/roihu/bf16 --structured

bash scripts/restart.sh fp8
bash scripts/measure.sh fp8 results/roihu/fp8
bash scripts/stop.sh

python3 summarize_results.py \
  --bf16-dir results/roihu/bf16 \
  --fp8-dir results/roihu/fp8 \
  --structured-dir results/roihu/bf16 \
  --output-dir results/roihu
```

`scripts/serve.sh bf16` and `scripts/serve.sh fp8` are foreground alternatives.
They are useful when you want the vLLM log in the terminal. `restart.sh`
records the process ID and log under `.server/`, stops only the server it
started, and waits up to five minutes for the replacement to become ready.

## What is measured

`benchmark.py` streams every response:

- **TTFT** is the interval from sending the request until the first non-empty
  content event.
- **TPOT** is the interval from first to last content event divided by
  `completion_tokens - 1`.
- **Output throughput** is the sum of completion tokens divided by wall-clock
  time for the complete concurrency batch.

The client requests streaming usage from the server and uses
`usage.completion_tokens`. If a server does not provide usage, the raw record
marks `content_events` as the fallback token-count source. An SSE content event
is not guaranteed to equal one tokenizer token, so results using that fallback
should be treated as approximate.

Concurrency is real: a thread pool releases requests together and keeps up to
1, 8 or 32 in flight. The default is 64 measured requests per level, enough to
make p95 interpretable, after 20 discarded warm-ups. Every speed request uses:

```text
temperature = 0
max_tokens = 128
ignore_eos = true
```

The raw log has one JSON object per request, including request ID, wall-clock
start, configuration, concurrency, TTFT, TPOT, latency, output tokens, token
count source, response hash, seed, model and revisions. Failures remain in the
log instead of silently disappearing.

## Quality and valid output

`score_task.py` sends each held-out example to the endpoint and records both
the raw output and extracted prediction. For a real project, reuse the same
data, prompt, decoding settings, extraction and metric as the bf16 baseline.
Changing the scorer would confound the precision comparison.

`evaluate_structured.py` sends the same JSON-requesting prompts twice:

- `free`: the prompt asks for JSON, but the server does not constrain it;
- `schema`: the request includes the exact JSON schema.

An output is invalid if it is not JSON, is not an object with exactly `label`
and `confidence`, uses a label outside the enum, or has a confidence outside
0–1. Task accuracy is calculated separately, because valid JSON can still
contain the wrong label.

The default request field is `guided_json`, matching the course example. Some
vLLM releases use the newer spelling:

```bash
python3 evaluate_structured.py ... --schema-field structured_outputs
```

Use the spelling supported by the installed Roihu vLLM module and record it in
the generated summary.

## Result layout

A full run produces:

```text
results/roihu/
  bf16/
    requests.jsonl
    benchmark-manifest.json
    benchmark-summary.json
    benchmark-summary.csv
    task-observations.jsonl
    task-summary.json
    structured-observations.jsonl
    structured-summary.json
  fp8/
    requests.jsonl
    benchmark-manifest.json
    benchmark-summary.json
    benchmark-summary.csv
    task-observations.jsonl
    task-summary.json
  speed-quality.csv
  experiment-summary.json
```

The manifest contains a snapshot of the server config and all benchmark
arguments. Raw observations preserve request count, seed, model name,
revisions, timestamps and failures. Summaries contain the sample sizes, warm-up
count, p50/p95 and aggregate throughput.

The compact CSV reports concurrency-1 TTFT/TPOT plus throughput at 1/8/32.
Per-concurrency TTFT and TPOT remain in `benchmark-summary.json`; inspect those
before making a deployment recommendation. Schema speed is left blank because
the core structured experiment compares invalid rate and content score. You
can benchmark schema decoding separately if its latency matters to the task.

## Slurm

Submit unattended jobs from the login node; Slurm starts them on a compute
node. The script already requests one GH200 with `--gres=gpu:gh200:1`.
Create its log directory and test the workflow using its short `gputest`
defaults:

```bash
mkdir -p logs
sbatch slurm/lab3.slurm
```

For the complete bf16 and FP8 sweep, override those defaults:

```bash
sbatch --partition=gpumedium --time=02:00:00 slurm/lab3.slurm
```

A server started by Slurm ends with the allocation. Check held jobs with
`squeue --me`, cancel unneeded jobs with `scancel JOB_ID`, and release
interactive GPU sessions when finished.

## Completion checklist

For the final report, keep:

- the exact bf16 and FP8 server configurations;
- raw request and task-observation files;
- benchmark and task summaries for both precisions;
- the structured-output comparison;
- `speed-quality.csv`; and
- a recommendation that discusses speed, quality, output validity and the
  limits of the measurements.

## Caveats worth reporting

- The mock server validates orchestration and file formats only. Its timing,
  throughput, FP8 score change and invalid rate are deliberately synthetic.
- FP8 support depends on the installed vLLM version, model architecture and
  GH200 kernels. A model may quantize at load time, require a pre-quantized
  checkpoint, or reject FP8 LoRA. Keep any workaround identical across the two
  configurations or rerun both.
- `ignore_eos` fixes output length for speed tests but changes normal generation
  behavior. Do not use it for held-out scoring.
- Client and server share one node in this reference, so timings include local
  HTTP and Python scheduling overhead but not a production network.
- p95 from 64 requests is a lab-scale estimate, not a production capacity
  study. Repeat runs and larger samples are needed for stable tail claims.
- Quantization can change greedy outputs even with a fixed seed. The seed is
  recorded for reproducibility; it does not guarantee batch-invariant GPU
  inference.
