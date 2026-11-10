# Lab 3: Hugging Face versus an inference engine

You run the same model in two ways and compare them:

- **Hugging Face `transformers`**: the model in your own Python process,
  answering with `model.generate()`;
- **vLLM**: an inference engine that runs the model as a server and handles
  many requests at once.

For each, you run a **speed benchmark** (how fast answers start, how fast they
continue, how many tokens per second in total) and an **evaluation** (accuracy
on your Lab 2 test set, with the Lab 2 prompt and scorer). Then you design one
small experiment of your own.

Everything runs **interactively** in one GPU session; each command takes
seconds to a few minutes. The same steps are in the notebook
[`lab3.ipynb`](lab3.ipynb). Details are in [REFERENCE.md](REFERENCE.md).

## What you need to do

**Minimum (enough for the report):** steps 1–7: Hugging Face and vLLM, each
benchmarked and evaluated, one experiment of your own (step 6), and a short
discussion.

**Optional:** FP8 weights, JSON output with a schema, serving your Lab 2
adapter, or running everything as a batch job (see the end).

## 1. Get the files and test them (login node)

```bash
ssh USERNAME@roihu-gpu.csc.fi
cd ~/LLM-Course-2026 && git pull
cd ~/LLM-Course-2026/Week-3/Lab-3
bash run_smoke_test.sh
```

✓ The last line is `Local smoke test completed.` (fake models, no GPU).

The Hugging Face part uses the packages from Lab 2. If you have not done
Lab 2 step 3, do it now:

```bash
module purge
module load python-pytorch/2.13
python3 -m pip install --user -r ../../Week-2/Lab-2/requirements-roihu.txt
```

## 2. Start an interactive GPU session

```bash
srun --account=project_2021893 --partition=gpuinteractive --gres=gpu:gh200:1 --time=01:00:00 --pty bash
cd ~/LLM-Course-2026/Week-3/Lab-3
TASK=../../Week-2/Lab-2/configs/my-task.json
```

`TASK` is the config of your Lab 2 task. Without your own task, use
`TASK=../../Week-2/Lab-2/configs/roihu.json` (the small example). If the
session ends, start a new one and set `TASK` again.

## 3. Hugging Face: speed and accuracy

```bash
module purge
module load python-pytorch/2.13
python3 benchmark.py --backend hf --output-dir results/hf
python3 score_task.py --backend hf --task-config $TASK --output-dir results/hf
```

✓ The benchmark prints one line per concurrency level (1 and 8): time to
first token, time per token and tokens per second. The evaluation prints the
accuracy on 100 test examples.

With Hugging Face, "concurrency 8" means a batch of 8 prompts in one
`generate()` call.

## 4. vLLM: speed and accuracy

```bash
bash scripts/start_server.sh
python3 benchmark.py --backend server --output-dir results/vllm
python3 score_task.py --backend server --task-config $TASK --output-dir results/vllm
```

✓ `start_server.sh` prints `Ready: http://127.0.0.1:8000/v1` after a minute or
two. The other two commands print the same kind of lines as in step 3.

With vLLM, "concurrency 8" means 8 requests sent to the server at the same
time. Keep the server running for step 6.

## 5. Compare

```bash
python3 summarize_results.py results
python3 show_differences.py results/hf results/vllm
```

- Which is faster for one request? For 8 at a time? Why?
- Is the accuracy the same? `show_differences.py` lists the examples where the
  two answered differently.

## 6. Your own experiment

Choose **one** question, run the commands, and write a table and a paragraph
that explain the result with the lecture's ideas (prefill and decode,
batching, KV cache, memory bandwidth).

**A. Where is the knee?** How do throughput and time per token change as more
requests run at once?
```bash
python3 benchmark.py --backend server --output-dir results/vllm-knee --concurrency 1 2 4 8 16 32 64
```
Compare with Hugging Face batches:
```bash
python3 benchmark.py --backend hf --output-dir results/hf-knee --concurrency 1 4 16
```

**B. Prompt length.** How does the time to first token depend on the length
of the prompt? Write a file with short and one with long prompts (one per
line, like `configs/benchmark-prompts.txt`) and run:
```bash
python3 benchmark.py --backend server --output-dir results/vllm-long --prompts my-long-prompts.txt
```

**C. Predict, then measure.** Before measuring, estimate the maximum tokens per
second for one user from the memory bandwidth (about 4 TB/s) and the model
size (about 1.6 GB in bf16). Then compare with your concurrency-1 number and
explain the gap.

**D. Your own workload.** Use prompts and an answer length typical of your
task (`--prompts`, `--max-tokens`). Does vLLM's advantage over Hugging Face
hold for your workload?

## 7. Stop and keep the results

```bash
bash scripts/stop_server.sh
exit
```

For the final report, keep the tables from `summarize_results.py`, the
`results/` folder, `configs/model.json`, and a short discussion: when is an
inference engine worth it compared with plain Hugging Face, and what did your
own experiment show?

## Optional extensions

Run these in the GPU session, with the server from step 4.

**FP8 weights.** Restart the server with 8-bit weights and repeat step 4:
```bash
bash scripts/start_server.sh configs/model-fp8.json
python3 benchmark.py --backend server --config configs/model-fp8.json --output-dir results/vllm-fp8
python3 score_task.py --backend server --config configs/model-fp8.json --task-config $TASK --output-dir results/vllm-fp8
```

**JSON with a schema.** Ask for `{"label": ...}` with and without a JSON schema:
```bash
python3 evaluate_structured.py --task-config $TASK --output-dir results/vllm-json
```

**Your Lab 2 adapter** and **a batch job instead of an interactive session**:
see [REFERENCE.md](REFERENCE.md).

## If something goes wrong

| Problem | Fix |
|---|---|
| `srun` waits a long time | the GPUs are busy; wait, or try again later |
| `ModuleNotFoundError` in step 3 | run the Lab 2 install in step 1 |
| `The server stopped during start-up` | read the lines it prints; the full log is `.server/vllm.log` |
| the vLLM log says the model architecture is not supported | the vLLM module is too old for Qwen3.5 (needs 0.17 or newer); tell the teacher |
| `Connection refused` | the server is not running: `bash scripts/start_server.sh` |
| out of GPU memory when Hugging Face runs | stop the vLLM server first: `bash scripts/stop_server.sh` |
| accuracy 0, or many answers without a label | check `TASK` and that `lab2lib/prompting.py` is the one you used in Lab 2 |

Stuck for more than ten minutes? Ask.
