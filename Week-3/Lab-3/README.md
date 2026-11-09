# Lab 3: serve your model and measure it

You run your model behind a server, as it would be used in practice, and
measure three things:

- **speed**: how fast answers start and how many tokens per second the server
  produces when 1, 8 or 32 requests arrive at the same time;
- **bf16 versus FP8**: whether storing the weights in 8 bits makes the server
  faster, and whether it changes the answers on your Lab 2 test set;
- **valid output**: how often the model's JSON answers are broken, with and
  without a JSON schema that constrains the output.

The server is [SGLang](https://docs.sglang.ai), an inference engine. Lab 3
uses the same task, test set, prompt and scorer as your Lab 2, so the quality
numbers are comparable.

Run every command from the `Week-3/Lab-3` directory, in an **SSH session on
the GPU side of Roihu** (`ssh USERNAME@roihu-gpu.csc.fi`). You do not have to
finish in the session: you can continue on your own time. Details are in
[REFERENCE.md](REFERENCE.md); you do not need them to complete the lab.

## What you need to do

**Minimum (enough for the report):** steps 1–6 with your Lab 2 task and the
default settings. That gives you a speed table, a quality table and a JSON
table, and a short discussion.

**If you want to go further** (pick any, after the minimum):

- serve your Lab 2 adapter and compare it with the plain model ([REFERENCE.md](REFERENCE.md#serving-your-lab-2-adapter));
- try a larger model, for example `Qwen/Qwen3.5-4B`, where FP8 usually makes a bigger difference;
- replace the benchmark prompts with prompts typical of your task;
- start the server interactively and send it your own requests ([REFERENCE.md](REFERENCE.md#using-the-server-interactively)).

## 1. Get the latest files

```bash
cd ~/LLM-Course-2026 && git pull
cd ~/LLM-Course-2026/Week-3/Lab-3
```

## 2. Run the smoke test

Checks the scripts with a fake server. No GPU needed.

```bash
bash run_smoke_test.sh
```

✓ The last line is `Local smoke test completed.` The numbers are meaningless.

## 3. Check that SGLang is installed

The course provides one shared SGLang installation. Check that it is there:

```bash
ls /projappl/project_2021893/sglang-env/bin/python
```

✓ The path is printed. If you get "No such file or directory", ask the
teacher, or install your own copy (about 15 minutes, several GB) and tell the
scripts where it is:

```bash
bash scripts/install_sglang.sh /scratch/project_2021893/$USER/sglang-env
echo 'export SGLANG_ENV=/scratch/project_2021893/$USER/sglang-env' >> ~/.bashrc
source ~/.bashrc
```

## 4. Run the experiment

Give the job the config file of your Lab 2 task. Without it, the job uses the
small example task from Lab 2.

```bash
mkdir -p logs
sbatch slurm/lab3.slurm ../../Week-2/Lab-2/configs/my-task.json
```

Follow it as in Lab 2:

```bash
squeue --me
tail -f logs/lab3-JOBID.out
```

The job starts the bf16 server, measures speed, scores your test set, compares
JSON with and without a schema, and then does the same for FP8 (without the
JSON part). It should take well under the one-hour limit; most of the time
goes into starting the server twice.

✓ The log ends with three tables.

## 5. Read the results

```bash
python3 summarize_results.py results/lab3-JOBID
python3 show_differences.py results/lab3-JOBID
```

- **Speed:** how do the time to first token, the time per token and the
  tokens per second change from 1 to 8 to 32 parallel requests? Is FP8 faster?
- **Quality:** is FP8's accuracy different? `show_differences.py` lists the
  test items where bf16 and FP8 disagree.
- **JSON:** how many answers are invalid without a schema, and with one?

## 6. Keep for the final report

- the three tables from `summarize_results.py`, with the model and task;
- the `results/lab3-JOBID/` folder and `configs/bf16.json` and `configs/fp8.json`;
- a short recommendation: would you serve this model in bf16 or FP8, with or
  without a JSON schema, and why? Mention what the measurements cannot show
  (see [REFERENCE.md](REFERENCE.md#limits-of-the-measurements)).

## If something goes wrong

| Problem | Fix |
|---|---|
| `SGLang environment not found` | see step 3 |
| `The server stopped during start-up` | read the log it prints; the full log is in `.server/sglang-bf16.log` or `.server/sglang-fp8.log` |
| out of GPU memory at start-up | lower `mem_fraction_static` (e.g. to `0.7`) in both configs |
| FP8 fails but bf16 works | report it: FP8 support depends on the model and the SGLang version |
| job waits as `PD` for long | `squeue --me --start` shows the estimated start |
| `ERROR: this job is not on a GH200 node` | submit from `ssh USERNAME@roihu-gpu.csc.fi` |
| accuracy 0, or many answers with no label | check that the Lab 2 config and `lab2lib/prompting.py` are the ones you used in Lab 2 |

Stuck for more than ten minutes? Ask.
