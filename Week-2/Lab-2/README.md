# Lab 2: does fine-tuning beat your best prompt?

You compare **prompting** (instructions and examples, model unchanged) with
**fine-tuning** (training the model on your labelled data). Both are scored on
the same test set. You also check whether fine-tuning made the model worse at
general knowledge.

Work through the steps in order. Run every command from the `Week-2/Lab-2`
directory. Details are in [REFERENCE.md](REFERENCE.md); you do not need them
to complete the lab.

You need access to Roihu (see the [course README](../../README.md#roihu-access))
and a **classification task**: each example has some text and one label from
a fixed set, such as positive / negative. You do not need to create a
dataset: an existing labelled dataset is enough (step 5).

## What you need to do

**Minimum (enough for the report):** steps 1–11 once, with your own task and
the default settings. That gives you the results table, the loss plot and a
short discussion.

**If you want to go further** (pick any, after the minimum):

- tune the learning rate, epochs or LoRA rank, one at a time ([REFERENCE.md](REFERENCE.md#tuning));
- improve the prompt further with `try_prompt.py`: other instructions, or more few-shot examples (`prompting.few_shot_count` in the config);
- analyse the errors: which labels are confused, and what the wrong examples have in common;
- try a larger base and instruct pair from the same model family;
- compare languages, for example the same task in English and Finnish;
- adapt the scorer to a task that is not classification ([REFERENCE.md](REFERENCE.md#tasks-that-are-not-classification));
- try DPO or GRPO ([REFERENCE.md](REFERENCE.md#optional-extensions)).

## 1. Get the files

```bash
git clone https://github.com/Helsinki-NLP/LLM-Course-2026.git   # first time only
cd ~/LLM-Course-2026 && git pull                                  # later updates
cd ~/LLM-Course-2026/Week-2/Lab-2
```

## 2. Run the smoke test

Checks the scripts with a fake model. No GPU needed.

```bash
bash run_smoke_test.sh
```

✓ The last line is `Local smoke test completed.` The scores are meaningless.

## 3. Install the packages (once)

```bash
module purge
module load python-pytorch/2.13
python3 -m pip install --user -r requirements-roihu.txt
```

## 4. Run the example on a GPU

```bash
mkdir -p logs
sbatch slurm/lab2.slurm configs/roihu.json
```

Note the job number that `sbatch` prints. Then:

```bash
squeue --me                          # PD = waiting, R = running, gone = finished
tail -f logs/lab2-lora-JOBID.out     # follow the output, Ctrl+C to stop
python3 summarize_results.py --config configs/roihu.json   # when finished
```

✓ The log shows stages `1/6` to `6/6` and the table prints. The example data
is tiny, so the scores mean nothing.

## 5. Get your data

You do **not** create a dataset. You take an existing labelled dataset and
split it into three files: **train** (to learn from), **dev** (to choose the
prompt and settings) and **test** (only for the final scores).

**Option A — no data of your own yet:** use SST-2, film reviews labelled
positive or negative.

```bash
python3 from_huggingface.py --dataset stanfordnlp/sst2 --out sst2.jsonl
python3 make_splits.py --input sst2.jsonl --fields sentence --label-field label \
    --train 2000 --dev 300 --test 500 --out data/my-task
```

Other tested datasets work the same way. Change the two commands and the
`TASK_FIELDS` setting in step 6:

| Task | `--dataset` | `--fields` and `TASK_FIELDS` | Labels |
|---|---|---|---|
| film review sentiment | `stanfordnlp/sst2` | `sentence` | negative, positive |
| emotion in tweets | `dair-ai/emotion` | `text` | sadness, joy, love, anger, fear, surprise |
| news topic | `fancyzhx/ag_news` | `text` | World, Sports, Business, Sci/Tech |
| inference (two sentences) | `nyu-mll/glue --config rte` | `sentence1 sentence2` | entailment, not_entailment |

For RTE, write `TASK_FIELDS = ("sentence1", "sentence2")` and use
`--train 1500 --dev 300 --test 500`, because the dataset is small.

**Option B — your own labelled file** (CSV, TSV or JSONL, for example from
your project): name the text column(s) and the label column.

```bash
python3 make_splits.py --input mydata.csv --fields text --label-field label \
    --train 2000 --dev 300 --test 500 --out data/my-task
```

✓ `make_splits.py` prints the number of examples and the labels it found.
Labels should be words (`positive`), not numbers (`1`).

## 6. Adapt the prompt

Open `lab2lib/prompting.py` and change the two settings at the top:

```python
TASK_FIELDS = ("sentence",)    # your text column(s); keep the comma for one column
TASK_SYSTEM = (
    "Is the review positive or negative? "
    "Use exactly one of: positive, negative. "
    "End with 'Label: <label>'."   # keep this sentence: the scorer needs it
)
```

## 7. Make your config and check everything

```bash
cp configs/roihu.json configs/my-task.json
```

In `configs/my-task.json`, change:

| Entry | Example |
|---|---|
| `experiment_name` | `"my-task"` |
| `output_dir` | `"results/my-task"` |
| `training.adapter_dir` | `"results/my-task/adapter"` |
| `task.labels` | `["positive", "negative"]` |
| `data.train`, `data.dev`, `data.test` | `"data/my-task/train.jsonl"` and so on |

Then check the data and read one prompt, still without a GPU:

```bash
python3 check_data.py --config configs/my-task.json
python3 run_experiment.py --config configs/my-task.json --mock --fresh > /dev/null
python3 show_examples.py --config configs/my-task.json --run instruct-5-shot --n 1 --prompt
```

✓ `check_data.py` ends with `No errors found.` and the prompt looks right.

## 8. Try prompts (interactive GPU)

Find a good instruction before the full run. Start an interactive GPU session
(15 minutes), then try your prompt on 50 dev examples:

```bash
srun --account=project_2021893 --partition=gputest --gres=gpu:gh200:1 \
    --cpus-per-task=16 --mem=64G --time=00:15:00 --pty bash
module load python-pytorch/2.13
python3 try_prompt.py --config configs/my-task.json          # instruct, 5 examples
python3 try_prompt.py --config configs/my-task.json --all    # all four prompts
exit                                                         # free the GPU when done
```

Edit `TASK_SYSTEM` in `lab2lib/prompting.py` and run `try_prompt.py` again.
Two or three tries are enough. The test set is not used, and every try is
logged in `results/my-task/prompt-trials/log.jsonl` for your report.

✓ A table of dev accuracy and some wrong answers.

## 9. Run your experiment

```bash
sbatch --partition=gpumedium --time=02:00:00 slurm/lab2.slurm configs/my-task.json
```

Follow it as in step 4. It can take an hour or more, so submit it by
minute 60 of the session. **Do not edit `prompting.py` or the config until
the job has finished**: the job reads them when it starts, which may be later
than you submit.

## 10. Read the results

```bash
python3 summarize_results.py --config configs/my-task.json
python3 show_examples.py --config configs/my-task.json --run lora-sft --wrong
```

- Compare **the baseline selected on dev** with **LoRA SFT** on the test set.
- Read some raw outputs. A score of exactly 0 or 100, or many
  `prediction=None`, means the scorer or prompt needs fixing.
- Open `results/my-task/loss.png`. If the training loss falls while the dev
  loss rises, try fewer epochs.

## 11. Keep for the final report

- the table from `summarize_results.py`, with dataset, split sizes and models;
- `configs/my-task.json`, `lab2lib/prompting.py` and the `results/my-task/` folder;
- a short discussion: did fine-tuning beat the best prompt, at what cost, and
  what happened to general knowledge? A negative result is a valid result.

## If something goes wrong

| Problem | Fix |
|---|---|
| no log file | `mkdir -p logs`, then submit again |
| job waits as `PD` for long | `squeue --me --start` shows the estimated start |
| job stops at the time limit | longer `--time`, or smaller dev and test sets |
| `ModuleNotFoundError` | repeat step 3 (in an interactive session: `module load python-pytorch/2.13`) |
| `KeyError` with a field name | `TASK_FIELDS` does not match your data; run `check_data.py` |
| `CUDA out of memory` | in the config, halve `per_device_train_batch_size` and double `gradient_accumulation_steps` |
| scores of 0 or 100, many `None` | read outputs with `show_examples.py` and check the labels |

Stuck for more than ten minutes? Ask.
