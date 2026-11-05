# Lab 2: does fine-tuning beat your best prompt?

In this lab you compare two ways of getting a language model to do your task:

- **prompting**: you write instructions, and optionally a few worked examples,
  and the model is not changed; and
- **fine-tuning**: you train the model on labelled examples of your task.

You score both on the same held-out test set and check whether fine-tuning
damaged the model's general knowledge. The result is one table, one loss plot
and a short discussion for the post-training section of your final report.
Lab 2 is not submitted separately.

The steps below take you from the bundled example to your own task. Each step
says what to run, what you should see, and what to do if something goes wrong.
If you are new to the command line, type the commands exactly as shown, one
line at a time.

## Contents

- [What you need](#what-you-need)
- [Terms used in this lab](#terms-used-in-this-lab)
- [Step 1: get the latest files](#step-1-get-the-latest-files)
- [Step 2: run the smoke test](#step-2-run-the-smoke-test)
- [Step 3: install the Python packages](#step-3-install-the-python-packages)
- [Step 4: run the example on a GPU](#step-4-run-the-example-on-a-gpu)
- [Step 5: prepare your own data](#step-5-prepare-your-own-data)
- [Step 6: adapt the prompt to your task](#step-6-adapt-the-prompt-to-your-task)
- [Step 7: create your configuration file](#step-7-create-your-configuration-file)
- [Step 8: run your experiment](#step-8-run-your-experiment)
- [Step 9: read and check your results](#step-9-read-and-check-your-results)
- [Step 10 (optional): tune the fine-tuning settings](#step-10-optional-tune-the-fine-tuning-settings)
- [Step 11: keep material for the final report](#step-11-keep-material-for-the-final-report)
- [Troubleshooting](#troubleshooting)
- [Reference](#reference)

## What you need

- Access to Roihu through course project `project_2021893`. The
  [course README](../../README.md#roihu-access) explains how to log in, either
  through the browser (JupyterLab terminal) or with SSH.
- A copy of the course repository on Roihu (Step 1).
- A **classification task**: every example has an input text (or a few text
  fields) and exactly one label from a fixed set, for example sentiment
  (positive / negative), natural language inference (entailment / neutral /
  contradiction) or topic labels. You can use the task from Lab 1 if it is a
  classification task, or any other labelled dataset. For other task types,
  see [Tasks that are not classification](#tasks-that-are-not-classification).

You do not need to write new Python code for a classification task. You change
two settings in one Python file and fill in one configuration file.

## Terms used in this lab

| Term | Meaning here |
|---|---|
| base model | a model after pretraining only; it continues text rather than answering |
| instruct model | the same model after post-training; it follows instructions in a chat format |
| zero-shot / few-shot | a prompt with no worked examples / with a few (here five) worked examples |
| train / dev / test split | examples for training / for choosing settings / for the final score only |
| LoRA | a way of fine-tuning that trains a small add-on (an *adapter*) instead of the whole model |
| SFT | supervised fine-tuning: training on input–answer pairs |
| epoch | one pass over the training data |
| learning rate | how large each training update is |
| loss | the training error; lower is better, but only the dev loss shows whether the model generalises |
| forgetting check | 50 general-knowledge questions scored before and after fine-tuning |
| login node | the Roihu machine you log in to; no GPU, shared by everyone |
| compute node | a machine with a GPU; you reach it by submitting a Slurm job |
| Slurm job | a script you hand to the queue with `sbatch`; it runs when a GPU is free |

## Step 1: get the latest files

Open a terminal on Roihu. If you have not cloned the repository yet:

```bash
git clone https://github.com/Helsinki-NLP/LLM-Course-2026.git
```

If you cloned it earlier, update it:

```bash
cd LLM-Course-2026
git pull
```

Then go to the lab directory. All commands in this README are run from here:

```bash
cd ~/LLM-Course-2026/Week-2/Lab-2
```

(Adjust the path if you cloned the repository somewhere other than your home
directory.)

## Step 2: run the smoke test

The smoke test checks that the scripts work, using a fake model and a tiny
bundled dataset. It needs no GPU and downloads nothing, so it is safe to run
on the login node:

```bash
bash run_smoke_test.sh
```

**What you should see:** a results table and, on the last line,
`Local smoke test completed.` The numbers in the table come from a fake model
and mean nothing; they only show that every part of the pipeline runs.

**If it fails:** make sure you are in `Week-2/Lab-2` (run `pwd` to check) and
that `python3 --version` shows Python 3.9 or newer.

## Step 3: install the Python packages

Do this once. On Roihu, load the course PyTorch module and install the other
packages into your home directory:

```bash
module purge
module load python-pytorch/2.13
python3 -m pip install --user -r requirements-roihu.txt
```

The Slurm script loads the same module automatically, so you do not need to
repeat these lines before each job.

*Not on Roihu?* Create a virtual environment and install everything, including
PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Step 4: run the example on a GPU

Before using your own data, run the full experiment once on the bundled
example. This checks that the models download, the GPU works and the job
finishes. It uses Qwen3.5-0.8B (base and instruct) and the short `gputest`
queue, which has a 15-minute limit.

```bash
mkdir -p logs
sbatch slurm/lab2.slurm configs/roihu.json
```

Slurm replies with `Submitted batch job 1234567`. Note the number: it is your
job ID. Then follow the job:

```bash
squeue --me                          # is the job waiting (PD) or running (R)?
tail -f logs/lab2-lora-1234567.out   # follow the output; press Ctrl+C to stop
```

**What you should see in the log:** six numbered stages,
`1/6 Scoring four prompting baselines` to `6/6 Plotting train and evaluation loss`,
followed by the results in JSON. When `squeue --me` no longer lists the job,
it has finished. Print the results table:

```bash
python3 summarize_results.py --config configs/roihu.json
```

The bundled data has only 12 training and 6 test examples, so these scores are
not meaningful. You only need the job to finish without errors. Afterwards,
`seff 1234567` shows how much time and memory the job used.

**If it fails:** look at the last 30 lines of the log with
`tail -n 30 logs/lab2-lora-1234567.out` and check the
[troubleshooting table](#troubleshooting).

## Step 5: prepare your own data

The scripts read three files in [JSON Lines](https://jsonlines.org/) format,
one example per line:

```json
{"id": "train-00001", "premise": "A dog sleeps.", "hypothesis": "An animal rests.", "label": "entailment"}
```

Every line needs an `id`, the input field or fields, and a `label`. The three
files have different jobs:

| File | Used for | Suggested size |
|---|---|---|
| `train.jsonl` | fine-tuning, and the source of few-shot examples | 1,000–5,000 |
| `dev.jsonl` | choosing the best prompt and watching the training loss | 200–500 |
| `test.jsonl` | the final scores only, after all choices are made | 500–1,000 |

The test set must not influence any decision. Do not look at test scores
while you change prompts or settings.

### Option A: start from a CSV, TSV or JSONL file

`make_splits.py` removes duplicate inputs, shuffles with a fixed seed, adds
IDs and writes the three files. For a sentiment file `reviews.csv` with the
columns `text` and `sentiment`:

```bash
python3 make_splits.py --input reviews.csv --fields text --label-field sentiment \
    --train 2000 --dev 300 --test 500 --out data/my-task
```

For two input columns, list both: `--fields premise hypothesis`. If your
dataset comes with an official test file, keep it as the test set:

```bash
python3 make_splits.py --input train.csv --test-input test.csv \
    --fields premise hypothesis --label-field label \
    --train 2000 --dev 300 --test 500 --out data/my-task
```

The script prints the labels it found. You need them in Step 7.

### Option B: start from a Hugging Face dataset

Export the dataset to a JSONL file first, then use Option A. Many datasets
store labels as numbers (`0`, `1`); the model has to write the label, so
convert the numbers to their names while exporting. For example, for SST-2 on
the login node (after the `module load` line from Step 3):

```bash
python3 - <<'EOF'
from datasets import load_dataset
data = load_dataset("stanfordnlp/sst2")["train"]        # replace with your dataset
names = data.features["label"].names                    # ["negative", "positive"]
data = data.map(lambda row: {"label_name": names[row["label"]]})
data.to_json("sst2-train.jsonl")
EOF
python3 make_splits.py --input sst2-train.jsonl --fields sentence --label-field label_name \
    --train 2000 --dev 300 --test 500 --out data/sst2
```

This example uses only the SST-2 training split, because its official test
labels are hidden. Check the dataset card on Hugging Face for the column names
and for whether the test split has labels.

### Check the data

After Steps 6 and 7, run:

```bash
python3 check_data.py --config configs/my-task.json
```

It shows the size and label distribution of each split. It reports errors
(missing fields, unknown labels, repeated IDs, the same input in two splits)
and warnings (very small splits, a label with no examples). Fix all errors
before you continue.

## Step 6: adapt the prompt to your task

Open `lab2lib/prompting.py` in an editor. In JupyterLab, double-click the file
in the file browser; in an SSH session, use `nano lab2lib/prompting.py`. Near
the top are the two things to change:

```python
TASK_FIELDS = ("premise", "hypothesis")

TASK_SYSTEM = (
    "Classify the relationship between the premise and hypothesis. "
    "Use exactly one of: entailment, neutral, contradiction. "
    "End with 'Label: <label>'."
)
```

1. **`TASK_FIELDS`** lists the input fields of your JSONL files, in the order
   they should appear in the prompt. For a single text field write
   `TASK_FIELDS = ("text",)`; the comma is needed when there is only one field.
2. **`TASK_SYSTEM`** is the instruction. Describe your task in one or two
   sentences, list the labels exactly as they appear in your data, and keep
   the last sentence `End with 'Label: <label>'.`, because the scorer looks
   for that line.

The rest of the prompt is built automatically. A five-shot prompt for a
sentiment task then looks like this:

```text
Examples:

Example 1
Text: The plot was predictable but the acting was superb.
Label: positive

... four more examples ...

Classify this item:
Text: I left before the end.

Label:
```

The base model receives this text directly. The instruct model receives it as
a user message, after the instruction as the system message, through its own
chat template. Fine-tuning trains the instruct model to answer
`Label: <label>` to the zero-shot version of the same prompt.

## Step 7: create your configuration file

All settings for one experiment are in one JSON file. Copy the Roihu example
and edit the copy:

```bash
cp configs/roihu.json configs/my-task.json
nano configs/my-task.json
```

Change these entries:

| Entry | What to write |
|---|---|
| `experiment_name` | a short name, for example `"sst2-qwen"` |
| `output_dir` | where results go, for example `"results/sst2-qwen"` |
| `training.adapter_dir` | for example `"results/sst2-qwen/adapter"` |
| `task.labels` | your labels, exactly as in the data, for example `["positive", "negative"]` |
| `data.train`, `data.dev`, `data.test` | the paths of your three files |
| `decoding.max_new_tokens` | `32` is enough for one label; increase it only if the model needs to write more |

Leave the models as they are unless you have a reason to change them. If you
do change them, choose a base and an instruct model **from the same family and
size**, and use the instruct model for `models.sft`. For reproducibility,
replace each `"revision": "main"` with the commit hash shown under *Files and
versions* on the model's Hugging Face page.

### Check the prompt before using a GPU

Make a mock copy of your configuration and run it on the login node. It uses
the fake model, so the scores are meaningless, but you can read the exact
prompts your models will receive:

```bash
python3 - <<'EOF'
import json
config = json.load(open("configs/my-task.json"))
config["backend"] = "mock"
config["output_dir"] = "results/my-task-mock"
config["training"]["adapter_dir"] = "results/my-task-mock/adapter"
config["plot"]["format"] = "svg"
json.dump(config, open("configs/my-task-mock.json", "w"), indent=2, ensure_ascii=False)
EOF
python3 check_data.py --config configs/my-task.json
python3 run_experiment.py --config configs/my-task-mock.json --fresh > /dev/null
python3 show_examples.py --config configs/my-task-mock.json --run instruct-5-shot --n 1 --prompt
```

Read the printed prompt. Check that the fields, the labels and the few-shot
examples look right. `run_smoke_test.sh` keeps using the bundled NLI data, so
after you change `TASK_FIELDS` it will fail; use this mock run instead.

## Step 8: run your experiment

Submit the job to the longer `gpumedium` queue. Options on the command line
override the 15-minute default in the Slurm script:

```bash
sbatch --partition=gpumedium --time=02:00:00 slurm/lab2.slurm configs/my-task.json
```

The job runs all six stages: four prompting baselines on dev and test, the
general-knowledge check before training, LoRA training, scoring of the
fine-tuned model, the general-knowledge check after training, and the loss
plot. Follow it as in Step 4.

The run time depends mostly on the size of the dev and test sets, because
every example is generated separately. If the job reaches its time limit,
increase `--time` or use smaller dev and test sets. Use `seff <jobid>`
afterwards to see how long the job took.

## Step 9: read and check your results

### The table

```bash
python3 summarize_results.py --config configs/my-task.json
```

This prints the scores for the four baselines on dev and test, marks the one
selected on dev, and adds the fine-tuned model and the general-knowledge check
before and after training. Your comparison is between **the selected
baseline** and **LoRA SFT** on the test set.

### The raw outputs

Before you trust a score, read what the models actually wrote:

```bash
python3 show_examples.py --config configs/my-task.json                     # list the runs
python3 show_examples.py --config configs/my-task.json --run base-zero     # ten examples
python3 show_examples.py --config configs/my-task.json --run lora-sft --wrong --n 20
```

Look for these problems:

- **Score of exactly 0 or 100:** almost always a parsing or prompt problem,
  not a real result. Check that the labels in `TASK_SYSTEM`, `task.labels` and
  the data match.
- **`prediction=None`:** the model did not write any of the labels. Is
  `max_new_tokens` large enough? Is the base model continuing the text instead
  of answering? This is common for zero-shot base models and is a finding in
  itself.
- **General knowledge near 25%:** that is chance level for four options, so
  check the knowledge outputs too (`--run knowledge-before`).

### The loss plot

`loss.png` in your output directory shows the training and dev loss. Open it
in the JupyterLab file browser. A training loss that keeps falling while the
dev loss rises means the model is memorising the training data; try fewer
epochs or a lower learning rate (Step 10).

## Step 10 (optional): tune the fine-tuning settings

You may tune three settings in `configs/my-task.json`, changing **one at a
time**:

1. `training.learning_rate` (start: `0.0002`);
2. `training.num_train_epochs` (start: `2`);
3. `training.lora_r`, the adapter size (start: `16`); keep `lora_alpha` at
   twice its value.

For each run, copy the configuration to a new file and give it a new
`experiment_name`, `output_dir` and `training.adapter_dir`, so that earlier
results are not overwritten. Keep a small table of every run: setting, value,
training loss, dev loss and test score. Choose settings on the dev loss, not
on the test score.

## Step 11: keep material for the final report

Keep the following for the post-training section of your report:

- your three data files, or the exact commands and seed that created them;
- `configs/my-task.json` and your edited `lab2lib/prompting.py`;
- the output directory: `results.json`, `predictions/` and `loss.png`;
- the table from `summarize_results.py`, with a caption naming the dataset,
  split sizes, metric and model IDs; and
- a short discussion: did fine-tuning beat the best prompt, by how much, at
  what cost in GPU time and effort, and what changed in general knowledge?

A negative result is a valid result. If the best prompt matches or beats
fine-tuning, report it and explain why you think that happened. Fifty
right-or-wrong questions give a standard error of about 7 percentage points,
so do not over-interpret a small change in general knowledge.

Keep the adapter (`training.adapter_dir`) if you want to serve your fine-tuned
model in Week 3.

## Troubleshooting

| Problem | What to do |
|---|---|
| `sbatch: error: ... account` or a job stuck in `PD` | Check the queue with `squeue --me --start`. Make sure you are a member of `project_2021893`. |
| The log file does not exist | Create the directory with `mkdir -p logs` and submit again. |
| The job stops at the time limit | Use `gpumedium` with a longer `--time`, or smaller dev and test sets. |
| `CUDA out of memory` | In your config, halve `per_device_train_batch_size` and double `gradient_accumulation_steps`, or reduce `max_length`. |
| `KeyError: 'text'` (or another field name) | `TASK_FIELDS` does not match the field names in your JSONL files. Run `check_data.py`. |
| `ModuleNotFoundError: No module named 'trl'` (or `peft`, `transformers`) | Repeat Step 3. |
| Loss is `nan` | Lower the learning rate. |
| An error mentioning the chat template or `assistant_only_loss` after you changed the models | The new instruct model's chat template does not mark the assistant turn. Go back to the default Qwen model or choose another instruct model whose template supports it. |
| Scores of exactly 0 or 100, or many `None` predictions | Read the raw outputs (Step 9). |

If you are stuck for more than ten minutes during the lab, ask.

## Reference

### Files in this directory

| File | Purpose |
|---|---|
| `run_smoke_test.sh` | local pipeline test with a fake model (Step 2) |
| `run_experiment.py` | runs all six stages for one config |
| `slurm/lab2.slurm` | Slurm script that runs `run_experiment.py` on one GH200 GPU |
| `make_splits.py` | creates train, dev and test files from one data file (Step 5) |
| `check_data.py` | checks the data files against a config (Step 5) |
| `summarize_results.py` | prints the results table (Step 9) |
| `show_examples.py` | prints raw model outputs (Step 9) |
| `evaluate.py`, `sft_lora.py`, `plot_losses.py` | run single stages (see below) |
| `lab2lib/prompting.py` | the prompt, the instruction and the answer extraction |
| `configs/smoke.json`, `configs/roihu.json` | the mock and the real example configs |
| `data/task/` | the bundled NLI example; `data/general_knowledge.jsonl` holds the 50 questions |

### What the scripts do

- **Baselines.** Four conditions: base and instruct model, each zero-shot and
  five-shot. The five examples are drawn from the training set with the fixed
  seed, never from dev or test. All four are scored on dev, the best on dev is
  selected, and all four are then scored on test. Base models receive plain
  text prompts; instruct models receive their own chat template.
- **Decoding.** Greedy decoding everywhere (`temperature=0`, `do_sample=false`,
  `num_beams=1`), so every condition is decoded in the same way.
- **Scoring.** One extraction function for every condition: it looks for
  `Label: <label>`, and otherwise takes the first label mentioned. Matching
  ignores upper and lower case. The metric is exact-match accuracy.
- **Fine-tuning.** bf16 LoRA on all linear layers with rank 16, alpha 32,
  dropout 0.05, cosine learning-rate schedule with 3% warm-up, and an
  effective batch size of 16. The loss is computed on the assistant's answer
  only (`assistant_only_loss=True`), which requires a chat template that marks
  the assistant turn. The fine-tuned model is scored zero-shot.
- **Forgetting check.** The same 50 questions are put to the instruct model
  before training and to the fine-tuned model after training. The useful
  number is the difference, not the absolute score.
- **Provenance.** `results.json` records the seed, split sizes, decoding
  settings, training settings, requested and resolved model revisions, and a
  hash of the configuration.

### Outputs

An output directory contains:

- `results.json`: all scores, the selected baseline, the forgetting check and
  the settings used;
- `predictions/*.jsonl`: for every condition, each example's ID, prompt, raw
  output, extracted prediction, gold label and whether it was correct;
- `training_metrics.jsonl`: training and dev loss by step;
- `loss.png` (real runs) or `loss.svg` (smoke test): the loss plot; and
- `adapter/`: the LoRA adapter and tokenizer files.

### Running single stages

Useful when debugging. Run the stages in this order; they all update the same
`results.json`:

```bash
python3 evaluate.py --config configs/my-task.json --target baselines
python3 evaluate.py --config configs/my-task.json --target forgetting-before
python3 sft_lora.py --config configs/my-task.json
python3 evaluate.py --config configs/my-task.json --target sft
python3 evaluate.py --config configs/my-task.json --target forgetting-after
python3 plot_losses.py --config configs/my-task.json
```

On Roihu, run these inside a Slurm job or an interactive GPU session, never on
the login node.

### Tasks that are not classification

The scripts assume one label per example from a fixed set. For other tasks,
such as short-answer question answering or generation, you need to change the
answer extraction (`extract_task_label` in `lab2lib/prompting.py`) and the
metric in `lab2lib/evaluation.py`. Use the same extraction and metric for
every condition. Ask during the lab if you plan to do this.

### What the comparison does and does not show

The test set, decoding settings and scorer are the same for all five task
scores, so the scores are directly comparable. The comparison does not rule
out that the test data was part of the models' pretraining data
(contamination), it does not give confidence intervals, and it does not show
that the chosen fine-tuning settings are the best possible ones. Mention these
limits in your discussion.

### Optional extensions

Attempt these only after the main table, loss plot and forgetting check are
complete. This directory does not include reference code for them. Report
either extension as extra rows in the same table, scored with the same
held-out set, parser, decoding settings and forgetting check.

**A — DPO.** Train with TRL's `DPOTrainer`, starting from your SFT adapter
rather than from the base model. Each example needs `prompt`, `chosen` and
`rejected` fields. Pairs can come from a public preference set or from your own
scorer: sample several completions per training prompt and label correct
completions as chosen and incorrect ones as rejected. Reasonable starting
values are `beta=0.1`, one epoch and a learning rate of `5e-6`. Watch for the
known failure mode in which the likelihoods of the chosen and rejected answers
both fall while the gap between them grows.

**B — GRPO.** Use TRL's `GRPOTrainer` with a programmatic reward: exact match
on the final answer plus a small bonus for the required output format. This is
most natural for tasks with a checkable answer, such as GSM8K. Start from your
SFT adapter with a group size of 8 and about 100 steps. Each step generates
many completions before it updates the model, so run GRPO only as a batch job:
debug on `gputest` with a few steps, then submit to `gpumedium`. One hundred
steps will not produce a reasoning model; report the reward curve and what you
learned about the method.
