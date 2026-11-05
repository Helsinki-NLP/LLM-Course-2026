# Lab 2 reference

Background for [README.md](README.md). You do not need to read this to
complete the lab.

## Terms

| Term | Meaning here |
|---|---|
| base model | a model after pretraining only; it continues text |
| instruct model | the same model after post-training; it follows instructions |
| zero-shot / few-shot | a prompt without / with (here five) worked examples |
| train / dev / test | for training / for choosing settings / for final scores only |
| LoRA | fine-tuning that trains a small add-on (an *adapter*) instead of the whole model |
| SFT | supervised fine-tuning: training on input–answer pairs |
| epoch | one pass over the training data |
| learning rate | how large each training update is |
| loss | how wrong the model's answers are; lower is better |
| login node / compute node | the shared machine you log in to / a GPU machine reached through Slurm |

## Data

Each split is a JSON Lines file, one example per line, with an `id`, the
input field(s) and a `label`:

```json
{"id": "train-00001", "text": "I left before the end.", "label": "negative"}
```

Suggested sizes: train 1,000–5,000, dev 200–500, test 500–1,000. The test
set must not influence any decision.

`make_splits.py` removes duplicate inputs, shuffles with a fixed seed (2026
by default) and adds IDs. To keep an official test file separate:

```bash
python3 make_splits.py --input train.csv --test-input test.csv --fields premise hypothesis --label-field label --out data/my-task
```

`check_data.py` reports missing fields, unknown labels, repeated IDs and
inputs that occur in more than one split.

### Data from Hugging Face

`from_huggingface.py` downloads one split of a dataset and replaces numeric
labels with their names:

```bash
python3 from_huggingface.py --dataset stanfordnlp/sst2 --split train --out sst2.jsonl
```

Use `--config` for datasets with several configurations and `--label-field`
if the label column is not called `label`. Examples without a label, such as
hidden test labels stored as `-1`, are skipped. Check the dataset card on
Hugging Face for the column names. SST-2's test labels are hidden, so README
step 5 splits its training set into train, dev and test.

## The prompt

The script builds prompts from `TASK_FIELDS` and `TASK_SYSTEM` in
`lab2lib/prompting.py`. A five-shot prompt looks like this:

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

The base model receives this text directly. The instruct model receives
`TASK_SYSTEM` as the system message and this text as the user message,
through its own chat template. Fine-tuning trains the instruct model to answer
`Label: <label>` to the zero-shot version of the prompt.

### Trying prompts

`try_prompt.py` scores the current prompt on the first 50 dev examples
(`--limit` changes the number) without training anything. `--all` tries all
four conditions; `--model base --shots 0` tries one specific condition;
`--mock` checks the script without a GPU. Each try is appended to
`<output_dir>-prompt-trials/log.jsonl` with the instruction and the score.
Choose the prompt on dev only. The full job (README step 9) uses the
instruction in `prompting.py` at the moment the job starts.

## Models and configuration

The default models are `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-0.8B`. If
you change them, use a base and an instruct model from the same family and
size, and the instruct model for `models.sft`. For reproducibility, replace
each `"revision": "main"` with the commit hash from the model's Hugging Face
page. `decoding.max_new_tokens` (32) is enough for one label.

## What the scripts do

- **Baselines:** base and instruct model, each zero-shot and five-shot. The
  five examples come from the training set. All four are scored on dev; the
  best is selected, and all four are then scored on test.
- **Decoding:** greedy everywhere (`temperature=0`, `do_sample=false`,
  `num_beams=1`).
- **Scoring:** one function for every condition. It reads `Label: <label>`,
  or otherwise the first label mentioned, ignoring case. The metric is
  exact-match accuracy.
- **Fine-tuning:** bf16 LoRA on all linear layers, rank 16, alpha 32, dropout
  0.05, cosine schedule with 3% warm-up, effective batch size 16. The loss
  covers only the answer (`assistant_only_loss=True`). The fine-tuned model is
  scored zero-shot.
- **General-knowledge check:** 50 questions, asked to the instruct model
  before training and to the fine-tuned model after. The change matters, not
  the score. With 50 questions, one standard error is about 7 points.
- **Provenance:** `results.json` records the seed, split sizes, settings,
  model revisions and a hash of the configuration.

## Where files are stored

Models and datasets downloaded from Hugging Face go to `HF_HOME`, which README
step 3 sets to `/scratch/project_2021893/$USER/hf`; the Slurm script uses the
same location if `HF_HOME` is not set. Everything the lab produces, including
the fine-tuned adapter, goes to the `output_dir` in your config, inside the lab
directory. Prompt tries from `try_prompt.py` go to a separate folder next to
it (`<output_dir>-prompt-trials`), because the full job clears `output_dir`
when it starts.

## Outputs

An output directory contains `results.json` (all scores and settings),
`predictions/*.jsonl` (prompt, raw output, prediction and gold label for
every example), `training_metrics.jsonl`, `loss.png` (or `loss.svg` for mock
runs) and `adapter/` (the LoRA adapter).

## Tuning

You may change `training.learning_rate` (start 0.0002),
`training.num_train_epochs` (start 2) and `training.lora_r` (start 16; keep
`lora_alpha` at twice its value). Change one at a time, give each run its own
`experiment_name`, `output_dir` and `training.adapter_dir`, and choose on the
dev loss, never on the test score. Record setting, value, training loss, dev
loss and test score for every run.

## Running single stages

For debugging, inside a Slurm job or an interactive GPU session:

```bash
python3 evaluate.py --config configs/my-task.json --target baselines
python3 evaluate.py --config configs/my-task.json --target forgetting-before
python3 sft_lora.py --config configs/my-task.json
python3 evaluate.py --config configs/my-task.json --target sft
python3 evaluate.py --config configs/my-task.json --target forgetting-after
python3 plot_losses.py --config configs/my-task.json
```

## Tasks that are not classification

For other tasks, change the answer extraction (`extract_task_label` in
`lab2lib/prompting.py`) and the metric in `lab2lib/evaluation.py`, and use
them for every condition. Ask during the lab first.

## Limits of the comparison

The five task scores use the same test set, decoding and scorer, so they are
comparable. The comparison does not rule out that the test data was in the
models' pretraining data, gives no confidence intervals, and does not show
that the training settings are the best possible. Mention these limits in
your report.

## Other environments

Outside Roihu, install everything including PyTorch in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Optional extensions

Only after the main table, loss plot and general-knowledge check are done.
No reference code is provided. Report them as extra rows in the same table.

**A — DPO.** Train with TRL's `DPOTrainer`, starting from your SFT adapter.
Each example needs `prompt`, `chosen` and `rejected` fields; you can build
pairs by sampling several outputs per training prompt and labelling correct
ones as chosen. Start with `beta=0.1`, one epoch and a learning rate of `5e-6`.
Watch for both likelihoods falling while the gap between them grows.

**B — GRPO.** Use TRL's `GRPOTrainer` with an exact-match reward plus a small
bonus for the required format; most natural for tasks with a checkable answer
such as GSM8K. Start from your SFT adapter, group size 8, about 100 steps, as a
batch job only (`gputest` with a few steps first, then `gpumedium`). Report the
reward curve.
