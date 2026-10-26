# Lab 2 reference: prompting versus LoRA

This directory is a complete, readable example of one controlled comparison:

1. run zero-shot and few-shot prompts on matching base and instruct models;
2. choose the strongest of those four on the development split;
3. train an instruct model with bf16 LoRA supervised fine-tuning (SFT);
4. score every method on the same held-out test split with the same parser and
   exact-match metric;
5. compare general-knowledge accuracy before and after SFT; and
6. save the results and a train/evaluation loss plot.

The bundled task is a tiny three-way natural-language inference example. It is
useful for checking code, but far too small for a scientific conclusion.
Replace its JSONL files with your task data once the pipeline works. Keeping the
same task or model in later course work can make comparisons easier, but it is
optional.

## The two execution paths

`configs/smoke.json` uses a deterministic heuristic backend, bundled data, and
only the Python standard library. It does not import PyTorch, contact Hugging
Face, download a model, or require a GPU. Its synthetic loss values and scores
are pipeline checks, not model results.

`configs/roihu.json` uses Hugging Face Transformers for generation and
TRL/PEFT for real bf16 LoRA training. Both paths call the same prompt builders,
answer extractors, held-out scorer, result writer, and plotting entry point.

## Run the local smoke test

From this directory:

```bash
bash run_smoke_test.sh
```

The command runs all four baselines, mock SFT, the before/after knowledge
check, loss plotting, and artifact validation. It writes to `results/smoke/`.
Running it again with the same files and seed gives the same predictions and
scores.

## Set up the real environment

For a machine where pip should install PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Roihu, load the course PyTorch module and install the dependency list that
does not contain PyTorch:

```bash
module load pytorch
python -m pip install --user -r requirements-roihu.txt
```

## Inspect and adapt the data

The task files use one JSON object per line:

```json
{"id":"train-001","premise":"A dog sleeps.","hypothesis":"An animal rests.","label":"entailment"}
```

The required splits are:

- `train.jsonl`: SFT examples and the source of few-shot demonstrations;
- `dev.jsonl`: prompt selection, evaluation loss and hyperparameter choices;
- `test.jsonl`: one held-out set used by every reported task score.

The scorer expects one of the labels listed in `task.labels` and extracts the
shared `Label: <label>` output format. For another classification task, change
the prompt text in `lab2lib/prompting.py`, the labels in the config, and your
JSONL fields. Keep one extraction function for all five task conditions.

`data/general_knowledge.jsonl` contains 50 unrelated multiple-choice questions.
The check is a regression test: the useful number is the after-minus-before
change, not the standalone benchmark score. Fifty binary observations are
noisy, so do not over-interpret a small difference.

## Configure models and reproducibility

Edit `configs/roihu.json` before a real run:

- choose matching base and instruct model IDs;
- choose the instruct checkpoint under `models.sft`;
- replace each `revision: "main"` with a Hugging Face commit hash when
  possible;
- point the split paths at your data;
- adjust `max_new_tokens` for your output format;
- choose an output and adapter directory.

The example records the requested revision and the resolved Hub commit exposed
by Transformers. It also stores the seed, split sizes, complete decoding
settings, training settings, prompts, raw generations, extracted predictions,
and config hash. Greedy decoding is fixed globally: `temperature=0`,
`do_sample=false`, and `num_beams=1`.

The supplied SFT settings are bf16, assistant-only loss, rank 16, alpha 32,
dropout 0.05, all linear layers, cosine scheduling, 3% warmup, and an effective
batch size of 16. TRL requires a chat template that can identify assistant
tokens for `assistant_only_loss=True`. If your model's template lacks that
support, choose a compatible instruct model or provide a correct template;
silently training on prompt tokens changes the experiment.

## Run on Roihu

Create the log directory, put your project number in the Slurm file, and first
use the short `gputest` queue:

```bash
mkdir -p logs
sbatch slurm/lab2.slurm configs/roihu.json
```

After that short run is healthy, choose a suitable longer partition and wall
time in `slurm/lab2.slurm`. The one-command runner executes:

```bash
python run_experiment.py --config configs/roihu.json --fresh
```

Model loading is repeated between phases to keep each evaluator independent.
That is simple and auditable, though not the fastest possible implementation.
For a large held-out set, separate the phases or cache already downloaded model
files.

Each phase is also available on its own, which is helpful while debugging:

```bash
python evaluate.py --config configs/roihu.json --target baselines
python evaluate.py --config configs/roihu.json --target forgetting-before
python sft_lora.py --config configs/roihu.json
python evaluate.py --config configs/roihu.json --target sft
python evaluate.py --config configs/roihu.json --target forgetting-after
python plot_losses.py --config configs/roihu.json
```

Run the phases in that order. The individual commands update one shared
`results.json`; the one-command runner initializes it automatically.

## Outputs

An experiment directory contains:

- `results.json`: machine-readable configuration, provenance, all scores, the
  selected best prompt (chosen on development data), test scores, forgetting
  delta, and artifact paths;
- `predictions/*.jsonl`: IDs, prompts, raw generations, extracted answers,
  gold labels, and correctness for every condition;
- `training_metrics.jsonl`: step-level train and evaluation losses;
- `loss.png` (real path) or `loss.svg` (smoke path): both losses on one plot;
- `adapter/`: PEFT adapter weights and tokenizer files.

Read a sample of raw generations before trusting a score. A score of exactly
zero or one often indicates an extraction or prompt-format problem. Base
models receive plain completion prompts; only instruct models use their own
chat template.

## What the comparison does and does not establish

The shared test set, decoding settings, and parser make the five task scores
directly comparable. The few-shot examples are sampled deterministically from
training data, never from the test set. The general-knowledge questions are
also identical before and after SFT.

The example does not remove possible benchmark contamination from model
pretraining, estimate uncertainty for a tiny task fixture, or prove that one
hyperparameter setting is optimal. A useful analysis reports these limits,
the GPU and human effort, and whether any score change justified the SFT run.
