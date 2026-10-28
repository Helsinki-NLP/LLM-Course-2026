# Lab 1 example: from text to a tiny GPT

This directory contains a reference implementation for Lab 1. It is deliberately
small enough to read: data preparation, tokenization, the decoder-only model,
training and plotting are separate scripts.

Lab 1 is not an intermediate submission. Keep the code and results for the
pretraining section of the final report.

## What the lab demonstrates

- **Part A — tokenization:** vocabulary size changes sequence length and the
  number of tokens used for different languages.
- **Part B — pretraining:** a decoder learns through next-token prediction, and
  its training and validation losses should fall smoothly.
- **Part C — scaling:** at a fixed token budget, changing model size changes
  both validation loss and estimated compute, \(C \approx 6ND\).
- **Part D — in-context learning:** the same few-shot prompt behaves differently
  in a tiny model and a larger pretrained base model.
- **Part E — project planning:** define the task, model and held-out data that
  you may use in later labs. Reusing them is recommended, not required.

## Core requirements

The minimum work for Lab 1 is:

1. Run `run_smoke_test.sh` to check that the pipeline works.
2. Compare tokenization for Finnish and English using at least two vocabulary
   sizes.
3. Train one of the provided tiny GPT configurations.
4. Plot its training and validation loss.
5. Record the parameter count, tokens processed, wall-clock time and estimated
   compute.
6. Briefly explain whether the model learned, what evidence supports your
   conclusion and what the experiment cannot show.

Keep the configuration, code, metrics and plots for the final submission. There
is no required loss threshold: reproducible evidence and a reasonable
interpretation matter more than obtaining a particular number.

## Optional extensions

For a more complete experiment, you can:

- compare three languages and all three vocabulary sizes;
- train the three supplied model sizes with the same tokenizer, token budget
  and seed;
- plot validation loss against estimated compute;
- compare the tiny model with a larger pretrained base model; and
- complete the project-planning file for a task you may use in later labs.

A strong Week 1 report section is usually one or two pages. It contains a
clearly labelled tokenization table and loss figure, states the main
experimental settings, interprets the results and discusses unexpected
outcomes or limitations. Include the scaling plot and generation comparison if
you complete those extensions.

## 1. Set up the environment

On a normal Linux machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Roihu, use the course PyTorch module and install
`requirements-roihu.txt`, which deliberately excludes PyTorch. Installing a
second CUDA build with pip can replace the module's compatible build. Do not
create a new environment inside a batch job.

```bash
module purge
module load python-pytorch/2.13
python3 -m pip install --user -r requirements-roihu.txt
```

Before downloading FLORES+, log in to Hugging Face and accept the dataset terms:

```bash
huggingface-cli login
```

## 2. Run the local smoke test first

The bundled sample text checks the complete pipeline without a GPU or network:

```bash
bash run_smoke_test.sh
```

It produces a tokenizer, token files, a very small model, JSONL training logs
and two plots/tables under `results/`. The sample is only for debugging; do not
use its numbers in the report.

## 3. Part A: tokenizer and fertility

Download a manageable Finnish Wikipedia slice and FLORES+ in three languages:

```bash
python download_data.py --max-articles 50000 --flores \
  --languages eng_Latn fin_Latn deu_Latn
```

Train byte-level BPE tokenizers:

```bash
python train_tokenizer.py data/wikipedia-train.txt \
  --vocab-sizes 8000 16000 32000
```

Compare tokens per word. These are decoder-only GPT-style model tokenizers, not
embedding models:

```bash
python measure_fertility.py \
  --language English=data/flores-eng_Latn.txt \
  --language Finnish=data/flores-fin_Latn.txt \
  --language German=data/flores-deu_Latn.txt \
  --tokenizer BPE-8k=tokenizers/bpe-8000/tokenizer.json \
  --tokenizer BPE-16k=tokenizers/bpe-16000/tokenizer.json \
  --tokenizer BPE-32k=tokenizers/bpe-32000/tokenizer.json \
  --hf-model Qwen/Qwen3.5-4B-Base \
  --hf-model google/gemma-4-E4B \
  --output results/fertility.csv
```

Gemma is gated: accept its licence on Hugging Face before running the command.
Fertility here means tokens divided by whitespace-delimited words. State that
definition in the report; word segmentation is not equally natural for every
language.

## 4. Part B: train one tiny GPT

Encode the corpus once:

```bash
python prepare_tokens.py \
  --tokenizer tokenizers/bpe-32000/tokenizer.json \
  --train data/wikipedia-train.txt \
  --valid data/wikipedia-valid.txt \
  --output-dir data/tokens-32k
```

Debug a short run before requesting a longer allocation:

```bash
mkdir -p logs
sbatch slurm/train.slurm configs/smoke.yaml
```

For the real run, change the Slurm partition to `gpumedium`, request enough
time, and use one of the real configurations:

```bash
sbatch slurm/train.slurm configs/medium.yaml
```

Each run writes:

- `metrics.jsonl`: step, tokens, learning rate, losses, elapsed time and
  throughput;
- `summary.json`: architecture, parameter count, token count and \(6ND\);
- `model.pt`: model weights and configuration.

Useful checks:

1. At initialization, loss should be close to \(\ln(V)\), where \(V\) is the
   vocabulary size.
2. Training and validation loss should both fall.
3. Validation loss should not start rising while training loss continues down.
4. Compare losses only for runs using the same tokenizer and validation data.

Plot the training and validation curves:

```bash
python3 plot_losses.py runs/medium --output results/medium-loss.png
```

The command also prints the final training and validation losses.

## 5. Part C: scaling experiment (optional extension)

The three supplied configurations use the same 32k tokenizer, sequence length,
token budget and seed. Their approximate sizes are 11M, 18M and 27M parameters.

```bash
sbatch slurm/train.slurm configs/small.yaml
sbatch slurm/train.slurm configs/medium.yaml
sbatch slurm/train.slurm configs/large.yaml
```

After all three finish:

```bash
python plot_scaling.py runs/small runs/medium runs/large \
  --output results/scaling.png
```

This also writes `results/scaling.csv`, which should be the source for the
table in the report. A three-point plot is an illustration, not enough evidence
to fit a new scaling law.

The stated run times are planning estimates. Measure the packed loader and all
three configurations on Roihu before promising a fixed duration.

## 6. Part D: what scale buys (optional extension)

Run the same few-shot sentiment prompt on your model and a larger base model:

```bash
python compare_generation.py \
  --run runs/medium \
  --tokenizer tokenizers/bpe-32000/tokenizer.json \
  --hf-model Qwen/Qwen2.5-0.5B \
  --output results/generation-comparison.json
```

The script uses greedy decoding and records the prompt hash, model revision and
decoding settings. The 0.5B model keeps the example quick. You may substitute a
larger base model that fits the available GPU. Use a **base** model, not an
instruct model.

This is qualitative rather than a controlled experiment in scale: the
architecture, tokenizer and training data also differ.

## 7. Part E: plan the final project (recommended, not required)

Copy `project-plan.yaml` and fill it in for yourself. It is a planning aid, not
a separate submission. Keep the held-out test set untouched until the final
evaluation.

## Material for the final report

For the core lab, include:

1. the Finnish–English fertility comparison and your interpretation;
2. training and validation loss curves;
3. the parameter count, token count, runtime and estimated compute;
4. a short explanation of what the results show and what they do not show; and
5. links to the exact scripts, configuration and result files.

If you complete the extensions, also include the scaling plot with model sizes,
token budgets and seeds, and the few-shot outputs with a short comparison.

Report unexpected results and what you checked. Reproducible evidence matters
more than obtaining a particular ordering of the three models.
