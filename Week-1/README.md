# Week 1

## Lecture overview

Lecture 1 introduces language models as next-token predictors. It covers
tokenization, cross-entropy and perplexity, attention, the transformer data
flow, common model families and the practical Roihu workflow.

Lecture 2 explains how language models are pretrained. Topics include the
training loop, data preparation and filtering, scaling laws, distributed
training, mid-training and the emergence of in-context learning.

## Lab 1 assignment

Build a small version of the pretraining pipeline:

1. Train byte-level BPE tokenizers at several vocabulary sizes.
2. Compare tokenization efficiency for English, Finnish and a third language.
3. Train a small decoder-only language model and plot training and validation
   loss.
4. Repeat the run at three model sizes with a fixed token budget, and compare
   validation loss with estimated compute.
5. Run the same few-shot prompt on the small model and a pretrained base
   model.
6. Identify a possible task, model and held-out dataset for the final project.

The runnable lab reference is in `Lab-1/`. Start with:

```bash
cd Lab-1
bash run_smoke_test.sh
```

Keep the tokenizer comparison, loss curves, scaling plot, model outputs,
configuration files and scripts for the pretraining section of the final
report. The project choices made here can be revised later.
