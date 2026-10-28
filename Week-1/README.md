# Week 1

## Lecture overview

Lecture 1 introduces language models as next-token predictors. It covers
tokenization, cross-entropy and perplexity, attention, the transformer data
flow, common model families and the practical Roihu workflow.

Lecture 2 explains how language models are pretrained. Topics include the
training loop, data preparation and filtering, scaling laws, distributed
training, mid-training and the emergence of in-context learning.

## Lab 1 assignment

The core assignment is to build and inspect a small version of the pretraining
pipeline:

1. Compare tokenization efficiency for English and Finnish with at least two
   vocabulary sizes.
2. Train one small decoder-only language model and plot training and validation
   loss.
3. Record the experimental settings and explain what the results show.

Optional extensions include a third language, three-model scaling experiment,
comparison with a pretrained base model and planning a possible final-project
task.

The runnable lab reference is in `Lab-1/`. Start with:

```bash
cd Lab-1
bash run_smoke_test.sh
```

Keep the tokenizer comparison, loss curves, scaling plot, model outputs,
configuration files and scripts for the pretraining section of the final
report. The project choices made here can be revised later.
