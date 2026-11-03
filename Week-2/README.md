# Week 2

## Lecture overview

The lecture explains how post-training turns a base model into a useful
assistant. It covers chat templates, supervised fine-tuning with
assistant-only loss, preference data and DPO, reinforcement learning with
verifiable rewards, LoRA and the choice between prompting, retrieval and
fine-tuning.

## Lab 2 assignment

Test whether fine-tuning improves on a strong prompting baseline:

1. Run zero-shot and few-shot prompts on matching base and instruct models.
2. Choose the strongest prompt using development data.
3. Fine-tune the instruct model with bf16 LoRA supervised fine-tuning.
4. Compare the locked prompting baseline and fine-tuned model on the same
   held-out test set.
5. Compare general-knowledge performance before and after fine-tuning as a
   simple forgetting check.

The runnable lab reference is in `Lab-2/`. Start with:

```bash
cd Lab-2
bash run_smoke_test.sh
```

Choose prompting settings on development data, lock the protocol, and then
report results on the held-out test set. Reusing the Week 1 task is optional.

Lab 2 starts from a matching pretrained base and instruct pair downloaded from
Hugging Face (`Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-0.8B` by default), not
from the Week 1 model. The tiny GPT from Lab 1 is a custom PyTorch checkpoint
without an instruct version or chat template, so it cannot be used in this
pipeline.

Keep the result table, raw generations, loss curve, model and data revisions,
configuration and a short analysis of whether fine-tuning was worth the
additional effort.
