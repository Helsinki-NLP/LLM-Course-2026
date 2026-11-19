# Example task for Lab 4

A ready-made task, so that Lab 4 runs without your own Lab 2 task: natural
language inference (does the hypothesis follow from the premise?) with the
labels entailment, neutral and contradiction.

- `configs/example.json`: the task config (the default in Lab 4);
- `lab2lib/prompting.py`: the unchanged Lab 2 prompt and answer parsing;
- `data/test.jsonl`: 200 test items (67 entailment, 66 neutral, 67
  contradiction) from the SNLI test set.

To use your own task instead, give Lab 4 your Lab 2 config, for example
`../../Week-2/Lab-2/configs/my-task.json`.

Data: Bowman, Angeli, Potts and Manning (2015), *A large annotated corpus for
learning natural language inference*, EMNLP. The SNLI corpus is licensed under
CC BY-SA 4.0; this file is a small extract, reformatted, under the same
licence.
