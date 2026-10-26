#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

"$PYTHON" train_tokenizer.py data/sample-train.txt \
  --vocab-sizes 512 \
  --output-root tokenizers-smoke

"$PYTHON" measure_fertility.py \
  --language English=data/sample-flores-eng_Latn.txt \
  --language Finnish=data/sample-flores-fin_Latn.txt \
  --language German=data/sample-flores-deu_Latn.txt \
  --tokenizer LabBPE=tokenizers-smoke/bpe-512/tokenizer.json \
  --output results/smoke-fertility.csv

"$PYTHON" prepare_tokens.py \
  --tokenizer tokenizers-smoke/bpe-512/tokenizer.json \
  --train data/sample-train.txt \
  --valid data/sample-valid.txt \
  --output-dir data/smoke-tokens

"$PYTHON" train_tiny_gpt.py --config configs/smoke.yaml --device cpu
"$PYTHON" plot_scaling.py runs/smoke --output results/smoke-scaling.png
"$PYTHON" smoke_checks.py

echo "Smoke test completed."
