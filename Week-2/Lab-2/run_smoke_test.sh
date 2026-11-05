#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

python3 check_data.py --config configs/smoke.json
python3 run_experiment.py --config configs/smoke.json --fresh
python3 smoke_checks.py --results results/smoke/results.json
python3 summarize_results.py --config configs/smoke.json
python3 show_examples.py --config configs/smoke.json --run instruct-zero --n 2 > /dev/null
python3 try_prompt.py --config configs/smoke.json --mock --all > /dev/null

echo "Local smoke test completed."
