#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

python3 run_experiment.py --config configs/smoke.json --fresh
python3 smoke_checks.py --results results/smoke/results.json

echo "Local smoke test completed."
