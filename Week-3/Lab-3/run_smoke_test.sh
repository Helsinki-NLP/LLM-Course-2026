#!/bin/bash
# Run the whole Lab 3 pipeline against a fake local server: no GPU, no downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RESULTS="results/smoke"
SERVER_PID=""

stop_mock() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_mock EXIT
rm -rf "$RESULTS"
mkdir -p "$RESULTS"

for LABEL in bf16 fp8; do
  python3 mock_openai_server.py --precision "$LABEL" --port 8765 >"$RESULTS/mock-$LABEL.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 50); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=1)" 2>/dev/null && break
    sleep 0.1
  done
  CONFIG="configs/mock-$LABEL.json"
  python3 benchmark.py --server-config "$CONFIG" --output-dir "$RESULTS/$LABEL" \
    --requests-per-level 32 --warmup-requests 2 --max-tokens 6
  python3 score_task.py --server-config "$CONFIG" --output-dir "$RESULTS/$LABEL"
  if [[ "$LABEL" == "bf16" ]]; then
    python3 evaluate_structured.py --server-config "$CONFIG" --output-dir "$RESULTS/$LABEL"
  fi
  stop_mock
done

python3 summarize_results.py "$RESULTS"
python3 show_differences.py "$RESULTS" > /dev/null

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path("results/smoke")
for label in ("bf16", "fp8"):
    benchmark = json.loads((root / label / "benchmark-summary.json").read_text())
    assert [row["concurrency"] for row in benchmark["levels"]] == [1, 8, 32]
    assert all(row["requests_failed"] == 0 for row in benchmark["levels"])
bf16 = json.loads((root / "bf16" / "task-summary.json").read_text())
fp8 = json.loads((root / "fp8" / "task-summary.json").read_text())
modes = {m["mode"]: m for m in json.loads((root / "bf16" / "structured-summary.json").read_text())["modes"]}
assert bf16["score"] == 1.0 and fp8["score"] < 1.0
assert modes["free"]["invalid_rate"] > 0 and modes["schema"]["invalid_rate"] == 0
assert len(list(csv.DictReader((root / "speed-quality.csv").open()))) == 6
print("Smoke checks passed.")
PY
echo "Local smoke test completed."
