#!/bin/bash
# Exercise the complete measurement pipeline without a GPU or model download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
RESULTS="$ROOT/results/smoke"
BASE_URL="http://127.0.0.1:8765/v1"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
rm -rf "$RESULTS"
mkdir -p "$RESULTS"

start_mock() {
  local precision="$1"
  cleanup
  "$PYTHON" mock_openai_server.py --precision "$precision" --port 8765 \
    >"$RESULTS/mock-${precision}.log" 2>&1 &
  SERVER_PID=$!
  for _ in {1..50}; do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$BASE_URL/models', timeout=1)" \
      >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
  echo "Mock server did not become ready." >&2
  exit 1
}

run_configuration() {
  local precision="$1"
  local config_dir="$RESULTS/$precision"
  start_mock "$precision"
  "$PYTHON" benchmark.py \
    --base-url "$BASE_URL" --model task --config-label "$precision" \
    --server-config "configs/mock-${precision}.json" \
    --model-revision mock-v1 --adapter-revision none \
    --output-dir "$config_dir" --concurrency 1 8 32 \
    --requests-per-level 32 --warmup-requests 2 --max-tokens 6
  "$PYTHON" score_task.py \
    --base-url "$BASE_URL" --model task --config-label "$precision" \
    --model-revision mock-v1 --adapter-revision none \
    --output-dir "$config_dir"
}

run_configuration bf16
"$PYTHON" evaluate_structured.py \
  --base-url "$BASE_URL" --model task --config-label bf16 \
  --model-revision mock-v1 --adapter-revision none \
  --output-dir "$RESULTS/bf16"
run_configuration fp8
cleanup
SERVER_PID=""

"$PYTHON" summarize_results.py \
  --bf16-dir "$RESULTS/bf16" \
  --fp8-dir "$RESULTS/fp8" \
  --structured-dir "$RESULTS/bf16" \
  --output-dir "$RESULTS"

"$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

root = Path("results/smoke")
for precision in ("bf16", "fp8"):
    benchmark = json.loads((root / precision / "benchmark-summary.json").read_text())
    assert [row["concurrency"] for row in benchmark["levels"]] == [1, 8, 32]
    assert all(row["requests_completed"] == 32 for row in benchmark["levels"])
    assert all(row["requests_failed"] == 0 for row in benchmark["levels"])

bf16 = json.loads((root / "bf16" / "task-summary.json").read_text())
fp8 = json.loads((root / "fp8" / "task-summary.json").read_text())
structured = json.loads((root / "bf16" / "structured-summary.json").read_text())
modes = {row["mode"]: row for row in structured["modes"]}
assert bf16["score"] == 1.0
assert fp8["score"] < bf16["score"]
assert modes["free"]["invalid_rate"] == 0.25
assert modes["schema"]["invalid_rate"] == 0.0
with (root / "speed-quality.csv").open(newline="") as handle:
    assert len(list(csv.DictReader(handle))) == 3
print("Smoke assertions passed.")
PY

echo "End-to-end smoke test completed: $RESULTS"
