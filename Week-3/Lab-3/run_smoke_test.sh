#!/bin/bash
# Test every Lab 3 script with fake models: no GPU, no downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="results/smoke"
rm -rf "$OUT"
mkdir -p "$OUT"
FAST="--requests 8 --warmup 1 --max-tokens 6"

# "Hugging Face" with a fake model
python3 benchmark.py --backend hf --config configs/mock.json --output-dir "$OUT/hf" $FAST
python3 score_task.py --backend hf --config configs/mock.json --output-dir "$OUT/hf"

# "vLLM" with a fake server
python3 mock_openai_server.py --port 8765 >"$OUT/mock-server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do
  python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=1)" 2>/dev/null && break
  sleep 0.1
done
python3 benchmark.py --backend server --config configs/mock.json --output-dir "$OUT/vllm" $FAST
python3 score_task.py --backend server --config configs/mock.json --output-dir "$OUT/vllm"
python3 evaluate_structured.py --config configs/mock.json --output-dir "$OUT/vllm"

python3 summarize_results.py "$OUT"
python3 show_differences.py "$OUT/hf" "$OUT/vllm" >/dev/null

python3 - <<'PY'
import json
from pathlib import Path

root = Path("results/smoke")
for run in ("hf", "vllm"):
    speed = json.loads((root / run / "speed-summary.json").read_text())
    assert [level["concurrency"] for level in speed["levels"]] == [1, 8]
    assert all(level["failed"] == 0 for level in speed["levels"])
    assert json.loads((root / run / "eval-summary.json").read_text())["accuracy"] == 1.0
modes = {m["mode"]: m for m in json.loads((root / "vllm" / "structured-summary.json").read_text())["modes"]}
assert modes["free"]["invalid_rate"] > 0 and modes["schema"]["invalid_rate"] == 0
assert modes["schema-reasoning"]["invalid_rate"] == 0
print("Smoke checks passed.")
PY
echo "Local smoke test completed."
