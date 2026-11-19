#!/bin/bash
# Test every Lab 4 script with fake models: no GPU, no downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="results/smoke"
rm -rf "$OUT"
mkdir -p "$OUT"
FAST="--requests 8 --warmup 1 --max-tokens 6"
# The Lab 2 task to test with: an argument, else your own task if you made
# one in Lab 2, else the bundled example.
LAB2="../../Week-2/Lab-2/configs"
if [[ -n "${1:-}" ]]; then TASK="$1"
elif [[ -f "$LAB2/my-task.json" ]]; then TASK="$LAB2/my-task.json"
else TASK="$LAB2/roihu.json"; fi
echo "Smoke test with the task $TASK"

# "Hugging Face" with a fake model
python3 benchmark.py --backend hf --config configs/mock.json --output-dir "$OUT/hf" $FAST
python3 score_task.py --backend hf --config configs/mock.json --task-config "$TASK" --output-dir "$OUT/hf"

# "vLLM" with a fake server, on a free port so that several people can run
# the smoke test on the same login node at once
PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
MOCK="$OUT/mock.json"
python3 -c 'import json, sys; c = json.load(open("configs/mock.json")); c["port"] = int(sys.argv[1]); json.dump(c, open(sys.argv[2], "w"))' "$PORT" "$MOCK"
python3 mock_openai_server.py --port "$PORT" --task-config "$TASK" >"$OUT/mock-server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
READY=0
for _ in $(seq 100); do
  if python3 -c "import sys, urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:$PORT/health', timeout=1)" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.1
done
if [[ "$READY" != 1 ]]; then
  echo "The fake server did not start. Its log:" >&2
  cat "$OUT/mock-server.log" >&2
  exit 1
fi
python3 benchmark.py --backend server --config "$MOCK" --output-dir "$OUT/vllm" $FAST
python3 score_task.py --backend server --config "$MOCK" --task-config "$TASK" --output-dir "$OUT/vllm"
python3 evaluate_structured.py --config "$MOCK" --task-config "$TASK" --output-dir "$OUT/vllm"

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
