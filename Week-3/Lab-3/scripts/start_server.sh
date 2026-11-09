#!/bin/bash
# Start the SGLang server for one config in the background and wait until it is ready.
# Usage: bash scripts/start_server.sh configs/bf16.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: start_server.sh configs/bf16.json}"
SGLANG_ENV="${SGLANG_ENV:-/projappl/project_2021893/sglang-env}"
export HF_HOME="${HF_HOME:-/scratch/project_2021893/${USER}/hf}"
mkdir -p "$HF_HOME" "$ROOT/.server"

if [[ ! -x "$SGLANG_ENV/bin/python" ]]; then
  echo "SGLang environment not found at $SGLANG_ENV." >&2
  echo "Ask the teacher, or install it yourself with: bash scripts/install_sglang.sh" >&2
  exit 2
fi

LABEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "$CONFIG")"
PORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$CONFIG")"
LOG="$ROOT/.server/sglang-$LABEL.log"
COMMAND="$(python3 "$ROOT/scripts/sglang_command.py" "$CONFIG")"

bash "$ROOT/scripts/stop_server.sh" >/dev/null
echo "Starting SGLang ($LABEL); log: $LOG"
echo "$COMMAND" >"$LOG"
(
  source "$SGLANG_ENV/bin/activate"
  exec bash -c "$COMMAND"
) >>"$LOG" 2>&1 &
echo $! >"$ROOT/.server/server.pid"

for _ in $(seq 600); do
  if ! kill -0 "$(cat "$ROOT/.server/server.pid")" 2>/dev/null; then
    echo "The server stopped during start-up. Last lines of the log:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "Ready: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 1
done
echo "The server was not ready after 10 minutes. See $LOG" >&2
exit 1
