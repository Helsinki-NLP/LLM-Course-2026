#!/bin/bash
# Stop this project's previous server, start a new one, and wait until it is ready.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-bf16}"
CONFIG="$ROOT/configs/${CONFIG_NAME}.json"
STATE_DIR="$ROOT/.server"
PID_FILE="$STATE_DIR/vllm.pid"
LOG_FILE="$STATE_DIR/vllm.log"

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG" >&2; exit 2; }
mkdir -p "$STATE_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Stopping prior server (PID $OLD_PID)..."
    kill "$OLD_PID"
    for _ in {1..30}; do
      kill -0 "$OLD_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$OLD_PID" 2>/dev/null; then
      echo "Server did not stop cleanly; inspect PID $OLD_PID." >&2
      exit 1
    fi
  fi
  rm -f "$PID_FILE"
fi

PORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$CONFIG")"
echo "Starting $CONFIG_NAME; log: $LOG_FILE"
nohup bash "$ROOT/scripts/serve.sh" "$CONFIG_NAME" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"

for _ in {1..300}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup. Last log lines:" >&2
    tail -n 30 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "Ready: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for vLLM. Inspect $LOG_FILE" >&2
exit 1
