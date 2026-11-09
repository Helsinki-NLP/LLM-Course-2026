#!/bin/bash
# Stop the server started by start_server.sh, if it is running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/.server/server.pid"
[[ -f "$PID_FILE" ]] || exit 0
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  pkill -TERM -P "$PID" 2>/dev/null || true
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  echo "Stopped the server (PID $PID)."
fi
rm -f "$PID_FILE"
