#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/.server/vllm.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No project server PID file found."
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped server PID $PID."
fi
rm -f "$PID_FILE"
