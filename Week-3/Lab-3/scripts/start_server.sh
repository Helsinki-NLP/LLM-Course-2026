#!/bin/bash
# Start the vLLM server in the background and wait until it is ready.
# Usage: bash scripts/start_server.sh [configs/model.json]
# Run it inside an interactive GPU session. Stop it with scripts/stop_server.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/model.json}"
LOG="$ROOT/.server/vllm.log"
export HF_HOME="${HF_HOME:-/scratch/project_2021893/${USER}/hf}"
mkdir -p "$HF_HOME" "$ROOT/.server"

PORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$CONFIG")"
COMMAND="$(python3 "$ROOT/scripts/vllm_command.py" "$CONFIG")"
bash "$ROOT/scripts/stop_server.sh" >/dev/null

echo "Starting vLLM; log: $LOG"
echo "$COMMAND" >"$LOG"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh  # non-interactive shells may lack "module"
fi
(
  module purge
  module load python-vllm
  # Keep packages installed with pip --user (e.g. for Lab 2) out of vLLM's way.
  export PYTHONNOUSERSITE=1
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
