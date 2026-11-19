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

bash "$ROOT/scripts/stop_server.sh" >/dev/null
# Use the port from the config, or a free one if someone else's server already
# uses it on this node. The other scripts read the port from .server/port.
PORT="$(python3 - "$CONFIG" <<'PY'
import json, socket, sys
def free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
port = json.load(open(sys.argv[1]))["port"]
if not free(port):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
print(port)
PY
)"
echo "$PORT" >"$ROOT/.server/port"
COMMAND="$(LAB4_PORT="$PORT" python3 "$ROOT/scripts/vllm_command.py" "$CONFIG")"

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
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "Ready: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 1
done
echo "The server was not ready after 10 minutes. See $LOG" >&2
exit 1
