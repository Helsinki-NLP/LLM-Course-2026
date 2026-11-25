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
# vLLM needs a GPU. On the login node, or in a session without one, stop here.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && ! compgen -G "/dev/nvidia[0-9]*" >/dev/null; then
  echo "ERROR: this session has no GPU, so vLLM cannot start." >&2
  echo "Start a GPU session first (README step 2):" >&2
  echo "  srun --account=project_2021893 --partition=gpuinteractive --gres=gpu:gh200:1 --time=01:00:00 --pty bash" >&2
  echo "or use the batch job: sbatch slurm/lab4.slurm" >&2
  exit 1
fi
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
    echo "The server stopped during start-up. The error lines in the log:" >&2
    grep -i -E "error|out of memory|not supported|exception" "$LOG" | grep -v 'File "' | head -n 20 >&2 || true
    echo "--- last lines of $LOG:" >&2
    tail -n 15 "$LOG" >&2
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
