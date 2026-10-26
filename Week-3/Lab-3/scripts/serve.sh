#!/bin/bash
# Start vLLM in the foreground from a clean Roihu module environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-bf16}"
CONFIG="$ROOT/configs/${CONFIG_NAME}.json"

if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown config '$CONFIG_NAME' (expected configs/$CONFIG_NAME.json)." >&2
  exit 2
fi
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # Non-interactive shells do not always initialize Environment Modules.
  source /etc/profile.d/modules.sh
fi
if ! command -v module >/dev/null 2>&1; then
  echo "The 'module' command is unavailable. Run this inside a Roihu allocation." >&2
  exit 2
fi

module purge
module load python-vllm
if [[ -z "${HF_HOME:-}" && -z "${PROJECT:-}" ]]; then
  echo "Set PROJECT to the Roihu project name, or set HF_HOME explicitly." >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/scratch/${PROJECT}/hf}"

exec python3 "$ROOT/scripts/serve_vllm.py" --config "$CONFIG"
