#!/bin/bash
# Measure one already-running server configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG_NAME="${1:?usage: measure.sh CONFIG_NAME OUTPUT_DIR [--structured]}"
OUTPUT_DIR="${2:?usage: measure.sh CONFIG_NAME OUTPUT_DIR [--structured]}"
STRUCTURED="${3:-}"
CONFIG="configs/${CONFIG_NAME}.json"

read_config() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))$2)" "$CONFIG"
}
PORT="$(read_config port '["port"]')"
MODEL="$(read_config model '["served_model_name"]')"
REVISION="$(read_config revision '["model_revision"]')"
ADAPTER_REVISION="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("adapter") or {}).get("revision", "none"))' "$CONFIG")"
BASE_URL="http://127.0.0.1:${PORT}/v1"

python3 benchmark.py \
  --base-url "$BASE_URL" --model "$MODEL" --config-label "$CONFIG_NAME" \
  --server-config "$CONFIG" --model-revision "$REVISION" \
  --adapter-revision "$ADAPTER_REVISION" --output-dir "$OUTPUT_DIR"

python3 score_task.py \
  --base-url "$BASE_URL" --model "$MODEL" --config-label "$CONFIG_NAME" \
  --model-revision "$REVISION" --adapter-revision "$ADAPTER_REVISION" \
  --output-dir "$OUTPUT_DIR"

if [[ "$STRUCTURED" == "--structured" ]]; then
  python3 evaluate_structured.py \
    --base-url "$BASE_URL" --model "$MODEL" --config-label "$CONFIG_NAME" \
    --model-revision "$REVISION" --adapter-revision "$ADAPTER_REVISION" \
    --output-dir "$OUTPUT_DIR"
fi
