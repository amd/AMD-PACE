#!/usr/bin/env bash
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
# MLPerf Server scenario. MODE=performance (default) or accuracy.
# Override any variable (e.g. MODE=accuracy TOTAL_SAMPLE_COUNT=32 ./run_mlperf.sh).
# Set LOG_FILE to tee output to a file.

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Model and data ---
CHECKPOINT_PATH="${CHECKPOINT_PATH:-meta-llama/llama-3.1-8b-instruct}"
MODEL_PATH="${MODEL_PATH:-$CHECKPOINT_PATH}"
DATASET_PATH="${DATASET_PATH:-data/cnn_eval.json}"
DTYPE="${DTYPE:-bfloat16}"

# --- LoadGen ---
USER_CONF="${USER_CONF:-user.conf}"
TOTAL_SAMPLE_COUNT="${TOTAL_SAMPLE_COUNT:-13368}"
LG_MODEL_NAME="${LG_MODEL_NAME:-llama3_1-8b}"
OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR:-output}"

# --- Server ---
ENDPOINT_URL="${ENDPOINT_URL:-http://localhost:8080/v1/completions}"
SERVER_TYPE="${SERVER_TYPE:-pace}"
TIMEOUT_SEC="${TIMEOUT_SEC:-28800}"
MAX_TOKENS="${MAX_TOKENS:-128}"

# --- Run options ---
MODE="${MODE:-performance}"
MODE_LOWER="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
case "$MODE_LOWER" in
  accuracy)     ACCURACY_ARG="--accuracy" ;;
  perf|performance) ACCURACY_ARG="" ;;
  *)            echo "MODE must be 'performance' or 'accuracy', got: $MODE" >&2; exit 1 ;;
esac
ENABLE_LOG_TRACE="${ENABLE_LOG_TRACE:-}"
LOG_FILE="${LOG_FILE:-}"

# --- Run ---
run_main() {
  python -u main.py \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --dtype "$DTYPE" \
    --user-conf "$USER_CONF" \
    --total-sample-count "$TOTAL_SAMPLE_COUNT" \
    --lg-model-name "$LG_MODEL_NAME" \
    --output-log-dir "$OUTPUT_LOG_DIR" \
    --endpoint-url "$ENDPOINT_URL" \
    --server-type "$SERVER_TYPE" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-tokens "$MAX_TOKENS" \
    $ACCURACY_ARG \
    ${ENABLE_LOG_TRACE:+--enable-log-trace} \
    "$@"
}

if [ -n "$LOG_FILE" ]; then
  run_main "$@" 2>&1 | tee "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
run_main "$@"
