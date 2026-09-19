#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-0}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/dynamics_raw/gpu0" "${EXP_DIR}/dynamics_raw/gpu1"

extra=()
if [[ "${LIMIT}" != "0" ]]; then
  extra=(--limit-prompts "${LIMIT}")
fi

CUDA_VISIBLE_DEVICES=0 python "${EXP_DIR}/generate_dynamics_bank_worker.py" \
  --worker-index 0 --num-workers 2 --resume "${extra[@]}" \
  >> "${EXP_DIR}/logs/dynamics_gpu0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python "${EXP_DIR}/generate_dynamics_bank_worker.py" \
  --worker-index 1 --num-workers 2 --resume "${extra[@]}" \
  >> "${EXP_DIR}/logs/dynamics_gpu1.log" 2>&1 &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
exit "${status}"
