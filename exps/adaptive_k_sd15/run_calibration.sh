#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/exps/adaptive_k_sd15"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
DENSE_BANK="${DENSE_BANK:-${ROOT}/exps/dense_calibration_200}"

mkdir -p "${EXP}/calibration/plots" "${EXP}/logs"
export HF_HOME="/workspace/.hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="/workspace/.cache"
export PYTHONUNBUFFERED=1

PY="${GEN_VENV}/bin/python"
[[ -x "${PY}" ]] || { echo "Missing environment ${GEN_VENV}" >&2; exit 1; }
"${PY}" -c "import sklearn, pandas, pyarrow" 2>/dev/null || "${PY}" -m pip install scikit-learn joblib

echo "[1/3] build compact calibration bank from the independent ImageReward corpus"
"${PY}" "${EXP}/prepare_bank.py" --source-dense-bank "${DENSE_BANK}" | tee "${EXP}/logs/prepare_bank.log"

echo "[2/3] offline 10-seed subset simulation (seed 20260919, 1000 subsets/prompt)"
"${PY}" "${EXP}/simulate_subsets.py" | tee "${EXP}/logs/simulate_subsets.log"

echo "[3/3] prompt-grouped 5-fold OOF, exact-budget allocation, gate"
"${PY}" "${EXP}/run_cv.py" | tee "${EXP}/logs/run_cv.log"

date -u +%Y-%m-%dT%H:%M:%SZ > "${EXP}/calibration/COMPLETE"
echo "=== ADAPTIVE-K CALIBRATION EVIDENCE ==="
cat "${EXP}/calibration/CALIBRATION_REPORT.md"
