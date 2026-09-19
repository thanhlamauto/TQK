#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/exps/voi_commit_vs_reobserve"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
DENSE_BANK="${DENSE_BANK:-${ROOT}/exps/dense_calibration_200}"

mkdir -p "${EXP}/calibration/plots" "${EXP}/logs"
export HF_HOME="/workspace/.hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="/workspace/.cache"
export PYTHONUNBUFFERED=1

PY="${GEN_VENV}/bin/python"
[[ -x "${PY}" ]] || { echo "Missing environment ${GEN_VENV}" >&2; exit 1; }
"${PY}" -c "import sklearn, pandas, pyarrow, matplotlib" 2>/dev/null || "${PY}" -m pip install scikit-learn joblib matplotlib

echo "[tests] NFE accounting, A/B/PSP brute-force equivalence"
"${PY}" "${EXP}/test_voi.py" | tee "${EXP}/logs/tests.log"

echo "[1/3] build compact calibration bank from the independent ImageReward corpus"
"${PY}" "${EXP}/prepare_bank.py" --source-dense-bank "${DENSE_BANK}" | tee "${EXP}/logs/prepare_bank.log"

echo "[2/3] ordered 10-seed pool simulation (seed 20260920) and A/B/PSP replay"
"${PY}" "${EXP}/simulate_pools.py" | tee "${EXP}/logs/simulate_pools.log"

echo "[3/3] oracle-headroom gate, then grouped OOF selector and gate"
"${PY}" "${EXP}/run_cv.py" | tee "${EXP}/logs/run_cv.log"

if [[ -f "${EXP}/calibration/STOP_NO_HEADROOM" ]]; then
  echo "=== VOI CALIBRATION: STOPPED, NO ACTION-FAMILY HEADROOM ==="
else
  echo "=== VOI CALIBRATION EVIDENCE ==="
fi
cat "${EXP}/calibration/CALIBRATION_REPORT.md"
