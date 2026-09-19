#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/exps/voi_predictor_v2"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
DENSE_BANK="${DENSE_BANK:-${ROOT}/exps/dense_calibration_200}"

mkdir -p "${EXP}/calibration/plots" "${EXP}/logs"
export PYTHONUNBUFFERED=1

PY="${GEN_VENV}/bin/python"
[[ -x "${PY}" ]] || { echo "Missing environment ${GEN_VENV}" >&2; exit 1; }
"${PY}" -c "import sklearn, pandas, pyarrow, matplotlib" 2>/dev/null || "${PY}" -m pip install scikit-learn joblib matplotlib

echo "[tests] v2 feature/objective invariants"
"${PY}" "${EXP}/test_voi_v2.py" | tee "${EXP}/logs/tests.log"

echo "[1/3] compact bank (steps 12/14/16/32/64) from the independent corpus"
"${PY}" "${EXP}/prepare_bank.py" --source-dense-bank "${DENSE_BANK}" | tee "${EXP}/logs/prepare_bank.log"

echo "[2/3] ordered pools, A/B/PSP replay, F0/F1 features"
"${PY}" "${EXP}/simulate_pools.py" | tee "${EXP}/logs/simulate_pools.log"

echo "[3/3] weighted-policy classification, grouped OOF, headroom capture"
"${PY}" "${EXP}/run_cv_v2.py" | tee "${EXP}/logs/run_cv_v2.log"

date -u +%Y-%m-%dT%H:%M:%SZ > "${EXP}/calibration/COMPLETE"
echo "=== PREDICTOR V2 (F0/F1) EVIDENCE ==="
cat "${EXP}/calibration/CALIBRATION_REPORT.md"
