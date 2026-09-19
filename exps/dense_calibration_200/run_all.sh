#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAL="${ROOT}/exps/dense_calibration_200"
CONF="${ROOT}/exps/confirmatory_553"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
GENEVAL_VENV="${GENEVAL_VENV:-/workspace/psp_geneval_env}"
GENEVAL_MMDET="${GENEVAL_MMDET:-/workspace/mmdetection-v2.28.2}"
GENEVAL_WEIGHTS="${GENEVAL_WEIGHTS:-/workspace/geneval_weights}"

mkdir -p "${CAL}"/{logs,bank_raw/gpu0,bank_raw/gpu1,replay,plots,metrics,prompts}
export HF_HOME="/workspace/.hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/workspace/.torch"
export XDG_CACHE_HOME="/workspace/.cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 2 ]]
[[ "${gpu_names[0]}" == "NVIDIA GeForce RTX 4090" ]]
[[ "${gpu_names[1]}" == "NVIDIA GeForce RTX 4090" ]]
nvidia-smi > "${CAL}/logs/nvidia_smi_preflight.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && {
  echo "GPU already has a compute process; refusing to mix workloads" >&2
  exit 1
} || true
[[ -x "${GEN_VENV}/bin/python" ]] || {
  echo "Missing generation environment ${GEN_VENV}" >&2
  exit 1
}
export PATH="${GEN_VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEN_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

"${GEN_VENV}/bin/python" - <<'PY'
import importlib.util, subprocess, sys
missing = [name for name in ("pandas", "pyarrow", "scipy") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
print("analysis dependencies ready")
PY

echo "[protocol] verify fixed 200 prompts and fixed 57-policy grid"
"${GEN_VENV}/bin/python" "${CAL}/select_prompts.py" --verify-existing
"${GEN_VENV}/bin/python" "${CAL}/budget_check.py"

echo "[tests] synthetic invariant tests for breadth value and subset regret"
"${GEN_VENV}/bin/python" "${CAL}/test_analysis.py"

echo "[phase1 preflight] two calibration prompts per GPU"
"${CAL}/run_dual_gpu.sh" 2
"${GEN_VENV}/bin/python" "${CAL}/validate_bank.py" --expected-prompts 2 \
  | tee "${CAL}/logs/preflight_validation.log"

echo "[phase1 full] 200 prompts x 25 complete trajectories"
phase1_started="$(date +%s)"
"${CAL}/run_dual_gpu.sh" 0
phase1_ended="$(date +%s)"
"${GEN_VENV}/bin/python" "${CAL}/validate_bank.py" --expected-prompts 200 \
  | tee "${CAL}/logs/bank_validation.log"
"${GEN_VENV}/bin/python" - "${CAL}/metrics/run_info.json" "${phase1_started}" "${phase1_ended}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
row = {}
row["phase1_started_unix"] = int(sys.argv[2]); row["phase1_ended_unix"] = int(sys.argv[3])
row["phase1_wall_s"] = int(sys.argv[3]) - int(sys.argv[2])
path.write_text(json.dumps(row, indent=2) + "\n")
PY

echo "[phase1 analysis] subset regret, shared surfaces, 5-fold OOF, LCB80 freeze"
"${GEN_VENV}/bin/python" "${CAL}/analyze_calibration.py" | tee "${CAL}/logs/analysis.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${CAL}/COMPLETE"

if [[ ! -f "${CAL}/FROZEN_SCHEDULE.json" ]]; then
  echo "[gate] OOF/LCB80 freeze gate did not pass; stop before confirmatory 553 (pre-registered stop rule)"
  echo "COMPLETE (no freeze): ${CAL}/CALIBRATION_REPORT.md"
  exit 0
fi

if [[ "${DENSE_PHASE1_ONLY:-0}" == "1" ]]; then
  echo "[gate] freeze gate passed; DENSE_PHASE1_ONLY=1 so the confirmatory 553 phase is deferred"
  echo "COMPLETE: ${CAL}/CALIBRATION_REPORT.md"
  exit 0
fi

echo "[phase2] freeze gate passed; running confirmatory 553 with three fresh seed bases"
bash "${CONF}/run_all.sh"

tar -C "${CAL}" -czf "${CAL}/results_bundle.tar.gz" \
  README.md PROTOCOL.md CALIBRATION_REPORT.md FROZEN_SCHEDULE.json COMPLETE \
  prompts replay logs metrics bank_raw
echo "=== PHASE1 EVIDENCE ==="
cat "${CAL}/CALIBRATION_REPORT.md"
