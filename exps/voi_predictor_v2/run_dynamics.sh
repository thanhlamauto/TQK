#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/exps/voi_predictor_v2"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"

mkdir -p "${EXP}/dynamics_raw/gpu0" "${EXP}/dynamics_raw/gpu1" "${EXP}/logs"
export HF_HOME="/workspace/.hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="/workspace/.cache"
export TORCH_HOME="/workspace/.torch"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 2 ]]
[[ "${gpu_names[0]}" == "NVIDIA GeForce RTX 4090" ]]
[[ "${gpu_names[1]}" == "NVIDIA GeForce RTX 4090" ]]
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && {
  echo "GPU already has a compute process; refusing to mix workloads" >&2
  exit 1
} || true
export PATH="${GEN_VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEN_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[preflight] one prompt on GPU0 (verifies hidden-feature extraction)"
CUDA_VISIBLE_DEVICES=0 "${GEN_VENV}/bin/python" "${EXP}/generate_dynamics_bank_worker.py" \
  --worker-index 0 --num-workers 1 --resume --limit-prompts 1 \
  | tee "${EXP}/logs/dynamics_preflight.log"
"${GEN_VENV}/bin/python" "${EXP}/validate_dynamics.py" --expected-prompts 1 \
  | tee "${EXP}/logs/dynamics_preflight_validation.log"

echo "[full] 200 prompts x 25 candidates, 2 GPUs"
started="$(date +%s)"
"${EXP}/run_dynamics_dual_gpu.sh" 0
ended="$(date +%s)"
"${GEN_VENV}/bin/python" "${EXP}/validate_dynamics.py" --expected-prompts 200 \
  | tee "${EXP}/logs/dynamics_validation.log"
echo "dynamics bank wall seconds: $((ended - started))"

echo "[offline] pools + F0/F1/F2, then F0-F3 OOF"
"${GEN_VENV}/bin/python" "${EXP}/simulate_pools_dynamics.py" | tee "${EXP}/logs/simulate_dynamics.log"
"${GEN_VENV}/bin/python" "${EXP}/run_cv_v2_dynamics.py" | tee "${EXP}/logs/run_cv_dynamics.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${EXP}/calibration/COMPLETE_DYNAMICS"
echo "=== PREDICTOR V2 (F0-F3) EVIDENCE ==="
cat "${EXP}/calibration/CALIBRATION_REPORT.md"
