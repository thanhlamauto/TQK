#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="${ROOT}/exps/confirmatory_553"
GEN_VENV="${GEN_VENV:-/workspace/psp_sd15_env}"
GENEVAL_VENV="${GENEVAL_VENV:-/workspace/psp_geneval_env}"
GENEVAL_MMDET="${GENEVAL_MMDET:-/workspace/mmdetection-v2.28.2}"
GENEVAL_WEIGHTS="${GENEVAL_WEIGHTS:-/workspace/geneval_weights}"

mkdir -p "${CONF}"/{logs,outputs,metadata/gpu0,metadata/gpu1,metrics,geneval_results}
for method in psp ours; do
  for rep in 0 1 2; do
    mkdir -p "${CONF}/outputs/${method}/rep${rep}"
  done
done
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
nvidia-smi > "${CONF}/logs/nvidia_smi_preflight.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && {
  echo "GPU already has a compute process; refusing to mix workloads" >&2
  exit 1
} || true

export PATH="${GEN_VENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GEN_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[protocol] locksum frozen schedule, 553 prompts and three fresh seed bases"
"${GEN_VENV}/bin/python" "${CONF}/prepare_protocol.py" | tee "${CONF}/logs/protocol.log"

echo "[preflight] four method-runs across two GPUs (two prompts x rep-0)"
"${CONF}/run_dual_gpu.sh" 2
"${GEN_VENV}/bin/python" "${CONF}/validate_generation.py" --expected-prompts 2 \
  | tee "${CONF}/logs/preflight_validation.log"

echo "[full] 553 prompts x 3 seeds x {PSP, frozen}"
gen_started="$(date +%s)"
"${CONF}/run_dual_gpu.sh" 0
gen_ended="$(date +%s)"
"${GEN_VENV}/bin/python" "${CONF}/validate_generation.py" --expected-prompts 553 \
  | tee "${CONF}/logs/generation_validation.log"
"${GEN_VENV}/bin/python" - "${CONF}/metrics/run_info.json" "${gen_started}" "${gen_ended}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "generation_started_unix": int(sys.argv[2]),
    "generation_ended_unix": int(sys.argv[3]),
    "generation_wall_s": int(sys.argv[3]) - int(sys.argv[2]),
}, indent=2) + "\n")
PY

echo "[hps] all final winners"
CUDA_VISIBLE_DEVICES=0 "${GEN_VENV}/bin/python" "${CONF}/evaluate_hps.py" | tee "${CONF}/logs/hps.log"

echo "[geneval] official evaluator over 3 seeds x 2 methods"
"${GEN_VENV}/bin/python" "${CONF}/export_geneval.py"
bash "${ROOT}/exps/confirmatory_553/setup_geneval_env.sh" | tee "${CONF}/logs/geneval_setup.log"
for method in psp ours; do
  for rep in 0 1 2; do
    name="${method}_rep${rep}"
    if [[ -s "${CONF}/geneval_results/${name}.jsonl" ]] && \
       [[ "$(wc -l < "${CONF}/geneval_results/${name}.jsonl")" -eq 553 ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${GENEVAL_MMDET}:${PYTHONPATH:-}" \
      "${GENEVAL_VENV}/bin/python" "${ROOT}/geneval/evaluation/evaluate_images.py" \
      "${CONF}/geneval_inputs/${name}" \
      --outfile "${CONF}/geneval_results/${name}.jsonl" \
      --model-config "${GENEVAL_MMDET}/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
      --model-path "${GENEVAL_WEIGHTS}" \
      2> "${CONF}/logs/geneval_${name}.log"
    [[ "$(wc -l < "${CONF}/geneval_results/${name}.jsonl")" -eq 553 ]]
  done
done

echo "[analysis] prompt-level paired bootstrap"
"${GEN_VENV}/bin/python" "${CONF}/aggregate.py" | tee "${CONF}/logs/aggregate.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${CONF}/COMPLETE"

tar -C "${CONF}" -czf "${CONF}/results_bundle.tar.gz" \
  README.md PROTOCOL.md FINAL_REPORT.md FROZEN_SCHEDULE.json protocol_manifest.json \
  COMPLETE metrics metadata logs geneval_results outputs
echo "COMPLETE: ${CONF}/FINAL_REPORT.md"
