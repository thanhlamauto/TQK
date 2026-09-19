#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${GENEVAL_VENV:-/workspace/psp_geneval_env}"
MMDET="${GENEVAL_MMDET:-/workspace/mmdetection-v2.28.2}"
WEIGHTS="${GENEVAL_WEIGHTS:-/workspace/geneval_weights}"

MODEL="${WEIGHTS}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
if [[ -x "${VENV}/bin/python" && -d "${MMDET}/mmdet" && -s "${MODEL}" ]]; then
  if PYTHONPATH="${MMDET}:${PYTHONPATH:-}" "${VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import mmdet, mmcv, open_clip, torch
assert mmdet.__version__ == "2.28.2"
assert mmcv.__version__ == "1.7.2"
assert torch.cuda.is_available()
PY
  then
    echo "Reusing validated GenEval environment"
    exit 0
  fi
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  uv python install 3.10
  uv venv --seed --python 3.10 "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade pip wheel "setuptools<70"
"${VENV}/bin/python" -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
"${VENV}/bin/python" -m pip install \
  "numpy<2" pandas==1.5.3 networkx==2.8.8 open-clip-torch==2.26.1 \
  clip-benchmark einops tqdm openmim==0.3.9 packaging
"${VENV}/bin/mim" install "mmengine<1" "mmcv-full==1.7.2"

if [[ ! -d "${MMDET}/.git" ]]; then
  git clone https://github.com/open-mmlab/mmdetection.git "${MMDET}"
fi
git -C "${MMDET}" checkout --detach e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1
# MMDetection 2.x predates PEP 660 and cannot be installed editable by modern
# pip. The evaluator only needs its Python sources plus MMCV's compiled ops, so
# use the pinned checkout directly on PYTHONPATH and install its runtime deps.
"${VENV}/bin/python" -m pip install -r "${MMDET}/requirements/runtime.txt"
# OpenMIM's dependency resolver may pull NumPy 2 through OpenCV/MMEngine, while
# this pinned GenEval stack (pandas 1.5 + MMDetection 2.x) requires NumPy 1.x.
"${VENV}/bin/python" -m pip install --force-reinstall \
  "numpy<2" pandas==1.5.3 opencv-python==4.10.0.84

mkdir -p "${WEIGHTS}"
if [[ ! -s "${MODEL}" ]]; then
  curl -fL --retry 5 -o "${MODEL}.tmp" \
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
  mv "${MODEL}.tmp" "${MODEL}"
fi

PYTHONPATH="${MMDET}:${PYTHONPATH:-}" "${VENV}/bin/python" - <<PY
import importlib.metadata, mmdet, mmcv, torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "mmcv": mmcv.__version__, "mmdet": mmdet.__version__, "open_clip": importlib.metadata.version("open-clip-torch")})
assert torch.cuda.is_available()
PY
