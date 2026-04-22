#!/usr/bin/env bash
set -euo pipefail

# One-click Qwen3-VL-8B-Instruct evaluation on VideoMMMU with public VLMEvalKit.
#
# Usage:
#   bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh smoke
#   bash /mnt/task_runtime/VLMEvalKit/scripts/run_qwen3_videommmu_oneclick.sh full
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash ... full
#   DATASET_PRESET=VideoMMMU_64frame bash ... full
#   WORK_DIR=/mnt/output/my_vlmeval_run bash ... smoke
#   REUSE_VIDEO_O3_ENV=1 bash ... smoke

RUN_MODE="${1:-smoke}"
if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/VLMEvalKit}"
REUSE_VIDEO_O3_ENV="${REUSE_VIDEO_O3_ENV:-0}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
if [[ "${REUSE_VIDEO_O3_ENV}" == "1" ]]; then
  PY_RUNTIME="${PY_RUNTIME:-/mnt/output/video_o3/eval/py312-runtime}"
  EVAL_SITE="${EVAL_SITE:-/mnt/output/video_o3/eval/.venv312/lib/python3.12/site-packages}"
else
  PY_RUNTIME="${PY_RUNTIME:-/mnt/output/vlmevalkit/envs/qwen3-videommmu-py312}"
  EVAL_SITE="${EVAL_SITE:-}"
fi
HF_HOME="${HF_HOME:-/mnt/output/cache/hf}"
FULL_VIDEOMMMU_ROOT="${FULL_VIDEOMMMU_ROOT:-/mnt/output/video_bench_datasets/VideoMMMU}"
SMOKE_VIDEOMMMU_ROOT="${SMOKE_VIDEOMMMU_ROOT:-/mnt/output/video_bench_datasets/VideoMMMU_smoke50}"
WORK_DIR="${WORK_DIR:-/mnt/output/vlmevalkit/qwen3_videommmu}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_KEY="${MODEL_KEY:-qwen3_vl_8b_instruct_flash}"
DATASET_PRESET="${DATASET_PRESET:-VideoMMMU_8frame}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
API_NPROC="${API_NPROC:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SMOKE_ROWS="${SMOKE_ROWS:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLMEVALKIT_BOOTSTRAP_DEPS="${VLMEVALKIT_BOOTSTRAP_DEPS:-vllm==0.19.0 transformers==4.57.6 accelerate qwen-vl-utils decord pandas pyarrow openpyxl xlsxwriter portalocker rich tabulate termcolor validators huggingface_hub pillow numpy tqdm requests python-dotenv Levenshtein jieba anls apted distance editdistance json_repair latex2sympy2-extended math-verify num2words polygon3 sty zss rouge timeout-decorator imageio av opencv-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REPO_ROOT PY_RUNTIME EVAL_SITE REUSE_VIDEO_O3_ENV BOOTSTRAP_ENV
export FULL_VIDEOMMMU_ROOT SMOKE_VIDEOMMMU_ROOT WORK_DIR
export MODEL_PATH MODEL_KEY DATASET_PRESET MAX_NEW_TOKENS API_NPROC NPROC_PER_NODE SMOKE_ROWS
export ATTN_IMPLEMENTATION MODEL_DTYPE PYTHON_VERSION VLMEVALKIT_BOOTSTRAP_DEPS
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PRED_FORMAT="${PRED_FORMAT:-xlsx}"
export EVAL_FORMAT="${EVAL_FORMAT:-csv}"

mkdir -p "${HF_HOME}" "${WORK_DIR}/configs"

if [[ ! -x "${PY_RUNTIME}/bin/python" ]]; then
  if [[ "${BOOTSTRAP_ENV}" != "1" ]]; then
    echo "[setup] Python runtime not found at ${PY_RUNTIME} and BOOTSTRAP_ENV=0" >&2
    exit 1
  fi
  echo "[setup] Python runtime not found at ${PY_RUNTIME}; creating it with mamba."
  if command -v mamba >/dev/null 2>&1; then
    mamba create -y -p "${PY_RUNTIME}" "python=${PYTHON_VERSION}"
  elif [[ -x /coreflow/mambaforge/bin/mamba ]]; then
    /coreflow/mambaforge/bin/mamba create -y -p "${PY_RUNTIME}" "python=${PYTHON_VERSION}"
  elif command -v conda >/dev/null 2>&1; then
    conda create -y -p "${PY_RUNTIME}" "python=${PYTHON_VERSION}"
  else
    echo "[setup] Need mamba or conda to create ${PY_RUNTIME}" >&2
    exit 1
  fi
fi

PYTHON="${PY_RUNTIME}/bin/python"
if [[ -z "${EVAL_SITE}" ]]; then
  EVAL_SITE="$("${PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  export EVAL_SITE
fi

export PYTHONPATH="${REPO_ROOT}:${EVAL_SITE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PY_RUNTIME}/lib:${LD_LIBRARY_PATH:-}"

if ! "${PYTHON}" -m pip --version >/dev/null 2>&1; then
  "${PYTHON}" -m ensurepip --upgrade
fi

if ! "${PYTHON}" - <<'PY'
import decord
import jieba
import Levenshtein
import pandas
import qwen_vl_utils
import termcolor
import torch
import transformers
import vllm
PY
then
  if [[ "${BOOTSTRAP_ENV}" != "1" ]]; then
    echo "[setup] Missing dependencies and BOOTSTRAP_ENV=0" >&2
    exit 1
  fi
  echo "[setup] Installing Qwen3/VideoMMMU dependencies into ${PY_RUNTIME}"
  "${PYTHON}" -m pip install --upgrade pip setuptools wheel
  if [[ "${REUSE_VIDEO_O3_ENV}" == "1" ]]; then
    "${PYTHON}" -m pip install --upgrade --target "${EVAL_SITE}" ${VLMEVALKIT_BOOTSTRAP_DEPS}
  else
    "${PYTHON}" -m pip install --upgrade ${VLMEVALKIT_BOOTSTRAP_DEPS}
    "${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
  fi
fi

if [[ ! -f "${FULL_VIDEOMMMU_ROOT}/VideoMMMU.tsv" ]]; then
  echo "[data] Missing VideoMMMU.tsv under ${FULL_VIDEOMMMU_ROOT}" >&2
  exit 1
fi

if [[ "${RUN_MODE}" == "smoke" ]]; then
  export VIDEOMMMU_ROOT="${SMOKE_VIDEOMMMU_ROOT}"
  export VLMEVAL_DATASET_NAME="VideoMMMU_smoke${SMOKE_ROWS}_8frame"
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

full = Path(os.environ["FULL_VIDEOMMMU_ROOT"])
smoke = Path(os.environ["SMOKE_VIDEOMMMU_ROOT"])
rows = int(os.environ["SMOKE_ROWS"])
smoke.mkdir(parents=True, exist_ok=True)

src = pd.read_csv(full / "VideoMMMU.tsv", sep="\t")
subset = src.head(rows).copy()
subset.to_csv(smoke / "VideoMMMU.tsv", sep="\t", index=False)

for name in ["videos", "images"]:
    target = full / name
    link = smoke / name
    if link.exists() or link.is_symlink():
        continue
    link.symlink_to(target, target_is_directory=True)

print(f"[data] Smoke dataset ready: {smoke} ({len(subset)} rows)")
PY
else
  export VIDEOMMMU_ROOT="${FULL_VIDEOMMMU_ROOT}"
  export VLMEVAL_DATASET_NAME="${DATASET_PRESET}"
fi

"${PYTHON}" - <<'PY'
import importlib.metadata as md
import os
import sys

import torch
import transformers
from transformers.utils import is_flash_attn_2_available

import flash_attn

print("[env] python:", sys.executable)
print("[env] torch:", torch.__version__)
print("[env] transformers:", transformers.__version__)
print("[env] flash_attn:", md.version("flash_attn"), flash_attn.__file__)
print("[env] transformers sees flash_attention_2:", is_flash_attn_2_available())
print("[env] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[env] VIDEOMMMU_ROOT:", os.environ["VIDEOMMMU_ROOT"])
PY

export CONFIG_PATH="${WORK_DIR}/configs/qwen3_videommmu_${RUN_MODE}.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

dataset_name = os.environ["VLMEVAL_DATASET_NAME"]
data_cfg = {"class": "VideoMMMU", "dataset": "VideoMMMU"}
if dataset_name.endswith("_64frame"):
    data_cfg["nframe"] = 64
elif dataset_name.endswith("_1fps"):
    data_cfg["fps"] = 1.0
elif dataset_name.endswith("_0.5fps"):
    data_cfg["fps"] = 0.5
else:
    data_cfg["nframe"] = 8

cfg = {
    "model": {
        os.environ["MODEL_KEY"]: {
            "class": "Qwen3VLChat",
            "model_path": os.environ["MODEL_PATH"],
            "use_custom_prompt": False,
            "use_vllm": False,
            "attn_implementation": os.environ["ATTN_IMPLEMENTATION"],
            "dtype": os.environ["MODEL_DTYPE"],
            "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
            "temperature": 0.01,
        }
    },
    "data": {dataset_name: data_cfg},
}
path = Path(os.environ["CONFIG_PATH"])
path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"[config] Wrote {path}")
print(json.dumps(cfg, indent=2))
PY

echo "[download] Ensuring model snapshot is present: ${MODEL_PATH}"
"${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(os.environ["MODEL_PATH"], repo_type="model", resume_download=True)
print("[download] Model snapshot ready")
PY

cd "${REPO_ROOT}"
echo "[run] Starting VLMEvalKit ${RUN_MODE} run"
"${PYTHON}" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE}" \
  run.py \
  --config "${CONFIG_PATH}" \
  --mode all \
  --work-dir "${WORK_DIR}" \
  --reuse \
  --api-nproc "${API_NPROC}" \
  --judge exact_matching

echo "[done] Outputs are under ${WORK_DIR}/${MODEL_KEY}/"
