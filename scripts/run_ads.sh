#!/usr/bin/env bash
# End-to-end ADS: format data -> build the ADS training set -> GRPO training.
#
#   DATASET=<hf-id|local .parquet/.json/.jsonl> bash scripts/run_ads.sh
#
# One DATASET is threaded through every stage, so paths line up automatically.
# Override CUDA_VISIBLE_DEVICES=0,1,2,3 to pick devices, NAME=<dir> to rename outputs, and
# STAGE=data|build|train to run a single stage (default: all three).
set -euo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}

DATASET=${DATASET:?Set DATASET=<hf-id or local .parquet/.json/.jsonl>}
gpus=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
stage=${STAGE:-all}

# Derive a filesystem-friendly run name from DATASET unless one is given.
if [[ -z "${NAME:-}" ]]; then
  _b="${DATASET%/}"; _b="${_b##*/}"; _b="${_b%.parquet}"; _b="${_b%.json}"; _b="${_b%.jsonl}"
  NAME=$(printf '%s' "${_b}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//')
  NAME=${NAME:-dataset}
fi

train_parquet="${root_dir}/dataset/${NAME}/train.parquet"
output_base="${root_dir}/outputs/${NAME}"
echo "ADS run: dataset='${DATASET}' name='${NAME}' gpus='${gpus}' stage='${stage}'"

if [[ "${stage}" == "all" || "${stage}" == "data" ]]; then
  echo "========== Stage 1/3: format data =========="
  "${python_bin}" "${root_dir}/scripts/prepare_datasets/prepare_custom_dataset.py" \
    --dataset "${DATASET}" --output-name "${NAME}" --output-root "${root_dir}/dataset"
  "${python_bin}" "${root_dir}/scripts/prepare_datasets/run_all.py" \
    --output-root "${root_dir}/dataset"
fi

if [[ "${stage}" == "all" || "${stage}" == "build" ]]; then
  echo "========== Stage 2/3: build ADS training set =========="
  INPUT_DATA="${train_parquet}" OUTPUT_BASE="${output_base}" CUDA_VISIBLE_DEVICES="${gpus}" \
    bash "${root_dir}/scripts/build_ads_dataset.sh"
fi

if [[ "${stage}" == "all" || "${stage}" == "train" ]]; then
  echo "========== Stage 3/3: train with ADS =========="
  TRAIN_DATASET_DIR="${output_base}/phase5" CUDA_VISIBLE_DEVICES="${gpus}" \
    bash "${root_dir}/verl/examples/grpo_trainer/train_ads.sh"
fi

echo "ADS run complete."
