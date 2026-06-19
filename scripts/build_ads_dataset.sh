#!/usr/bin/env bash
# Build the ADS training set: embedding -> cluster -> difficulty -> sort.
# Produces the parquet (cluster_id / sample_id / rank_in_cluster / cluster_size)
# consumed by verl/examples/grpo_trainer/train_ads.sh.
# Override INPUT_DATA / OUTPUT_BASE / MODEL / NUM_CLUSTERS / CUDA_VISIBLE_DEVICES via env.
set -euo pipefail

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}

MODEL="${MODEL:-RoadQAQ/Qwen2.5-Math-1.5B-16k-think}"
INPUT_DATA="${INPUT_DATA:-${root_dir}/dataset/train.parquet}"
OUTPUT_BASE="${OUTPUT_BASE:-${root_dir}/outputs/ads_run}"
NUM_CLUSTERS="${NUM_CLUSTERS:-64}"
# Saved here because CPU phases below temporarily clear CUDA_VISIBLE_DEVICES.
_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Default NUM_PROCESSES to the number of GPUs listed in CUDA_VISIBLE_DEVICES.
IFS=',' read -r -a _gpu_arr <<< "${_gpus}"
NUM_PROCESSES="${NUM_PROCESSES:-${#_gpu_arr[@]}}"

CACHE_ROOT="${root_dir}/.cache/huggingface"
export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

# ── Generate config files ────────────────────────────────────────────
config_dir="${OUTPUT_BASE}/configs"
mkdir -p "${config_dir}"

cat > "${config_dir}/extract.yaml" <<EOF
model_name_or_path: ${MODEL}
input_path: ${INPUT_DATA}
input_format: parquet
text_field: prompt
id_field: sample_id
output_dir: ${OUTPUT_BASE}
max_length: 4096
batch_size: 16
dtype: bfloat16
pooling_methods:
  - mean
num_proc: 8
save_format: npy
normalize_embeddings: false
prompt_text_mode: user_only
input_text_mode: question_plus_cot
datasets_cache_dir: ${HF_DATASETS_CACHE}
skip_if_exists: true
overwrite: false
export_parquet_aux: false
expected_num_processes: ${NUM_PROCESSES}
seed: 42
EOF

cat > "${config_dir}/cluster.yaml" <<EOF
ids_path: ${OUTPUT_BASE}/all_ids.npy
embedding_path: ${OUTPUT_BASE}/all_emb_mean.npy
raw_data_path: ${INPUT_DATA}
raw_data_format: parquet
output_dir: ${OUTPUT_BASE}/phase2a
embedding_name: mean
normalize_embeddings: true
pca_dim: 128
clustering_method: kmeans
num_clusters: ${NUM_CLUSTERS}
random_seed: 42
text_field: prompt
id_field: sample_id
prompt_text_mode: user_only
input_text_mode: question_plus_cot
num_proc: 8
datasets_cache_dir: ${HF_DATASETS_CACHE}
top_k_nearest: 10
top_k_random: 10
export_examples_parquet: false
silhouette_max_samples: 5000
EOF

cat > "${config_dir}/difficulty.yaml" <<EOF
model_name_or_path: ${MODEL}
input_path: ${INPUT_DATA}
input_format: parquet
text_field: prompt
id_field: sample_id
output_dir: ${OUTPUT_BASE}/phase4_difficulty
max_length: 4096
batch_size: 8
dtype: bfloat16
datasets_cache_dir: ${HF_DATASETS_CACHE}
skip_if_exists: true
expected_num_processes: ${NUM_PROCESSES}
EOF

echo "Configs written to ${config_dir}/"

# ── Phase 1: Extract embeddings (GPU) ────────────────────────────────
echo ""
echo "========== Phase 1: Extract embeddings =========="
export CUDA_VISIBLE_DEVICES="${_gpus}"
"${python_bin}" -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" \
  "${root_dir}/src/extract_embeddings.py" \
  --config "${config_dir}/extract.yaml"

# ── Phase 1.5: Merge embedding shards (CPU) ──────────────────────────
echo ""
echo "========== Phase 1.5: Merge embedding shards =========="
export CUDA_VISIBLE_DEVICES=""
"${python_bin}" "${root_dir}/src/merge_embeddings.py" \
  --config "${config_dir}/extract.yaml"

# ── Phase 2: Cluster (CPU) ───────────────────────────────────────────
echo ""
echo "========== Phase 2a: Inspect embeddings =========="
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
"${python_bin}" "${root_dir}/src/inspect_embeddings.py" \
  --config "${config_dir}/cluster.yaml"

echo ""
echo "========== Phase 2b: K-Means K=${NUM_CLUSTERS} =========="
"${python_bin}" "${root_dir}/src/cluster_embeddings.py" \
  --config "${config_dir}/cluster.yaml"

echo ""
echo "========== Phase 2c: Export cluster examples =========="
"${python_bin}" "${root_dir}/src/export_cluster_examples.py" \
  --config "${config_dir}/cluster.yaml"

# ── Phase 3: Join cluster assignments to raw data (CPU) ──────────────
echo ""
echo "========== Phase 3: Join cluster assignments =========="
CLUSTER_PARQUET="${OUTPUT_BASE}/phase2a/cluster_assignments_mean_k${NUM_CLUSTERS}.parquet"
PHASE3_OUTPUT="${OUTPUT_BASE}/phase3/train_with_cluster_cot_k${NUM_CLUSTERS}.parquet"
mkdir -p "${OUTPUT_BASE}/phase3"
"${python_bin}" "${root_dir}/src/prepare_verl_cluster_data.py" \
  --raw_data_path "${INPUT_DATA}" \
  --cluster_assignments_path "${CLUSTER_PARQUET}" \
  --output_path "${PHASE3_OUTPUT}" \
  --cluster_source "cot_mean_kmeans_k${NUM_CLUSTERS}_roadqaq_1_5b"

# ── Phase 4: Compute NLL difficulty (GPU) ────────────────────────────
echo ""
echo "========== Phase 4: Compute NLL difficulty =========="
export CUDA_VISIBLE_DEVICES="${_gpus}"
"${python_bin}" -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" \
  "${root_dir}/src/compute_difficulty.py" \
  --config "${config_dir}/difficulty.yaml"

# ── Phase 4.5: Merge difficulty shards (CPU) ─────────────────────────
echo ""
echo "========== Phase 4.5: Merge difficulty shards =========="
export CUDA_VISIBLE_DEVICES=""
"${python_bin}" "${root_dir}/src/merge_difficulty.py" \
  --config "${config_dir}/difficulty.yaml"

# ── Phase 5: Sort within clusters by difficulty (CPU) ────────────────
echo ""
echo "========== Phase 5: Sort clusters by difficulty =========="
PHASE5_OUTPUT="${OUTPUT_BASE}/phase5/train_clustered_sorted_$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${PHASE3_OUTPUT}').num_rows)").parquet"
mkdir -p "${OUTPUT_BASE}/phase5"
"${python_bin}" "${root_dir}/src/sort_clusters_by_difficulty.py" \
  --clustered_parquet "${PHASE3_OUTPUT}" \
  --difficulty_ids_path "${OUTPUT_BASE}/phase4_difficulty/all_difficulty_ids.npy" \
  --difficulty_scores_path "${OUTPUT_BASE}/phase4_difficulty/all_difficulty_scores.npy" \
  --output_path "${PHASE5_OUTPUT}"

echo ""
echo "========== Pipeline complete =========="
echo "Final output: ${PHASE5_OUTPUT}"
