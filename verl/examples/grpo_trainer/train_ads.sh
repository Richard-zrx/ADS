#!/usr/bin/env bash
set -xeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "${script_dir}/../.." && pwd)
project_root=$(cd -- "${root_dir}/.." && pwd)
cd "${root_dir}"

script_name=$(basename -- "${BASH_SOURCE[0]}" .sh)
timestamp=$(date +%Y%m%d.%H%M%S)
experiment_name=${EXPERIMENT_NAME:-${script_name}-${timestamp}}

export PYTHONPATH="${root_dir}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Some Slurm gres plugins also export ROCR/HIP_VISIBLE_DEVICES, which can collide
# with CUDA_VISIBLE_DEVICES inside verl workers (worker.py raises a ValueError).
# On NVIDIA GPUs these AMD/ROCm vars are spurious, so drop them if present.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONUNBUFFERED=1
# HuggingFace caches. Override to point at a shared/pre-populated cache if you have
# one; otherwise they default to the per-user cache under $HOME/.cache.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"
export WANDB_PROJECT="${WANDB_PROJECT:-ads}"
# Provide your own key via the environment (`export WANDB_API_KEY=...`) when running
# with wandb logging, or set ENABLE_WANDB=0 to log to the console only.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
trap 'unset WANDB_API_KEY' EXIT

enable_wandb=${ENABLE_WANDB:-1}
trainer_loggers='["console"]'
if [[ "${enable_wandb}" == "1" ]]; then
  trainer_loggers='["console","wandb"]'
fi

python_bin=${PYTHON_BIN:-python3}
ppo_config_name=${PPO_CONFIG_NAME:-_generated_ppo_trainer}
model_path=${MODEL_PATH:-RoadQAQ/Qwen2.5-Math-1.5B-16k-think}
reward_fn_path=${REWARD_FN_PATH:-${script_dir}/reward_boxed_binary.py}
vllm_backend=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-uni}
attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}
use_remove_padding=${USE_REMOVE_PADDING:-true}
flash_attn_required=${FLASH_ATTN_REQUIRED:-1}
case "${use_remove_padding,,}" in
  true|false) use_remove_padding="${use_remove_padding,,}" ;;
  *) echo "Invalid USE_REMOVE_PADDING='${use_remove_padding}'. Use true|false." >&2; exit 1 ;;
esac

resolve_project_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then printf '%s\n' "${path}"
  else printf '%s\n' "${project_root}/${path#./}"
  fi
}

train_dataset_dir=$(resolve_project_path "${TRAIN_DATASET_DIR:-outputs/ads_run/phase5}")
train_file=$(resolve_project_path "${TRAIN_FILE:-${train_dataset_dir}}")
val_dataset_aime2024_dir=$(resolve_project_path "${VAL_DATASET_AIME2024_DIR:-dataset/aime2024}")
val_dataset_math500_dir=$(resolve_project_path "${VAL_DATASET_MATH500_DIR:-dataset/math500}")
val_dataset_aime2025_dir=$(resolve_project_path "${VAL_DATASET_AIME2025_DIR:-dataset/aime25}")

val_file_aime2024=$(resolve_project_path "${VAL_FILE_AIME2024:-${val_dataset_aime2024_dir}/test.parquet}")
val_file_math500=$(resolve_project_path "${VAL_FILE_MATH500:-${val_dataset_math500_dir}/test.parquet}")
val_file_aime2025=$(resolve_project_path "${VAL_FILE_AIME2025:-${val_dataset_aime2025_dir}/test.parquet}")

# Allow passing directory paths directly. The pipeline writes
# train_clustered_sorted_<N>.parquet (N = training-set row count), so glob for it.
if [[ -d "${train_file}" ]]; then
  shopt -s nullglob
  _sorted_matches=( "${train_file%/}"/train_clustered_sorted_*.parquet )
  shopt -u nullglob
  if (( ${#_sorted_matches[@]} > 0 )); then
    train_file="${_sorted_matches[0]}"
  else
    train_file="${train_file%/}/train.parquet"
  fi
fi
[[ -d "${val_file_aime2024}" ]] && val_file_aime2024="${val_file_aime2024%/}/test.parquet"
[[ -d "${val_file_math500}" ]]  && val_file_math500="${val_file_math500%/}/test.parquet"
[[ -d "${val_file_aime2025}" ]] && val_file_aime2025="${val_file_aime2025%/}/test.parquet"

if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
  n_gpus_per_node="${N_GPUS_PER_NODE}"
else
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  n_gpus_per_node="${#_visible_gpus[@]}"
fi
nnodes=${NNODES:-1}
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$((train_batch_size / 16))}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
val_response_length=${VAL_RESPONSE_LENGTH:-14336}
train_max_tokens=${TRAIN_MAX_TOKENS:-$((max_prompt_length + max_response_length))}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-16384}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-10240}
actor_calculate_entropy=${ACTOR_CALCULATE_ENTROPY:-true}
rollout_n=${ROLLOUT_N:-8}
val_n=${VAL_N:-16}
val_temperature=${VAL_TEMPERATURE:-0.7}
val_top_p=${VAL_TOP_P:-0.8}
aime_eval_n=${AIME_EVAL_N:-16}
math500_eval_n=${MATH500_EVAL_N:-4}
total_training_steps=${TOTAL_TRAINING_STEPS:-800}
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-25}
test_freq=${TEST_FREQ:-25}
project_name=${PROJECT_NAME:-${WANDB_PROJECT}}
resume_mode=${RESUME_MODE:-disable}

model_max_position_embeddings=${MODEL_MAX_POSITION_EMBEDDINGS:-}
if [[ -z "${model_max_position_embeddings}" ]]; then
  model_max_position_embeddings=$("${python_bin}" - "${model_path}" 2>/dev/null <<'PY'
import sys
try:
    from transformers import AutoConfig
except Exception:
    sys.exit(0)
model_path = sys.argv[1]
try:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
except Exception:
    sys.exit(0)
values = []
for key in ("max_position_embeddings", "n_positions", "seq_length", "model_max_length"):
    value = getattr(cfg, key, None)
    if isinstance(value, int) and 0 < value < 10_000_000:
        if key == "model_max_length" and value > 1_000_000:
            continue
        values.append(value)
rope_scaling = getattr(cfg, "rope_scaling", None)
if isinstance(rope_scaling, dict):
    for key in ("original_max_position_embeddings", "max_position_embeddings"):
        value = rope_scaling.get(key)
        if isinstance(value, (int, float)) and 0 < value < 10_000_000:
            values.append(int(value))
if values:
    print(max(values))
PY
)
fi
if [[ -z "${model_max_position_embeddings}" || ! "${model_max_position_embeddings}" =~ ^[0-9]+$ ]]; then
  model_max_position_embeddings=16384
  echo "Could not infer MODEL_MAX_POSITION_EMBEDDINGS, fallback to ${model_max_position_embeddings}." >&2
fi

(( rollout_max_model_len > model_max_position_embeddings )) && rollout_max_model_len=${model_max_position_embeddings}
if (( max_prompt_length >= rollout_max_model_len )); then
  echo "Invalid: MAX_PROMPT_LENGTH=${max_prompt_length} must be < ROLLOUT_MAX_MODEL_LEN=${rollout_max_model_len}." >&2; exit 1
fi
max_val_response_override=$((rollout_max_model_len - max_prompt_length))
(( max_response_length > rollout_max_model_len )) && max_response_length=${rollout_max_model_len}
(( val_response_length > max_val_response_override )) && val_response_length=${max_val_response_override}

echo "Length config: train_max_tokens=${train_max_tokens}, prompt_max=${max_prompt_length}, response_max=${max_response_length}, val_response_max=${val_response_length}, rollout_max_model_len=${rollout_max_model_len}, model_max=${model_max_position_embeddings}"

checkpoint_dir=${CHECKPOINT_DIR:-"${root_dir}/verl/checkpoints/${script_name}"}
mkdir -p "${checkpoint_dir}"
run_dir="${checkpoint_dir}/${experiment_name}"
mkdir -p "${run_dir}"
train_log="${run_dir}/train.log"

for f in "${train_file}" "${val_file_aime2024}" "${val_file_aime2025}" "${val_file_math500}" "${reward_fn_path}"; do
  [[ ! -f "${f}" ]] && { echo "Required file not found: ${f}" >&2; exit 1; }
done

"${python_bin}" - "${train_file}" <<'PY'
import sys
import pyarrow.parquet as pq
path = sys.argv[1]
table = pq.read_table(path)
cols = set(table.column_names)
required = {"cluster_id", "sample_id", "rank_in_cluster", "cluster_size"}
missing = required - cols
if missing:
    raise SystemExit(f"Training parquet missing required columns: {sorted(missing)}")
print(f"Frontier schema check OK: rows={table.num_rows}, cols={sorted(required)} in {path}")
PY

visible_gpu_count=$("${python_bin}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [[ "${visible_gpu_count}" -lt "${n_gpus_per_node}" ]]; then
  echo "Visible GPUs (${visible_gpu_count}) < N_GPUS_PER_NODE (${n_gpus_per_node}). CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'" >&2; exit 1
fi

has_flash_attn=$("${python_bin}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("flash_attn") else "0")
PY
)

resolved_attn_implementation="${attn_implementation}"
case "${attn_implementation}" in
  auto) [[ "${has_flash_attn}" == "1" ]] && resolved_attn_implementation="flash_attention_2" || resolved_attn_implementation="eager" ;;
  flash|flash2|fa2) resolved_attn_implementation="flash_attention_2" ;;
  flash_attention_2|eager|sdpa) ;;
  *) echo "Invalid ATTN_IMPLEMENTATION='${attn_implementation}'." >&2; exit 1 ;;
esac

if [[ "${resolved_attn_implementation}" == "flash_attention_2" && "${has_flash_attn}" != "1" ]]; then
  if [[ "${flash_attn_required}" == "1" ]]; then
    echo "flash_attn is required but not installed." >&2; exit 1
  fi
  echo "flash_attn not installed, fallback to eager." >&2
  resolved_attn_implementation="eager"
fi

if [[ "${use_remove_padding}" == "true" && "${has_flash_attn}" != "1" ]]; then
  echo "USE_REMOVE_PADDING=true but flash_attn not installed, fallback to false." >&2
  use_remove_padding="false"
fi

echo "Attention: ${resolved_attn_implementation} (flash_attn=${has_flash_attn}), use_remove_padding=${use_remove_padding}"

val_files_hydra="['${val_file_aime2024}','${val_file_aime2025}','${val_file_math500}']"

"${python_bin}" -m verl.trainer.main_ppo --config-name "${ppo_config_name}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${train_file}" \
    data.val_files="${val_files_hydra}" \
    data.prompt_key=prompt \
    data.train_batch_size="${train_batch_size}" \
    data.dataloader_num_workers="${dataloader_num_workers}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.sampler.class_name="ADSMiniClusterSampler" \
    data.sampler.class_path="pkg://verl.experimental.dataset.ads_sampler" \
    +data.ads_sampler.cluster_key=cluster_id \
    +data.ads_sampler.sample_id_key=sample_id \
    +data.ads_sampler.rank_key=rank_in_cluster \
    +data.ads_sampler.cluster_size_key=cluster_size \
    +data.ads_sampler.seed=42 \
    +data.ads_sampler.active_clusters=4 \
    +data.ads_sampler.mini_cluster_size=32 \
    +data.ads_sampler.boundary_eps=0.17 \
    +data.ads_sampler.alpha=0.3 \
    +data.ads_sampler.r_init=0.5 \
    +data.ads_sampler.prob_snapshot_log_interval=1 \
    actor_rollout_ref.model.path="${model_path}" \
    ++actor_rollout_ref.model.override_config.attn_implementation="${resolved_attn_implementation}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${use_remove_padding}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy="${actor_calculate_entropy}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend="${vllm_backend}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    ++actor_rollout_ref.rollout.val_response_length="${val_response_length}" \
    actor_rollout_ref.rollout.max_model_len="${rollout_max_model_len}" \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
    +actor_rollout_ref.rollout.val_kwargs.n_per_data_source.aime2024="${aime_eval_n}" \
    +actor_rollout_ref.rollout.val_kwargs.n_per_data_source.aime2025="${aime_eval_n}" \
    +actor_rollout_ref.rollout.val_kwargs.n_per_data_source.math500="${math500_eval_n}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${reward_fn_path}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger="${trainer_loggers}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.nnodes="${nnodes}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.default_local_dir="${run_dir}" \
    trainer.val_before_train=True  \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    +trainer.save_best_val_checkpoint=true \
    +trainer.best_ckpt_metric_key='val-core/aime2025/acc/best@16/mean' \
    +trainer.best_ckpt_mode=max \
    "$@" 2>&1 | tee "${train_log}"

"${python_bin}" - "${train_log}" "${aime_eval_n}" "${math500_eval_n}" <<'PY'
import re, sys
from collections import defaultdict

train_log_path, aime_n, math_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def load_metrics(log_path):
    metrics = defaultdict(dict)
    patterns = {
        f"mean{math_n}": re.compile(rf"val-core/([^/\s]+)/acc/mean@{math_n}[^0-9-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        f"mean{aime_n}": re.compile(rf"val-core/([^/\s]+)/acc/mean@{aime_n}[^0-9-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        f"best{aime_n}": re.compile(rf"val-core/([^/\s]+)/acc/best@{aime_n}/mean[^0-9-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    }
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pat in patterns.items():
                for m in pat.finditer(line):
                    metrics[m.group(1)][key] = float(m.group(2))
    return metrics

metrics = load_metrics(train_log_path)
find = lambda name: next((k for k in metrics if name in k.lower()), None)
a24, a25, m5 = find("aime2024"), find("aime2025"), find("math500")
print("\nValidation summary:")
print(f"- aime2024: avg@{aime_n}={metrics[a24].get(f'mean{aime_n}') if a24 else None}  pass@{aime_n}={metrics[a24].get(f'best{aime_n}') if a24 else None}")
print(f"- aime2025: avg@{aime_n}={metrics[a25].get(f'mean{aime_n}') if a25 else None}  pass@{aime_n}={metrics[a25].get(f'best{aime_n}') if a25 else None}")
print(f"- math500:  avg@{math_n}={metrics[m5].get(f'mean{math_n}') if m5 else None}")
PY
