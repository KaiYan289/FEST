#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEST_ROOT="${FEST_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${FEST_ROOT}"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}}"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-10000000000}"
RAY_CPUS="${SLURM_CPUS_PER_TASK:-64}"
unset ROCR_VISIBLE_DEVICES
export WANDB_ENTITY="${WANDB_ENTITY:-kaiyan3}"
ray stop --force || true

export DATA_DIR=./dataset
export WANDB_INIT_TIMEOUT=300
export MAX_RESPONSE_LENGTH=8192
export MAX_BATCHED_TOKENS=16384

USER_ROOT="${USER_ROOT:-/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/${USER}}"
export MODEL_PATH="${FEST_MODEL_PATH:-${USER_ROOT}/Qwen2.5-Math-1.5B}"
export FEST_INIT_CKPT="${FEST_INIT_CKPT:-${FEST_ROOT}/train_results/relift/SFT_128shot_Qwen2.5-Math-1.5B__seed1/global_step_40}"
export FEST_SEED="${FEST_SEED:-1}"
export EXP_NAME="${FEST_EXP_NAME:-RL_from_SFT128_step40_seed${FEST_SEED}}"
export WANDB_DIR="${USER_ROOT}/wandb_runs"
export WANDB_CACHE_DIR="${USER_ROOT}/wandb_cache"
export WANDB_CONFIG_DIR="${USER_ROOT}/wandb_config"
export XDG_CACHE_HOME="${USER_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${USER_ROOT}/xdg_config"
mkdir -p "$RAY_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
export WANDB_PROJECT="relift"

CKPT_DIR="./train_results/${WANDB_PROJECT}/${EXP_NAME}"
RESUME_CKPT="${FEST_INIT_CKPT}"
latest_step=-1
for candidate in "${CKPT_DIR}"/global_step_*; do
    [[ -d "${candidate}" ]] || continue
    step="${candidate##*/global_step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > latest_step )); then
        latest_step="${step}"
        RESUME_CKPT="${candidate}"
    fi
done
echo "Loading checkpoint: ${RESUME_CKPT}"

python -u -m verl.ternary_dpo.main_ppo \
    ray_init.num_cpus="${RAY_CPUS}" \
    +data.seed="${FEST_SEED}" \
    +actor_rollout_ref.rollout.seed="${FEST_SEED}" \
    trainer.soft_update_factor=0.05 \
    +actor_rollout_ref.actor.sft.sft_loss_type="v0" \
    +actor_rollout_ref.actor.beta_gold_to_win=0.01 \
    +actor_rollout_ref.actor.beta_win_to_lose=0.01 \
    +actor_rollout_ref.actor.beta_gold_to_lose=0.1 \
    +actor_rollout_ref.actor.grpo_clip_flag=True \
    +actor_rollout_ref.actor.dpo_clamp_value=-50.0 \
    +actor_rollout_ref.actor.token_flag="sum" \
    +actor_rollout_ref.actor.dpo_loss_ratio=0.01 \
    +actor_rollout_ref.actor.ref_prob_type="ref" \
    +actor_rollout_ref.actor.logprob_agg_type="sum" \
    actor_rollout_ref.actor.sft.sft_epochs=1 \
    actor_rollout_ref.actor.sft.sft_data_size=512 \
    critic.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.sft.sft_mini_batch_size=256 \
    actor_rollout_ref.actor.sft.sft_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.sft.entropy_coeff=0.0001 \
    actor_rollout_ref.actor.sft.grad_clip=0.5 \
    actor_rollout_ref.actor.optim.sft.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    algorithm.adv_estimator=grpo \
    +data.gold_files=$DATA_DIR/train_data/openr1_128_shot_sample_1_hpt_format.parquet \
    data.train_files=$DATA_DIR/train_data/openr1_hpt.parquet \
    data.val_files=$DATA_DIR/valid_hpt.parquet \
    data.truncation='left' \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_target_length=$MAX_RESPONSE_LENGTH \
    data.target_key="target" \
    reward_model.reward_manager="naive" \
    custom_reward_function.path="./utils/reward_utils/my_reward_func_v2.py" \
    custom_reward_function.name="my_reward_func" \
    custom_reward_function.overlong_buffer.enable=False \
    custom_reward_function.overlong_buffer.len=4096 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.actor.special_mode="RL_nogoldrl" \
    +actor_rollout_ref.actor.gold_data_weight=1 \
    +actor_rollout_ref.actor.rl_data_weight=1 \
    +actor_rollout_ref.actor.sft_weight_for_weighted=-1 \
    +actor_rollout_ref.actor.rl_weight_for_weighted=-1 \
    +trainer.special_mode="RL_nogoldrl" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_BATCHED_TOKENS \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.max_grad_norm=80.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.sft_param_offload=False \
    actor_rollout_ref.actor.fsdp_config.sft_optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.sft_use_lora=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_BATCHED_TOKENS \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_BATCHED_TOKENS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.update_ref_freq=-2 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir=./rollout_data/${EXP_NAME} \
    trainer.default_local_dir=./train_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.total_epochs=2 \
    trainer.resume_mode=resume_path \
    "trainer.resume_from_path=${RESUME_CKPT}"
