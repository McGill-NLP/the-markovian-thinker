# !/usr/bin/env bash

set -e


#################################
# Modifying the config
#################################


# User-tunable variables (Hydra overrides)
# Base model and data lengths
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
max_prompt=2048
max_response=24576

# Data loader
train_batch_size=128
val_batch_size=16

# Policy (actor) optimization and loss
lr=1e-6
weight_decay=0.0
use_kl_loss=false
kl_coef=0.000
entropy_coeff=0
clip_ratio_high=0.26
ppo_mini_batch_size=64
ppo_max_token_len_per_gpu=32768
ppo_epochs=1

# Truncated Importance Sampling (TIS)
tis_imp_ratio_cap=2
loss_mode=vanilla_with_trace_lengths

# Sampling (rollout) params
rollout_n=8
temp=0.6
top_k=-1
top_p=1.0
calculate_log_probs=true
gpu_memory_utilization=0.8

# Validation sampling params
val_top_k=-1
val_top_p=1.0
val_temperature=0.6
val_n=32
val_do_sample=true

# Trainer schedule and logging
train_steps=1000
save_freq=25
test_freq=50
rollout_dump_freq=25
full_episodes_dump_freq=50
log_val_generations=1000

# Checkpoint policy
keep_every_n_saves=2
keep_only_hf_in_prev=false
push_to_hub_freq=50

# Build Hydra overrides
overrides=(
  data.max_prompt_length=$max_prompt
  data.max_response_length=$max_response
  data.train_batch_size=$train_batch_size
  data.val_batch_size=$val_batch_size

  actor_rollout_ref.model.path=$model_path

  actor_rollout_ref.actor.entropy_coeff=$entropy_coeff
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss
  actor_rollout_ref.actor.kl_loss_coef=$kl_coef
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
  actor_rollout_ref.actor.optim.lr=$lr
  actor_rollout_ref.actor.optim.weight_decay=$weight_decay
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu
  actor_rollout_ref.actor.ppo_epochs=$ppo_epochs
  actor_rollout_ref.actor.tis_imp_ratio_cap=$tis_imp_ratio_cap
  actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode

  actor_rollout_ref.actor.checkpoint.keep_every_n_saves=$keep_every_n_saves
  actor_rollout_ref.actor.checkpoint.keep_only_hf_in_previous_saves=$keep_only_hf_in_prev
  actor_rollout_ref.actor.checkpoint.push_to_hub_freq=$push_to_hub_freq

  actor_rollout_ref.rollout.n=$rollout_n
  actor_rollout_ref.rollout.temperature=$temp
  actor_rollout_ref.rollout.top_k=$top_k
  actor_rollout_ref.rollout.top_p=$top_p
  actor_rollout_ref.rollout.calculate_log_probs=$calculate_log_probs
  actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization

  actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k
  actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p
  actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature
  actor_rollout_ref.rollout.val_kwargs.n=$val_n
  actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample

  trainer.total_training_steps=$train_steps
  trainer.save_freq=$save_freq
  trainer.test_freq=$test_freq
  trainer.rollout_dump_freq=$rollout_dump_freq
  trainer.full_episodes_dump_freq=$full_episodes_dump_freq
  trainer.log_val_generations=$log_val_generations
)

#################################
# End of modifying the config
#################################

export VERL_LOGGING_LEVEL=INFO

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CONFIG_NAME="r1d-1.5b_deepscaler_longcot_24k"

export TREETUNEV__EXP_NAME=$CONFIG_NAME
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

python -m verl.trainer.main_policy_iteration \
    --config-name=$CONFIG_NAME \
    ${overrides[@]} \
    $@