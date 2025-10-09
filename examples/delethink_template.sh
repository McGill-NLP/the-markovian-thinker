# !/usr/bin/env bash

set -e

#################################
# Modifying the config
#################################

# user-tunable variables (hydra overrides)
# base model and data lengths
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
overlong_prompt_length=2048

# data loader
train_batch_size=128
val_batch_size=16

# policy (actor) optimization and loss
lr=1e-6
weight_decay=0.0
use_kl_loss=false
kl_coef=0.000
entropy_coeff=0
clip_ratio_high=0.26
ppo_mini_batch_size=128  # delethink config will dynamically adjust
ppo_max_token_len_per_gpu=32768
ppo_epochs=1

# sampling (rollout) params
rollout_mode=async
calculate_log_probs=true
rollout_n=8
temp=0.6
top_k=-1
top_p=1.0
gpu_memory_utilization=0.8

# validation sampling params
val_top_k=-1
val_top_p=1.0
val_temperature=0.6
val_n=32
val_do_sample=true

# trainer schedule and logging
train_steps=1000
save_freq=10

# checkpoint policy
keep_every_n_saves=5
push_to_hub_freq=20

# delethink parameters
# Total reasoning budget (C + (I-1) * (C-m)) = 8192 + (5-1) * (8192-8192/2) = 24576
max_response=8192 # thinking budget (C)
keep_head=100 # number of tokens to fold into the query from the first chunk
fixed_num_optim_steps=2 # number of optimization steps per each iteration
multi_turn_max_assistant_turns=5 # max number of delethink turns (I)

use_trimmer_class=true
trimmer_name=progressive
trimmer_kwargs={}  
agent_worker_name=with_extra_fields
agent_num_workers=1
use_flat_batch_correction_ratio=true

# build hydra overrides
overrides=(
  data.overlong_prompt_length=$overlong_prompt_length
  data.max_response_length=$max_response
  data.train_batch_size=$train_batch_size
  data.val_batch_size=$val_batch_size

  actor_rollout_ref.model.path=$model_path

  algorithm.adv_estimator=$adv_estimator

  algorithm.delethink.keep_head=$keep_head
  algorithm.delethink.fixed_num_optim_steps=$fixed_num_optim_steps
  algorithm.delethink.use_flat_batch_correction_ratio=$use_flat_batch_correction_ratio
  algorithm.delethink.use_trimmer_class=$use_trimmer_class
  algorithm.delethink.trimmer_name=$trimmer_name
  algorithm.delethink.trimmer_kwargs=$trimmer_kwargs

  reward_model.reward_manager=delethink
  reward_model.launch_reward_fn_async=false

  actor_rollout_ref.rollout.mode=$rollout_mode
  actor_rollout_ref.rollout.calculate_log_probs=$calculate_log_probs
  actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$multi_turn_max_assistant_turns
  actor_rollout_ref.rollout.agent.worker_name=$agent_worker_name
  actor_rollout_ref.rollout.agent.num_workers=$agent_num_workers

  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-norm-trace-length
  actor_rollout_ref.actor.entropy_coeff=$entropy_coeff
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss
  actor_rollout_ref.actor.kl_loss_coef=$kl_coef
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
  actor_rollout_ref.actor.optim.lr=$lr
  actor_rollout_ref.actor.optim.weight_decay=$weight_decay
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu
  actor_rollout_ref.actor.ppo_epochs=$ppo_epochs
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla_with_trace_lengths

  actor_rollout_ref.actor.checkpoint.keep_every_n_saves=$keep_every_n_saves
  actor_rollout_ref.actor.checkpoint.push_to_hub_freq=$push_to_hub_freq

  actor_rollout_ref.rollout.n=$rollout_n
  actor_rollout_ref.rollout.temperature=$temp
  actor_rollout_ref.rollout.top_k=$top_k
  actor_rollout_ref.rollout.top_p=$top_p

  actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k
  actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p
  actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature
  actor_rollout_ref.rollout.val_kwargs.n=$val_n
  actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample

  trainer.total_training_steps=$train_steps
  trainer.save_freq=$save_freq
)

#################################
# End of modifying the config
#################################

export VERL_LOGGING_LEVEL=INFO

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CONFIG_NAME="r1d-1.5b_deepscaler_delethink_24k"

export TREETUNEV__EXP_NAME=$CONFIG_NAME
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

python -m verl.trainer.main_policy_iteration \
    --config-name=$CONFIG_NAME \
    ${overrides[@]} \
    $@