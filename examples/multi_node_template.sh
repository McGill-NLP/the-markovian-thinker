# !/usr/bin/env bash

set -e

# Assert that NODE_RANK, RAY_HEAD_IP, and RAY_HEAD_PORT are set for multi-node training
if [ -z "$NODE_RANK" ]; then
    echo "Error: NODE_ID environment variable must be set for multi-node training"
    exit 1
fi

if [ -z "$RAY_HEAD_IP" ]; then
    echo "Error: RAY_HEAD_IP environment variable must be set for multi-node training"
    exit 1
fi

if [ -z "$RAY_HEAD_PORT" ]; then
    echo "Error: RAY_HEAD_PORT environment variable must be set for multi-node training"
    exit 1
fi

export VERL_LOGGING_LEVEL=INFO

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CONFIG_NAME="r1d-1.5b_deepscaler_longcot_24k"

export TREETUNEV__EXP_NAME=$CONFIG_NAME
export TREETUNEV__NNODE=2
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

if [ "${NODE_RANK:-0}" -eq 0 ]; then
    echo "Starting Ray head node on $RAY_HEAD_IP:$RAY_HEAD_PORT"
    ray start --head --port="$RAY_HEAD_PORT" \
      --num-cpus "16" --num-gpus "$NUM_GPUS"
    sleep 10
  else
    sleep 10
    echo "[worker] waiting for head at $RAY_HEAD_IP:$RAY_HEAD_PORT"
    ray start --address="${RAY_HEAD_IP}:${RAY_HEAD_PORT}" \
      --num-cpus "16" --num-gpus "$NUM_GPUS" --block
    exit 0
  fi

python -m verl.trainer.main_policy_iteration \
    --config-name=$CONFIG_NAME \
    $@