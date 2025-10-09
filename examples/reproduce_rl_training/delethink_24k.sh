# !/usr/bin/env bash

set -e

export VERL_LOGGING_LEVEL=INFO

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CONFIG_NAME="r1d-1.5b_deepscaler_delethink_24k"

export TREETUNEV__EXP_NAME=$CONFIG_NAME
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

python -m verl.trainer.main_policy_iteration \
    --config-name=$CONFIG_NAME \
    $@