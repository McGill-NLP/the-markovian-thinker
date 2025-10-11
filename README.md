<div align="center">

# The Markovian Thinker

[Milad Aghajohari*](https://miladink.github.io), [Kamran Chitsaz*](https://kmchiti.github.io), [Amirhossein Kazemnejad*](https://kazemnejad.com/),  
[Sarath Chandar](https://sarathchandar.in), [Alessandro Sordoni†](https://www.microsoft.com/en-us/research/people/alsordon/), [Aaron Courville†](https://mila.quebec/en/directory/aaron-courville), [Siva Reddy†](https://sivareddy.in)

<sub>
*Equal Contribution &nbsp; †Equal Advising
</sub>

<br>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2510.06557v1) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/McGill-NLP/the-markovian-thinker-68debd2919c4ae47f50706cd)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#tldr" style="text-decoration: none; font-weight: bold;">TL;DR</a> •
    <a href="#installation" style="text-decoration: none; font-weight: bold;">Installation</a>
  </p>
  <p>
    <a href="#usage" style="text-decoration: none; font-weight: bold;">Usage </a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a>
  </p>
</div>

</div>




## Table of Contents

- [Updates](#updates)
- [Links](#links)
- [TL;DR](#tldr)
- [Usage](#usage)
  - [Install](#install)
  - [Quick Start](#quick-start)
  - [Reproduce RL experiments](#reproduce-rl-experiments)
  - [Evaluation](#evaluation)
  - [Delethink Tracing Demo](#delethink-tracing-demo)
- [Delethink Implementation Reference](#delethink-implementation-reference)
- [Codebase Overview](#codebase-overview)
- [Trajectory Visualization](#trajectory-visualization)
- [FAQ](#faq)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## Updates

* October 2025: 🎉 We release our paper, models and codebase. 

## Links

* 📄 [Paper](https://arxiv.org/abs/2510.06557v1)
* 🤗 Hugging Face
  * 💻 [The Markovian Thinker Collection](https://huggingface.co/collections/McGill-NLP/the-markovian-thinker-68debd2919c4ae47f50706cd)
  * 🤖 Models
    * [`delethink-24k-1.5b`](https://huggingface.co/McGill-NLP/delethink-24k-1.5b)
    * [`delethink-96k-1.5b`](https://huggingface.co/McGill-NLP/delethink-96k-1.5b)
    * [`longcot-24k-1.5b`](https://huggingface.co/McGill-NLP/longcot-24k-1.5b)
    * [`longcot-8k-1.5b`](https://huggingface.co/McGill-NLP/longcot-8k-1.5b)
* 🚀 [Release tweet]()
* 📝 [Blog post]()


## TL;DR

RL for reasoning LLMs has a trivial underlying RL environment (MDP) that treats the state as the whole prompt plus all past thinking tokens. That state keeps growing, which make the computation cost quadratic.

We propose Markovian Thinking paradigm, where the state size remains bounded/fixed.
This by design, and no matter the policy architecture, makes the compute cost linear with the number of thinking tokens, and memory stays flat.

### Delethink:
<p align="center">
  <img src="./assets/method.gif" width=80%/>
</p>

Delethink makes the environment chunked: generation happens in fixed-size chunks. At each boundary we reset the context to a fresh prompt with the original question plus a short carryover. The model learns to carry progress forward in text — becoming a Markovian thinker. LongCoT, in contrast, keeps concatenating tokens so context grows.
<p align="center">
  <img src="./assets/main_results.png" width=100%/>
</p>

- While only having 8K active context, Delethink 24K matches or beats LongCoT-RL 24K in accuracy while using less compute. 
- Beyond the trained budget, Delethink keeps improving while others plateau. Within the budget, both scale similarly under sequential sampling.
- Training cost vs. average thinking length on H100s: LongCoT ~ quadratic, Delethink ~ linear.

### Scaling Delethink to 96K

<p align="center">
  <img src="./assets/delethink_96k.png" width=80%/>
</p>

### GPT-OSS & Qwen3 Thinking show signs of Markovian Thinking

<p align="center">
  <img src="./assets/markovian_thinking_in_gpt_oss.png" width=80%/>
</p>

State-of-the-art reasoning LLMs, GPT-OSS-120B and Qwen3-30B-A3B, are capable of Markovian Thinking zero-shot, providing a strong initialization for training, signaling scalability. Delethink closely tracks LongCoT and recovers most of its final accuracy (the curves nearly coincide on Qwen3). Overall, Delethink looks promising for scaling.

### Takeaways

- RLVR is governed by a trivial MDP that everyone has forgotten. We can actually design any MDP we want.
- By making the state size bounded, we propose the Markovian Thinking paradigm, where the model learn to advance its reasoning by only conditioning on fixed-size states.
- Delethink is simple and effective: with an 8K fixed state it matches or beats LongCoT-RL and can think up to 128K tokens.
- GPT-OSS 120B and Qwen3 30B-A3B already show strong signs of Markovian thinking.

If you are interested in more details, please check out our [paper](https://arxiv.org/pdf/2503.20783)!

## Usage

### Install
Our codebase is built on top of [verl](https://github.com/volcengine/verl) and [SGLang](https://github.com/sgl-project/sglang). We provide a pre-built docker image and an instalation recipe based on `uv`. For more details, please see [INSTALLATION.md](./INSTALLATION.md).

### Quick Start
Use the following script to quickly start a Delethink training on DeepScaleR dataset with 24K response length with linear compute!

```bash
#!/bin/bash

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CONFIG_NAME="r1d-1.5b_deepscaler_delethink_24k"

export VERL_LOGGING_LEVEL=INFO
export TREETUNEV__EXP_NAME=$CONFIG_NAME
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

# base model
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# delethink parameters
# Total reasoning budget (C + (I-1) * (C-m)) = 8192 + (5-1) * (8192-8192/2) = 24576
max_response=8192
intermediate_max_new_tokens=4096
keep_head=100
fixed_num_optim_steps=2
multi_turn_max_assistant_turns=5

# rollout and sampling
gpu_memory_utilization=0.8
rollout_n=8
temp=0.6
top_k=-1
top_p=1.0

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

# actor optimization
clip_ratio_high=0.26
ppo_max_token_len_per_gpu=32768
ppo_epochs=1

# build hydra overrides
overrides=(
  actor_rollout_ref.model.path=$model_path

  data.max_response_length=$max_response
  algorithm.delethink.intermediate_max_new_tokens=$intermediate_max_new_tokens
  algorithm.delethink.keep_head=$keep_head
  algorithm.delethink.fixed_num_optim_steps=$fixed_num_optim_steps
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$multi_turn_max_assistant_turns

  actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization

  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-norm-trace-length
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu
  actor_rollout_ref.actor.ppo_epochs=$ppo_epochs

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

python -m verl.trainer.main_policy_iteration \
    --config-name=$CONFIG_NAME \
    ${overrides[@]}
```

For more tunable scripts, please see:
- [`examples/longcot_template.sh`](examples/longcot_template.sh)
- [`examples/delethink_template.sh`](examples/delethink_template.sh)
- [`examples/multi_node_template.sh`](examples/multi_node_template.sh) - For multi-node training

### Reproduce RL experiments

To reproduce the RL experiments in the paper, please use the following scripts:
```bash
# LongCoT-RL with 8K thinking tokens
sh examples/reproduce_rl_training/longcot_8k.sh

# LongCoT-RL with 24K thinking tokens
sh examples/reproduce_rl_training/longcot_24k.sh

# Delethink-RL with 24K thinking tokens
sh examples/reproduce_rl_training/delethink_24k.sh

# Delethink-RL with 96K thinking tokens
sh examples/reproduce_rl_training/delethink_96k.sh
```
These scripts are tested on single 8xH100 GPU node. For multi-node training, please see [./examples/multi_node_template.sh](./examples/multi_node_template.sh).


### Evaluation
TBD

### Delethink Tracing Demo
We provide a single-file, self-contained Delethink tracing implementation in [`delethink_tracing_demo.py`](./delethink_tracing_demo.py).
You can use the following command to run the demo:
```bash
python delething_tracing_demo.py
```
This will run Delethink tracing on the entire AIME 2024 eval set using SGLang and R1-Distill-Qwen-1.5B.
Options:

 - `--model` (str, default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B): Model path to use for sglang.Engine
 - `--tp` (int, default: 1): Tensor model parallel size
 - `--dp` (int, default: 1): Data parallel size; parallel inference engines
 - `--samples` (int, default: -1): Number of samples (all if -1)
 - `--delethink_context_size` (int, default: 8192): Delethink context size (C)
 - `--delethink_markovian_size` (int, default: 4096): Delethink markovian size (m)
 - `--delethink_iteration_cap` (int, default: 5): Delethink iteration cap (I)


## Delethink Implementation Reference

The core parts of the Delethink implementation are:

1. **Inference Loop (verl calls this "Agent Loop")**: [`verl/experiments/agent_loop/delethink_agent_loop.py`](verl/experiments/agent_loop/delethink_agent_loop.py) - Implements the logic for generating Delethink traces.
2. **Trainer**: [`verl/trainer/delethink_ppo/ray_trainer.py`](verl/trainer/delethink_ppo/ray_trainer.py) - Implements training using Delethink traces
3. **Configs**:
   - [`verl/trainer/config/r1d-1.5b_deepscaler_delethink_24k.yaml`](verl/trainer/config/r1d-1.5b_deepscaler_delethink_24k.yaml) - The config file for Delethink 24K
   - [`verl/trainer/config/r1d-1.5b_openmath_delethink_96k.yaml`](verl/trainer/config/r1d-1.5b_openmath_delethink_96k.yaml) - The config file for Delethink 96K

## Codebase Overview

#### Directory Structure
```bash
├── examples/                   # Example scripts for training
├── verl/                       # the verl codebase
│   ├── trainer/
│   │   └── config/             # Configuration files for different experiments
├── scripts/                    # Utility scripts
├── delethink_tracing_demo.py   # Single-file, self-contained Delethink tracing implementation
├── INSTALLATION.md             # Installation guide
├── README.md                   # README for the codebase
```

#### Important Config Files
| Config File | Description |
|------------|-------------|
| [`verl/trainer/config/r1d-1.5b_deepscaler.yaml`](verl/trainer/config/r1d-1.5b_deepscaler.yaml) | The parent config file for all DeepScaleR experiments |
| [`verl/trainer/config/r1d-1.5b_deepscaler_longcot_24k.yaml`](verl/trainer/config/r1d-1.5b_deepscaler_longcot_24k.yaml) | The config file for LongCoT-RL 24K |
| [`verl/trainer/config/r1d-1.5b_deepscaler_longcot_8k.yaml`](verl/trainer/config/r1d-1.5b_deepscaler_longcot_8k.yaml) | The config file for LongCoT-RL 8K |
| [`verl/trainer/config/r1d-1.5b_deepscaler_delethink_24k.yaml`](verl/trainer/config/r1d-1.5b_deepscaler_delethink_24k.yaml) | The config file for Delethink 24K |
| [`verl/trainer/config/r1d-1.5b_openmath_delethink_96k.yaml`](verl/trainer/config/r1d-1.5b_openmath_delethink_96k.yaml) | The config file for Delethink 96K |


#### Core Components

**Main Entry Point:**
- [`verl/trainer/main_policy_iteration.py`](verl/trainer/main_policy_iteration.py)

**Trainers:**
- [`verl/trainer/<algo_name>/ray_trainer.py`](verl/trainer/) - The main logic is inside the `trainer.fit()` method

**Actor (Policy backward):**
- [`verl/workers/actor/dp_actor.py`](verl/workers/actor/dp_actor.py)

**Rollout (SGLang/Inference Engine):**
- [`verl/rollout/sglang_rollout/sglang_rollout.py`](verl/rollout/sglang_rollout/sglang_rollout.py)
- [`verl/rollout/sglang_rollout/custom_sglang_rollout.py`](verl/rollout/sglang_rollout/custom_sglang_rollout.py) - SGLang with custom modifications

**Dataset:**
- [`verl/utils/dataset/rl_dataset.py`](verl/utils/dataset/rl_dataset.py) - Dataset implementation
- [`verl/tasks/`](verl/tasks/) - Tasks that generate input data for `rl_dataset.py`

**Reward Function/Mechanism:**
- [`verl/utils/reward_score/treetune_math_verify.py`](verl/utils/reward_score/treetune_math_verify.py) - Actual reward function for math problems
- [`verl/workers/reward_manager/naive.py`](verl/workers/reward_manager/naive.py) - Naive reward manager (thin wrapper)
- [`verl/workers/reward_manager/delethink.py`](verl/workers/reward_manager/delethink.py) - Delethink reward manager

Reward managers wrap the reward function and handle some custom logics like length penalty, etc.
We mostly use `naive.py`/`delethink.py` reward managers which are thin wrappers and don't modify the reward computation.

**ActorRolloutRefWorker:**
- [`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)


## Trajectory Visualization

Use the following command to visualize the LongCoT trajectories

```bash
pip install textual==0.52.1
python scripts/rollout_viewer.py experiments/<exp_name>/train_rollouts
```

To visualize the unflattened Delethink trajectories, use the following command:

```bash
python scripts/rollout_viewer.py experiments/<exp_name>/train_rollouts_delethink
```

## FAQ

### How to setup the VSCode ray debugger?

<details>
<summary>Show answer</summary>

Follow the guide: [VERL Ray Debug Tutorial](https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html)

</details>

### Where is the config file for Delethink?

<details>
<summary>Show answer</summary>

The config file is located at: [`configs/polIter_r1d-1.5b_deepscaler_delethink.yaml`](configs/polIter_r1d-1.5b_deepscaler_delethink.yaml)

</details>

### Where is loss aggregation implemented?

<details>
<summary>Show answer</summary>

Loss aggregation is implemented in [`verl/trainer/ppo/core_algos.py`](verl/trainer/ppo/core_algos.py) using the `agg_loss` and `agg_loss_with_trace_lengths` functions.

To change the aggregation mode, use `actor_rollout_ref.actor.loss_agg_mode` in the config.

</details>

### What does all the batch size configuration mean in the VERL universe?

<details>
<summary>Show answer</summary>

- `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` are with respect to the number of prompts
- To map to TreeTune Next convention:
  - `number_of_episodes_per_iteration = data.train_batch_size * actor_rollout_ref.rollout.n`
  - `target_batch_size = actor_rollout_ref.actor.ppo_mini_batch_size * actor_rollout_ref.rollout.n`
- The number of gradient updates per iteration is `data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size`
- We also have `algorithm.delethink.fixed_num_optim_steps` which allows us to dynamically adjust the batch size to have a fixed number of gradient updates per iteration

</details>

### How to use smaller batch sizes for debugging/development?

<details>
<summary>Show answer</summary>

Use configuration overrides like this:

```bash
python -m verl.trainer.main_policy_iteration \
    --config-name="<config_name>" \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enable_debug=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    data.train_batch_size=4
```

</details>

### How can I add a new config?

<details>
<summary>Show answer</summary>

Simply define it in YAML format. You can also assign a target config class to perform config validation. 

See `algorithm.delethink._target_` in [`configs/polIter_r1d-1.5b_deepscaler_delethink.yaml`](configs/polIter_r1d-1.5b_deepscaler_delethink.yaml) for an example.

</details>

### Where are all important components in VERL implemented?

<details>
<summary>Show answer</summary>

**Main Entry Point:**
- [`verl/trainer/main_policy_iteration.py`](verl/trainer/main_policy_iteration.py)

**Trainers:**
- [`verl/trainer/<algo_name>/ray_trainer.py`](verl/trainer/) - The main logic is inside the `trainer.fit()` method

**Actor (Policy backward):**
- [`verl/workers/actor/dp_actor.py`](verl/workers/actor/dp_actor.py)

**Rollout (SGLang/Inference Engine):**
- [`verl/rollout/sglang_rollout/sglang_rollout.py`](verl/rollout/sglang_rollout/sglang_rollout.py)
- [`verl/rollout/sglang_rollout/custom_sglang_rollout.py`](verl/rollout/sglang_rollout/custom_sglang_rollout.py) - SGLang with custom modifications

**Dataset:**
- [`verl/utils/dataset/rl_dataset.py`](verl/utils/dataset/rl_dataset.py) - Dataset implementation
- [`verl/tasks/`](verl/tasks/) - Tasks that generate input data for `rl_dataset.py`

**Reward Function/Mechanism:**
- [`verl/utils/reward_score/treetune_math_verify.py`](verl/utils/reward_score/treetune_math_verify.py) - Actual reward function for math problems
- [`verl/workers/reward_manager/naive.py`](verl/workers/reward_manager/naive.py) - Naive reward manager (thin wrapper)
- [`verl/workers/reward_manager/delethink.py`](verl/workers/reward_manager/delethink.py) - Delethink reward manager

Reward managers wrap the reward function and handle some custom logics like length penalty, etc.
We mostly use `naive.py`/`delethink.py` reward managers which are thin wrappers and don't modify the reward computation.

**ActorRolloutRefWorker:**
- [`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)

</details>

## Citation

```bibtex
@misc{Aghajohari2025:TheMarkovianThinker,
      title={The Markovian Thinker}, 
      author={Milad Aghajohari and Kamran Chitsaz and Amirhossein Kazemnejad and Sarath Chandar and Alessandro Sordoni and Aaron Courville and Siva Reddy},
      year={2025},
      eprint={2510.06557},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.06557}, 
}
```

## Acknowledgement
* The training infra is built on top of [verl](https://github.com/volcengine/verl), which employs [SGLang](https://github.com/sgl-project/sglang)
* Parts of this code is borrowed from [VinePPO](https://github.com/McGill-NLP/VinePPO).
