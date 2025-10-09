# Installation

- [Using `uv`](#using-uv)
- [Using Singularity](#using-singularity)
- [Using Docker](#using-docker)

## Using `uv`

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Python 3.10

```bash
uv python install 3.10
```

### 3. Create a virtual environment

```bash
cd /path/to/markovian-thinker
uv venv --python=3.10
source .venv/bin/activate
```

### 4. Install dependencies

```bash
# Make sure you have cuda 12.6 and cudnn 9.10.0 installed
uv pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-cache-dir packaging wheel psutil
ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
uv pip install --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# Install apex 
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=4 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -v --no-build-isolation .

uv pip install --no-cache-dir "tensordict==0.6.2" torchdata "transformers[hf_xet]>=4.52.3" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas cuda-bindings \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff

uv pip install --no-cache-dir --no-build-isolation flashinfer-python==0.2.9rc1
uv pip install --no-cache-dir --no-build-isolation --prerelease=allow "sglang[all]==0.4.9.post6"

# Fix packages
uv pip install --no-cache-dir "tensordict==0.6.2" "transformers[hf_xet]==4.54.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff InquirerPy math-verify fire

uv pip uninstall pynvml nvidia-ml-py
uv pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

uv pip install --no-cache-dir nvidia-cudnn-cu12==9.8.0.87
```

## Using Singularity

Download the singularity image from DockerHub and copy it to your scratch directory:

```bash
singularity pull docker://kazemnejad/treetune_verl:v2
mkdir -p ~/scratch/containers
cp treetune_verl_v2.sif ~/scratch/containers/
```

## Using Docker

```bash
docker pull kazemnejad/treetune_verl:v2
```