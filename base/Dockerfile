# A CUDA+PyTorch "devel" base is the easiest path for compiling flash-attn, deepspeed, etc.
# Pick a tag that matches what you want (CUDA version + Python version).
# Example shown in RunPod docs/guides: PyTorch 2.4 + CUDA 12.4.1 + Python 3.11 on Ubuntu 22.04.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TOKENIZERS_PARALLELISM=false

# ---- system deps for building common ML/CUDA python packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    curl wget ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
    python3-dev \
    openssh-client \
    libaio-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Upgrade basic python build tooling
RUN python -m pip install -U pip setuptools wheel packaging

# ---- core ML / LLM training stack ----
# (pin versions in a requirements.txt if you want maximum reproducibility)
RUN pip install -U \
    runpod \
    "transformers>=4.41.0" \
    accelerate \
    datasets \
    peft \
    trl \
    bitsandbytes \
    safetensors \
    sentencepiece \
    protobuf \
    einops \
    huggingface_hub \
    evaluate \
    tqdm \
    rich \
    wandb \
    tensorboard

# ---- flash-attn (often compiles; limit jobs to avoid OOM during build) ----
# flash-attn install guidance: pip install flash-attn --no-build-isolation
ENV MAX_JOBS=4
RUN pip install -U ninja && \
    pip install flash-attn --no-build-isolation

# ---- Unsloth (choose the right extra for your CUDA+torch combo) ----
# From Unsloth docs: e.g. torch 2.4 + CUDA 12.1 => unsloth[cu121-torch240]
# We'll default to CUDA 12.4 + torch 2.4 => try cu124-torch240.
ARG UNSLOTH_VARIANT="cu124-torch240"
RUN pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git && \
    pip install "unsloth[${UNSLOTH_VARIANT}] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation

# ---- Optional: common training/serving extras (enable if you need them) ----
# RUN pip install -U deepspeed
# RUN pip install -U xformers
# RUN pip install -U lightning
# RUN pip install -U gradio fastapi uvicorn[standard]
