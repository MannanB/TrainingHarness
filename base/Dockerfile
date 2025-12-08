# ----------------------------------------------------------------------
# Single-stage image: CUDA 12.8 + Python 3.10 + PyTorch nightly + flash-attn
# ----------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# System deps + Python 3.10 + pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-venv \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && python3 -m pip install --upgrade pip \
    && rm -rf /root/.cache/pip

# Install PyTorch nightly with CUDA 12.x wheels (cu121 is current nightly tag)
RUN python3 -m pip install --no-cache-dir \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# flash-attn prebuilt wheel for CUDA 12.8, torch 2.9, Python 3.10
ENV WHEEL_NAME="flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
ENV DOWNLOAD_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/${WHEEL_NAME}"

RUN echo "Downloading: ${DOWNLOAD_URL}"
RUN wget "${DOWNLOAD_URL}" -O "/tmp/${WHEEL_NAME}"
RUN python3 -m pip install --no-cache-dir --no-deps "/tmp/${WHEEL_NAME}"
RUN rm "/tmp/${WHEEL_NAME}"

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

# # Default command – override in your serverless layer
# CMD ["python3"]
