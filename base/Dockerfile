# This image comes pre-baked with CUDA 12.6/12.8 support and optimized Unsloth for Blackwell
FROM unsloth/unsloth:latest

# Switch to root if you need to install extra system deps
USER root

# Install any extras you need that aren't in standard unsloth
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    scipy \
    "transformers>=4.48.0" 

# Set workspace
WORKDIR /workspace
CMD ["/bin/bash"]
