# Use your slim CUDA 12.8 + PyTorch 2.7 + HF + Unsloth base image
FROM mannanb/mltraining:latest

WORKDIR /

COPY rp_handler.py /rp_handler.py
COPY ./projects /projects

# RunPod Serverless executes this when the container starts
CMD ["python3", "-u", "/rp_handler.py"]
