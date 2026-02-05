FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install TRELLIS dependencies
RUN pip install --no-cache-dir \
    runpod \
    torch torchvision \
    transformers \
    diffusers \
    accelerate \
    trimesh \
    pygltflib \
    pillow \
    numpy \
    safetensors \
    huggingface_hub

# Clone and install TRELLIS
RUN git clone https://github.com/microsoft/TRELLIS.git /app/trellis_repo \
    && cd /app/trellis_repo \
    && pip install -e .

# Model downloads on first run (~8GB, adds to cold start but faster build)
# HuggingFace cache dir
ENV HF_HOME=/runpod-volume/hf_cache

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
