FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Hunyuan3D 2.0 Full (not mini)
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
    huggingface_hub \
    hy3dgen

# Model downloads on first run
ENV HF_HOME=/runpod-volume/hf_cache

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
