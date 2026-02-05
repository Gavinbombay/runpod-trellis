# Use community TRELLIS 2 image as base (includes all CUDA deps)
FROM camenduru/trellis2:latest

WORKDIR /app

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# H100 optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ATTN_BACKEND=flash-attn
ENV HF_HOME=/runpod-volume/hf_cache

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
