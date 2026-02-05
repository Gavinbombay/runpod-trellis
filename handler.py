"""
RunPod Serverless Handler — Hunyuan3D 2.0 Full
High-quality 3D asset generation optimized for H100
"""

import runpod
import torch
import os
import time
import base64
import tempfile
from PIL import Image
import io

# H100 optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Lazy load model
_pipeline = None
_texgen = None

def get_pipeline():
    global _pipeline, _texgen
    if _pipeline is None:
        print("Loading Hunyuan3D 2.0 Full pipeline (H100 optimized)...")
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        # Detect H100/Hopper for BF16 (better than FP16 on H100)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        use_bf16 = "H100" in gpu_name or "H800" in gpu_name or "Hopper" in gpu_name
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"GPU: {gpu_name}, using dtype: {dtype}")

        # Full model (not mini)
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-dit-v2-0",
            torch_dtype=dtype,
        )
        _pipeline.to("cuda")

        # Texture generator for full quality
        _texgen = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-paint-v2-0",
            torch_dtype=dtype,
        )
        _texgen.to("cuda")

        print("Hunyuan3D 2.0 Full loaded successfully (H100 optimized)")
    return _pipeline, _texgen

def handler(job):
    """
    Hunyuan3D 2.0 Full Image-to-3D Generation

    Input:
        image_b64: Base64 encoded input image
        steps: Inference steps (default 50 for quality)
        seed: Random seed (-1 for random)
        texture: Generate textures (default true)
        output_format: 'glb' or 'obj' - default 'glb'

    Output:
        mesh_b64: Base64 encoded GLB/OBJ file
        generation_time: Time taken in seconds
    """
    start_time = time.time()

    job_input = job["input"]
    image_b64 = job_input.get("image_b64")
    steps = job_input.get("steps", 50)  # Higher for quality
    seed = job_input.get("seed", -1)
    with_texture = job_input.get("texture", True)
    output_format = job_input.get("output_format", "glb")

    if not image_b64:
        return {"error": "image_b64 required"}

    # Decode input image
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # Set seed
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Generating 3D with {steps} steps, texture={with_texture}, seed={seed}")

    try:
        pipeline, texgen = get_pipeline()

        # Generate mesh
        with torch.no_grad():
            mesh = pipeline(
                image=image,
                num_inference_steps=steps,
                generator=generator,
            )[0]

        # Generate texture if requested
        if with_texture and texgen is not None:
            print("Generating textures...")
            mesh = texgen(
                mesh=mesh,
                image=image,
                generator=generator,
            )[0]

        # Export to file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as f:
            output_path = f.name

        mesh.export(output_path)

        # Read and encode
        with open(output_path, "rb") as f:
            mesh_data = f.read()
        mesh_b64 = base64.b64encode(mesh_data).decode("utf-8")

        # Cleanup
        os.unlink(output_path)

        generation_time = round(time.time() - start_time, 2)

        print(f"Generation complete: {generation_time}s, {len(mesh_data)} bytes")

        return {
            "mesh_b64": mesh_b64,
            "generation_time": generation_time,
            "seed": seed,
            "format": output_format,
            "size_bytes": len(mesh_data),
            "with_texture": with_texture,
        }

    except Exception as e:
        import traceback
        return {"error": f"Generation failed: {str(e)}", "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
