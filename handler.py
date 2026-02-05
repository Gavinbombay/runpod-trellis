"""
RunPod Serverless Handler — TRELLIS 2 (Microsoft)
High-quality 3D asset generation with PBR textures
"""

import runpod
import torch
import os
import time
import base64
import tempfile
from PIL import Image
import io

# Lazy load model
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading TRELLIS 2 pipeline...")
        from trellis.pipelines import TrellisImageTo3DPipeline
        _pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        _pipeline.to("cuda")
        print("TRELLIS 2 loaded successfully")
    return _pipeline

def handler(job):
    """
    TRELLIS 2 Image-to-3D Generation

    Input:
        image_b64: Base64 encoded input image
        resolution: Output resolution (512, 1024, or 1536) - default 512
        seed: Random seed (-1 for random)
        output_format: 'glb' or 'obj' - default 'glb'

    Output:
        mesh_b64: Base64 encoded GLB/OBJ file
        generation_time: Time taken in seconds
        resolution: Actual resolution used
    """
    start_time = time.time()

    job_input = job["input"]
    image_b64 = job_input.get("image_b64")
    resolution = job_input.get("resolution", 512)
    seed = job_input.get("seed", -1)
    output_format = job_input.get("output_format", "glb")

    if not image_b64:
        return {"error": "image_b64 required"}

    # Validate resolution
    if resolution not in [512, 1024, 1536]:
        resolution = 512

    # Decode input image
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # Set seed
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating 3D at {resolution}³ resolution, seed={seed}")

    try:
        pipeline = get_pipeline()

        # Generate 3D asset
        with torch.no_grad():
            outputs = pipeline.run(
                image,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3,
                },
            )

        # Extract mesh
        mesh = outputs["mesh"][0]

        # Export to file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as f:
            output_path = f.name

        if output_format == "glb":
            mesh.export(output_path)
        else:
            mesh.export(output_path, file_type="obj")

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
            "resolution": resolution,
            "seed": seed,
            "format": output_format,
            "size_bytes": len(mesh_data),
        }

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

runpod.serverless.start({"handler": handler})
