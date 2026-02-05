"""
RunPod Serverless Handler — Microsoft TRELLIS 2
High-quality 3D asset generation (~3 sec on H100)
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
os.environ["ATTN_BACKEND"] = "flash-attn"  # Use flash attention on H100

# Lazy load model
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading TRELLIS 2 pipeline (H100 optimized)...")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _pipeline.cuda()

        print("TRELLIS 2 loaded successfully")
    return _pipeline

def handler(job):
    """
    TRELLIS 2 Image-to-3D Generation

    Input:
        image_b64: Base64 encoded input image
        resolution: Output resolution - 512, 1024, or 1536 (default 512)
        texture_size: Texture resolution (default 4096)
        decimation_target: Mesh face count target (default 1000000)
        seed: Random seed (-1 for random)
        output_format: 'glb' or 'obj' (default 'glb')

    Output:
        mesh_b64: Base64 encoded GLB/OBJ file
        generation_time: Time taken in seconds
    """
    start_time = time.time()

    job_input = job["input"]
    image_b64 = job_input.get("image_b64")
    resolution = job_input.get("resolution", 512)
    texture_size = job_input.get("texture_size", 4096)
    decimation_target = job_input.get("decimation_target", 1000000)
    seed = job_input.get("seed", -1)
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
    torch.manual_seed(seed)

    print(f"Generating 3D with TRELLIS 2: resolution={resolution}, texture={texture_size}, seed={seed}")

    try:
        pipeline = get_pipeline()

        # Generate mesh
        with torch.no_grad():
            mesh = pipeline.run(image)[0]
            mesh.simplify(16777216)  # nvdiffrast limit

        # Export to GLB
        import o_voxel

        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as f:
            output_path = f.name

        if output_format == "glb":
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True
            )
            glb.export(output_path, extension_webp=True)
        else:
            # OBJ export
            mesh.export(output_path)

        # Read and encode
        with open(output_path, "rb") as f:
            mesh_data = f.read()
        mesh_b64 = base64.b64encode(mesh_data).decode("utf-8")

        # Cleanup
        os.unlink(output_path)

        generation_time = round(time.time() - start_time, 2)

        print(f"TRELLIS 2 generation complete: {generation_time}s, {len(mesh_data)} bytes")

        return {
            "mesh_b64": mesh_b64,
            "generation_time": generation_time,
            "seed": seed,
            "format": output_format,
            "size_bytes": len(mesh_data),
            "resolution": resolution,
            "texture_size": texture_size,
        }

    except Exception as e:
        import traceback
        return {"error": f"Generation failed: {str(e)}", "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
