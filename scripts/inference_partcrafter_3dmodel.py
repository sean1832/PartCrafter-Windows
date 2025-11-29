import argparse
import os
import sys
import time
from typing import Any, Tuple, Union

import numpy as np
import pyrender
import torch
import trimesh
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from PIL import Image

# Add parent directory to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import (
    export_renderings,
    make_grid_for_images_or_videos,
    render_normal_views_around_mesh,
    render_views_around_mesh,
)


def normalize_mesh(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    """Centers and scales the mesh to fit within the view."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Center mesh
    mesh.apply_translation(-mesh.centroid)

    # Scale to fit in unit box with margin
    extents = mesh.extents
    scale_factor = 0.8 / np.max(extents)
    mesh.apply_scale(scale_factor)
    return mesh


def get_best_render_view(mesh: trimesh.Trimesh) -> Image.Image:
    """Renders multiple views and selects the one with the most visible object surface."""
    rendered_views = render_views_around_mesh(
        mesh,
        num_views=16,
        radius=1.5,
        image_size=(512, 512),
        light_intensity=8.0,
        flags=pyrender.constants.RenderFlags.RGBA,
    )

    best_idx = 0
    max_pixels = 0
    processed_views = []

    for view in rendered_views:
        # Composite over white background
        if view.mode == "RGBA":
            background = Image.new("RGBA", view.size, (255, 255, 255, 255))
            final_view = Image.alpha_composite(background, view).convert("RGB")
        else:
            final_view = view.convert("RGB")

        processed_views.append(final_view)

        # Calculate visibility (heuristic: count non-white pixels)
        gray = final_view.convert("L")
        non_white = np.sum(np.array(gray) < 250)
        if non_white > max_pixels:
            max_pixels = non_white
            best_idx = len(processed_views) - 1

    return processed_views[best_idx]


@torch.no_grad()
def run_partcrafter_mesh(
    pipe: Any,
    mesh_path: str,
    num_parts: int,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
) -> Tuple[Any, Image.Image]:
    # 1. Load and Normalization
    print(f"Loading and normalizing mesh: {mesh_path}")
    input_mesh = trimesh.load(mesh_path)
    input_mesh = normalize_mesh(input_mesh)

    # 2. Render Virtual View
    print("Rendering optimal virtual view...")
    img_pil = get_best_render_view(input_mesh)

    # 3. Run Inference
    start_time = time.time()
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")

    for i in range(len(outputs)):
        if outputs[i] is None:
            # If decoding error, use dummy mesh
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

    return outputs, img_pil


MAX_NUM_PARTS = 16

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="Path to input 3D mesh (obj, glb)")
    parser.add_argument("--num_parts", type=int, required=True, help="number of parts to generate")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"

    # Download pretrained weights
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)

    # Init Pipeline
    pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
    set_seed(args.seed)

    # Run Inference
    outputs, processed_image = run_partcrafter_mesh(
        pipe,
        mesh_path=args.mesh,
        num_parts=args.num_parts,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
    )

    # Save Results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.tag is None:
        mesh_basename = os.path.splitext(os.path.basename(args.mesh))[0]
        args.tag = f"{mesh_basename}_{time.strftime('%Y%m%d_%H_%M_%S')}"

    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)

    # Save the input view used for generation (for debugging)
    processed_image.save(os.path.join(export_dir, "input_view.png"))

    for i, mesh in enumerate(outputs):
        mesh.export(os.path.join(export_dir, f"part_{i:02}.glb"))

    merged_mesh = get_colored_mesh_composition(outputs)
    merged_mesh.export(os.path.join(export_dir, "object.glb"))
    print(f"Generated {len(outputs)} parts and saved to {export_dir}")

    # Visualization
    if args.render:
        print("Start rendering...")
        num_views = 36
        radius = 4
        fps = 18

        # We explicitly pass image_size here to match your render_utils.py expectations
        rendered_images = render_views_around_mesh(
            merged_mesh, num_views=num_views, radius=radius, image_size=(512, 512)
        )
        rendered_normals = render_normal_views_around_mesh(
            merged_mesh, num_views=num_views, radius=radius, image_size=(512, 512)
        )
        rendered_grids = make_grid_for_images_or_videos(
            [
                [processed_image] * num_views,
                rendered_images,
                rendered_normals,
            ],
            nrow=3,
        )
        export_renderings(
            rendered_images,
            os.path.join(export_dir, "rendering.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_normals,
            os.path.join(export_dir, "rendering_normal.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_grids,
            os.path.join(export_dir, "rendering_grid.gif"),
            fps=fps,
        )

        rendered_image, rendered_normal, rendered_grid = (
            rendered_images[0],
            rendered_normals[0],
            rendered_grids[0],
        )
        rendered_image.save(os.path.join(export_dir, "rendering.png"))
        rendered_normal.save(os.path.join(export_dir, "rendering_normal.png"))
        rendered_grid.save(os.path.join(export_dir, "rendering_grid.png"))
        print("Rendering done.")
