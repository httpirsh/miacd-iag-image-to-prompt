import gc
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, LCMScheduler

from .targets import safe_stem, seed_from_filename


def load_lcm_pipeline(config, cache_dir):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=cache_dir,
    )

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Low-memory options
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if torch.cuda.is_available():
        # Important: do not call pipe.to("cuda") on a small GPU.
        # CPU offload keeps only the active parts on GPU.
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    return pipe


def render_prompt(pipe, prompt, seed, config):
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            lcm_origin_steps=config.lcm_origin_steps,
            width=config.width,
            height=config.height,
            output_type="pil",
            generator=generator,
        ).images[0]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image


def render_prompt_for_target(pipe, prompt, target_path, config):
    seed = seed_from_filename(target_path)
    return render_prompt(pipe, prompt, seed=seed, config=config)


def save_generated_image(image, run_dir, target_path, prompt_index=1, prompt=None):
    """Save a generated image (and its prompt, if given) under run_dir/<target_stem>/."""
    target_dir = Path(run_dir) / safe_stem(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"candidate_{prompt_index:03d}.png"
    image.save(path)

    if prompt is not None:
        meta_path = target_dir / f"candidate_{prompt_index:03d}_prompt.txt"
        meta_path.write_text(prompt)

    return path
