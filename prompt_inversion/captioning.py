import gc
from dataclasses import dataclass

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from .targets import load_image

VLM_MODEL_ID = "Salesforce/blip-image-captioning-base"


@dataclass
class VLMCaptioner:
    processor: BlipProcessor
    model: BlipForConditionalGeneration
    device: str
    dtype: torch.dtype


def load_vlm_captioner(cache_dir, model_id=VLM_MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=cache_dir
    ).to(device)
    model.eval()

    return VLMCaptioner(processor=processor, model=model, device=device, dtype=dtype)


def generate_vlm_caption(captioner, image_path, max_new_tokens=60, num_beams=5):
    """Generate a short BLIP caption for one target image."""
    image = load_image(image_path)

    inputs = captioner.processor(images=image, return_tensors="pt")
    inputs = {key: value.to(captioner.device) for key, value in inputs.items()}
    if captioner.device == "cuda":
        inputs = {
            key: value.to(dtype=captioner.dtype) if torch.is_floating_point(value) else value
            for key, value in inputs.items()
        }

    with torch.no_grad():
        output_ids = captioner.model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
        )

    caption = captioner.processor.decode(output_ids[0], skip_special_tokens=True)
    return " ".join(caption.strip().split())


def unload_vlm_captioner(captioner):
    """Free the VLM before loading the (much larger) LCM generator."""
    del captioner.model
    del captioner.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
