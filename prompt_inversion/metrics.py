from dataclasses import dataclass

import lpips
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@dataclass
class MetricModels:
    clip_processor: CLIPProcessor
    clip_model: CLIPModel
    lpips_model: lpips.LPIPS
    device: str = "cpu"


def load_metric_models(cache_dir, device="cpu"):
    """CLIP + LPIPS default to CPU to save GPU memory for the LCM generator."""
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir).to(device)
    clip_model.eval()

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    return MetricModels(clip_processor=clip_processor, clip_model=clip_model, lpips_model=lpips_model, device=device)


def clip_image_similarity(models, img_a, img_b):
    inputs = models.clip_processor(images=[img_a, img_b], return_tensors="pt").to(models.device)

    with torch.no_grad():
        outputs = models.clip_model.vision_model(pixel_values=inputs["pixel_values"])
        features = models.clip_model.visual_projection(outputs.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)

    return torch.matmul(features[0], features[1]).item()


def _image_to_lpips_tensor(img, size, device):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0).to(device)


def lpips_distance(models, img_a, img_b, size):
    """Perceptual distance; lower means more similar."""
    tensor_a = _image_to_lpips_tensor(img_a, size, models.device)
    tensor_b = _image_to_lpips_tensor(img_b, size, models.device)

    with torch.no_grad():
        distance = models.lpips_model(tensor_a, tensor_b).item()

    return distance


def pixel_mse(img_a, img_b, size):
    """Pixel-level error; lower means more similar."""
    img_a = img_a.resize(size)
    img_b = img_b.resize(size)

    arr_a = np.asarray(img_a).astype(np.float32) / 255.0
    arr_b = np.asarray(img_b).astype(np.float32) / 255.0

    return np.mean((arr_a - arr_b) ** 2)


def evaluate_image_pair(models, target_img, generated_img, size):
    return {
        "clip_similarity": clip_image_similarity(models, target_img, generated_img),
        "lpips": lpips_distance(models, target_img, generated_img, size),
        "mse": pixel_mse(target_img, generated_img, size),
    }
