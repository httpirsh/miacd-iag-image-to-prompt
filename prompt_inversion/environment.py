import os
from pathlib import Path


def setup_cache_dir(workspace_root=None):
    """Point HuggingFace/torch caches at a local model_cache/ dir and create it."""
    workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
    cache_dir = workspace_root / "model_cache"
    cache_dir.mkdir(exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    import torch
    torch.hub.set_dir(str(cache_dir / "torch_hub"))

    return cache_dir
