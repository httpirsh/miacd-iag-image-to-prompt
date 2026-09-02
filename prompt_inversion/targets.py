import re
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_target_images(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_target_dir(workspace_root, dir_candidates, zip_candidates):
    """Find (or extract from a zip) the directory holding target images."""
    if not any(list_target_images(candidate) for candidate in dir_candidates):
        for zip_path in zip_candidates:
            if zip_path.exists():
                extract_dir = workspace_root / "tp2-chosen"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
                break

    for candidate in dir_candidates:
        if list_target_images(candidate):
            return candidate

    raise FileNotFoundError(
        "No target images found. Put images in tp2-chosen/ directory, "
        "or place tp2-chosen.zip in the workspace root to auto-extract."
    )


def seed_from_filename(path, fallback=2026):
    match = re.match(r"^(\d+)", Path(path).stem)
    return int(match.group(1)) if match else fallback


def safe_stem(path):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(path).stem)


def load_image(path):
    return Image.open(path).convert("RGB")
