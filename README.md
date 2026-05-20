# TP2: Image-to-Prompt Inversion with Metric-Guided Search

**Student IDs**: 2021231014, 2021221971

## Project Overview

This project implements **VLM-initialized, metric-guided iterative prompt inversion with a global visual modifier bank** for image-to-prompt generation. The goal is to find text prompts that, when rendered with a fixed diffusion model and known seed, produce images visually similar to target images.

### Key Insight

This is not simple image captioning. A prompt is considered successful only if rendering it with the fixed `SimianLuo/LCM_Dreamshaper_v7` model using the encoded seed produces an image that closely matches the target.

## Pipeline Architecture

The optimization loop follows:

```
target image → VLM caption → structured visual analysis → modifier bank expansion 
→ deterministic LCM render → image-side metrics → ranking → refinement
```

## Implementation Details

### Core Components

- **Local image loading**: Reads target PNG images with seed information encoded in filenames
- **BLIP captioning**: Initial visual understanding using Vision-Language Model
- **Global modifier banks**: Style, lighting, camera, and background modifiers
- **LCM rendering**: Fast, deterministic image generation (8 steps, 768x768)
- **Multi-metric evaluation**: CLIP similarity, LPIPS perceptual loss, MSE pixel-level error
- **Iterative refinement**: Metric-guided candidate expansion and ranking
- **Batch processing**: Parallel evaluation across multiple target images

### Model Configuration

- **Model**: `SimianLuo/LCM_Dreamshaper_v7`
- **Inference steps**: 8
- **Guidance scale**: 8.0
- **LCM origin steps**: 50
- **Resolution**: 768×768
- **Seeds**: Extracted from target filename (e.g., `1159_25.png` → seed `1159`)

## Project Structure

```
IAGTP2_2021231014_2021221971/
├── README.md                           # This file
├── TP2_Project.ipynb                   # Main implementation notebook
├── tp2-chosen/                         # Target images
├── model_cache/                        # Cached models (CLIP, BLIP, LCM, etc.)
└── outputs/                            # Generated results
    ├── all_ranked_results.csv          # Aggregated rankings across all runs
    ├── final_prompts.csv               # Best prompts per target image
    ├── round_comparison.csv            # Metrics comparison between rounds
    └── [timestamp]_round[N]_[stage]/   # Individual run outputs
```

## Results

### Target Images Processed: 6
- `1159_25.png` - Glass with orange juice and fruit
- `1159_29.png` - Palm tree in ocean at sunset
- `1159_3.png` - Anime warrior with armor and sword
- `1159_7.png` - Hedgehog-like creature with spiky fur
- `7836.png` - Astronaut on alien planet
- `9338.png` - Fantasy creature with dragon scales

### Output Files

- **final_prompts.csv**: Best refined prompts for each target with metrics
  - CLIP similarity scores: 0.81–0.94
  - LPIPS (perceptual loss): 0.53–0.67
  - MSE (pixel-level): 0.026–0.065

- **Rankings & Analysis**: Per-image candidate rankings with full metric breakdowns

### Execution Log

Multiple optimization rounds completed:
- Round 0: Initial prompt expansion (low VRAM mode)
- Round 1: Metric-guided refinement

## Dependencies

- PyTorch with CUDA support
- Diffusers, Transformers (HuggingFace)
- BLIP for captioning
- CLIP for semantic similarity
- LPIPS for perceptual loss
- PIL, NumPy, Pandas, Matplotlib

## Notes

- Cached models stored locally in `model_cache/` for offline access
- GPU required for reasonable inference speed
- The notebook includes cleanup steps to manage VRAM efficiently
- Fully local execution (no Google Drive dependency required)
