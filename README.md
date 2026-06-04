# TP2: Image-to-Prompt Inversion with Metric-Guided Search

**Student IDs**: 2021231014, 2021221971

## Project Overview

This project implements **VLM-initialized, metric-guided iterative prompt inversion with a global visual modifier bank** for image-to-prompt generation. The goal is to find text prompts that, when rendered with a fixed diffusion model and known seed, produce images visually similar to target images.

### Key Insight

This is not simple image captioning. A prompt is considered successful only if rendering it with the fixed `SimianLuo/LCM_Dreamshaper_v7` model using the encoded seed produces an image that closely matches the target.

## Model Versions & Configuration

### Core Models

| Component | Model ID / Version | Purpose |
|-----------|-------------------|---------|
| **LCM (Image Generator)** | `SimianLuo/LCM_Dreamshaper_v7` | Deterministic image rendering from text prompts |
| **VLM (Caption Initialization)** | `Salesforce/blip-image-captioning-base` | Initial visual understanding via BLIP |
| **Semantic Similarity** | `openai/clip-vit-base-patch32` | Image-image semantic matching metric |
| **Perceptual Distance** | LPIPS (AlexNet backbone) | Deep feature-based visual similarity |
| **LLM (Prompt Refinement)** | `llama-3.1-8b-instant` (Groq API) | LLM-driven prompt optimization |

### Precision & Device Configuration

- **CUDA-enabled**: All models use `torch.float16` for GPU memory efficiency
- **CPU fallback**: `torch.float32` precision when CUDA unavailable
- **Memory optimization**: CPU offload enabled for LCM; metrics computed on CPU to preserve GPU for generation

## Hyperparameters

### LCM Rendering Configuration

```python
class LCMConfig:
    model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    num_inference_steps: int = 8              # Fast LCM inference
    guidance_scale: float = 8.0               # Classifier-free guidance strength
    lcm_origin_steps: int = 50                # Equivalent steps for guidance calculation
    width: int = 384                          # Generated image width (pixels)
    height: int = 384                         # Generated image height (pixels)
```

### VLM Caption Generation

- **Max new tokens**: 60 (limit to summary-length captions)
- **Num beams**: 5 (beam search for caption generation)
- **Model precision**: float16 on CUDA, float32 on CPU

### Prompt Processing

- **Max prompt tokens**: 75 (CLIP text encoder limit)
- **Tokenizer**: LCM model's tokenizer for token counting
- **Truncation**: Applied to all generated prompts to ensure CLIP compatibility

### Metric Scoring & Normalization

The combined score normalizes and weights three complementary metrics **per target**:

```python
score = (
    0.50 * CLIP_similarity_normalized +
    0.40 * (1.0 - LPIPS_normalized) +
    0.10 * (1.0 - MSE_normalized)
)
```

- **CLIP weight**: 0.50 (semantic content alignment)
- **LPIPS weight**: 0.40 (perceptual visual quality)
- **MSE weight**: 0.10 (pixel-level accuracy)
- **Normalization**: Min-max scaling per target to ensure equal metric influence

### Optimization Rounds

1. **Round 0 - Initial Expansion**: 10 candidates per target (VLM-initialized, structured analysis, global modifier combinations)
2. **Round 1 - Metric-Guided Refinement**: 10 mutations per top-3 candidate
3. **Round 2 - LLM Refinement**: 4 variants per top-3 candidate via Groq API

## Random Seeds & Deterministic Generation

### Render Seeds (Per Target Image)

Seeds are extracted from target filename prefixes:

| Target Image | Encoded Seed | Used For |
|--------------|-------------|----------|
| `1159_3.png` | 1159 | Anime warrior rendering |
| `1159_7.png` | 1159 | Hedgehog creature rendering |
| `1159_25.png` | 1159 | Orange juice still life rendering |
| `1159_29.png` | 1159 | Palm tree seascape rendering |
| `7836.png` | 7836 | Astronaut sci-fi rendering |
| `9338.png` | 9338 | Fantasy dragon hamster rendering |

### Stochastic Components & Seeding

#### 1. LCM Image Generation (Deterministic)
```python
generator = torch.Generator(device=device).manual_seed(seed)
image = pipe(
    prompt=prompt,
    generator=generator,
    ...
)
```
- **Seed source**: Extracted from target filename
- **Generator device**: CUDA if available, else CPU
- **Result**: Identical image for same prompt + seed across runs

#### 2. VLM Caption Generation (Partially Stochastic)
- **Beam search with num_beams=5**: Introduces some variability in caption selection
- **No explicit seed setting**: Uses PyTorch's default RNG (not seeded per caption)
- **Impact**: Minimal; captions are used only as initialization, later overridden by metric-driven expansion

#### 3. Prompt Expansion (Deterministic)
- All candidate generation is deterministic (combinatorial expansion from fixed modifier banks)
- No randomization in modifier selection; all combinations are generated
- Metric-driven ranking (not random) determines which modifiers are retained

#### 4. Metric Computation (Deterministic)
- CLIP similarity: Deterministic feature extraction and cosine similarity
- LPIPS: Deterministic feature extraction (frozen networks)
- MSE: Direct pixel comparison, fully deterministic

### Reproducibility

To reproduce exact results for a target image:

```python
# Render with known seed
seed = seed_from_filename("1159_3.png")  # seed = 1159
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator, ...)

# Metrics are deterministic
metrics = evaluate_image_pair(target_img, generated_img)
```

**Note**: VLM captions (`Salesforce/blip-image-captioning-base`) may vary slightly between runs due to beam search stochasticity, but this does not affect final results since VLM captions are only used as initialization.

## Setup

### Environment Configuration

To use the LLM-based refinement features, you need to set up a `.env` file in the root directory:

1. Create a file named `.env` in the project root.
2. Add the following line to it:
   ```env
   GROQ_API_KEY=SUA_API_KEY
   ```

### Obtaining a Groq API Key

1. Go to the [Groq Console](https://console.groq.com/).
2. Sign in or create an account.
3. Navigate to the **API Keys** section in the sidebar.
4. Click **Create API Key**.
5. Copy the generated key and paste it into your `.env` file.

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
- Round 2: LLM-based refinement (based on `llm_refinement_implementation_guide.pdf`)

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
