import gc
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .io_utils import create_run_dir, write_csv
from .metrics import MetricModels, evaluate_image_pair
from .prompt_utils import unpack_prompt_entry
from .rendering import render_prompt_for_target, save_generated_image
from .scoring import add_combined_score
from .targets import load_image, safe_stem, seed_from_filename


@dataclass
class PipelineContext:
    """Loaded models + config shared across a run: the LCM pipe, metric models,
    the LCM tokenizer (used to token-limit refined prompts), and render config."""

    pipe: object
    config: object
    metric_models: MetricModels
    prompt_tokenizer: object = None


def evaluate_prompt_for_target(context, prompt_entry, target_path, run_dir, prompt_index, iteration=0, family="initial"):
    prompt, prompt_type = unpack_prompt_entry(prompt_entry)

    target_img = load_image(target_path)
    generated_img = render_prompt_for_target(context.pipe, prompt, target_path, context.config)
    size = (context.config.width, context.config.height)
    metrics = evaluate_image_pair(context.metric_models, target_img, generated_img, size)

    output_path = save_generated_image(generated_img, run_dir, target_path, prompt_index, prompt)

    row = {
        "target": Path(target_path).name,
        "target_key": safe_stem(target_path),
        "seed": seed_from_filename(target_path),
        "iteration": int(iteration),
        "family": family,
        "prompt_type": prompt_type,
        "prompt_index": int(prompt_index),
        "prompt": prompt,
        "generated_path": str(output_path),
        "clip_similarity": float(metrics["clip_similarity"]),
        "lpips": float(metrics["lpips"]),
        "mse": float(metrics["mse"]),
    }

    del generated_img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def prompts_for_target(candidate_prompts, target_path):
    """
    Accept either:
    - a list of prompt entries shared by all targets;
    - a dict mapping target_key or target filename to prompt entries.
    """
    if isinstance(candidate_prompts, list):
        return candidate_prompts

    target_key = safe_stem(target_path)
    target_name = Path(target_path).name

    if target_key in candidate_prompts:
        return candidate_prompts[target_key]
    if target_name in candidate_prompts:
        return candidate_prompts[target_name]

    raise KeyError(f"No prompts found for target {target_name} / {target_key}")


def run_evaluation(context, candidate_prompts, target_images, output_dir, identity="prompt_inversion", iteration=0, family="initial", verbose=True):
    """
    Render and evaluate all candidate prompts for all target images.
    Saves generated images, raw_results.csv/json, ranked_results.csv, top3_results.csv
    under a new timestamped run directory, and returns (ranked_df, top3_df, run_dir).
    """
    run_dir = create_run_dir(output_dir, identity=identity)
    rows = []

    for target_path in target_images:
        target_prompt_list = prompts_for_target(candidate_prompts, target_path)
        if verbose:
            print(f"\nEvaluating {Path(target_path).name}: {len(target_prompt_list)} prompts")

        for prompt_index, prompt_entry in enumerate(target_prompt_list, start=1):
            prompt, prompt_type = unpack_prompt_entry(prompt_entry)
            if verbose:
                print(f"  [{prompt_index:03d}/{len(target_prompt_list):03d}] ({prompt_type}) {prompt[:90]}")
            row = evaluate_prompt_for_target(
                context, prompt_entry, target_path, run_dir, prompt_index, iteration, family
            )
            rows.append(row)

    raw_df = pd.DataFrame(rows)
    ranked_df = add_combined_score(raw_df)
    ranked_df = ranked_df.sort_values(
        ["target", "score", "clip_similarity"], ascending=[True, False, False]
    ).reset_index(drop=True)

    top3_df = ranked_df.groupby("target", group_keys=False).head(3).reset_index(drop=True)

    write_csv(run_dir / "raw_results.csv", rows)
    ranked_df.to_csv(run_dir / "ranked_results.csv", index=False)
    top3_df.to_csv(run_dir / "top3_results.csv", index=False)
    (run_dir / "raw_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    if verbose:
        print("\nSaved results to:", run_dir)

    return ranked_df, top3_df, run_dir
