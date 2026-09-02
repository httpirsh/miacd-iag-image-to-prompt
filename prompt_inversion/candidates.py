import pandas as pd

from .modifiers import LIGHTING_MODIFIERS, REFINEMENT_REPLACEMENTS, REFINEMENT_SUFFIXES, STYLE_MODIFIERS
from .prompt_utils import join_prompt_parts, make_entry, unique_prompt_entries, unpack_prompt_entry
from .targets import safe_stem


def make_compact_variants(analysis, max_prompts=10):
    subject = analysis.get("subject", "")
    composition = analysis.get("composition", "")
    background = analysis.get("background", "")
    style = analysis.get("style", "")
    lighting = analysis.get("lighting", "")
    camera = analysis.get("camera", "")
    details = analysis.get("details", "")
    vlm_caption = analysis.get("vlm_caption", "")

    if isinstance(details, list):
        details_text = ", ".join(details[:4])
    else:
        details_text = str(details)

    main_subject = subject or vlm_caption

    variants = []

    def add(prompt_type, parts):
        parts = [p.strip(" ,.") for p in parts if isinstance(p, str) and p.strip()]
        prompt = ", ".join(parts)

        if prompt:
            variants.append({"prompt": prompt, "prompt_type": prompt_type})

    add("vlm_only", [vlm_caption])
    add("subject_only", [main_subject])
    add("subject_style", [main_subject, style])
    add("subject_composition", [main_subject, composition, style])
    add("subject_lighting", [main_subject, lighting, style])
    add("subject_background", [main_subject, background, style])
    add("subject_camera", [main_subject, camera, composition, style])
    add("detail_focus", [main_subject, details_text, style])
    add("compact_full", [main_subject, composition, background, lighting, style])
    add("photo_realism", [main_subject, "realistic photography", lighting, camera])

    # Remove duplicate prompts while preserving order.
    seen = set()
    unique_variants = []
    for entry in variants:
        key = entry["prompt"].lower()
        if key not in seen:
            seen.add(key)
            unique_variants.append(entry)

    return unique_variants[:max_prompts]


def build_candidate_bank(target_analysis, target_images, max_prompts_per_target=10):
    """
    Convert a per-target visual analysis dict into:
    {target_stem: [{"prompt": ..., "prompt_type": ...}, ...]}
    """
    candidate_bank = {}

    for target_path in target_images:
        key = safe_stem(target_path)

        if key not in target_analysis:
            raise KeyError(f"Missing analysis for target: {key}")

        candidate_bank[key] = make_compact_variants(
            target_analysis[key], max_prompts=max_prompts_per_target
        )

    return candidate_bank


def candidate_bank_to_dataframe(candidate_bank):
    rows = []
    for target, entries in candidate_bank.items():
        for i, entry in enumerate(entries, start=1):
            prompt, prompt_type = unpack_prompt_entry(entry)
            rows.append({
                "target_key": target,
                "candidate_id": i,
                "prompt_type": prompt_type,
                "prompt": prompt,
            })
    return pd.DataFrame(rows)


def preview_candidate_bank(candidate_bank, n=5):
    for key, entries in candidate_bank.items():
        print(f"\n--- {key}: {len(entries)} prompts ---")
        for i, entry in enumerate(entries[:n], start=1):
            if isinstance(entry, str):
                print(f"[{i:02d}] [unspecified] {entry}")
            else:
                print(f"[{i:02d}] [{entry.get('prompt_type', 'unspecified')}] {entry.get('prompt', '')}")


def mutate_prompt(prompt, tokenizer=None):
    """Metric-guided mutations of a single prompt: suffixes, replacements, modifier additions."""
    mutations = [make_entry(prompt, "refined_original", tokenizer)]

    for suffix in REFINEMENT_SUFFIXES:
        mutations.append(make_entry(join_prompt_parts(prompt, suffix), "refined_suffix", tokenizer))

    for old, new in REFINEMENT_REPLACEMENTS:
        if old in prompt:
            mutations.append(make_entry(prompt.replace(old, new), "refined_replacement", tokenizer))

    for style in STYLE_MODIFIERS[:6]:
        mutations.append(make_entry(join_prompt_parts(prompt, style), "refined_global_style", tokenizer))

    for light in LIGHTING_MODIFIERS[:4]:
        mutations.append(make_entry(join_prompt_parts(prompt, light), "refined_global_lighting", tokenizer))

    parts = [part.strip() for part in prompt.split(",") if part.strip()]
    if len(parts) > 5:
        mutations.append(make_entry(", ".join(parts[:5]), "refined_compact", tokenizer))

    return unique_prompt_entries(mutations)


def build_metric_guided_bank(top_df, top_k, max_prompts_per_target, tokenizer=None):
    """Mutate each target's top-k prompts (by score) into a fresh bank of candidates."""
    refined_bank = {}

    for target_name in top_df["target"].unique():
        rows = top_df[top_df["target"] == target_name].head(top_k)
        target_key = safe_stem(target_name)

        mutations_per_prompt = [mutate_prompt(p, tokenizer) for p in rows["prompt"].tolist()]

        prompts = []
        max_len = max((len(m) for m in mutations_per_prompt), default=0)
        for i in range(max_len):
            for mutations in mutations_per_prompt:
                if i < len(mutations):
                    prompts.append(mutations[i])

        refined_bank[target_key] = unique_prompt_entries(prompts)[:max_prompts_per_target]

    return refined_bank
