import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def call_llm(instruction, api_key=None):
    """Calls the Groq API (Llama 3.1 8B Instant) for prompt refinement."""
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")

    client = Groq(api_key=api_key)
    model_name = "llama-3.1-8b-instant"

    try:
        print(f"  > Attempting {model_name}...", end=" ")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": instruction}],
            model=model_name,
            response_format={"type": "json_object"},
        )
        print("✅")
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"⚠️ (Error: {str(e)[:50]})")
        raise e


def analysis_to_text(analysis):
    parts = []
    for key in ["subject", "composition", "background", "style", "lighting", "camera", "details", "vlm_caption"]:
        value = analysis.get(key, "")
        if isinstance(value, list):
            value = ", ".join(value)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def make_metric_feedback(row):
    return f"""
Current score: {row.get("score")}
CLIP similarity: {row.get("clip_similarity")} higher is better
LPIPS: {row.get("lpips")} lower is better
MSE: {row.get("mse")} lower is better
""".strip()


def build_refinement_instruction(target_key, analysis, row, n_variants=4):
    target_description = analysis_to_text(analysis)
    metric_feedback = make_metric_feedback(row)
    current_prompt = row["prompt"]
    return f"""
# Task: Image-to-Prompt Inversion Refinement
You are an expert prompt engineer. Your goal is to refine a text prompt to more accurately reconstruct a target image using a Latent Consistency Model (LCM).

## Target Image Description (Ground Truth):
{target_description}

## Current Prompt (The one that produced the metrics below):
"{current_prompt}"

## Current Performance Metrics:
{metric_feedback}

## Instructions:
1. Compare the 'Target Image Description' with the 'Current Prompt'.
2. Identify and REMOVE any stylistic modifiers in the Current Prompt that contradict the Target Description (e.g., if the target is a 'fantasy creature' but the prompt says 'realistic food photography', REMOVE the food photography part).
3. Use the 'Metric Feedback' to guide your changes:
   - If 'CLIP similarity' is low (< 0.7), the prompt is missing core subjects. Add precise nouns/verbs from the Target Description.
   - If 'LPIPS' or 'MSE' is high, the visual structure/colors are wrong. Adjust lighting, composition, and style keywords based ONLY on the Target Description.
4. Keep prompts focused. Avoid generic "quality" buzzwords (e.g., "highly detailed", "masterpiece") unless they are essential to the style in the Target Description.
5. Do NOT add any new elements (subjects, objects, colors) that are not in the Target Image Description.
6. CRITICAL: Each prompt MUST be under 60 words to fit within the 77-token limit of the CLIP text encoder.
7. Output EXACTLY {n_variants} improved prompt variants.

## Response Format:
Return a JSON object with a single key 'prompts' containing an array of strings.
Example: {{"prompts": ["refined prompt 1", "refined prompt 2", ...]}}
"""


def parse_llm_prompt_variants(response_text):
    try:
        # Find JSON block
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        json_str = match.group(0) if match else response_text
        data = json.loads(json_str)
        prompts = [p.strip() for p in data.get("prompts", []) if p.strip()]
        return prompts[:4]  # Limit to requested number
    except Exception as e:
        print(f"    [Parsing Error] {e}")
        return []


def build_llm_refined_bank(top_df, target_analysis, api_key=None, iteration=1, top_k_per_target=3, variants_per_prompt=4):
    """Ask the LLM to refine each target's top-k prompts, using the metrics as feedback."""
    refined_bank = {}
    for target, group in top_df.groupby("target"):
        target_key = str(target).replace(".png", "").replace(".jpg", "")
        refined_bank[target_key] = []

        best_prompts = group.sort_values("score", ascending=False).head(top_k_per_target)
        print(f"  > Refining {target_key} (using top {len(best_prompts)} prompts)...")

        for _, row in best_prompts.iterrows():
            instruction = build_refinement_instruction(target_key, target_analysis.get(target_key, {}), row, variants_per_prompt)
            try:
                response = call_llm(instruction, api_key)
                new_prompts = parse_llm_prompt_variants(response)
                for p in new_prompts:
                    refined_bank[target_key].append({
                        "prompt": p,
                        "prompt_type": f"groq_llama_3.1_refinement_r{iteration}_from_{row.get('prompt_type', 'unknown')}",
                    })
            except Exception as e:
                print(f"    Error on {target_key}: {e}")
    return refined_bank
