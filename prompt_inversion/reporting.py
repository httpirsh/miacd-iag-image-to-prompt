import shutil
from pathlib import Path

import pandas as pd

from .scoring import add_combined_score


def build_round_comparison(round_dfs):
    """
    round_dfs: dict of {round_name: results_df}, e.g.
    {"initial": results_df, "refined": refined_results_df, "llm": llm_results_df}.

    Returns one row per target with each round's best prompt/metrics side by
    side, plus the overall best_score and best_round.
    """
    combined = pd.concat(list(round_dfs.values()), ignore_index=True)
    combined = add_combined_score(combined)

    comparison_df = None
    for round_name, round_df in round_dfs.items():
        mask = combined["family"].isin(round_df["family"].unique())
        best = (
            combined[mask]
            .sort_values(["target", "score", "clip_similarity"], ascending=[True, False, False])
            .groupby("target", as_index=False)
            .head(1)
            [["target", "prompt_type", "prompt", "clip_similarity", "lpips", "mse", "score"]]
            .rename(columns={
                "prompt_type": f"{round_name}_prompt_type",
                "prompt": f"{round_name}_best_prompt",
                "clip_similarity": f"{round_name}_clip",
                "lpips": f"{round_name}_lpips",
                "mse": f"{round_name}_mse",
                "score": f"{round_name}_score",
            })
        )
        comparison_df = best if comparison_df is None else comparison_df.merge(best, on="target")

    score_cols = [f"{name}_score" for name in round_dfs]
    comparison_df["best_score"] = comparison_df[score_cols].max(axis=1)
    comparison_df["best_round"] = comparison_df[score_cols].idxmax(axis=1).str.replace("_score", "")

    return comparison_df


def export_final_results(round_dfs, output_dir):
    """Re-normalize scores across all rounds and export final CSVs."""
    all_results_df = pd.concat(list(round_dfs.values()), ignore_index=True)
    all_results_df = add_combined_score(all_results_df)

    ranked = all_results_df.sort_values(
        ["target", "score", "clip_similarity"], ascending=[True, False, False]
    )

    final_top3_df = ranked.groupby("target", as_index=False).head(3).reset_index(drop=True)
    final_df = ranked.groupby("target", as_index=False).head(1).reset_index(drop=True)

    output_dir = Path(output_dir)
    final_df.to_csv(output_dir / "final_prompts.csv", index=False)
    final_top3_df.to_csv(output_dir / "final_top3_prompts.csv", index=False)
    all_results_df.to_csv(output_dir / "all_ranked_results.csv", index=False)

    return final_df, final_top3_df, all_results_df


def build_gallery(final_top3_df, target_images, gallery_dir):
    """Copy each target's top-3 renders + target image + prompts.txt into gallery_dir/<target>/."""
    gallery_dir = Path(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    for target_name in final_top3_df["target"].unique():
        target_gallery = gallery_dir / Path(target_name).stem
        target_gallery.mkdir(parents=True, exist_ok=True)

        target_src = next(p for p in target_images if Path(p).name == target_name)
        shutil.copy2(target_src, target_gallery / f"00_target_{target_name}")

        rows = final_top3_df[final_top3_df["target"] == target_name].reset_index(drop=True)
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            src = row["generated_path"]
            ext = Path(src).suffix
            dst = target_gallery / f"top{rank}_score{row['score']:.4f}{ext}"
            shutil.copy2(src, dst)

        with open(target_gallery / "prompts.txt", "w") as f:
            f.write(f"Target: {target_name}\n")
            f.write("=" * 60 + "\n\n")
            for rank, (_, row) in enumerate(rows.iterrows(), start=1):
                f.write(f"Top-{rank}  (score={row['score']:.4f}, family={row['family']}, type={row['prompt_type']})\n")
                f.write(f"CLIP={row['clip_similarity']:.4f}  LPIPS={row['lpips']:.4f}  MSE={row['mse']:.5f}\n")
                f.write(f"Prompt: {row['prompt']}\n\n")

    return gallery_dir


def print_summary_table(final_top3_df):
    summary_cols = ["target", "family", "prompt_type", "clip_similarity", "lpips", "mse", "score"]
    summary_df = final_top3_df[summary_cols].copy()
    summary_df.columns = ["Target", "Round", "Prompt Type", "CLIP", "LPIPS", "MSE", "Score"]

    summary_df["Rank"] = summary_df.groupby("Target").cumcount() + 1
    summary_df = summary_df[["Target", "Rank", "Round", "Prompt Type", "CLIP", "LPIPS", "MSE", "Score"]]

    n_targets = summary_df["Target"].nunique()
    max_rank = summary_df["Rank"].max()

    print("=" * 90)
    print("PLAIN-TEXT SUMMARY: Top-3 Candidates per Target")
    print("=" * 90)

    with pd.option_context(
        "display.max_rows", 20, "display.max_columns", 10, "display.width", 140,
        "display.float_format", "{:.4f}".format, "display.max_colwidth", 60,
    ):
        print(summary_df.to_string(index=False))

    print(f"\nTotal rows: {len(summary_df)} ({n_targets} targets x {max_rank} candidates)")

    top1_df = summary_df[summary_df["Rank"] == 1]
    metric_cols = ["CLIP", "LPIPS", "MSE", "Score"]

    means = top1_df[metric_cols].mean()
    stds = top1_df[metric_cols].std()

    print("\n" + "=" * 90)
    print("TEST SET METRICS (Best candidate per target): Mean +/- Std")
    print("=" * 90)
    for col in metric_cols:
        print(f"  {col:>8s}:  {means[col]:.4f} +/- {stds[col]:.4f}")
    print("=" * 90)

    return summary_df
