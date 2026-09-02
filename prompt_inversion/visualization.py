from pathlib import Path

import matplotlib.pyplot as plt

from .targets import load_image


def show_images(paths, cols=3, title=None):
    paths = list(paths)
    if not paths:
        print("No images to show.")
        return
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for ax in [ax for row in axes for ax in row]:
        ax.axis("off")
    for ax, path in zip([ax for row in axes for ax in row], paths):
        ax.imshow(load_image(path))
        ax.set_title(Path(path).name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_topk(top_df, target_images, k=3, show_prompts=True):
    for target_name in top_df["target"].unique():
        target_path = next(p for p in target_images if Path(p).name == target_name)
        target_img = load_image(target_path)
        rows = top_df[top_df["target"] == target_name].head(k).reset_index(drop=True)

        fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4))

        axes[0].imshow(target_img)
        axes[0].set_title(f"Target\n{target_name}")
        axes[0].axis("off")

        for i, row in rows.iterrows():
            img = load_image(row["generated_path"])
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(
                f"#{i+1} | Score {row['score']:.3f}\n"
                f"CLIP {row['clip_similarity']:.3f} | LPIPS {row['lpips']:.3f} | MSE {row['mse']:.4f}"
            )
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

        if show_prompts:
            print(f"\n--- Prompts for {target_name} ---")
            for i, row in rows.iterrows():
                print(f"  #{i+1} (score={row['score']:.4f}, family={row['family']}):")
                print(f"    {row['prompt']}")
            print()
