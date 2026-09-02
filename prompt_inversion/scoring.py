def add_combined_score(df, clip_weight=0.50, lpips_weight=0.40, mse_weight=0.10):
    """
    Normalize CLIP/LPIPS/MSE per target and compute one combined score.
    Higher score = better prompt.
    """
    df = df.copy()

    def normalize_group(group):
        group = group.copy()

        def minmax(series):
            if series.max() == series.min():
                return series * 0.0 + 1.0  # If all same, return 1.0
            return (series - series.min()) / (series.max() - series.min() + 1e-8)

        group["clip_norm"] = minmax(group["clip_similarity"])
        group["lpips_norm"] = minmax(group["lpips"])
        group["mse_norm"] = minmax(group["mse"])

        group["score"] = (
            clip_weight * group["clip_norm"]
            + lpips_weight * (1.0 - group["lpips_norm"])
            + mse_weight * (1.0 - group["mse_norm"])
        )
        return group

    return df.groupby("target", group_keys=False).apply(normalize_group)
