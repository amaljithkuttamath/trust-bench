"""Feature analysis functions."""

from trust_bench.models.base import FeatureActivations


def selectivity_score(
    fa: FeatureActivations,
    feature_idx: int,
    target_indices: list[int],
    control_indices: list[int],
) -> float:
    """How selective is a feature for target vs control tokens? Returns 0-1."""
    acts = fa.features[:, feature_idx]
    target_mean = acts[target_indices].abs().mean().item()
    control_mean = acts[control_indices].abs().mean().item()

    total = target_mean + control_mean
    if total == 0:
        return 0.0
    return (target_mean - control_mean) / total


def sparsity_profile(fa: FeatureActivations) -> dict:
    """L0 sparsity stats for the feature activations."""
    n_tokens, n_features = fa.features.shape
    active = (fa.features.abs() > 0).float()

    l0_mean = (active.sum(dim=0) > 0).float().mean().item()

    active_fraction = (active.sum(dim=0) / n_tokens).tolist()

    return {
        "l0_mean": l0_mean,
        "active_fraction": active_fraction,
        "n_features": n_features,
        "n_tokens": n_tokens,
    }


def top_features_by_category(
    activations: dict[str, FeatureActivations],
    k: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """For each category, which features have the highest mean activation?"""
    result = {}
    for category, fa in activations.items():
        mean_acts = fa.features.abs().mean(dim=0)
        topk_vals, topk_idx = mean_acts.topk(min(k, len(mean_acts)))
        result[category] = [
            (idx.item(), val.item()) for idx, val in zip(topk_idx, topk_vals) if val.item() > 0
        ]
    return result
