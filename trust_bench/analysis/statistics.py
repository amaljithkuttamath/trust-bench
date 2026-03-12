"""Statistical analysis functions for activation comparisons."""

import math

import torch
from scipy import stats as scipy_stats


def mann_whitney_test(group_a: torch.Tensor, group_b: torch.Tensor) -> dict:
    """Mann-Whitney U test for significance between two groups."""
    a = group_a.detach().cpu().numpy()
    b = group_b.detach().cpu().numpy()
    stat, p_value = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p_value)}


def effect_size(group_a: torch.Tensor, group_b: torch.Tensor) -> float:
    """Cohen's d effect size."""
    a = group_a.float()
    b = group_b.float()
    mean_diff = a.mean() - b.mean()
    pooled_std = math.sqrt((a.var().item() + b.var().item()) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_diff / pooled_std).item()


def confidence_interval(activations: torch.Tensor, alpha: float = 0.05) -> dict:
    """Confidence interval for mean activation."""
    acts = activations.float()
    n = len(acts)
    mean = acts.mean().item()
    std = acts.std().item()
    se = std / math.sqrt(n) if n > 0 else 0
    t_val = scipy_stats.t.ppf(1 - alpha / 2, df=max(n - 1, 1))
    margin = t_val * se
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin, "alpha": alpha}


def consistency_score(
    results_a: dict[int, float],
    results_b: dict[int, float],
) -> dict:
    """Measure consistency between two sets of feature findings."""
    shared = set(results_a.keys()) & set(results_b.keys())
    all_features = set(results_a.keys()) | set(results_b.keys())

    if not all_features:
        return {"jaccard": 0.0, "rank_correlation": 0.0, "n_shared": 0}

    jaccard = len(shared) / len(all_features) if all_features else 0.0

    if len(shared) >= 3:
        vals_a = torch.tensor([results_a[f] for f in shared])
        vals_b = torch.tensor([results_b[f] for f in shared])
        corr = scipy_stats.spearmanr(vals_a.numpy(), vals_b.numpy())
        rank_corr = float(corr.statistic)
    else:
        rank_corr = float("nan")

    return {
        "jaccard": round(jaccard, 3),
        "rank_correlation": round(rank_corr, 3) if not math.isnan(rank_corr) else None,
        "n_shared": len(shared),
        "n_only_a": len(set(results_a.keys()) - shared),
        "n_only_b": len(set(results_b.keys()) - shared),
    }
