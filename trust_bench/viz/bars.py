"""Bar chart visualizations."""

import matplotlib.pyplot as plt
import numpy as np

from trust_bench.models.base import FeatureActivations
from trust_bench.viz.style import SINGLE_COL, apply_style, save_figure


def top_features_bar(
    fa: FeatureActivations,
    k: int = 20,
    save_path: str | None = None,
    title: str | None = None,
):
    """Bar chart of top-k features by mean activation."""
    apply_style()

    top = fa.top_features(k=k)
    if not top:
        return None

    indices = [str(idx) for idx, _ in top]
    values = [val for _, val in top]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.barh(indices, values)
    ax.set_xlabel("Max activation")
    ax.set_ylabel("Feature #")
    ax.set_title(title or f"Top {len(top)} features, layer {fa.layer}")
    ax.invert_yaxis()

    if save_path:
        save_figure(fig, save_path)
    return fig


def selectivity_bars(
    feature_idx: int,
    target_acts,
    control_acts,
    labels: tuple[str, str] = ("Target", "Control"),
    save_path: str | None = None,
    title: str | None = None,
):
    """Side-by-side bars comparing target vs control activation."""
    apply_style()

    target_mean = (
        float(target_acts.mean()) if hasattr(target_acts, "mean") else np.mean(target_acts)
    )
    control_mean = (
        float(control_acts.mean()) if hasattr(control_acts, "mean") else np.mean(control_acts)
    )

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.bar(labels, [target_mean, control_mean], color=["#2196F3", "#FF5722"])
    ax.set_ylabel("Mean activation")
    ax.set_title(title or f"Feature #{feature_idx} selectivity")

    if save_path:
        save_figure(fig, save_path)
    return fig
