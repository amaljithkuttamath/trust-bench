"""Token x feature activation heatmaps."""

import matplotlib.pyplot as plt
import seaborn as sns

from trust_bench.models.base import FeatureActivations
from trust_bench.viz.style import HEATMAP_CMAP, SINGLE_COL, apply_style, save_figure


def token_feature_heatmap(
    fa: FeatureActivations,
    features_to_show: list[int],
    save_path: str | None = None,
    title: str | None = None,
):
    """Heatmap showing which features fire on which tokens."""
    apply_style()

    data = fa.features[:, features_to_show].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    sns.heatmap(
        data.T,
        xticklabels=fa.tokens,
        yticklabels=[f"#{i}" for i in features_to_show],
        cmap=HEATMAP_CMAP,
        ax=ax,
    )
    ax.set_xlabel("Token")
    ax.set_ylabel("Feature")
    ax.set_title(title or f"Feature activations, layer {fa.layer}")

    if save_path:
        save_figure(fig, save_path)
    return fig
