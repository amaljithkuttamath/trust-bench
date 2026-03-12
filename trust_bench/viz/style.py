"""Shared visualization style configuration."""

import matplotlib.pyplot as plt
import seaborn as sns

DPI = 300
SINGLE_COL = (6, 4)
DOUBLE_COL = (10, 4)
HEATMAP_CMAP = "viridis"
CATEGORICAL_PALETTE = "tab10"


def apply_style():
    """Apply Trust Bench plot style."""
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


def save_figure(fig, path: str):
    """Save figure as PNG and SVG."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    svg_path = path.rsplit(".", 1)[0] + ".svg"
    if not path.endswith(".svg"):
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
