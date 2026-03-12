import os
import tempfile

import matplotlib
import torch

matplotlib.use("Agg")

from trust_bench.models.base import FeatureActivations
from trust_bench.viz.heatmaps import token_feature_heatmap


class TestTokenFeatureHeatmap:
    def test_generates_png(self):
        fa = FeatureActivations(
            tokens=["The", " cat", " sat"],
            features=torch.rand(3, 10),
            layer=16,
            model_name="test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "heatmap.png")
            token_feature_heatmap(fa, features_to_show=[0, 1, 2], save_path=path)
            assert os.path.exists(path)

    def test_generates_svg(self):
        fa = FeatureActivations(
            tokens=["The", " cat", " sat"],
            features=torch.rand(3, 10),
            layer=16,
            model_name="test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "heatmap.svg")
            token_feature_heatmap(fa, features_to_show=[0, 1, 2], save_path=path)
            assert os.path.exists(path)
