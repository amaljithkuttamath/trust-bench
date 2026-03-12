import os
import tempfile

import matplotlib
import torch

matplotlib.use("Agg")

from trust_bench.models.base import FeatureActivations
from trust_bench.viz.bars import selectivity_bars, top_features_bar


class TestTopFeaturesBar:
    def test_generates_png(self):
        fa = FeatureActivations(
            tokens=["a", "b", "c"],
            features=torch.rand(3, 20),
            layer=16,
            model_name="test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "top_features.png")
            top_features_bar(fa, k=10, save_path=path)
            assert os.path.exists(path)


class TestSelectivityBars:
    def test_generates_png(self):
        target = torch.tensor([10.0, 8.0, 12.0])
        control = torch.tensor([0.5, 0.1, 0.3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "selectivity.png")
            selectivity_bars(
                feature_idx=42,
                target_acts=target,
                control_acts=control,
                labels=("Conjunction", "Control"),
                save_path=path,
            )
            assert os.path.exists(path)
