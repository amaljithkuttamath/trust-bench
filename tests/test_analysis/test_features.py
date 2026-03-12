import pytest
import torch

from trust_bench.analysis.features import (
    selectivity_score,
    sparsity_profile,
    top_features_by_category,
)
from trust_bench.models.base import FeatureActivations


class TestSelectivityScore:
    def test_perfect_selectivity(self):
        fa = FeatureActivations(
            tokens=["a", "b", "c", "d"],
            features=torch.tensor(
                [
                    [10.0, 0.0],
                    [12.0, 0.0],
                    [0.0, 5.0],
                    [0.0, 3.0],
                ]
            ),
            layer=16,
            model_name="test",
        )
        score = selectivity_score(fa, feature_idx=0, target_indices=[0, 1], control_indices=[2, 3])
        assert score == 1.0

    def test_no_selectivity(self):
        fa = FeatureActivations(
            tokens=["a", "b", "c", "d"],
            features=torch.tensor(
                [
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                ]
            ),
            layer=16,
            model_name="test",
        )
        score = selectivity_score(fa, feature_idx=0, target_indices=[0, 1], control_indices=[2, 3])
        assert score == 0.0


class TestSparsityProfile:
    def test_sparsity(self):
        fa = FeatureActivations(
            tokens=["a", "b", "c", "d"],
            features=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            layer=16,
            model_name="test",
        )
        profile = sparsity_profile(fa)
        assert profile["l0_mean"] == pytest.approx(1 / 3, abs=0.01)
        assert profile["active_fraction"][0] == 0.5
        assert profile["active_fraction"][1] == 0.0


class TestTopFeaturesByCategory:
    def test_category_specific(self):
        activations = {
            "english": FeatureActivations(
                tokens=["The", "cat"],
                features=torch.tensor([[10.0, 1.0], [0.0, 0.0]]),
                layer=16,
                model_name="test",
            ),
            "code": FeatureActivations(
                tokens=["def", "foo"],
                features=torch.tensor([[0.0, 0.0], [0.0, 15.0]]),
                layer=16,
                model_name="test",
            ),
        }
        result = top_features_by_category(activations, k=1)
        assert result["english"][0][0] == 0
        assert result["code"][0][0] == 1
