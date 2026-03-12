import pytest
import torch

from trust_bench.analysis.statistics import (
    confidence_interval,
    consistency_score,
    effect_size,
    mann_whitney_test,
)


class TestMannWhitneyTest:
    def test_significant_difference(self):
        group_a = torch.tensor([10.0, 12.0, 11.0, 13.0, 10.5])
        group_b = torch.tensor([1.0, 2.0, 1.5, 0.5, 2.5])
        result = mann_whitney_test(group_a, group_b)
        assert result["p_value"] < 0.05
        assert "statistic" in result

    def test_no_difference(self):
        group_a = torch.tensor([5.0, 5.1, 4.9, 5.0, 5.2])
        group_b = torch.tensor([5.0, 4.9, 5.1, 5.0, 4.8])
        result = mann_whitney_test(group_a, group_b)
        assert result["p_value"] > 0.05


class TestEffectSize:
    def test_large_effect(self):
        group_a = torch.tensor([10.0, 11.0, 12.0, 10.0])
        group_b = torch.tensor([1.0, 2.0, 1.0, 2.0])
        d = effect_size(group_a, group_b)
        assert abs(d) > 0.8

    def test_zero_effect(self):
        group_a = torch.tensor([5.0, 5.0, 5.0])
        group_b = torch.tensor([5.0, 5.0, 5.0])
        d = effect_size(group_a, group_b)
        assert d == pytest.approx(0.0, abs=0.01)


class TestConfidenceInterval:
    def test_returns_ci(self):
        acts = torch.tensor([10.0, 11.0, 9.0, 10.5, 10.0])
        ci = confidence_interval(acts)
        assert "mean" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] < ci["mean"] < ci["upper"]


class TestConsistencyScore:
    def test_perfect_agreement(self):
        a = {1: 10.0, 2: 8.0, 3: 5.0}
        b = {1: 10.0, 2: 8.0, 3: 5.0}
        result = consistency_score(a, b)
        assert result["jaccard"] == 1.0
        assert result["rank_correlation"] == 1.0
        assert result["n_shared"] == 3

    def test_no_overlap(self):
        a = {1: 10.0, 2: 8.0}
        b = {3: 5.0, 4: 3.0}
        result = consistency_score(a, b)
        assert result["jaccard"] == 0.0
        assert result["n_shared"] == 0

    def test_partial_overlap(self):
        a = {1: 10.0, 2: 8.0, 3: 5.0}
        b = {2: 7.0, 3: 6.0, 4: 3.0}
        result = consistency_score(a, b)
        assert result["jaccard"] == pytest.approx(2 / 4)
        assert result["n_shared"] == 2
