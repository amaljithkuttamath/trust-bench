import pytest
import torch

from trust_bench.models.base import (
    ConfigError,
    FeatureActivations,
    ModelLoadError,
    ProbeError,
    ProbeResult,
    ResultMetadata,
    TokenizedInput,
    TrustBenchError,
)
from trust_bench.models.registry import REGISTRY, get_model


class TestTokenizedInput:
    def test_creation(self):
        ti = TokenizedInput(ids=torch.tensor([1, 2, 3]), strings=["a", "b", "c"])
        assert ti.ids.shape == (3,)
        assert ti.strings == ["a", "b", "c"]


class TestFeatureActivations:
    def setup_method(self):
        self.features = torch.zeros(4, 10)
        self.features[0, 3] = 5.0
        self.features[1, 3] = 8.0
        self.features[2, 7] = 12.0
        self.features[3, 1] = 3.0
        self.fa = FeatureActivations(
            tokens=["The", " cat", " sat", " down"],
            features=self.features,
            layer=16,
            model_name="test-model",
        )

    def test_top_features_returns_by_max_activation(self):
        top = self.fa.top_features(k=3)
        assert len(top) == 3
        assert top[0] == (7, 12.0)
        assert top[1] == (3, 8.0)
        assert top[2] == (1, 3.0)

    def test_tokens_for_feature(self):
        result = self.fa.tokens_for_feature(3, threshold=0.0)
        assert len(result) == 2
        assert (" cat", 8.0) in result
        assert ("The", 5.0) in result

    def test_tokens_for_feature_with_threshold(self):
        result = self.fa.tokens_for_feature(3, threshold=6.0)
        assert len(result) == 1
        assert result[0] == (" cat", 8.0)

    def test_feature_at_token(self):
        result = self.fa.feature_at_token(2, threshold=0.0)
        assert len(result) == 1
        assert result[0] == (7, 12.0)

    def test_top_features_k_larger_than_active(self):
        top = self.fa.top_features(k=5)
        assert len(top) == 3


class TestResultMetadata:
    def test_creation(self):
        meta = ResultMetadata(
            timestamp="2026-03-12T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="abc1234",
            device="cpu",
            duration_seconds=1.5,
        )
        assert meta.canary == "canary-trust-bench-2026"
        assert meta.git_hash == "abc1234"


class TestProbeResult:
    def test_creation(self):
        meta = ResultMetadata(
            timestamp="2026-03-12T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="abc1234",
            device="cpu",
            duration_seconds=1.5,
        )
        pr = ProbeResult(
            schema_version="1.0",
            probe_name="test",
            model_name="test-model",
            config={"probe": "test"},
            data={"features": [1, 2, 3]},
            result_metadata=meta,
        )
        assert pr.schema_version == "1.0"
        assert pr.probe_name == "test"
        assert pr.result_metadata.canary == "canary-trust-bench-2026"


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(ModelLoadError, TrustBenchError)
        assert issubclass(ConfigError, TrustBenchError)
        assert issubclass(ProbeError, TrustBenchError)

    def test_raise_model_load_error(self):
        with pytest.raises(ModelLoadError, match="Failed"):
            raise ModelLoadError("Failed to load model")


class TestModelRegistry:
    def test_registry_has_llama(self):
        assert "llama-3.1-8b" in REGISTRY

    def test_get_model_returns_backend(self):
        model = get_model("llama-3.1-8b")
        assert model.name == "llama-3.1-8b"
        assert model.d_model == 4096
        assert model.n_layers == 32

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent-model")
