import pytest
import torch

from trust_bench.models.base import (
    ConfigError,
    FeatureActivations,
    ModelBackend,
    ProbeResult,
    SAEWrapper,
    TokenizedInput,
)
from trust_bench.probes.feature_survey import FeatureSurveyProbe


class MockSAE(SAEWrapper):
    n_features = 100

    def encode(self, activations):
        batch_size = activations.shape[0]
        features = torch.zeros(batch_size, self.n_features)
        features[:, 3] = 5.0
        if batch_size > 1:
            features[1, 42] = 15.0
        return features

    def decode(self, features):
        return torch.zeros(features.shape[0], 64)


class MockBackend(ModelBackend):
    name = "mock-model"
    d_model = 64
    n_layers = 4

    def load(self, device="auto"):
        pass

    def tokenize(self, text):
        words = text.split()
        return TokenizedInput(
            ids=torch.arange(len(words)),
            strings=words,
        )

    def get_activations(self, tokens, layers):
        seq_len = tokens.ids.shape[0]
        return {layer_idx: torch.randn(seq_len, self.d_model) for layer_idx in layers}

    def get_sae(self, layer):
        return MockSAE()


class TestFeatureSurveyProbe:
    def setup_method(self):
        self.probe = FeatureSurveyProbe()
        self.model = MockBackend()
        self.config = {
            "probe": "feature_survey",
            "model": "mock-model",
            "layer": 2,
            "prompts": [
                {"category": "english", "texts": ["The cat sat"]},
                {"category": "code", "texts": ["def hello world"]},
            ],
        }

    def test_validate_config_valid(self):
        self.probe.validate_config(self.config)

    def test_validate_config_missing_prompts(self):
        bad = {"probe": "feature_survey", "model": "mock-model", "layer": 2}
        with pytest.raises(ConfigError):
            self.probe.validate_config(bad)

    def test_validate_config_missing_layer(self):
        bad = {"probe": "feature_survey", "model": "mock-model", "prompts": []}
        with pytest.raises(ConfigError):
            self.probe.validate_config(bad)

    def test_run_returns_probe_result(self):
        result = self.probe.run(self.model, self.config)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "feature_survey"
        assert result.schema_version == "1.0"

    def test_run_identifies_narrow_and_broad(self):
        result = self.probe.run(self.model, self.config)
        assert "narrow_features" in result.data
        assert "broad_features" in result.data

    def test_run_skips_bos_token(self):
        result = self.probe.run(self.model, self.config)
        assert result.result_metadata.bos_skipped is True

    def test_process_prompt_directly(self):
        features = torch.zeros(4, 10)
        features[1, 5] = 8.0
        features[2, 5] = 6.0
        features[3, 9] = 3.0
        fa = FeatureActivations(
            tokens=["<bos>", "The", "cat", "sat"],
            features=features, layer=16, model_name="test",
        )
        result = self.probe.process_prompt(fa, {"category": "english"})
        assert result["category"] == "english"
        assert result["n_tokens"] == 3
        assert 5 in result["features"]
        assert result["features"][5]["count"] == 2
        assert result["features"][5]["max_act"] == 8.0

    def test_max_prompts_limits_processing(self):
        config = {**self.config, "max_prompts": 1}
        result = self.probe.run(self.model, config)
        assert result.result_metadata.n_prompts == 1
