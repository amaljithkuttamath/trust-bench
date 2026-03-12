import pytest
import torch

from trust_bench.models.base import (
    ConfigError,
    ModelBackend,
    ProbeResult,
    SAEWrapper,
    TokenizedInput,
)
from trust_bench.probes.cross_lingual import CrossLingualProbe


class MockSAECrossLingual(SAEWrapper):
    n_features = 50

    def encode(self, activations):
        n_tokens = activations.shape[0]
        features = torch.zeros(n_tokens, self.n_features)
        if n_tokens > 2:
            features[2, 20] = 25.0
        return features

    def decode(self, features):
        return torch.zeros(features.shape[0], 64)


class MockBackendCrossLingual(ModelBackend):
    name = "mock-model"
    d_model = 64
    n_layers = 4

    def load(self, device="auto"):
        pass

    def tokenize(self, text):
        words = text.split()
        return TokenizedInput(ids=torch.arange(len(words)), strings=words)

    def get_activations(self, tokens, layers):
        seq_len = tokens.ids.shape[0]
        return {layer_idx: torch.randn(seq_len, self.d_model) for layer_idx in layers}

    def get_sae(self, layer):
        return MockSAECrossLingual()


class TestCrossLingualProbe:
    def setup_method(self):
        self.probe = CrossLingualProbe()
        self.model = MockBackendCrossLingual()
        self.config = {
            "probe": "cross_lingual",
            "model": "mock-model",
            "layer": 2,
            "prompts": {
                "parallel_sentences": [
                    {
                        "concept": "conjunction",
                        "sentences": {
                            "en": "The dog and the cat",
                            "fr": "Le chien et le chat",
                            "de": "Der Hund und die Katze",
                        },
                    },
                ],
                "control_sentences": {
                    "en": "The sky is blue today",
                    "fr": "Le ciel est bleu",
                },
            },
        }

    def test_validate_config_valid(self):
        self.probe.validate_config(self.config)

    def test_validate_config_missing_parallel(self):
        bad = {
            "probe": "cross_lingual",
            "model": "m",
            "layer": 2,
            "prompts": {"control_sentences": {}},
        }
        with pytest.raises(ConfigError):
            self.probe.validate_config(bad)

    def test_run_returns_probe_result(self):
        result = self.probe.run(self.model, self.config)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "cross_lingual"
        assert "cross_lingual_features" in result.data

    def test_run_finds_cross_lingual_feature(self):
        result = self.probe.run(self.model, self.config)
        features = result.data["cross_lingual_features"]
        feature_ids = [f["feature_idx"] for f in features]
        assert 20 in feature_ids
