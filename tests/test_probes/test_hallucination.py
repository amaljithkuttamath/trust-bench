import pytest
import torch

from trust_bench.models.base import (
    ConfigError,
    ModelBackend,
    ProbeResult,
    SAEWrapper,
    TokenizedInput,
)
from trust_bench.probes.hallucination import HallucinationProbe


class MockSAEHallucination(SAEWrapper):
    n_features = 50

    def __init__(self, hallucination_mode=False):
        self._hallucination_mode = hallucination_mode

    def encode(self, activations):
        n_tokens = activations.shape[0]
        features = torch.zeros(n_tokens, self.n_features)
        if self._hallucination_mode:
            features[-1, 10] = 20.0
        else:
            features[-1, 10] = 1.0
        features[:, 5] = 3.0
        return features

    def decode(self, features):
        return torch.zeros(features.shape[0], 64)


class MockBackendHallucination(ModelBackend):
    name = "mock-model"
    d_model = 64
    n_layers = 4

    def __init__(self):
        self._call_count = 0

    def load(self, device="auto"):
        pass

    def tokenize(self, text):
        words = text.split()
        return TokenizedInput(ids=torch.arange(len(words)), strings=words)

    def get_activations(self, tokens, layers):
        seq_len = tokens.ids.shape[0]
        return {layer: torch.randn(seq_len, self.d_model) for layer in layers}

    def get_sae(self, layer):
        self._call_count += 1
        return MockSAEHallucination(hallucination_mode=self._call_count > 2)


class TestHallucinationProbe:
    def setup_method(self):
        self.probe = HallucinationProbe()
        self.model = MockBackendHallucination()
        self.config = {
            "probe": "hallucination",
            "model": "mock-model",
            "layers": [2],
            "prompts": {
                "facts": [
                    {"text": "The capital of France is", "expected": "Paris"},
                    {"text": "Water boils at", "expected": "100"},
                ],
                "controls": [
                    {"text": "The capital of Freedonia is"},
                    {"text": "The drug Zyloxitab treats"},
                ],
            },
        }

    def test_validate_config_valid(self):
        self.probe.validate_config(self.config)

    def test_validate_config_missing_prompts(self):
        bad = {"probe": "hallucination", "model": "m", "layers": [2]}
        with pytest.raises(ConfigError):
            self.probe.validate_config(bad)

    def test_validate_config_missing_facts(self):
        bad = {
            "probe": "hallucination",
            "model": "m",
            "layers": [2],
            "prompts": {"controls": [{"text": "x"}]},
        }
        with pytest.raises(ConfigError):
            self.probe.validate_config(bad)

    def test_run_returns_probe_result(self):
        result = self.probe.run(self.model, self.config)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "hallucination"
        assert "differential_features" in result.data
