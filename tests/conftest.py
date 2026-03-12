"""Shared test factories. Inspired by CircuitLab's test utils."""

import torch

from trust_bench.models.base import (
    FeatureActivations,
    ProbeResult,
    ResultMetadata,
)


def build_config(**overrides) -> dict:
    """Build a minimal valid config dict."""
    base = {
        "probe": "feature_survey",
        "model": "llama-3.1-8b",
        "layers": [16],
        "prompts": [{"category": "test", "texts": ["The cat sat on the mat."]}],
    }
    base.update(overrides)
    return base


def fake_feature_activations(
    n_tokens: int = 5,
    n_features: int = 100,
    layer: int = 16,
    active_features: dict[int, list[tuple[int, float]]] | None = None,
) -> FeatureActivations:
    """Create synthetic FeatureActivations for testing probes without a real model."""
    tokens = [f"tok_{i}" for i in range(n_tokens)]
    features = torch.zeros(n_tokens, n_features)
    if active_features:
        for token_idx, feature_vals in active_features.items():
            for feat_idx, val in feature_vals:
                features[token_idx, feat_idx] = val
    return FeatureActivations(
        tokens=tokens, features=features, layer=layer, model_name="test-model",
    )


def fake_result(**overrides) -> ProbeResult:
    """Build a minimal ProbeResult for testing."""
    base = dict(
        schema_version="1.0",
        probe_name="test_probe",
        model_name="test-model",
        config={"probe": "test"},
        data={"features": []},
        result_metadata=ResultMetadata(
            timestamp="2026-01-01T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="test123",
            device="cpu",
            duration_seconds=0.1,
        ),
    )
    base.update(overrides)
    return ProbeResult(**base)
