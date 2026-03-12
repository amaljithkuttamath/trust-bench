"""Model abstraction layer: ABCs, data objects, and error types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

# --- Errors ---

class TrustBenchError(Exception):
    """Base error for Trust Bench."""


class ModelLoadError(TrustBenchError):
    """Failed to load a model or SAE."""


class ConfigError(TrustBenchError):
    """Invalid experiment configuration."""


class ProbeError(TrustBenchError):
    """Error during probe execution."""


# --- Data Objects ---

@dataclass
class TokenizedInput:
    """Tokenized text. Batch dim squeezed by backend."""

    ids: Tensor          # (seq_len,)
    strings: list[str]   # Per-token decoded strings


@dataclass
class FeatureActivations:
    """SAE feature activations for a single prompt."""

    tokens: list[str]
    features: Tensor     # (n_tokens, n_features)
    layer: int
    model_name: str

    def top_features(self, k: int = 10) -> list[tuple[int, float]]:
        """Top-k features by MAX activation across all tokens."""
        max_acts, _ = self.features.max(dim=0)
        active_mask = max_acts > 0
        active_indices = torch.where(active_mask)[0]
        active_vals = max_acts[active_mask]

        actual_k = min(k, len(active_indices))
        if actual_k == 0:
            return []

        topk_vals, topk_local = active_vals.topk(actual_k)
        topk_indices = active_indices[topk_local]
        return [(idx.item(), val.item()) for idx, val in zip(topk_indices, topk_vals)]

    def tokens_for_feature(
        self, feature_idx: int, threshold: float = 0.0
    ) -> list[tuple[str, float]]:
        """Tokens where feature activates above threshold (absolute value)."""
        acts = self.features[:, feature_idx]
        mask = acts.abs() > threshold
        return [
            (self.tokens[i], acts[i].item())
            for i in torch.where(mask)[0]
        ]

    def feature_at_token(self, token_idx: int, threshold: float = 0.0) -> list[tuple[int, float]]:
        """Features active at a specific token position above threshold (absolute value)."""
        acts = self.features[token_idx]
        mask = acts.abs() > threshold
        return [
            (i.item(), acts[i].item())
            for i in torch.where(mask)[0]
        ]


@dataclass
class ResultMetadata:
    """Reproducibility metadata attached to every result.
    Inspired by lm-evaluation-harness EvalResults + SAELens EvalConfig."""

    timestamp: str
    trust_bench_version: str
    git_hash: str
    device: str
    duration_seconds: float
    canary: str = "canary-trust-bench-2026"
    # Probe-specific fields (optional, populated by individual probes)
    layer: int | None = None
    total_tokens: int | None = None
    bos_skipped: bool | None = None
    n_prompts: int | None = None
    n_categories: int | None = None


@dataclass
class ProbeResult:
    """Structured result from a probe run."""

    schema_version: str
    probe_name: str
    model_name: str
    config: dict
    data: dict
    result_metadata: ResultMetadata


# --- ABCs ---

class SAEWrapper(ABC):
    """Wraps a sparse autoencoder for encoding/decoding activations."""

    n_features: int

    @abstractmethod
    def encode(self, activations: Tensor) -> Tensor:
        """Encode activations to sparse features. (seq_len, d_model) -> (seq_len, n_features)"""
        ...

    @abstractmethod
    def decode(self, features: Tensor) -> Tensor:
        """Decode sparse features to activations. (seq_len, n_features) -> (seq_len, d_model)"""
        ...

    def get_feature_label(self, idx: int) -> str | None:
        """Return auto-interp label if available. None otherwise."""
        return None


class ModelBackend(ABC):
    """Abstract model backend. All probes interact through this interface."""

    name: str
    d_model: int
    n_layers: int

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        ...

    @abstractmethod
    def tokenize(self, text: str) -> TokenizedInput:
        ...

    @abstractmethod
    def get_activations(self, tokens: TokenizedInput, layers: list[int]) -> dict[int, Tensor]:
        """Get residual stream activations. Returns {layer: (seq_len, d_model)}."""
        ...

    @abstractmethod
    def get_sae(self, layer: int) -> SAEWrapper:
        ...

    def get_feature_activations(self, tokens: TokenizedInput, layer: int) -> FeatureActivations:
        """Default: activations -> SAE encode."""
        acts = self.get_activations(tokens, [layer])[layer]
        sae = self.get_sae(layer)
        features = sae.encode(acts)
        return FeatureActivations(
            tokens=tokens.strings,
            features=features,
            layer=layer,
            model_name=self.name,
        )
