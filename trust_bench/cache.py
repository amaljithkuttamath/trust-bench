"""Activation caching for reuse across probes.
Inspired by SAELens ActivationsStore."""

import hashlib
from pathlib import Path

import torch
from torch import Tensor


class ActivationCache:
    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, model_name: str, prompt: str, layer: int) -> str:
        """Deterministic cache key from model + prompt + layer."""
        raw = f"{model_name}|{prompt}|{layer}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def save(self, model_name: str, prompt: str, layer: int, activations: Tensor) -> str:
        """Save activations to cache. Returns cache key."""
        key = self.make_key(model_name, prompt, layer)
        path = self._dir / f"{key}.pt"
        torch.save(activations, path)
        return key

    def load(self, key: str) -> Tensor | None:
        """Load activations from cache. Returns None on miss."""
        path = self._dir / f"{key}.pt"
        if not path.exists():
            return None
        return torch.load(path, weights_only=True)

    def has(self, model_name: str, prompt: str, layer: int) -> bool:
        """Check if activations are cached."""
        key = self.make_key(model_name, prompt, layer)
        return (self._dir / f"{key}.pt").exists()
