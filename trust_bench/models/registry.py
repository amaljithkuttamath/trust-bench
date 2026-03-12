"""Model backend registry."""

from trust_bench.models.base import ModelBackend
from trust_bench.models.llama import LlamaBackend

REGISTRY: dict[str, type[ModelBackend]] = {
    "llama-3.1-8b": LlamaBackend,
}


def get_model(name: str) -> ModelBackend:
    """Get a model backend by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]()
