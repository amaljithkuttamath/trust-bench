"""Llama 3.1 8B backend via TransformerLens + Llama Scope SAEs."""

import torch
from torch import Tensor

from trust_bench.models.base import (
    ModelBackend,
    ModelLoadError,
    SAEWrapper,
    TokenizedInput,
)


class SAELensWrapper(SAEWrapper):
    """Wraps an SAELens SAE object."""

    def __init__(self, sae):
        self._sae = sae
        self.n_features = sae.cfg.d_sae

    def encode(self, activations: Tensor) -> Tensor:
        return self._sae.encode(activations)

    def decode(self, features: Tensor) -> Tensor:
        return self._sae.decode(features)


class LlamaBackend(ModelBackend):
    name = "llama-3.1-8b"
    d_model = 4096
    n_layers = 32

    def __init__(self):
        self.model = None
        self._saes: dict[int, SAEWrapper] = {}

    def load(self, device: str = "auto") -> None:
        try:
            from transformer_lens import HookedTransformer

            self.model = HookedTransformer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B",
                device=device,
            )
        except OSError as e:
            raise ModelLoadError(
                f"Failed to load Llama 3.1 8B. Ensure you have:\n"
                f"  1. Accepted the license at huggingface.co/meta-llama/Meta-Llama-3.1-8B\n"
                f"  2. Run: huggingface-cli login\n"
                f"  3. At least 16GB free memory\n"
                f"Original error: {e}"
            ) from e

    def tokenize(self, text: str) -> TokenizedInput:
        tokens = self.model.to_tokens(text)
        strings = [self.model.tokenizer.decode(t.item()) for t in tokens[0]]
        return TokenizedInput(ids=tokens[0], strings=strings)

    @torch.no_grad()
    def get_activations(self, tokens: TokenizedInput, layers: list[int]) -> dict[int, Tensor]:
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        _, cache = self.model.run_with_cache(
            tokens.ids.unsqueeze(0),
            names_filter=hook_names,
        )
        result = {
            layer: cache[f"blocks.{layer}.hook_resid_post"].squeeze(0)
            for layer in layers
        }
        del cache
        return result

    def get_sae(self, layer: int) -> SAEWrapper:
        if layer not in self._saes:
            try:
                from sae_lens import SAE

                sae = SAE.from_pretrained(
                    release="llama-scope-lxr-8x",
                    sae_id=f"l{layer}r_8x",
                )
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load SAE for layer {layer}. "
                    f"Check that llama-scope-lxr-8x is available on HuggingFace.\n"
                    f"Original error: {e}"
                ) from e
            self._saes[layer] = SAELensWrapper(sae)
        return self._saes[layer]

    def clear_sae_cache(self) -> None:
        """Free SAE memory."""
        self._saes.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
