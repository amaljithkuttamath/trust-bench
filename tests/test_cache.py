import torch

from trust_bench.cache import ActivationCache


class TestActivationCache:
    def test_save_and_load(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        tensor = torch.randn(10, 4096)
        key = cache.save("test-model", "Hello world", layer=16, activations=tensor)
        loaded = cache.load(key)
        assert loaded is not None
        assert torch.allclose(tensor, loaded)

    def test_cache_miss_returns_none(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        assert cache.load("nonexistent-key") is None

    def test_cache_key_is_deterministic(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        key1 = cache.make_key("model", "prompt", 16)
        key2 = cache.make_key("model", "prompt", 16)
        assert key1 == key2

    def test_different_prompts_different_keys(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        key1 = cache.make_key("model", "prompt A", 16)
        key2 = cache.make_key("model", "prompt B", 16)
        assert key1 != key2

    def test_has_returns_true_after_save(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        cache.save("model", "prompt", 16, torch.randn(5, 64))
        assert cache.has("model", "prompt", 16)

    def test_has_returns_false_for_missing(self, tmp_path):
        cache = ActivationCache(cache_dir=str(tmp_path))
        assert not cache.has("model", "prompt", 16)
