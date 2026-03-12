import pytest

from trust_bench.models.base import ConfigError
from trust_bench.probes.base import Probe
from trust_bench.probes.registry import PROBE_REGISTRY, get_probe


class TestProbeRegistry:
    def test_feature_survey_registered(self):
        assert "feature_survey" in PROBE_REGISTRY

    def test_get_probe_returns_probe(self):
        probe = get_probe("feature_survey")
        assert isinstance(probe, Probe)
        assert probe.name == "feature_survey"

    def test_get_probe_unknown_raises(self):
        with pytest.raises(ConfigError, match="Unknown probe"):
            get_probe("nonexistent")
