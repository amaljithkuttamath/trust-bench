"""Probe registry."""

from trust_bench.models.base import ConfigError
from trust_bench.probes.base import Probe
from trust_bench.probes.feature_survey import FeatureSurveyProbe

PROBE_REGISTRY: dict[str, type[Probe]] = {
    "feature_survey": FeatureSurveyProbe,
}


def get_probe(name: str) -> Probe:
    if name not in PROBE_REGISTRY:
        raise ConfigError(f"Unknown probe: {name}. Available: {list(PROBE_REGISTRY.keys())}")
    return PROBE_REGISTRY[name]()
