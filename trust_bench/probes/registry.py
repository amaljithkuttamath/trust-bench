"""Probe registry."""

from trust_bench.models.base import ConfigError
from trust_bench.probes.base import Probe
from trust_bench.probes.cross_lingual import CrossLingualProbe
from trust_bench.probes.feature_survey import FeatureSurveyProbe
from trust_bench.probes.hallucination import HallucinationProbe

PROBE_REGISTRY: dict[str, type[Probe]] = {
    "cross_lingual": CrossLingualProbe,
    "feature_survey": FeatureSurveyProbe,
    "hallucination": HallucinationProbe,
}


def get_probe(name: str) -> Probe:
    if name not in PROBE_REGISTRY:
        raise ConfigError(f"Unknown probe: {name}. Available: {list(PROBE_REGISTRY.keys())}")
    return PROBE_REGISTRY[name]()
