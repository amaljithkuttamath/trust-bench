"""Probe abstract base class."""

from abc import ABC, abstractmethod
from typing import Any

from trust_bench.models.base import FeatureActivations, ModelBackend, ProbeResult


class Probe(ABC):
    name: str
    description: str

    @abstractmethod
    def validate_config(self, config: dict) -> None:
        ...

    @abstractmethod
    def process_prompt(self, fa: FeatureActivations, context: dict) -> dict[str, Any]:
        ...

    @abstractmethod
    def run(self, model: ModelBackend, config: dict) -> ProbeResult:
        ...
