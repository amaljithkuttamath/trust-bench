"""Cross-lingual probe: find SAE features that fire on the same concept across languages."""

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from trust_bench.models.base import (
    ConfigError,
    FeatureActivations,
    ModelBackend,
    ProbeResult,
    ResultMetadata,
)
from trust_bench.probes.base import Probe

_TRUST_BENCH_VERSION = "0.1.0"
_UNKNOWN_GIT_HASH = "unknown"
_UNKNOWN_DEVICE = "unknown"

# Minimum number of languages a feature must appear in to be considered cross-lingual.
_MIN_LANGUAGE_COUNT = 3


class CrossLingualProbe(Probe):
    """Find SAE features that activate on the same concept across multiple languages.

    For each group of parallel sentences (same meaning, different languages), the probe
    collects the features active at each non-BOS token. A feature is considered
    cross-lingual if it fires in at least _MIN_LANGUAGE_COUNT distinct languages for
    a given concept. Control sentences are used to measure selectivity: features that
    also fire heavily in control text are flagged as non-selective.
    """

    name = "cross_lingual"
    description = (
        "Find SAE features that fire on the same concept across multiple languages, "
        "indicating language-independent representations."
    )

    def validate_config(self, config: dict) -> None:
        if "layer" not in config:
            raise ConfigError("cross_lingual probe requires 'layer' field")
        prompts = config.get("prompts", {})
        if "parallel_sentences" not in prompts:
            raise ConfigError("cross_lingual probe requires 'prompts.parallel_sentences' field")
        for i, group in enumerate(prompts["parallel_sentences"]):
            if "concept" not in group:
                raise ConfigError(f"parallel_sentences[{i}] missing 'concept'")
            if "sentences" not in group or not group["sentences"]:
                raise ConfigError(f"parallel_sentences[{i}] missing or empty 'sentences'")

    def process_prompt(self, fa: FeatureActivations, context: dict) -> dict[str, Any]:
        """Collect active features for a single sentence, skipping BOS at position 0."""
        features: dict[int, float] = {}
        for tok_idx in range(1, len(fa.tokens)):
            active = fa.feature_at_token(tok_idx, threshold=0.0)
            for feat_idx, act_val in active:
                current_max = features.get(feat_idx, 0.0)
                features[feat_idx] = max(current_max, abs(act_val))
        return {
            "features": features,
            "lang": context.get("lang", ""),
            "concept": context.get("concept", ""),
        }

    def _get_active_features(self, model: ModelBackend, text: str, layer: int) -> dict[int, float]:
        """Return {feature_idx: max_activation} for non-BOS tokens in text."""
        tokens = model.tokenize(text)
        fa = model.get_feature_activations(tokens, layer)
        result = self.process_prompt(fa, {})
        return result["features"]

    def run(self, model: ModelBackend, config: dict) -> ProbeResult:
        start = time.monotonic()
        layer = config["layer"]
        prompts = config["prompts"]
        parallel_sentences = prompts["parallel_sentences"]
        control_sentences: dict[str, str] = prompts.get("control_sentences", {})

        # Build control feature set: features active in any control sentence.
        control_feature_max: dict[int, float] = defaultdict(float)
        for _lang, text in control_sentences.items():
            for feat_idx, max_act in self._get_active_features(model, text, layer).items():
                control_feature_max[feat_idx] = max(control_feature_max[feat_idx], max_act)

        # For each concept, collect which languages each feature fires in and max activation.
        # Structure: concept -> feature_idx -> {langs: set, max_act: float}
        concept_feature_data: dict[str, dict[int, dict]] = {}

        for group in parallel_sentences:
            concept = group["concept"]
            sentences: dict[str, str] = group["sentences"]
            feature_langs: dict[int, set] = defaultdict(set)
            feature_max_act: dict[int, float] = defaultdict(float)

            for lang, text in sentences.items():
                active = self._get_active_features(model, text, layer)
                for feat_idx, max_act in active.items():
                    feature_langs[feat_idx].add(lang)
                    feature_max_act[feat_idx] = max(feature_max_act[feat_idx], max_act)

            concept_feature_data[concept] = {
                feat_idx: {
                    "langs": langs,
                    "max_act": feature_max_act[feat_idx],
                }
                for feat_idx, langs in feature_langs.items()
            }

        # Aggregate cross-lingual features across all concepts.
        # A feature is cross-lingual if it fires in >= _MIN_LANGUAGE_COUNT languages
        # for at least one concept.
        cross_lingual_index: dict[int, dict] = {}

        for concept, feat_data in concept_feature_data.items():
            for feat_idx, info in feat_data.items():
                lang_count = len(info["langs"])
                if lang_count < _MIN_LANGUAGE_COUNT:
                    continue

                if feat_idx not in cross_lingual_index:
                    cross_lingual_index[feat_idx] = {
                        "feature_idx": feat_idx,
                        "lang_count": lang_count,
                        "languages": sorted(info["langs"]),
                        "max_activation": round(info["max_act"], 4),
                        "concepts": [concept],
                        "control_max_activation": round(control_feature_max.get(feat_idx, 0.0), 4),
                        "selective": control_feature_max.get(feat_idx, 0.0) < info["max_act"],
                    }
                else:
                    entry = cross_lingual_index[feat_idx]
                    entry["lang_count"] = max(entry["lang_count"], lang_count)
                    entry["languages"] = sorted(set(entry["languages"]) | info["langs"])
                    entry["max_activation"] = round(
                        max(entry["max_activation"], info["max_act"]), 4
                    )
                    if concept not in entry["concepts"]:
                        entry["concepts"].append(concept)

        cross_lingual_features = sorted(
            cross_lingual_index.values(),
            key=lambda x: (x["lang_count"], x["max_activation"]),
            reverse=True,
        )

        duration = time.monotonic() - start
        metadata = ResultMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trust_bench_version=_TRUST_BENCH_VERSION,
            git_hash=_UNKNOWN_GIT_HASH,
            device=_UNKNOWN_DEVICE,
            duration_seconds=round(duration, 4),
            layer=layer,
            bos_skipped=True,
            n_prompts=sum(len(g["sentences"]) for g in parallel_sentences) + len(control_sentences),
            n_categories=len(parallel_sentences),
        )

        return ProbeResult(
            schema_version="1.0",
            probe_name=self.name,
            model_name=model.name,
            config=config,
            data={
                "cross_lingual_features": cross_lingual_features,
                "n_concepts": len(parallel_sentences),
                "n_control_sentences": len(control_sentences),
                "min_language_threshold": _MIN_LANGUAGE_COUNT,
            },
            result_metadata=metadata,
        )
