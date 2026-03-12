"""Feature survey probe: catalog SAE features across diverse prompts."""

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


class FeatureSurveyProbe(Probe):
    name = "feature_survey"
    description = (
        "Run diverse prompts, catalog which features fire, "
        "identify narrow/selective features."
    )

    def validate_config(self, config: dict) -> None:
        if "layer" not in config:
            raise ConfigError("feature_survey probe requires 'layer' field")
        if "prompts" not in config:
            raise ConfigError("feature_survey probe requires 'prompts' field")
        for i, group in enumerate(config["prompts"]):
            if "category" not in group:
                raise ConfigError(f"prompts[{i}] missing 'category'")
            if "texts" not in group or not group["texts"]:
                raise ConfigError(f"prompts[{i}] missing or empty 'texts'")

    def process_prompt(self, fa: FeatureActivations, context: dict) -> dict[str, Any]:
        """Process one prompt: extract per-feature stats, skipping BOS."""
        category = context["category"]
        features: dict[int, dict] = {}

        for tok_idx in range(1, len(fa.tokens)):  # Skip BOS
            active = fa.feature_at_token(tok_idx, threshold=0.0)
            for feat_idx, act_val in active:
                if feat_idx not in features:
                    features[feat_idx] = {"count": 0, "max_act": 0.0}
                features[feat_idx]["count"] += 1
                features[feat_idx]["max_act"] = max(
                    features[feat_idx]["max_act"], abs(act_val)
                )

        return {
            "features": features,
            "category": category,
            "n_tokens": len(fa.tokens) - 1,
        }

    def run(self, model: ModelBackend, config: dict) -> ProbeResult:
        import time

        start = time.monotonic()
        layer = config["layer"]
        max_prompts = config.get("max_prompts")

        feature_token_counts: dict[int, int] = defaultdict(int)
        feature_categories: dict[int, set] = defaultdict(set)
        feature_max_act: dict[int, float] = defaultdict(float)
        total_tokens = 0
        prompts_processed = 0

        for group in config["prompts"]:
            category = group["category"]
            for text in group["texts"]:
                if max_prompts and prompts_processed >= max_prompts:
                    break

                tokens = model.tokenize(text)
                fa = model.get_feature_activations(tokens, layer)
                result = self.process_prompt(fa, {"category": category})

                total_tokens += result["n_tokens"]
                for feat_idx, stats in result["features"].items():
                    feature_token_counts[feat_idx] += stats["count"]
                    feature_categories[feat_idx].add(category)
                    feature_max_act[feat_idx] = max(
                        feature_max_act[feat_idx], stats["max_act"]
                    )
                prompts_processed += 1

        n_categories = len(config["prompts"])
        broad_features = []
        narrow_features = []

        for feat_idx in sorted(feature_token_counts.keys()):
            count = feature_token_counts[feat_idx]
            max_act = feature_max_act[feat_idx]
            entry = {
                "feature_idx": feat_idx,
                "token_count": count,
                "category_count": len(feature_categories[feat_idx]),
                "categories": sorted(feature_categories[feat_idx]),
                "max_activation": round(max_act, 2),
            }

            token_frac = count / total_tokens if total_tokens > 0 else 0
            if token_frac > 0.8:
                broad_features.append(entry)
            elif count >= 1:
                narrow_features.append(entry)

        narrow_features.sort(key=lambda x: x["max_activation"], reverse=True)

        duration = time.monotonic() - start
        metadata = ResultMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trust_bench_version=_TRUST_BENCH_VERSION,
            git_hash=_UNKNOWN_GIT_HASH,
            device=_UNKNOWN_DEVICE,
            duration_seconds=round(duration, 4),
            layer=layer,
            total_tokens=total_tokens,
            bos_skipped=True,
            n_prompts=prompts_processed,
            n_categories=n_categories,
        )

        return ProbeResult(
            schema_version="1.0",
            probe_name=self.name,
            model_name=model.name,
            config=config,
            data={
                "narrow_features": narrow_features,
                "broad_features": broad_features,
                "total_unique_features": len(feature_token_counts),
            },
            result_metadata=metadata,
        )
