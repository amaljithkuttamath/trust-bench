"""Hallucination probe: compare SAE feature activations on facts vs controls."""

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


class HallucinationProbe(Probe):
    name = "hallucination"
    description = (
        "Compare SAE feature activations at the answer position for factual "
        "prompts vs hallucination-inducing controls. Returns features that "
        "differentiate the two groups."
    )

    def validate_config(self, config: dict) -> None:
        if "prompts" not in config:
            raise ConfigError("hallucination probe requires 'prompts' field")
        prompts = config["prompts"]
        if "facts" not in prompts:
            raise ConfigError("hallucination probe requires 'prompts.facts' list")
        if not prompts["facts"]:
            raise ConfigError("hallucination probe requires at least one fact prompt")
        if "controls" not in prompts:
            raise ConfigError("hallucination probe requires 'prompts.controls' list")
        if not prompts["controls"]:
            raise ConfigError("hallucination probe requires at least one control prompt")
        for i, entry in enumerate(prompts["facts"]):
            if "text" not in entry:
                raise ConfigError(f"prompts.facts[{i}] missing 'text'")
        for i, entry in enumerate(prompts["controls"]):
            if "text" not in entry:
                raise ConfigError(f"prompts.controls[{i}] missing 'text'")

    def process_prompt(self, fa: FeatureActivations, context: dict) -> dict[str, Any]:
        """Extract last-token feature activations (answer position)."""
        last_token_idx = len(fa.tokens) - 1
        active = fa.feature_at_token(last_token_idx, threshold=0.0)
        return {
            "feature_activations": {feat_idx: act_val for feat_idx, act_val in active},
            "group": context.get("group", "unknown"),
        }

    def _collect_group_activations(
        self,
        model: ModelBackend,
        prompts: list[dict],
        layer: int,
        group_name: str,
    ) -> list[dict[int, float]]:
        """Run all prompts in a group, return per-prompt last-token feature dicts."""
        results = []
        for entry in prompts:
            text = entry["text"]
            tokens = model.tokenize(text)
            fa = model.get_feature_activations(tokens, layer)
            processed = self.process_prompt(fa, {"group": group_name})
            results.append(processed["feature_activations"])
        return results

    def _mean_activations(self, activation_list: list[dict[int, float]]) -> dict[int, float]:
        """Compute mean activation per feature across a list of prompt results."""
        if not activation_list:
            return {}
        totals: dict[int, float] = {}
        counts: dict[int, int] = {}
        for act_dict in activation_list:
            for feat_idx, val in act_dict.items():
                totals[feat_idx] = totals.get(feat_idx, 0.0) + val
                counts[feat_idx] = counts.get(feat_idx, 0) + 1
        n = len(activation_list)
        return {feat_idx: totals[feat_idx] / n for feat_idx in totals}

    def run(self, model: ModelBackend, config: dict) -> ProbeResult:
        import time

        start = time.monotonic()

        layers: list[int] = config.get("layers", [config.get("layer", 0)])
        layer = layers[0]

        fact_prompts: list[dict] = config["prompts"]["facts"]
        control_prompts: list[dict] = config["prompts"]["controls"]

        fact_activations = self._collect_group_activations(model, fact_prompts, layer, "fact")
        control_activations = self._collect_group_activations(
            model, control_prompts, layer, "control"
        )

        fact_means = self._mean_activations(fact_activations)
        control_means = self._mean_activations(control_activations)

        all_feature_idxs = set(fact_means.keys()) | set(control_means.keys())
        differential_features = []
        for feat_idx in all_feature_idxs:
            fact_val = fact_means.get(feat_idx, 0.0)
            control_val = control_means.get(feat_idx, 0.0)
            diff = control_val - fact_val
            differential_features.append(
                {
                    "feature_idx": feat_idx,
                    "fact_mean": round(fact_val, 4),
                    "control_mean": round(control_val, 4),
                    "difference": round(diff, 4),
                    "abs_difference": round(abs(diff), 4),
                }
            )

        differential_features.sort(key=lambda x: x["abs_difference"], reverse=True)

        duration = time.monotonic() - start
        metadata = ResultMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trust_bench_version=_TRUST_BENCH_VERSION,
            git_hash=_UNKNOWN_GIT_HASH,
            device=_UNKNOWN_DEVICE,
            duration_seconds=round(duration, 4),
            layer=layer,
            n_prompts=len(fact_prompts) + len(control_prompts),
        )

        return ProbeResult(
            schema_version="1.0",
            probe_name=self.name,
            model_name=model.name,
            config=config,
            data={
                "differential_features": differential_features,
                "n_fact_prompts": len(fact_prompts),
                "n_control_prompts": len(control_prompts),
                "n_differential_features": len(differential_features),
            },
            result_metadata=metadata,
        )
