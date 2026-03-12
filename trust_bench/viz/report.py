"""Markdown report generator from ProbeResult."""

import json
from pathlib import Path

from trust_bench.models.base import ProbeResult


def generate_report(result: ProbeResult, config_path: str) -> str:
    """Generate a markdown report next to the config file. Returns path."""
    out_dir = Path(config_path).parent
    report_path = out_dir / "report.md"

    lines = [
        f"# {result.probe_name.replace('_', ' ').title()}: {result.model_name}",
        "",
        f"**Model:** {result.model_name}",
    ]

    # Add metadata from result_metadata
    meta = result.result_metadata
    if meta.layer is not None:
        lines.append(f"**Layer:** {meta.layer}")
    if meta.total_tokens is not None:
        lines.append(f"**Total Tokens:** {meta.total_tokens}")
    if meta.n_prompts is not None:
        lines.append(f"**Prompts:** {meta.n_prompts}")
    lines.append("")

    # Probe-specific sections
    if result.probe_name == "feature_survey":
        _render_feature_survey(lines, result)
    elif result.probe_name == "hallucination":
        _render_hallucination(lines, result)
    elif result.probe_name == "cross_lingual":
        _render_cross_lingual(lines, result)
    else:
        lines.append("## Results")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.data, indent=2, default=str))
        lines.append("```")

    lines.append("")
    lines.append("## Raw Data")
    lines.append("[results.json](results.json)")
    lines.append("")

    content = "\n".join(lines)
    report_path.write_text(content)
    return str(report_path)


def _render_feature_survey(lines, result):
    data = result.data
    lines.append("## Summary")
    lines.append(f"- {data.get('total_unique_features', '?')} unique features fired")
    lines.append(f"- {len(data.get('broad_features', []))} broad features")
    lines.append(f"- {len(data.get('narrow_features', []))} narrow features")
    lines.append("")
    narrow = data.get("narrow_features", [])
    if narrow:
        lines.append("## Notable Narrow Features")
        lines.append("")
        for feat in narrow[:20]:
            idx = feat["feature_idx"]
            max_act = feat.get("max_activation", "?")
            cats = ", ".join(feat.get("categories", []))
            lines.append(f"### Feature #{idx}")
            lines.append(f"- Max activation: {max_act}")
            lines.append(f"- Categories: {cats}")
            lines.append(f"- Token count: {feat.get('token_count', '?')}")
            lines.append("")


def _render_hallucination(lines, result):
    diff = result.data.get("differential_features", {})
    for layer_str, features in diff.items():
        lines.append(f"## Layer {layer_str}")
        lines.append("")
        if not features:
            lines.append("No differential features found.")
            lines.append("")
            continue
        lines.append("| Feature | Fact mean | Control mean | Difference |")
        lines.append("|---------|-----------|-------------|------------|")
        for feat in features[:20]:
            lines.append(
                f"| #{feat['feature_idx']} | {feat['fact_mean']} | "
                f"{feat['control_mean']} | {feat['difference']} |"
            )
        lines.append("")


def _render_cross_lingual(lines, result):
    features = result.data.get("cross_lingual_features", [])
    lines.append("## Cross-Lingual Features")
    lines.append("")
    if not features:
        lines.append("No cross-lingual features found.")
        return
    for feat in features[:20]:
        idx = feat["feature_idx"]
        concept = feat["concept"]
        langs = ", ".join(feat["languages"])
        lines.append(f"### Feature #{idx}: {concept}")
        lines.append(f"- Languages: {langs} ({feat['n_languages']})")
        lines.append(f"- Max activation: {feat['max_activation']}")
        lines.append(f"- Fires on control: {feat.get('fires_on_control', 'N/A')}")
        lines.append("")
