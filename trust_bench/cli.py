"""Trust Bench CLI."""

from pathlib import Path

import click

from trust_bench.config import load_config
from trust_bench.models.registry import get_model
from trust_bench.probes.registry import get_probe
from trust_bench.results.io import save_result


def generate_report(result, config_path):
    """Placeholder for report generation. Implemented in P1."""
    pass


@click.group()
@click.version_option()
def cli():
    """Trust Bench: interpretability toolkit for LLMs."""


@cli.command()
@click.argument("config_path")
@click.option("--device", default="auto", help="Device: auto, cpu, mps, cuda")
@click.option("--max-prompts", default=None, type=int, help="Limit prompts for quick iteration")
@click.option(
    "--cache-dir", default=None, type=click.Path(),
    help="Cache activations to disk for reuse across probes",
)
def run(config_path, device, max_prompts, cache_dir):
    """Run an experiment from a YAML config."""
    config = load_config(config_path)
    if max_prompts:
        config["max_prompts"] = max_prompts
    if cache_dir:
        config["cache_dir"] = cache_dir
    model = get_model(config["model"])
    click.echo(f"Loading {config['model']}...")
    model.load(device=device)
    probe = get_probe(config["probe"])
    probe.validate_config(config)
    click.echo(f"Running {config['probe']} probe...")
    result = probe.run(model, config)
    out = save_result(result, config_path)
    click.echo(f"Results saved to {out}")
    generate_report(result, config_path)


@cli.command()
@click.option("--model", "model_name", default="llama-3.1-8b")
@click.option("--layer", default=16, type=int)
@click.option("--device", default="auto", help="Device: auto, cpu, mps, cuda")
def survey(model_name, layer, device):
    """Quick feature survey with default prompts."""
    from trust_bench.probes.feature_survey import DEFAULT_SURVEY_PROMPTS

    model = get_model(model_name)
    click.echo(f"Loading {model_name}...")
    model.load(device=device)

    config = {
        "probe": "feature_survey",
        "model": model_name,
        "layer": layer,
        "prompts": DEFAULT_SURVEY_PROMPTS,
    }
    probe = get_probe("feature_survey")
    click.echo(f"Surveying layer {layer}...")
    result = probe.run(model, config)

    n_narrow = len(result.data.get("narrow_features", []))
    n_broad = len(result.data.get("broad_features", []))
    total = result.data.get("total_unique_features", 0)
    click.echo(f"Found {total} features: {n_narrow} narrow, {n_broad} broad")


@cli.command()
@click.argument("results_path")
def report(results_path):
    """Generate report from existing results."""
    from trust_bench.results.io import load_result

    result = load_result(results_path)
    config_path = str(Path(results_path).parent / "config.yaml")
    out = generate_report(result, config_path)
    click.echo(f"Report written to {out}")


@cli.command()
@click.option("--model", "model_name", default="llama-3.1-8b")
@click.option("--layer", default=16, type=int)
@click.option("--feature", "feature_idx", required=True, type=int)
@click.option("--device", default="auto", help="Device: auto, cpu, mps, cuda")
def feature(model_name, layer, feature_idx, device):
    """Show info about a specific SAE feature."""
    model = get_model(model_name)
    click.echo(f"Loading {model_name}...")
    model.load(device=device)
    sae = model.get_sae(layer)
    label = sae.get_feature_label(feature_idx)
    click.echo(f"Feature #{feature_idx}, layer {layer}")
    if label:
        click.echo(f"Label: {label}")
    click.echo(f"SAE has {sae.n_features} features total")
