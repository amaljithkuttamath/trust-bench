# Trust Bench

Interpretability toolkit for understanding what happens inside language models. Probes SAE features, activations, and circuits.
## Install

```bash
# From PyPI
pip install trust-bench

# From source (recommended for development)
git clone https://github.com/amaljithkuttamath/trust-bench.git
cd trust-bench
uv sync --extra dev
```

## Quick start

```bash
# Feature survey on Llama 3.1 8B, layer 16
uv run trust-bench survey --model llama-3.1-8b --layer 16

# Run an experiment from config
uv run trust-bench run experiments/my-experiment/config.yaml
```

## Supported models

- Llama 3.1 8B (via TransformerLens + Llama Scope SAEs)

## License

MIT
