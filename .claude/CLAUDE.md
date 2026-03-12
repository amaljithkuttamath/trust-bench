# Trust Bench

Interpretability toolkit for LLMs. Probes model internals, generates findings.

## Build & Test
uv sync --extra dev
uv run pytest

## Run an experiment
trust-bench run experiments/<name>/config.yaml

## Eval sets
Experiments can use CSV eval sets (inspired by Anthropic's paired prompts pattern):
- Reference via `eval_set: data.csv` in config YAML
- CSV rows become prompt items, probes interpret columns per their schema
- Paired prompts: use `prompt_a`/`prompt_b` columns for contrastive experiments

## Project structure
- trust_bench/models/ -- Model backends. Never import TransformerLens outside this dir.
- trust_bench/probes/ -- Research probes. Each returns ProbeResult.
- trust_bench/analysis/ -- Post-probe analysis functions.
- trust_bench/viz/ -- Plot generation. All matplotlib/seaborn.
- trust_bench/results/ -- Serialization and report generation.
- experiments/ -- One folder per experiment with config + results + figures.
- .claude/skills/ -- Claude Code skills for research automation.

## Key rules
- Probes work with FeatureActivations, never with raw model objects
- Skip BOS token (position 0) in all feature analysis
- All plots save PNG (300 DPI) + SVG
- Every experiment must have a config.yaml that fully reproduces it
- Use model.tokenizer.decode(t.item()) for per-token strings, never model.to_str_tokens()

## Writing
- No em dashes
- Technical, accessible, curious
- Lead with findings, not setup
