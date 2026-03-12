---
name: explore-features
description: Explore SAE features interactively using Trust Bench
---

# Explore Features

Investigate specific SAE features or find interesting ones.

## Find interesting features

1. Run a feature survey first:
   ```bash
   uv run trust-bench survey --model llama-3.1-8b --layer 16
   ```
2. Look at the narrow features list. These are the most interpretable.
3. For each candidate feature, check what tokens it fires on:
   ```bash
   uv run trust-bench feature --model llama-3.1-8b --layer 16 --feature <idx>
   ```

## Investigate a specific feature

1. Write a config targeting the feature with diverse prompts
2. Run the experiment to see activation patterns
3. Check selectivity: does it fire on the target concept and nothing else?
4. Check cross-lingual: does it fire on the same concept in other languages?

## What makes a feature interesting?

- Narrow: fires on few token types but consistently
- Selective: high activation on target, zero on control
- Cross-lingual: same concept across 3+ languages
- Unexpected: fires on a category you did not design the prompt for
