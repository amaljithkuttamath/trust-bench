---
name: run-experiment
description: Run a Trust Bench experiment from a YAML config file
---

# Run Experiment

Run a probe experiment and save results.

## Usage

1. Check that the config YAML exists and is valid
2. Run the experiment:
   ```bash
   cd ~/Developer/trust-bench
   uv run trust-bench run <config-path> --device auto
   ```
3. Check the output directory for results.json and report.md
4. If the run fails, check:
   - Is the model name valid? (Currently only llama-3.1-8b)
   - Is the layer in range? (0-31 for Llama 3.1 8B)
   - Are prompts formatted correctly for the probe type?

## Quick survey shortcut

For a quick feature survey without writing a config:
```bash
uv run trust-bench survey --model llama-3.1-8b --layer 16
```
