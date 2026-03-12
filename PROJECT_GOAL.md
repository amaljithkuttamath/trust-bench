# Trust Bench Project Goal and MVP Contract

## Goal

Trust Bench is an open-source profiler that measures and compares LLM trustworthiness, first on Qwen3.5-0.8B, by combining output-level metrics (truthfulness, calibration, safety) with architecture-aware internal signals from hybrid attention layers.

## Phase 1 Mission

Ship extraction and scoring for one model end-to-end with reproducible outputs.

- First model: Qwen3.5-0.8B (from-scratch implementation path)
- First scope: trust signal extraction plus evaluation
- Improvement loop (retraining and interventions): out of scope for Phase 1

## MVP Success Criteria

The MVP is complete when all of the following are true:

1. A CLI run completes end-to-end for one model and one evaluation task.
2. The run produces output-level trust metrics:
   - Truthfulness score(s)
   - Calibration metrics (at minimum ECE plus one companion metric)
3. The run produces internal signal summaries grouped by layer type:
   - `linear_attention`
   - `full_attention`
4. Results are saved as structured artifacts (`json` or `jsonl`) for reproducibility.
5. At least one baseline comparison is included:
   - Output-only signals versus output plus internal signals.

## Core Novelty

Within a single model architecture, compare trust-relevant signal quality across Qwen3.5's hybrid attention pattern:

- 75% linear-attention layers
- 25% full-attention layers

This is the central research angle for the initial release.

## Non-Goals (Phase 1)

- Building the full improve/retrain loop
- Supporting many model families before Qwen3.5 path is stable
- Making causal claims about internal mechanisms
- Treating attention visualizations as sufficient for trust diagnosis

## Evidence Standard

- Prefer peer-reviewed papers and official model/framework documentation
- Separate established findings from hypotheses
- Record assumptions explicitly when evidence is limited

## Decision Filter

Before adding any feature, check:

1. Does this help produce a reproducible trust profile for Qwen3.5 now?
2. Does this strengthen truthfulness, calibration, or layer-type-aware signal analysis?
3. Is this required for Phase 1 delivery, or can it wait for Phase 2+?

If a proposed task fails this filter, defer it.
