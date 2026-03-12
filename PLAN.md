# Trust Bench — Design Decisions

## What is this?

An open-source tool that profiles LLM trustworthiness, diagnoses failures, and improves models through targeted retraining. The full loop: train → evaluate → diagnose → improve → re-evaluate.

## Why build this?

### The gap (from research review)

We reviewed the existing landscape:

**Evaluation tools exist** (DeepEval, TrustLLM, lm-eval-harness, RAGAS, Confident AI, Braintrust). They score model outputs. All black-box — none connect trust scores to what's happening inside the model.

**Safety training research exists** (Constitutional AI, RepE, SafeLadder/SafeWork-R1, safety-aware fine-tuning). Labs use these internally. None are packaged as reusable tools.

**Nobody has built the closed loop as a tool.** Profile a model's trust → understand why it fails → apply a targeted fix → verify the fix worked. Labs do this internally (Anthropic's Constitutional AI process, Shanghai AI Lab's SafeLadder). It doesn't exist as open-source infrastructure.

### Key research findings that shaped the design

1. **TrustLLM (ICML 2024):** Established 6 evaluation dimensions — truthfulness, safety, fairness, robustness, privacy, machine ethics. 30+ datasets. We use their dimensions and datasets, not reinvent them.

2. **Calibration paper (JAMIA Open 2025):** Out-of-the-box LLM calibration is poor (23-46% across biomedical tasks). Raw logprobs are unreliable trust signals without post-hoc correction. This means we need calibration as a first-class concern, not an afterthought.

3. **Anthropic's alignment faking paper:** Models can selectively comply during training while preserving misaligned preferences. Surface-level evaluation misses this. Internal signals (layer activations, attention patterns) may catch what output-level scoring cannot.

4. **Anthropic's reasoning faithfulness work:** Chain-of-thought doesn't always reflect actual reasoning. Monitoring CoT alone is insufficient for trust.

5. **Safety drift research (ACL 2025):** Fine-tuning degrades safety 20-80%, but safety behaviors are suppressed not destroyed — they persist in low-curvature parameter subspaces and can be restored. This means the "improve" step is feasible.

6. **Qwen3.5 hybrid attention architecture:** Alternates linear attention (Gated DeltaNet) and full attention (GQA) in a 3:1 pattern. Nobody has studied how trust signals differ across these two attention types. Novel research opportunity.

7. **Raschka's from-scratch Qwen3.5 implementation:** 0.8B model, readable PyTorch, full access to internals. Small enough to run and retrain locally. This is our first target model.

## Architecture decisions

### Decision: One repo, not four
**Considered:** Separate repos for signals/eval/improve/orchestrator.
**Chose:** Single repo with subdirectories.
**Why:** Nothing exists yet. Splitting prematurely creates overhead (4 READMEs, 4 CI configs, cross-repo versioning) with no benefit. Subdirectories have clean interfaces so we can split later if a piece gets real adoption independently. Ship first, split later.

### Decision: uv, not pip/poetry
**Why:** Already in Amal's workflow. Fast. Lock file. No virtualenv friction.

### Decision: Use existing datasets, not custom prompts
**Considered:** Writing our own clinical prompt sets.
**Chose:** TrustLLM datasets (30+, peer-reviewed), TruthfulQA, HalluLens.
**Why:** Credibility. Peer-reviewed datasets mean results are comparable to published work. Custom prompts can be added later for domain-specific evaluation (clinical, legal) but the foundation must be established benchmarks.

### Decision: Wrap lm-eval-harness for model loading, not write our own
**Considered:** Custom model wrappers for each architecture.
**Chose:** Use lm-eval-harness model interface where possible, add thin hooks for internal signal extraction.
**Why:** lm-eval-harness already handles HuggingFace models, vLLM, APIs. Don't rewrite model loading. Our value-add is the trust signal extraction layer on top, not another model runner.

### Decision: Raschka's Qwen3.5 0.8B as first model, not a larger model
**Considered:** Starting with Qwen3-32B or Llama 3.
**Chose:** Qwen3.5-0.8B via Raschka's from-scratch implementation.
**Why:** (a) Small enough to run and retrain locally. (b) Hybrid attention architecture (3:1 linear/full) is architecturally novel — nobody has studied trust signals across attention types. (c) From-scratch code means full visibility, easy to add hooks. (d) Full production architecture (RoPE, GQA, RMSNorm, Gated DeltaNet) — not a toy model.

### Decision: Start with extraction and evaluation, not improvement
**Considered:** Building the full loop immediately.
**Chose:** Ship signals + eval first, then add improve.
**Why:** The improvement techniques (RepE, safety fine-tuning, Constitutional AI loops) are complex and depend on having good evaluation to measure their effect. You need a working profiler before you can close the loop. Also, the profiler alone is publishable — the improvement loop is the second blog post.

## Project structure

```
trust-bench/
├── PLAN.md                          # this file
├── pyproject.toml                   # uv project config
├── README.md
├── prompts/                         # test cases (jsonl)
│   └── clinical_basic.jsonl         # first prompt set
├── results/                         # output from runs (gitignored)
├── src/trust_bench/
│   ├── __init__.py
│   ├── cli.py                       # entry point: trust-bench run
│   ├── types.py                     # ModelOutput, TrustProfile, shared types
│   ├── signals/                     # EXTRACT — pull trust signals from models
│   │   ├── __init__.py
│   │   ├── base.py                  # ModelRunner protocol
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── qwen35.py            # first model: Raschka's from-scratch
│   │       └── api.py               # API models (Claude, GPT-4) via logprobs
│   ├── eval/                        # SCORE — evaluate trust dimensions
│   │   ├── __init__.py
│   │   ├── base.py                  # Scorer protocol
│   │   └── scorers/
│   │       ├── __init__.py
│   │       ├── calibration.py       # Flex-ECE, post-hoc calibration
│   │       ├── truthfulness.py      # wraps TrustLLM truthfulness
│   │       └── safety.py            # wraps TrustLLM safety
│   └── improve/                     # FIX — apply targeted techniques
│       ├── __init__.py
│       ├── base.py                  # Technique protocol
│       └── techniques/
│           ├── __init__.py
│           ├── repe.py              # representation engineering
│           ├── calibrate.py         # post-hoc calibration (isotonic, histogram)
│           └── constitutional.py    # self-critique + retrain loop
└── tests/
```

## Implementation order

### Phase 1: Extract + Score (weeks 1-3)
Ship: `trust-bench run --model qwen35 --eval truthfulness`
1. `types.py` — ModelOutput, TrustProfile dataclasses
2. `signals/base.py` — ModelRunner protocol
3. `signals/models/qwen35.py` — load Raschka's Qwen3.5, extract logprobs + entropy + layer signals per attention type
4. `eval/scorers/calibration.py` — Flex-ECE, confidence vs correctness
5. `eval/scorers/truthfulness.py` — wrap TrustLLM truthfulness datasets
6. `cli.py` — wire it together
7. Blog post: "What I Found Inside Qwen3.5's Hybrid Attention"

### Phase 2: Compare (weeks 3-5)
Ship: `trust-bench run --model qwen35,llama3 --eval truthfulness,safety`
1. `signals/models/api.py` — Claude/GPT-4 via logprobs
2. Second open model adapter (Llama or Gemma)
3. `eval/scorers/safety.py` — wrap TrustLLM safety
4. Comparison output: trust profiles side-by-side
5. Blog post: "Trust Profiles Across Architectures: Hybrid vs Standard Attention"

### Phase 3: Improve (weeks 5-8)
Ship: `trust-bench improve --model qwen35 --technique repe --target safety`
1. `improve/techniques/calibrate.py` — isotonic regression on confidence scores
2. `improve/techniques/repe.py` — activation steering for safety dimensions
3. Re-evaluation after improvement: before/after trust profiles
4. Blog post: "Closing the Loop: From Trust Profile to Model Improvement"

### Phase 4: Polish (weeks 8-10)
1. `improve/techniques/constitutional.py` — self-critique + retrain
2. Research browser: index relevant papers per failure type
3. README, docs, contribution guide
4. Blog post: "Introducing Trust Bench"

## Content strategy

Each phase produces one blog post on portfolio site + Substack. Twitter threads from key findings. LinkedIn posts from the "why this matters" angle.

Portfolio site gets a new project entry under `trust-research/` branch. README links to the blog posts. Blog posts link back to the repo. Everything cross-linked.

## Research references

- TrustLLM (ICML 2024): https://github.com/HowieHwong/TrustLLM
- Anthropic alignment faking: https://arxiv.org/abs/2412.14093
- Anthropic Constitutional AI: https://arxiv.org/abs/2212.08073
- Anthropic circuit tracing: https://transformer-circuits.pub/
- Scaling monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- SafeWork-R1 / SafeLadder: https://ai45.shlab.org.cn/research/posts/safework-r1/
- Calibration in biomedical NLP: https://pmc.ncbi.nlm.nih.gov/articles/PMC12249208/
- Safety-aware fine-tuning: https://www.emergentmind.com/topics/safety-aware-llm-fine-tuning
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- DeepEval: https://github.com/confident-ai/deepeval
- Raschka Qwen3.5 from scratch: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/16_qwen3.5
- Karpathy microGPT: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- HalluLens (ACL 2025): https://arxiv.org/abs/2504.17550
- Anthropic Fellows Program: https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/
