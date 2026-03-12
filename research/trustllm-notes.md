# TrustLLM Reference Notes

Source: https://github.com/HowieHwong/TrustLLM
Package: `trustllm_pkg/trustllm/task/`

## Evaluation Dimensions (6 total)

| Dimension | Module | Key Subtasks |
|-----------|--------|-------------|
| **Truthfulness** | truthfulness.py | External knowledge (fact-checking), sycophancy, hallucination, internal knowledge (QA) |
| **Safety** | safety.py | Jailbreak resistance, toxicity, misuse, exaggerated safety |
| **Fairness** | fairness.py | Stereotype agreement, preference bias, disparagement |
| **Robustness** | robustness.py | AdvGLUE adversarial NLU benchmark |
| **Privacy** | privacy.py | ConfAIDe privacy awareness scoring |
| **Ethics** | ethics.py | Explicit moral choice, implicit ethics (SocialChemistry101, ETHICS dataset), emotional awareness |

## Truthfulness Scorer - How It Works

`TruthfulnessEval` has four evaluation modes:

### 1. External Knowledge (`external_eval`)
- Tests fact-checking against claims from 4 domains: **climate, healthver, covid, scifact**
- Model outputs parsed for SUPPORT/REFUTE classification
- Metric: **macro F1** via sklearn `classification_report`
- Simple string matching to extract predictions ("Answer: SUPPORT/REFUTE")

### 2. Sycophancy (`sycophancy_eval`)
- **Persona sycophancy**: Compares model response embedding to sycophantic vs. non-sycophantic completions using cosine similarity (`metrics.sycophancy_sim`)
- **Preference sycophancy**: Pairs responses, uses GPT auto-eval to check if model flip-flops. Metric: proportion of "no" (non-sycophantic) responses

### 3. Hallucination (`hallucination_eval`)
- Three subtasks: **halu_qa, halu_summ, halu_dial** - binary yes/no accuracy
- One multiple-choice subtask (**mc**) - accuracy on A/B/C/D answers
- Simple string parsing, no LLM-as-judge for these

### 4. Internal Knowledge (`internal_eval`)
- Datasets: **CODAH** (commonsense, regex digit extraction), **SQuAD** (GPT auto-eval), **adversarial QA** (GPT auto-eval), **HotpotQA** (GPT auto-eval)
- `advfact_eval`: Checks if model corrects factual errors, scored by GPT judge looking for [CORRECTED] tag

## Safety Scorer - How It Works

`SafetyEval` has four evaluation modes:

### 1. Jailbreak (`jailbreak_eval`)
- Uses **Longformer classifier** (HuggingFace model) to detect if response complied with jailbreak attempt
- Metric: **RtA (Refuse to Answer)** rate - higher is safer
- Two modes: `total` (single RtA score) or `single` (per-attack-label breakdown)

### 2. Toxicity (`toxicity_eval`)
- Only evaluates responses that Longformer classified as LABEL_1 (non-refusal)
- Uses **Google Perspective API** for toxicity scoring
- Metric: average and max toxicity scores

### 3. Misuse (`misuse_eval`)
- Same Longformer classifier as jailbreak
- Metric: RtA score

### 4. Exaggerated Safety (`exaggerated_eval`)
- Same Longformer classifier
- Metric: **1 - RtA** (inverted, because over-refusing safe prompts is the failure mode)

## Key Dependencies / External Services

- **GPT auto-eval** (`gpt_auto_eval.AutoEvaluator`): Used as LLM-judge for sycophancy preference, SQuAD, adversarial QA, HotpotQA, advfact
- **Longformer** (`longformer.HuggingFaceEvaluator`): Binary classifier for jailbreak/misuse/exaggerated safety detection
- **Perspective API** (`perspective.PerspectiveEval`): Google's toxicity scorer
- **Embedder** (`embedder.DataEmbedder`): Embedding model for sycophancy persona similarity
- **sklearn**: classification_report for F1 computation

## What We Can Wrap vs. Build Custom

### Can wrap/adapt directly
- **Metric functions**: RtA, toxicity calculation, F1 scoring are all straightforward utilities
- **Evaluation structure**: Their task-based eval pattern (load data, run model, score) maps cleanly to a pipeline
- **Hallucination eval**: Simple string matching, easy to reuse or reimplement
- **External knowledge eval**: Standard fact-checking F1, portable

### Need to build custom
- **Longformer dependency**: Their safety scoring is tightly coupled to a specific HF classifier. We should abstract this behind a pluggable classifier interface so we can swap in other safety classifiers (LlamaGuard, Aegis, etc.)
- **GPT-as-judge**: They hardcode OpenAI's GPT for auto-eval. Trust Bench should support any judge model (local or API) with configurable prompts
- **Perspective API**: Paid/rate-limited external service. Need a local toxicity alternative (detoxify, HF toxicity models) with Perspective as optional
- **Dataset coupling**: Their eval expects specific JSON schemas tied to their dataset downloads. We need a dataset adapter layer
- **Scoring aggregation**: Their averaging is naive (unweighted mean across subtasks). We should support weighted scoring and confidence intervals
- **Missing dimensions for Trust Bench**: They don't cover calibration/uncertainty, instruction following fidelity, or multi-turn consistency, which we likely want
