# Qwen3.5 Hybrid Attention Architecture

Research notes for Trust Bench. Last updated: 2026-03-08.

## Overview

Qwen3.5 is Alibaba's production-scale LLM series that replaces the uniform softmax attention stack with a **hybrid attention architecture**: a repeating 3:1 pattern of Gated DeltaNet (linear attention) and full attention (GQA with softmax). This is the first major open model family to deploy linear attention variants at scale.

- Total parameters: up to 397B (MoE), 17B active per token
- Context window: up to 1M tokens (262K in smaller models)
- Modality: natively multimodal (vision + language)
- License: Apache 2.0
- Predecessor: Qwen3-Next (September 2025) previewed this direction

---

## 1. Hybrid Attention Design: The 3:1 Pattern

The `layer_types` field in the model config defines the interleaving:

```
[linear_attention, linear_attention, linear_attention, full_attention,
 linear_attention, linear_attention, linear_attention, full_attention,
 ...]
```

- **75% of layers** use Gated DeltaNet (linear attention, O(n) per token)
- **25% of layers** use full softmax attention with GQA (O(n^2) but provides global context)

This pattern repeats uniformly. For the 0.8B model (24 layers), there are 6 full-attention layers and 18 linear-attention layers.

**Why this ratio?** Full attention layers provide strong retrieval and global context modeling. Linear layers provide efficient O(1)-per-token inference and scale near-linearly with sequence length. The periodic full attention prevents the quality degradation seen in pure linear attention models, while the linear layers keep compute tractable for long contexts.

---

## 2. Gated DeltaNet Mechanism

Source: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang, Kautz, Hatamizadeh, 2024). Accepted at ICLR 2025.

### From softmax attention to linear attention

Standard softmax attention computes:

```
O = softmax(QK^T . M) V       # O(L^2 d) complexity
```

Linear attention removes the softmax, enabling an equivalent recurrent form:

```
S_t = S_{t-1} + v_t k_t^T     # state accumulation
o_t = S_t q_t                  # output = state x query
```

This gives O(Ld^2) inference complexity, but **blind accumulation** means the state matrix grows without correction, leading to poor retrieval performance.

### The delta rule (error-correcting updates)

DeltaNet replaces blind accumulation with gradient descent on a per-step MSE loss:

```
Loss: L_t(S) = 0.5 * ||S k_t - v_t||^2

Update: S_t = S_{t-1} - beta_t * (S_{t-1} k_t - v_t) k_t^T
```

Where:
- `S_t` is the d x d state matrix (acts as a fast-weight memory)
- `beta_t` is a learned per-step learning rate
- `S_{t-1} k_t` is the **predicted** value for key k_t
- `v_t` is the **target** value
- The update corrects the state based on the prediction error (the "delta")

When beta_t = 0, memory is unchanged. When beta_t = 1, the old value for that key is completely replaced. This gives DeltaNet precise, targeted memory modification rather than the blind additive accumulation of vanilla linear attention.

### Gating (from Mamba2)

Gated DeltaNet adds exponential gating for adaptive memory decay:

```
S_t = G_t . S_{t-1} + v_t k_t^T    # general gated form
```

The gating mechanism serves two purposes:
1. **Rapid memory erasure** when information becomes irrelevant
2. **Eliminating attention sinks** (massive activations on specific tokens)

### Additional components

- **Causal Conv1D** (kernel size 4): provides local context, replaces positional encoding in linear attention layers
- **L2 normalization on Q/K**: replaces softmax normalization
- **No KV cache growth**: DeltaNet layers maintain a fixed-size d x d state matrix instead of a growing KV cache

### Key advantage over vanilla linear attention

DeltaNet achieves 100% on MQAR (Multi-Query Associative Recall) and in-context retrieval benchmarks where vanilla linear attention fails. The delta rule makes it a proper error-correcting associative memory.

---

## 3. GQA (Grouped Query Attention)

The full attention layers (every 4th layer) use standard softmax attention with **Grouped Query Attention**:

- Multiple query heads share fewer key-value heads
- 0.8B config: 8 attention heads, 2 KV heads (4:1 GQA ratio)
- Head dimension: 256
- No attention bias, no attention dropout

GQA reduces the KV cache size proportional to the grouping ratio while maintaining near-MHA quality. This is the same mechanism used in Llama 2/3, Mistral, and Qwen3.

---

## 4. RoPE (Rotary Position Embeddings)

Qwen3.5 uses multi-dimensional RoPE with notable differences from standard implementations:

```json
{
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10]
}
```

- **Partial rotation**: only 25% of dimensions are rotated (partial_rotary_factor = 0.25), leaving 75% position-independent
- **Multi-dimensional RoPE (M-RoPE)**: splits rotary dimensions into 3 sections [11, 11, 10] for spatial/temporal awareness (supports multimodal inputs with height, width, time)
- **High theta**: 10M (vs standard 10K), extending effective context length
- **Applied only to full attention layers** (linear attention layers use Conv1D for local position information instead)

---

## 5. RMSNorm

Standard pre-norm architecture. RMSNorm is applied before both attention and MLP sublayers:

```
rms_norm_eps: 1e-6
```

RMSNorm (Root Mean Square Layer Normalization) normalizes by the RMS of activations without centering, which is computationally cheaper than LayerNorm. This is consistent with Qwen3 and most modern LLM architectures.

---

## 6. Linear Attention Layer Config

The linear attention (Gated DeltaNet) layers have their own head configuration:

```json
{
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_conv_kernel_dim": 4
}
```

Note: the linear layers use **more heads** (16) with **smaller head dim** (128) compared to the full attention layers (8 heads, 256 dim). Both result in the same total dimension but partition it differently. The Conv1D kernel size of 4 provides local context within the linear attention layers.

---

## 7. Key Differences from Standard Transformers

| Aspect | Standard Transformer | Qwen3.5 |
|--------|---------------------|---------|
| Attention | Uniform softmax attention | 3:1 hybrid (DeltaNet + GQA) |
| Complexity | O(n^2) all layers | O(n) for 75% of layers |
| KV cache | Grows linearly with seq len | Fixed-size state for 75% of layers |
| Memory update | Blind accumulation | Error-correcting delta rule |
| Position encoding | RoPE everywhere | RoPE on full attn, Conv1D on linear |
| State representation | Q, K, V per token | d x d state matrix (linear layers) |
| Memory control | None (all info retained) | Gated decay + targeted updates |

---

## 8. Why This Matters for Trust Signal Extraction

Different attention mechanisms encode information differently, which has direct implications for probing trust-relevant signals from model internals:

1. **Dual representation pathways**: Trust signals (uncertainty, hedging, confidence calibration) may be encoded differently in linear vs. full attention layers. The fixed-size state matrix in DeltaNet layers compresses information differently than the explicit token-by-token attention pattern in full attention layers. Probing strategies need to account for both.

2. **Attention pattern analysis limitations**: Traditional attention-based interpretability (attention head visualization, attention rollout) only works on the 25% of layers using full attention. The 75% using DeltaNet have no explicit attention matrix to inspect. New interpretability approaches targeting the state matrix S_t are needed.

3. **Error-correcting memory may affect trust encoding**: The delta rule means the model actively corrects its internal representations. This could mean trust-relevant signals (e.g., the model recognizing it is uncertain) are encoded more precisely in the state matrix than in vanilla linear attention, but less transparently than in full softmax attention.

4. **Layer-type-aware probing**: Trust Bench probes should be designed to handle hybrid architectures. A probe trained on full attention layer activations may not transfer to DeltaNet layer activations and vice versa. The 3:1 pattern creates a natural experimental setup: compare trust signal quality across layer types within the same model.

5. **Gating as confidence signal**: The learned gating values (beta_t, G_t) in DeltaNet layers could themselves be informative about the model's confidence. High gating (memory erasure) on a token might indicate the model is revising its understanding, which correlates with uncertainty.

6. **KV cache vs. state matrix**: For inference-time trust monitoring, the fixed-size state matrix in DeltaNet layers may require different monitoring strategies than KV-cache-based approaches used for standard transformers.

---

## Sources

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Gated Delta Networks paper: https://arxiv.org/abs/2412.06464
- Qwen3.5 architecture blog (Maxime Labonne): https://huggingface.co/blog/mlabonne/qwen35
- DeltaNet Explained (Songlin Yang): https://sustcsonglin.github.io/blog/2024/deltanet-1/
- Gated DeltaNet walkthrough (Sebastian Raschka): https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md
- Qwen3.5-0.8B config: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3.5-397B-A17B model card: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.5 GitHub: https://github.com/QwenLM/Qwen3.5
