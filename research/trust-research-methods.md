# Building a Transformer-Based LLM from Scratch

A learning guide using Qwen3.5-0.8B as the case study.

This document teaches every component of a modern transformer LLM deeply enough that you could write one from scratch without referencing existing code. It follows the actual architecture of Qwen3.5-0.8B, a hybrid model that combines full softmax attention with linear attention (Gated DeltaNet), representing the state of the art as of early 2026.

**Qwen3.5-0.8B key config values** (referenced throughout):

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 1536 |
| `num_hidden_layers` | 24 |
| `vocab_size` | 151,936 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 (for full attention) |
| `intermediate_size` | 8960 |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 10,000,000 |
| `partial_rotary_factor` | 0.25 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 16 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| Layer pattern | 3 linear : 1 full (18 DeltaNet + 6 GQA) |

---

## 1. The Basics: What Is a Transformer, Really?

### The core insight: attention as learned retrieval

A transformer is, at its heart, a learned retrieval system. Given a sequence of tokens, each token asks a question ("query"), every other token advertises what it contains ("key"), and offers content to retrieve ("value"). The dot product between a query and a key measures relevance, and the output is a relevance-weighted sum of values. This is exactly how a database lookup works, except every component is learned.

The original insight from Vaswani et al. (2017) was that this mechanism, combined with residual connections and normalization, is sufficient to model arbitrary sequence-to-sequence transformations without any recurrence or convolution.

### The residual stream view

A useful mental model (due to Elhage et al., 2021) is the **residual stream**: there is a single vector per token position that flows through the entire model. Each layer reads from this stream and writes an additive update back into it. Attention layers route information between positions. Feed-forward layers transform information within each position. The final output is the sum of the original embedding plus all layer contributions.

In code (from Raschka's GPT implementation):

```python
shortcut = x
x = self.norm1(x)
x = self.att(x)       # attention reads and writes
x = x + shortcut      # additive update to residual stream

shortcut = x
x = self.norm2(x)
x = self.ff(x)        # FFN reads and writes
x = x + shortcut      # additive update to residual stream
```

Without residual connections, gradients vanish in deep networks, and individual layers cannot specialize since they must preserve all information in their output.

### Why layer norm matters

Deep networks suffer from internal covariate shift: the distribution of activations drifts during training, making optimization unstable. Normalization constrains activations to a predictable range before each sublayer processes them. Remove normalization, and training diverges in most architectures deeper than a few layers.

### Why position encoding is needed

Self-attention is **permutation invariant**: shuffling the input tokens and correspondingly shuffling the output gives the same result. But language is inherently sequential, so position must be injected explicitly. Without position encoding, the model cannot distinguish "the cat sat on the mat" from "mat the on sat cat the."

**Source**: Vaswani et al., "Attention Is All You Need" (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 2. Tokenization and Embeddings

### How text becomes numbers

Raw text is split into **tokens** using Byte Pair Encoding (BPE). BPE starts with individual bytes and iteratively merges the most frequent adjacent pairs into new tokens. The result is a fixed vocabulary (Qwen3.5 uses 151,936 tokens) where common words are single tokens and rare words are split into subword pieces.

The tokenizer converts a string into a sequence of integer IDs, each indexing into the vocabulary. This is a deterministic, non-learned step.

### What the embedding matrix does

The embedding matrix is a lookup table of shape `(vocab_size, hidden_size)`, so `(151936, 1536)` in Qwen3.5-0.8B. Each row is a learned 1536-dimensional vector representing one token. Looking up token ID 42 means selecting row 42.

```python
self.tok_emb = nn.Embedding(vocab_size, hidden_size)  # (151936, 1536)
tok_embeds = self.tok_emb(input_ids)  # (batch, seq_len, 1536)
```

These vectors are learned during training. Initially random, they converge to encode semantic and syntactic properties: similar words end up with similar vectors.

### Embedding dimension

The `hidden_size` of 1536 is the width of the residual stream. Every layer operates on vectors of this size. Larger dimensions give the model more capacity to represent distinctions but cost more compute and memory. The dimension must be divisible by the number of attention heads so each head gets an equal slice.

---

## 3. Attention: From First Principles to GQA

### Single-head attention

Given input `X` of shape `(seq_len, d_model)`, we project into three spaces:

```
Q = X W_Q    # queries:  (seq_len, d_k)
K = X W_K    # keys:     (seq_len, d_k)
V = X W_V    # values:   (seq_len, d_v)
```

The attention output is:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**Deriving each term:**

1. `Q K^T` produces a `(seq_len, seq_len)` matrix where entry (i, j) is the dot product between query i and key j, measuring how much token i should attend to token j.
2. `/ sqrt(d_k)` is critical. Without scaling, for large `d_k`, the dot products have variance proportional to `d_k`, pushing softmax into extreme regions where gradients vanish. Dividing by `sqrt(d_k)` keeps the variance at 1 regardless of dimension.
3. `softmax` converts raw scores to a probability distribution over positions. Each row sums to 1, so the output is a proper weighted average.
4. Multiplying by `V` produces the output: a weighted sum of value vectors, where the weights come from the attention scores.

### Multi-head attention

Instead of one set of Q, K, V projections, we use `h` heads, each with dimension `d_k = d_model / h`. Each head can specialize: one might track syntactic relationships, another coreference, another positional patterns.

In Qwen3.5's full attention layers: 8 heads with dimension 256 each, giving `8 * 256 = 2048`. Note this exceeds `hidden_size` (1536), meaning the head dimension is larger than a simple split, and the attention operates in a higher-dimensional space before projecting back.

```python
# Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
# Transpose: -> (batch, num_heads, seq_len, head_dim)
# Now each head computes attention independently
attn_scores = queries @ keys.transpose(-2, -1)  # (batch, heads, seq, seq)
```

### Grouped Query Attention (GQA)

Standard multi-head attention (MHA) has separate K, V projections per head. This means the KV cache during inference grows as `num_heads * head_dim * seq_len * 2`, which is expensive.

GQA (Ainslie et al., 2023) shares K and V heads across groups of query heads. Qwen3.5-0.8B uses 8 query heads but only 2 KV heads, a 4:1 ratio. Each KV head serves 4 query heads.

```python
# 2 KV heads, each shared by 4 query heads
# KV cache size reduced by 4x vs MHA
# Quality is very close to MHA (within 0.5% on most benchmarks)
```

The tradeoff: 4x less KV cache memory for a negligible quality hit. This is why GQA has become standard in Llama 2/3, Mistral, and Qwen.

### Causal masking

For autoregressive generation, token i must not attend to tokens j > i (future tokens). We enforce this with a causal mask: an upper-triangular matrix of negative infinity values added to the attention scores before softmax.

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
attn_scores.masked_fill_(mask.bool(), -torch.inf)
# After softmax, masked positions become 0
```

### KV cache

During autoregressive generation, we produce one token at a time. Without caching, generating token t requires recomputing K and V for all t-1 previous tokens. The KV cache stores the K, V tensors from previous steps and appends the new token's K, V, avoiding redundant computation.

Memory cost: `2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param`. For Qwen3.5-0.8B's full attention layers (6 layers with GQA), this is manageable. The DeltaNet layers avoid this entirely with fixed-size state matrices.

**Source**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

---

## 4. Position Encoding: RoPE

### Why absolute position embeddings are limited

The original transformer adds a learned or sinusoidal vector to each position. The problem: the model sees positions as fixed identities, not relative distances. It cannot generalize to sequences longer than training length, and it encodes "position 5" rather than "3 tokens apart."

### Rotary embeddings: encoding position through rotation

RoPE (Su et al., 2021) encodes position by rotating query and key vectors in 2D subspaces. The key insight: if you rotate Q and K vectors by angles proportional to their positions, the dot product `Q_m . K_n` naturally depends on the relative position `m - n`, not the absolute positions.

### The math

Split each head's dimension into pairs `(x_0, x_1), (x_2, x_3), ...`. For each pair at position `m`, apply a 2D rotation:

```
[cos(m*theta_i)  -sin(m*theta_i)] [x_{2i}  ]
[sin(m*theta_i)   cos(m*theta_i)] [x_{2i+1}]
```

Where `theta_i = 1 / (base^(2i/d))` and `base` is `rope_theta`.

**Why this gives relative position**: The dot product of a rotated query at position m and a rotated key at position n equals the dot product of the original vectors rotated by `(m-n) * theta_i`. Rotation is a group action, and the difference of rotations is a rotation by the difference.

**Complex number view**: Treating each pair `(x_{2i}, x_{2i+1})` as a complex number `z = x_{2i} + i * x_{2i+1}`, RoPE is simply multiplication by `e^{i*m*theta_i}`. The dot product of `z_q * e^{i*m*theta}` and `z_k * e^{i*n*theta}` depends on `e^{i*(m-n)*theta}`.

### Partial rotary factor

Qwen3.5 only rotates 25% of dimensions (`partial_rotary_factor = 0.25`). With head dimension 256, only 64 dimensions receive rotary encoding. The remaining 192 dimensions are position-independent.

Why? Not all dimensions need position information. Leaving most dimensions unrotated allows the model to use them for position-invariant semantic features. This is especially important in the multimodal setting where different modalities have different spatial structures.

### Multi-dimensional RoPE (M-RoPE)

The 64 rotated dimensions are split into three sections: `[11, 11, 10]` pairs (totaling 32 pairs = 64 dimensions). Each section encodes a different spatial dimension (height, width, time), enabling the model to handle vision tokens (which have 2D spatial position) alongside text tokens (which have 1D sequential position).

### rope_theta and context length

`rope_theta = 10,000,000` (vs the original 10,000). A higher base stretches the rotation frequencies, making the lowest-frequency rotations cycle over much longer periods. This effectively extends the context length the model can handle: with theta = 10K, attention patterns degrade beyond ~4K tokens; with theta = 10M, the model supports 262K+ tokens.

**Source**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## 5. Feed-Forward Networks (MLP)

### The role of FFN layers

While attention routes information between token positions, the feed-forward network processes each position independently. Research suggests FFN layers function as key-value memories (Geva et al., 2021): the first projection's rows are "keys" that pattern-match against the input, and the second projection's columns are "values" that write factual knowledge back into the residual stream.

### SwiGLU activation

Qwen3.5 uses SwiGLU (Shazeer, 2020), which replaces the standard two-layer FFN:

```
Standard:  FFN(x) = W_2 * GELU(W_1 * x)
SwiGLU:    FFN(x) = W_down * (SiLU(W_gate * x) . (W_up * x))
```

Where `.` is element-wise multiplication and `SiLU(x) = x * sigmoid(x)`.

Three projections instead of two:
- **Gate projection** `W_gate`: `(hidden_size, intermediate_size)` = `(1536, 8960)`. Produces the gating signal.
- **Up projection** `W_up`: same shape. Produces the value to be gated.
- **Down projection** `W_down`: `(intermediate_size, hidden_size)` = `(8960, 1536)`. Projects back.

The gating mechanism allows the network to selectively pass or block information, providing more expressive power than a simple nonlinearity. The intermediate dimension of 8960 gives an expansion ratio of approximately 5.8x (vs the traditional 4x), which compensates for the three-way split.

**Source**: Shazeer, "GLU Variants Improve Transformer" (2020). [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

---

## 6. Normalization: RMSNorm

### Why normalization is needed

Without normalization, activations in deep networks can grow or shrink exponentially across layers. This causes gradient explosion/vanishing and makes training unstable. Normalization constrains each layer's input to a predictable scale.

### LayerNorm vs RMSNorm

**LayerNorm** normalizes by subtracting the mean and dividing by the standard deviation:

```
LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta
```

**RMSNorm** drops the mean-centering and bias, normalizing only by the root mean square:

```
RMSNorm(x) = x / RMS(x) * gamma
where RMS(x) = sqrt(mean(x^2) + eps)
```

In code:

```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states  # gamma (learnable scale)
```

RMSNorm is simpler (no mean computation, no bias term), roughly 10% faster, and empirically performs comparably. Qwen3.5 uses `eps = 1e-6`.

### Pre-norm vs post-norm

The original transformer applied normalization after the residual addition (post-norm). Modern architectures apply it before the sublayer (pre-norm):

```
Post-norm:  x = LayerNorm(x + Sublayer(x))
Pre-norm:   x = x + Sublayer(RMSNorm(x))
```

Pre-norm produces better training stability because the residual connection carries unnormalized activations, providing a direct gradient path. Post-norm architectures require careful learning rate warmup and are harder to train at scale.

---

## 7. Linear Attention: The DeltaNet Revolution

This is the most novel component of Qwen3.5, and the most important to understand deeply.

### Why standard attention is expensive

Standard softmax attention computes an `(n, n)` attention matrix, costing O(n^2 d) in compute and O(n^2) in memory. For a 262K context, this matrix has 68 billion entries per layer per head. This is prohibitive.

### The recurrent view of attention

Linear attention (Katharopoulos et al., 2020) removes the softmax, enabling a recurrent reformulation. Without softmax:

```
O = (Q K^T) V = Q (K^T V)
```

By associativity, we can compute `K^T V` first (a `d_k x d_v` matrix), then multiply by Q. This changes the complexity from O(n^2 d) to O(n d^2).

Even better, we can process tokens one at a time with a **state matrix** `S_t`:

```
S_t = S_{t-1} + v_t k_t^T     # accumulate key-value associations
o_t = S_t q_t                  # retrieve using query
```

This is O(1) per token (excluding the d^2 state update), perfect for inference.

### Why vanilla linear attention fails

The state matrix accumulates everything and forgets nothing. Every key-value pair ever seen contributes equally. Old, irrelevant associations contaminate retrieval. On the Multi-Query Associative Recall (MQAR) benchmark, vanilla linear attention scores near zero where softmax attention scores 100%.

### The delta rule: error-correcting updates

DeltaNet replaces blind accumulation with gradient descent on a per-token associative memory. Consider the state matrix S as mapping keys to values. At each step, we want S to map the current key k_t to the current value v_t. The error is:

```
error_t = S_{t-1} k_t - v_t    # predicted value minus target value
```

We define a per-step loss:

```
L_t(S) = 0.5 * ||S k_t - v_t||^2
```

Taking the gradient with respect to S:

```
dL/dS = (S k_t - v_t) k_t^T
```

The update rule (gradient descent with learning rate beta_t):

```
S_t = S_{t-1} - beta_t * (S_{t-1} k_t - v_t) k_t^T
```

Rearranging:

```
S_t = S_{t-1} - beta_t * (S_{t-1} k_t) k_t^T + beta_t * v_t k_t^T
```

When `beta_t = 1`, the old association for key k_t is completely erased and replaced with v_t. When `beta_t = 0`, nothing changes. The model learns beta_t per step, giving it fine-grained control over memory updates.

This is what makes DeltaNet an **error-correcting associative memory** rather than a blind accumulator. It achieves 100% on MQAR.

### Gating from Mamba2

Gated DeltaNet adds exponential decay gating:

```
S_t = G_t * S_{t-1} - beta_t * (G_t * S_{t-1} k_t - v_t) k_t^T
```

where `G_t = exp(g_t)` and g_t is derived from:

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

`A_log` is a learned parameter (initialized uniformly in [0, 16]) whose exponential gives a decay rate. `softplus(a + dt_bias)` is a learned, input-dependent modulation. The result is a per-head, per-timestep decay factor.

This serves two purposes:
1. **Forgetting**: When G_t is small, old state is rapidly erased. The model can "clear its memory" when context shifts.
2. **Eliminating attention sinks**: Without decay, certain tokens accumulate disproportionate influence. Gating prevents this.

### Conv1D for local context

Linear attention layers use a causal Conv1D (kernel size 4) applied to the concatenated Q, K, V projections before splitting them:

```python
self.conv1d = nn.Conv1d(
    in_channels=self.conv_dim,  # key_dim*2 + value_dim
    out_channels=self.conv_dim,
    kernel_size=4,
    groups=self.conv_dim,       # depthwise: each channel independently
    padding=3,                  # causal padding
)
```

This provides local positional context (a window of 4 tokens) without requiring RoPE. It replaces positional encoding entirely in the linear attention layers. The depthwise convolution is efficient: each channel is convolved independently.

### L2 normalization on Q/K

Instead of the softmax-based normalization in standard attention, DeltaNet L2-normalizes queries and keys:

```python
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
```

This constrains dot products to the range [-1, 1], preventing the state matrix from growing unboundedly.

### Why DeltaNet layers do not need a KV cache

Full attention layers store past K and V vectors (growing linearly with sequence length). DeltaNet layers maintain a fixed-size state matrix S of shape `(num_heads, key_dim, value_dim)` = `(16, 128, 128)` in Qwen3.5-0.8B. This is 16 * 128 * 128 = 262,144 parameters per layer, constant regardless of sequence length. For 18 DeltaNet layers, total state is about 18 MB in fp32, far less than the KV cache of equivalent softmax layers on long sequences.

### The intuition

Think of DeltaNet as a fast-weight associative memory. The state matrix S is a dictionary that maps keys to values. Unlike a hash table, this dictionary is dense and approximate. The delta rule does gradient descent on this dictionary at every token, correcting errors. Gating lets the model decay old entries. The result is a memory system that can store, retrieve, update, and forget, all in O(1) per token.

**Source**: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024). [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

---

## 8. Hybrid Architecture: Putting It All Together

### The 3:1 pattern

Qwen3.5-0.8B alternates 3 DeltaNet layers with 1 full attention (GQA) layer:

```
Layers 0, 1, 2:    DeltaNet (linear attention)
Layer 3:            GQA (full softmax attention)
Layers 4, 5, 6:    DeltaNet
Layer 7:            GQA
...
Layers 20, 21, 22: DeltaNet
Layer 23:           GQA
```

18 DeltaNet layers + 6 GQA layers = 24 total.

### What each type contributes

**DeltaNet layers** handle efficient local processing and sequential pattern recognition. Their O(1) per-token inference cost makes them ideal for long contexts. The fixed-size state acts as a compressed, updateable memory. They handle the bulk of language modeling but struggle with tasks requiring precise retrieval over long distances.

**Full attention layers** provide global retrieval. Every token can attend to every other token with exact dot-product scoring. These layers "clean up" the representations every 4 layers, ensuring the model maintains precise long-range dependencies.

### The full forward pass

```
Input: token_ids (batch, seq_len)
    |
    v
Embedding lookup: (batch, seq_len, 1536)
    |
    v
For layer_idx in range(24):
    |
    +-- RMSNorm
    |
    +-- if layer_type[layer_idx] == "linear_attention":
    |       Gated DeltaNet (Conv1D -> Q,K,V split -> delta rule -> gated norm -> out_proj)
    |   else:
    |       GQA (Q,K,V projections -> RoPE -> grouped attention -> out_proj)
    |
    +-- Residual add
    |
    +-- RMSNorm
    |
    +-- SwiGLU MLP (gate_proj, up_proj -> SiLU gate -> down_proj)
    |
    +-- Residual add
    |
    v
Final RMSNorm
    |
    v
Output projection: (batch, seq_len, 151936) -> logits
```

### Weight tying

The output projection matrix (mapping 1536 -> 151,936 for vocabulary logits) shares its weights with the embedding matrix. This halves the parameter cost of the vocabulary-related weights and enforces a useful constraint: tokens with similar embeddings should have similar output probabilities. In code:

```python
self.out_head.weight = self.tok_emb.weight  # shared
```

---

## 9. Training Fundamentals

### Cross-entropy loss for next-token prediction

The training objective is next-token prediction: given tokens `[t_0, t_1, ..., t_{n-1}]`, predict `[t_1, t_2, ..., t_n]`. The loss is the cross-entropy between the predicted probability distribution over the vocabulary and the one-hot target:

```
L = -sum(y_true * log(softmax(logits)))
```

For a single position with correct token c, this simplifies to `-log(p_c)`, the negative log-probability of the correct token.

### AdamW optimizer

Adam tracks exponential moving averages of the gradient (first moment) and squared gradient (second moment) to adapt the learning rate per parameter. **AdamW** decouples weight decay from the gradient update: instead of adding an L2 penalty to the loss (which interacts poorly with Adam's adaptive rates), it directly shrinks weights each step:

```
theta_t = theta_{t-1} - lr * (adam_update + weight_decay * theta_{t-1})
```

This prevents large weights from accumulating and improves generalization.

### Learning rate schedules

Training typically uses:
1. **Linear warmup**: LR ramps from 0 to peak over the first ~2000 steps, preventing early instability.
2. **Cosine decay**: LR decreases following a cosine curve from peak to near-zero over the remaining steps. This provides a natural annealing schedule.

### Gradient accumulation

Large batch sizes improve training stability but exceed GPU memory. Gradient accumulation simulates large batches by accumulating gradients over N forward passes before applying one optimizer step. Effective batch size = micro-batch size * accumulation steps * num GPUs.

### Mixed precision training

Modern training uses bf16 (bfloat16) for most operations: same exponent range as fp32 but half the memory and 2x the throughput on modern GPUs. Critical operations (loss computation, normalization) remain in fp32 for numerical stability. The Qwen3.5 DeltaNet implementation explicitly casts to fp32 for the delta rule computation:

```python
query, key, value, beta, g = [
    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
]
```

---

## 10. From Weights to Inference

### Loading pretrained weights

HuggingFace checkpoints store weights with specific naming conventions. The mapping from checkpoint names to your architecture involves matching parameter names:

```
model.embed_tokens.weight          -> token embedding
model.layers.{i}.self_attn.*       -> attention sublayer
model.layers.{i}.mlp.*             -> FFN sublayer
model.layers.{i}.input_layernorm.* -> pre-attention RMSNorm
model.layers.{i}.post_attention_layernorm.* -> pre-FFN RMSNorm
model.norm.weight                  -> final RMSNorm
lm_head.weight                     -> output projection (or tied to embed_tokens)
```

For Qwen3.5, the layer type determines whether `self_attn` contains GQA parameters or DeltaNet parameters (with Conv1D, gating projections, etc.).

### Sampling strategies

Given output logits (151,936 raw scores), generation requires converting them to a token choice:

**Temperature**: Divide logits by T before softmax. T < 1 sharpens the distribution (more deterministic). T > 1 flattens it (more random). T = 0 is equivalent to greedy decoding (argmax).

**Top-k**: Zero out all logits except the k highest, then renormalize. This prevents sampling low-probability garbage tokens.

**Top-p (nucleus sampling)**: Sort tokens by probability, keep the smallest set whose cumulative probability exceeds p. This adapts the number of candidates to the model's confidence: when the model is sure, few tokens are considered; when uncertain, many are.

### The autoregressive loop

Generation is iterative. Starting from a prompt:

```python
for _ in range(max_new_tokens):
    logits = model(input_ids)          # forward pass
    next_logits = logits[:, -1, :]     # take last position
    next_token = sample(next_logits)   # temperature + top-k/top-p
    input_ids = cat(input_ids, next_token)  # append and repeat
```

### KV cache management

For the 6 GQA layers, past K and V tensors are cached and extended each step. For the 18 DeltaNet layers, the fixed-size state matrix and Conv1D state are updated in place. The DeltaNet inference path uses the recurrent formulation:

```python
# Single-token inference (recurrent mode)
last_recurrent_state = last_recurrent_state * g_t      # decay
kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # retrieve
delta = (v_t - kv_mem) * beta_t                        # error
last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # update
output = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)  # read
```

This processes each new token in O(d^2) per head, with no growing cache.

---

## Summary: What Would Break If You Removed Each Component

| Component | What breaks without it |
|-----------|----------------------|
| Residual connections | Gradients vanish, deep layers learn nothing |
| RMSNorm | Training diverges from activation scale drift |
| RoPE | Model cannot distinguish token order |
| Causal mask | Model cheats by reading future tokens |
| Multi-head attention | Single head bottleneck, poor specialization |
| GQA | KV cache 4x larger, slower inference |
| SwiGLU | Lower quality than gated activation (measurable on benchmarks) |
| DeltaNet delta rule | Falls back to vanilla linear attention, fails at retrieval |
| DeltaNet gating | Attention sinks, inability to forget old context |
| Conv1D in DeltaNet | No local positional information in linear layers |
| Weight tying | 233M extra parameters, less regularization |
| KV cache | Inference is O(n^2) per generated token instead of O(n) |

---

## Sources

- Vaswani et al., "Attention Is All You Need" (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024). [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Shazeer, "GLU Variants Improve Transformer" (2020). [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021). Anthropic.
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021). [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)
- Raschka, "Build a Large Language Model From Scratch" (Manning, 2024). [GitHub](https://github.com/rasbt/LLMs-from-scratch)
- Qwen3.5-0.8B model config: [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-0.8B)
