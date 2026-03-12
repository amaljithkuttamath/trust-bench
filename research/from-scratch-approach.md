# The "From Scratch" Approach to Building and Studying LLMs

Research document for Trust Bench. Covers the philosophy, implementation details, and practical
signal-extraction patterns that make from-scratch LLM code valuable for interpretability research.

---

## 1. What "From Scratch" Means

### The core idea

Sebastian Raschka's *Build a Large Language Model (From Scratch)* teaches LLM internals by
reimplementing every architectural component in plain PyTorch. The only dependencies are PyTorch
itself, `huggingface_hub` (to download pretrained weights), and `tokenizers` (for BPE). The model
classes, attention mechanisms, normalization layers, positional encodings, and training loops are
all written out explicitly, with no calls to `transformers.AutoModel`,
`transformers.PreTrainedModel`, or any other high-level wrapper.

The book covers the full lifecycle:

1. Tokenization and data loading
2. Attention mechanism implementation (single-head, multi-head, grouped-query)
3. Full GPT-style model construction from components
4. Pretraining on unlabeled data
5. Finetuning for classification and instruction-following

Everything runs on a laptop CPU. GPU acceleration is optional.

### Why this matters for interpretability

When you load a model through HuggingFace's `AutoModelForCausalLM.from_pretrained()`, the forward
pass is buried inside:

- `modeling_*.py` files with thousands of lines of generalized code
- Flash attention dispatchers that choose kernels at runtime
- Generation mixins that wrap the actual model call
- Config-driven conditional logic for dozens of model variants

The result: you cannot easily add a hook between the QKV projection and the attention score
computation, or inspect the recurrent state of a linear attention layer mid-sequence, or modify
how gating values are computed for a single layer while leaving others untouched.

With from-scratch code, every operation is a line of Python you can read, modify, and instrument.

### How it differs from lm-eval-harness

Tools like EleutherAI's `lm-eval-harness` treat models as black boxes. They send prompts in, read
logits or generated text out, and compute benchmark scores. This is useful for comparing models
but tells you nothing about *why* a model behaves a certain way. Trust Bench needs both:

- Black-box evaluation (does the model give truthful answers?)
- White-box probing (what internal signals correlate with truthful vs. fabricated outputs?)

The from-scratch approach enables the white-box side.

---

## 2. How Raschka's Qwen3.5 Implementation Works

The implementation lives in two files:

- `raschka-qwen35.ipynb` - the main notebook with architecture, weight loading, tokenizer, and generation
- `raschka-qwen35-transformers.py` - helper module containing the `Qwen3_5GatedDeltaNet` linear attention block (adapted from HuggingFace's transformers library)

There is also `raschka-qwen35-plus-kv-cache.ipynb`, which adds KV caching for faster autoregressive generation.

### Architecture overview

Qwen3.5 is a hybrid model that alternates between two types of layers:

- **Full attention layers** (standard grouped-query attention with softmax)
- **Linear attention layers** (Gated DeltaNet, a recurrent linear attention variant)

In the 0.8B model, the pattern across all 24 layers is: 3 linear, 1 full, repeated 6 times.

```
Layer types: [L, L, L, F, L, L, L, F, L, L, L, F, L, L, L, F, L, L, L, F, L, L, L, F]
```

### Key classes and their roles

#### `Qwen3_5Model` (top-level model)

The entry point. Owns the token embedding, all transformer blocks, final RMSNorm, and the
output projection head (which is weight-tied with the embedding).

```python
class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_type, idx)
             for idx, layer_type in enumerate(layer_types)]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # RoPE buffers
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
```

Forward pass:

```python
def forward(self, in_idx):
    x = self.tok_emb(in_idx)
    mask = torch.triu(torch.ones(num_tokens, num_tokens, ...), diagonal=1)
    for block in self.trf_blocks:
        x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
```

#### `TransformerBlock`

A dispatch layer. Based on `layer_type`, it instantiates either a `GroupedQueryAttention` or a
`Qwen3_5GatedDeltaNet` as the token mixer. Both are followed by a `FeedForward` block. Both use
pre-norm (RMSNorm before the sublayer) and residual connections.

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        if self.layer_type == "full_attention":
            x = self.token_mixer(x, mask, cos, sin)
        else:
            x = self.token_mixer(x)  # linear attention ignores mask/RoPE
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x
```

#### `GroupedQueryAttention` (full attention layers)

Standard grouped-query attention with several Qwen3.5-specific details:

- **Gated Q projection**: The query linear layer outputs 2x the normal dimension. The second half
  is used as a sigmoid gate on the attention output. This is a distinctive Qwen3.5 feature.
- **QK normalization**: RMSNorm applied to queries and keys before RoPE.
- **Partial rotary embeddings**: Only 25% of the head dimension gets RoPE (`partial_rotary_factor=0.25`).
- **Grouped KV**: 8 query heads, 2 KV groups (group size = 4).

```python
# Gated Q: project to 2x, split, use second half as gate
q_and_gate = self.W_query(x)  # shape: [b, seq, n_heads, head_dim * 2]
queries, gate = torch.chunk(q_and_gate, 2, dim=-1)

# ... standard attention computation ...
context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)

# Apply the gate
context = context * torch.sigmoid(gate)
return self.out_proj(context)
```

#### `Qwen3_5GatedDeltaNet` (linear attention layers)

This is the most architecturally interesting component for Trust Bench. It implements a gated
delta rule, which is a form of recurrent linear attention with an explicit state matrix.

Key parameters (from `raschka-qwen35-transformers.py`):

| Parameter | Value (0.8B) | Role |
|-----------|-------------|------|
| `linear_num_value_heads` | 16 | Number of value heads |
| `linear_num_key_heads` | 16 | Number of key heads |
| `linear_key_head_dim` | 128 | Key head dimension |
| `linear_value_head_dim` | 128 | Value head dimension |
| `linear_conv_kernel_dim` | 4 | 1D causal conv kernel size |

The forward pass:

1. **Project QKV**: `in_proj_qkv` maps hidden states to concatenated Q, K, V
2. **Causal conv1d**: Applied to QKV jointly (depthwise, kernel=4, with SiLU activation)
3. **Split Q, K, V**: After conv, split along the channel dimension
4. **Compute gating signals**:
   - `beta = sigmoid(in_proj_b(hidden_states))` - write gate (controls how much to write to state)
   - `g = -exp(A_log) * softplus(in_proj_a(hidden_states) + dt_bias)` - decay gate (controls how fast the state forgets)
5. **Run chunked delta rule**: Processes the sequence in chunks of 64, maintaining a recurrent
   state matrix of shape `[batch, heads, key_dim, value_dim]`
6. **Gated output norm**: `Qwen3_5RMSNormGated` normalizes then multiplies by `silu(z)` where
   `z = in_proj_z(hidden_states)` is a separate gating projection
7. **Output projection**: `out_proj` maps back to model dimension

#### `RMSNorm`

Qwen3.5's variant uses `(1 + weight)` scaling with zero-initialized weights, rather than the
standard `weight` scaling with ones-initialized weights. This means at initialization, the norm
has no effect (multiplies by 1.0).

```python
def forward(self, x):
    x_norm = self._norm(x.float())
    x_norm = x_norm * (1.0 + self.weight.float())
    return x_norm.to(dtype=x.dtype)
```

#### `FeedForward`

Standard SwiGLU: gate projection + up projection with SiLU gating, then down projection.

```python
def forward(self, x):
    x_fc1 = self.fc1(x)   # gate_proj
    x_fc2 = self.fc2(x)   # up_proj
    x = F.silu(x_fc1) * x_fc2
    return self.fc3(x)     # down_proj
```

### How weights are loaded

The `load_weights_into_qwen3_5` function maps HuggingFace checkpoint keys to from-scratch model
attributes. It downloads safetensor shards from `Qwen/Qwen3.5-0.8B` via `snapshot_download`,
then does explicit `tensor.copy_()` for each parameter with shape validation.

Key name mappings:

| HuggingFace key | From-scratch attribute |
|----------------|----------------------|
| `model.embed_tokens.weight` | `model.tok_emb.weight` |
| `model.layers.{l}.self_attn.q_proj.weight` | `block.token_mixer.W_query.weight` |
| `model.layers.{l}.linear_attn.in_proj_qkv.weight` | `block.token_mixer.in_proj_qkv.weight` |
| `model.layers.{l}.linear_attn.A_log` | `block.token_mixer.A_log` |
| `model.layers.{l}.mlp.gate_proj.weight` | `block.ff.fc1.weight` |
| `model.norm.weight` | `model.final_norm.weight` |

Weight tying: `out_head.weight` shares the same tensor as `tok_emb.weight` (no separate `lm_head`
in the checkpoint).

### The forward pass flow

```
Input token IDs
    |
    v
tok_emb (Embedding) --> [batch, seq_len, 1024]
    |
    v
For each of 24 TransformerBlocks:
    |-- norm1 (RMSNorm)
    |-- token_mixer:
    |     If full_attention (layers 3, 7, 11, 15, 19, 23):
    |         Q/K/V projections -> QK norm -> RoPE -> GQA softmax attn -> gate -> out_proj
    |     If linear_attention (layers 0-2, 4-6, 8-10, ...):
    |         QKV proj -> causal conv1d -> split Q/K/V -> compute beta, g -> delta rule -> gated norm -> out_proj
    |-- residual add
    |-- norm2 (RMSNorm)
    |-- FeedForward (SwiGLU)
    |-- residual add
    |
    v
final_norm (RMSNorm)
    |
    v
out_head (Linear, weight-tied) --> [batch, seq_len, vocab_size] logits
```

---

## 3. From-Scratch vs. Framework-Based: Tradeoffs

### When from-scratch is better

**Research and interpretability.** If you need to:

- Extract the recurrent state matrix from a specific DeltaNet layer after processing a specific token
- Compare attention weight distributions between "truthful" and "hallucinated" continuations
- Add custom loss terms that depend on intermediate activations
- Modify a single component (e.g., replace the gating function) while keeping everything else identical
- Log every intermediate tensor during a forward pass for post-hoc analysis

From-scratch code makes these operations straightforward. You modify the `forward` method of the
relevant class, or add a few lines around the call site. No need to understand framework
dispatch logic, no risk of accidentally breaking caching or gradient computation in unrelated
components.

**Debugging.** When outputs look wrong, you can set breakpoints anywhere in the forward pass and
inspect shapes, values, dtypes, and NaN propagation directly.

**Education.** The code reads like a specification of the architecture. Every matrix multiply,
every activation function, every reshape is explicit.

### When frameworks are better

**Production inference.** HuggingFace transformers integrates with:

- Flash Attention 2/3 for memory-efficient attention
- vLLM and TGI for batched serving
- Quantization (GPTQ, AWQ, bitsandbytes)
- ONNX/TensorRT export

**Multi-model support.** If you need to evaluate 50 different models, frameworks provide a
uniform API. The from-scratch approach requires a separate implementation for each architecture.

**Inference optimization.** The from-scratch Qwen3.5 notebook generates at ~8-9 tokens/sec on
GPU. The same model through vLLM or TGI would be significantly faster due to kernel fusion,
continuous batching, and PagedAttention.

**Maintenance.** Framework code is maintained by large teams and stays in sync with upstream
model releases.

### How Trust Bench should handle this

**Phase 1 (current): From-scratch for Qwen3.5.**

Qwen3.5 is the primary target for deep interpretability work. The from-scratch implementation
gives us direct access to:

- DeltaNet state matrices (the `last_recurrent_state` tensor)
- Gating values (`beta`, `g`) at every layer and position
- Full attention weights from the softmax attention layers
- All intermediate hidden states between layers

This is where the novel research happens: correlating internal signals with trustworthiness.

**Phase 2 (future): Framework-based for API and closed models.**

For models where we only have API access (GPT-4, Claude, Gemini) or where reimplementation is
impractical (70B+ models), Trust Bench should use:

- HuggingFace transformers with `output_hidden_states=True` and `output_attentions=True` for
  open-weight models where deep hooks are not needed
- API-based probing (logprobs, token probabilities) for closed models
- `lm-eval-harness` style black-box benchmarking as a compatibility layer

The architecture should make it easy to swap between "deep probe" mode (from-scratch) and
"surface probe" mode (framework/API) depending on the model and the research question.

---

## 4. Key Code Patterns for Signal Extraction

### Pattern 1: PyTorch forward hooks

The simplest way to extract intermediate activations without modifying model code. Works with
both from-scratch and framework-based models.

```python
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on specific layers
model.trf_blocks[3].token_mixer.register_forward_hook(hook_fn("layer3_attn"))
model.trf_blocks[0].token_mixer.register_forward_hook(hook_fn("layer0_deltanet"))

# Run forward pass
with torch.no_grad():
    logits = model(input_ids)

# Now activations["layer3_attn"] contains the attention output
# and activations["layer0_deltanet"] contains the DeltaNet output
```

For more granular access (e.g., hooking into submodules of a DeltaNet layer):

```python
# Hook the QKV projection inside a DeltaNet layer
model.trf_blocks[0].token_mixer.in_proj_qkv.register_forward_hook(
    hook_fn("layer0_deltanet_qkv")
)

# Hook the output normalization
model.trf_blocks[0].token_mixer.norm.register_forward_hook(
    hook_fn("layer0_deltanet_norm")
)
```

### Pattern 2: Extracting hidden states between layers

Modify the top-level forward to collect all intermediate representations.

```python
def forward_with_hidden_states(model, in_idx):
    """Modified forward that returns hidden states at every layer boundary."""
    hidden_states = []
    x = model.tok_emb(in_idx)
    hidden_states.append(x.detach().cpu())

    num_tokens = x.shape[1]
    mask = torch.triu(
        torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
        diagonal=1,
    )

    for block in model.trf_blocks:
        x = block(x, mask, model.cos, model.sin)
        hidden_states.append(x.detach().cpu())

    x = model.final_norm(x)
    logits = model.out_head(x.to(model.cfg["dtype"]))
    return logits, hidden_states
```

### Pattern 3: Extracting DeltaNet state matrices and gating values

This requires modifying the `Qwen3_5GatedDeltaNet.forward` method. The key tensors to capture:

```python
# Inside Qwen3_5GatedDeltaNet.forward, after the delta rule computation:

# 1. The recurrent state matrix: shape [batch, heads, key_dim, value_dim]
#    This is the model's "memory" - what it has accumulated from the sequence so far.
core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
    query, key, value, g=g, beta=beta,
    initial_state=None,
    output_final_state=True,  # <-- must be True to get the state
    use_qk_l2norm_in_kernel=True,
)
# last_recurrent_state: [batch, num_heads, key_head_dim, value_head_dim]
# For 0.8B: [batch, 16, 128, 128] = 16 heads, each with a 128x128 state matrix

# 2. The gating values, already computed earlier in forward:
beta = b.sigmoid()  # write gate: [batch, seq_len, num_heads]
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # decay: [batch, seq_len, num_heads]
```

A practical extraction pattern:

```python
class InstrumentedDeltaNet(Qwen3_5GatedDeltaNet):
    """Wraps DeltaNet to capture internal signals during forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_beta = None
        self.last_g = None
        self.last_state = None

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        # Run the parent's projections to get gating values
        hidden_states_clean = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states_clean.shape

        b = self.in_proj_b(hidden_states_clean)
        a = self.in_proj_a(hidden_states_clean)

        self.last_beta = b.sigmoid().detach().cpu()
        self.last_g = (-self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias
        )).detach().cpu()

        # Run the full forward pass, forcing state output
        output = super().forward(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        return output
```

### Pattern 4: Extracting attention weights from full attention layers

The `GroupedQueryAttention` class computes attention weights explicitly:

```python
attn_scores = queries @ keys.transpose(2, 3)
attn_scores = attn_scores.masked_fill(mask, -torch.inf)
attn_weights = torch.softmax(
    attn_scores * (self.head_dim ** -0.5),
    dim=-1,
    dtype=torch.float32,
).to(queries.dtype)
```

To capture these, either:

(a) Modify the forward to return them:

```python
def forward(self, x, mask, cos, sin):
    # ... existing code ...
    self._last_attn_weights = attn_weights.detach().cpu()
    context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
    # ...
```

(b) Use a forward hook on the softmax output by wrapping it in a submodule (more invasive but
avoids modifying the class).

### Pattern 5: Comparing signals at truthful vs. fabricated outputs

The full workflow for Trust Bench signal extraction:

```python
def extract_signals(model, tokenizer, prompt):
    """Extract all internal signals for a single prompt."""
    signals = {}

    # Set up hooks
    hooks = []
    for i, block in enumerate(model.trf_blocks):
        layer_type = model.cfg["layer_types"][i]

        if layer_type == "linear_attention":
            # Capture DeltaNet output
            h = block.token_mixer.register_forward_hook(
                lambda mod, inp, out, idx=i: signals.update({
                    f"layer{idx}_deltanet_out": out.detach().cpu()
                })
            )
            hooks.append(h)

            # Capture gating projections
            h = block.token_mixer.in_proj_b.register_forward_hook(
                lambda mod, inp, out, idx=i: signals.update({
                    f"layer{idx}_beta_pre_sigmoid": out.detach().cpu()
                })
            )
            hooks.append(h)

        elif layer_type == "full_attention":
            h = block.token_mixer.register_forward_hook(
                lambda mod, inp, out, idx=i: signals.update({
                    f"layer{idx}_attn_out": out.detach().cpu()
                })
            )
            hooks.append(h)

    # Forward pass
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids)

    signals["logits"] = logits.detach().cpu()
    signals["top_probs"] = torch.softmax(logits[:, -1, :], dim=-1).detach().cpu()

    # Clean up hooks
    for h in hooks:
        h.remove()

    return signals
```

### Pattern 6: Inspecting the delta rule recurrence step by step

The `torch_recurrent_gated_delta_rule` function (used during single-token generation) makes
the recurrence explicit. Each step is visible:

```python
for i in range(sequence_length):
    q_t = query[:, :, i]
    k_t = key[:, :, i]
    v_t = value[:, :, i]
    g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # decay factor
    beta_t = beta[:, :, i].unsqueeze(-1)                 # write strength

    # Decay the state
    last_recurrent_state = last_recurrent_state * g_t

    # Read from state using key
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)

    # Compute delta: difference between current value and what state predicts
    delta = (v_t - kv_mem) * beta_t

    # Write delta back to state
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    # Read from state using query
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

This recurrence is the heart of what makes DeltaNet interpretable. At each step:

- `g_t` controls how much the state decays (forgets)
- `beta_t` controls how aggressively the new token is written into state
- `delta` is the prediction error: how much the state's prediction for key `k_t` differs from the actual value `v_t`
- The state matrix `last_recurrent_state` is a key-value associative memory

For Trust Bench, the hypothesis is that these signals (especially the prediction error `delta`
and the decay/write gate values) will differ systematically between factual and fabricated
outputs.

---

## Summary: What This Means for Trust Bench

The from-scratch Qwen3.5 implementation gives Trust Bench three capabilities that framework-based
approaches cannot provide easily:

1. **Direct state matrix access.** The DeltaNet state matrix is a [batch, 16, 128, 128] tensor
   per layer, updated at every token. This is the model's working memory. Changes in how this
   state evolves during truthful vs. fabricated generation are a primary research target.

2. **Gating signal analysis.** The `beta` (write) and `g` (decay) gates are learned functions of
   the input. Their distributions across layers and positions may serve as lightweight
   trustworthiness indicators that do not require running a full interpretability pipeline.

3. **Prediction error tracking.** The `delta` value in the recurrent update is a direct measure
   of how "surprised" the state is by each new token. A model that is fabricating may show
   different delta patterns than one that is recalling learned facts.

The implementation is already functional (generates coherent text at ~8-9 tok/sec on GPU, 0.8B
parameters, 3.81 GB in bfloat16). The next step is to instrument it for systematic signal
collection across a benchmark dataset.
