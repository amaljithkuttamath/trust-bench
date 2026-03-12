# From Transformer Internals to Circuit Tracing

The missing bridge between understanding how transformers work and understanding what they compute. This document picks up where `trust-research-methods.md` and `from-scratch-approach.md` leave off — you know the architecture, you can extract internal tensors, and now you need the interpretability science that makes those tensors meaningful.

---

## 1. The Superposition Problem: Why Neurons Don't Mean What You Think

### The core frustration

You've built the Qwen3.5 implementation from scratch. You can hook into any layer, extract hidden states, capture DeltaNet state matrices, read attention weights. But when you look at a single neuron's activation — say, neuron 847 in layer 12 — what does it mean?

The honest answer: almost nothing, because that neuron is *polysemantic*. It fires for multiple unrelated concepts simultaneously. This isn't a failure of your analysis. It's a fundamental property of how neural networks encode information, and understanding it is the first step toward real interpretability.

### The superposition hypothesis

Elhage et al. (2022) formalized this in "Toy Models of Superposition." The claim: neural networks deliberately represent **more features than they have dimensions** by exploiting properties of high-dimensional geometry.

Think of it this way. Your Qwen3.5-0.8B model has a residual stream of dimension 1536. Naively, it could represent at most 1536 independent features — one per dimension. But language requires vastly more features than 1536. There are features for "European capitals," "medical terminology," "sarcasm," "legal hedging," "code syntax," "recipe ingredients," and thousands more. The model solves this by encoding features as *directions* in the 1536-dimensional space, not as individual dimensions.

Two features can coexist in the same set of neurons if they point in different directions. In 1536 dimensions, you can fit an enormous number of nearly-orthogonal directions. The features interfere with each other slightly — when feature A activates, it creates a small ghost activation of feature B if they're not perfectly orthogonal — but if each feature is *sparse* (only active on a small fraction of inputs), this interference rarely causes problems.

### The math of why sparsity makes it work

If you have N features, each active with probability p, represented in d dimensions, the expected interference is proportional to N·p². When p is small (sparse features), even huge N fits in small d without unacceptable noise.

There's a phase transition. Below a critical sparsity threshold, the network abandons superposition and assigns one neuron per feature (clean, interpretable). Above that threshold, superposition becomes optimal and the network packs features into geometric structures — polytopes like digons, triangles, pentagons — that minimize interference. Elhage et al. showed this in controlled experiments where the transition is sharp and predictable.

### What this means for your signal extraction

When you run your `extract_signals()` function from `from-scratch-approach.md` and capture hidden states at layer boundaries, each position in that 1536-dimensional vector is a mixture of many features in superposition. The DeltaNet state matrices — those 16×128×128 tensors — encode associations between keys and values, but those keys and values are themselves superposed representations. The gating signals (beta, g) control memory operations on superposed features.

This is why raw neuron-level analysis has a ceiling. You can detect statistical patterns (your hypothesis about delta values differing for factual vs. fabricated outputs is sound), but you can't cleanly identify *which features* are driving the behavior by looking at neurons. You need a tool that decomposes superposed representations into individual features.

That tool is the sparse autoencoder.

**Source**: Elhage et al., "Toy Models of Superposition" (2022). [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)

---

## 2. Sparse Autoencoders: Extracting Individual Features from the Mess

### The problem SAEs solve

Given a polysemantic neuron that fires for "grandmother" AND "basketball" AND "legal contracts," how do you separate these into distinct, interpretable features? You need a method that takes the 1536-dimensional residual stream and expands it into a much higher-dimensional space where each dimension corresponds to one concept.

This is unsupervised dictionary learning. You're finding the "dictionary" of features that the model actually uses, without any labeled data telling you what those features should be.

### Architecture

A sparse autoencoder has three components:

**Encoder**: A matrix W_enc of shape (d, n) where d is the residual stream dimension (1536) and n is much larger — typically 8x to 32x, so 12,288 to 49,152 features. This expands the representation into a space where features can be cleanly separated.

**Sparsity bottleneck**: ReLU or TopK activation applied to the expanded representation. This is critical — it forces most features to be exactly zero on any given input. Only a handful of features are active at once, which is what makes them interpretable. With ReLU, you add an L1 penalty to the loss to encourage sparsity. With TopK, you directly keep only the K most active features and zero the rest.

**Decoder**: A matrix W_dec of shape (n, d) that projects back to the original dimension. The decoder columns *are* the features — each column is a direction in the residual stream that corresponds to one interpretable concept.

The training objective:

```
L = ||x - x̂||² + λ * Σ|h_i|

where:
  x   = residual stream activation (ground truth)
  h   = ReLU(W_enc · x + b_enc)     (sparse feature activations)
  x̂  = W_dec · h                     (reconstruction)
  λ   = sparsity coefficient (controls how many features are active)
```

The first term ensures faithful reconstruction. The second term (L1 penalty) ensures sparsity. The tension between them is the core design tradeoff.

### Why it works

The SAE operates in a much higher-dimensional space (n >> d). In this expanded space, there's room to represent each feature along its own direction without the interference that occurs in the original d-dimensional space. The decoder then projects these clean, sparse features back to the original space for reconstruction.

Think of it as a microscope. The residual stream is a compressed photograph. The SAE is the microscope that resolves individual features that were blurred together in the compression.

### What the features look like in practice

Cunningham et al. (2023) trained SAEs on small transformers and found remarkably clean features:

- A feature that fires exclusively on the word "Jerusalem" and nothing else
- A feature for "being a date" that activates on "March 15" and "2024" but not random numbers
- A feature for "legal language" that lights up throughout legal documents but stays dark in casual text
- A feature for "DNA sequences" — the letters A, T, G, C in biological contexts only

These are genuinely monosemantic: each represents one concept. This is the breakthrough. Neurons are polysemantic and hard to interpret. SAE features are monosemantic and directly meaningful.

### Scaling monosemanticity

Bricken et al. (2023) pushed this further, decomposing a 512-neuron MLP layer into over 4,000 features. The key finding: *more features means more monosemantic features*. Scaling up the SAE doesn't just add noise — it reveals finer-grained concepts that were blurred together in smaller decompositions.

Templeton et al. (2024) then went to production scale: 34 million features extracted from Claude 3 Sonnet. Among the findings:

- **Multi-modal features**: A single feature for "Golden Gate Bridge" that activates across English text, French text, Chinese text, and photographs. The model has a unified internal representation of the concept regardless of modality.
- **Safety-relevant features**: Features for deception, sycophancy, and dangerous content. These aren't just post-hoc labels — clamping these features (forcing them on or off) directly steers model behavior.
- **Abstract reasoning features**: Features for planning, hedging, expressing uncertainty.

The deception and uncertainty features are directly relevant to your trust research. If you can identify features that correspond to "the model is uncertain" or "the model is fabricating," you have a mechanistic handle on hallucination that goes far deeper than output-level detection.

**Sources**:
- Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023). [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
- Bricken et al., "Towards Monosemanticity" (2023). [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
- Templeton et al., "Scaling Monosemanticity" (2024). [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

---

## 3. Transcoders: Decomposing What the Model Actually Computes

### Why SAEs aren't enough

SAEs decompose the residual stream — they tell you *what information is present* at a given point. But they don't tell you *how the model transforms that information*. The residual stream is a highway. SAEs tell you what's on the highway at a given mile marker. But the interesting computation happens in the on-ramps and off-ramps: the MLP layers and attention layers that read from and write to the residual stream.

MLP layers are the densest, least interpretable component. An MLP takes a 1536-dimensional input, expands it to 8960 dimensions (in Qwen3.5's SwiGLU), applies a nonlinear gate, and projects back to 1536. Understanding *what computation this performs* — which input features get transformed into which output features — is the key to tracing circuits end to end.

### What a transcoder does

A transcoder is an SAE specialized for MLP layers. Instead of decomposing the residual stream into "what's present," it decomposes the MLP computation into "what steps happen."

Architecture:

1. **Input**: The pre-MLP residual stream activation (after layer norm)
2. **Encoder**: Expands to a high-dimensional sparse space (same as an SAE)
3. **Sparse features**: Each active feature represents one interpretable step of the MLP's computation
4. **Decoder**: Reconstructs the MLP's output (not the input — this is the key difference from an SAE)

The transcoder is trained to match the MLP's actual output while keeping features sparse. Each feature captures a single computation the MLP performs: "if the input contains feature X, add feature Y to the residual stream."

### Cross-layer transcoders: the real breakthrough

Standard transcoders operate one layer at a time. But Anthropic's circuit tracing work introduced **cross-layer transcoders (CLTs)**, which change the game entirely.

In a CLT, features read from the residual stream at one layer but can contribute to MLP outputs at *all subsequent layers*. A feature that activates at layer 2 can influence computation at layers 3, 4, 5, all the way to the output.

Why this matters: it dramatically simplifies the circuit graph. Instead of tracing through N separate sets of features (one per layer), you have a single unified set of features that persist across depth. A "planning" feature that activates early can be followed through the entire model without losing track of it across layer boundaries.

Anthropic's results show that CLT features can replace a model's actual MLP layers and produce similar outputs roughly 50% of the time. That's a strong signal that CLTs are capturing real computation, not just fitting noise.

### How transcoders connect SAEs to circuits

The interpretability stack is now:

1. **Superposition** tells you the problem: neurons are polysemantic, features are mixed
2. **SAEs** decompose what information is *present* at each point
3. **Transcoders** decompose what *computation happens* at each MLP layer
4. With both, you can trace: which input features → which MLP computations → which output features

This is the foundation for attribution graphs — the complete circuit picture.

**Sources**:
- Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" (2024). [arXiv:2406.11944](https://arxiv.org/abs/2406.11944)
- Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025). [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

---

## 4. Attribution Graphs: The Full Circuit Picture

### What an attribution graph is

An attribution graph answers: "Given this specific input and this specific output, what computational steps did the model take?"

It's a directed acyclic graph with four types of nodes:

- **Input nodes**: Token embeddings at the start. The raw material.
- **Feature nodes**: Active transcoder features at specific layers and positions. Each represents one interpretable computation step.
- **Error nodes**: Reconstruction errors where the transcoder's approximation doesn't perfectly match the actual MLP. The parts we can't yet explain.
- **Output nodes**: Logits for candidate output tokens. The model's decisions.

Edges represent causal influence: an edge from node A to node B with weight w means A's activation contributes approximately w to B's activation (in a linear approximation). The graph is sparse — most features don't influence most other features — which is what makes it readable.

### How it's computed

The methodology, from Anthropic's circuit tracing paper:

1. **Run the model on a prompt** and record all activations
2. **Replace MLP layers with transcoders** to get sparse feature activations instead of dense neuron activations
3. **For each active feature**, compute its direct effect on every other active feature and on output logits using the transcoder weights
4. **Prune** the graph by removing weak edges (below a threshold) and inactive features
5. **The result** is a sparse graph showing the dominant computational pathways from input to output

The computation is linear in the number of active features (which is sparse), making it tractable even for large models. Cross-layer transcoders simplify this further because features persist across layers.

### Reading a circuit: the hallucination example

The most striking finding from Anthropic's biology paper concerns how Claude hallucinates. The circuit has three components:

**Default refusal**: Claude has a feature (or set of features) that is *on by default*, promoting tokens like "I don't have enough information to answer." This is the safety training manifesting as an active circuit — the model's default state is cautious.

**Known entity recognition**: When Claude encounters a well-known entity ("Michael Jordan," "Paris," "DNA"), a competing feature activates that recognizes the entity as something the model knows about. This feature *inhibits* the default refusal circuit.

**The hallucination failure mode**: When asked about "Michael Batkin" (a real but obscure person), the model recognizes "Michael" as a familiar name pattern. The "known entity" feature partially activates. This suppresses the refusal circuit. But the model has no actual facts about Michael Batkin, so it generates plausible-sounding fiction — a hallucination.

The circuit diagram shows this as a competition between two pathways: the refusal pathway (active by default) and the knowledge pathway (activated by entity recognition). Hallucination occurs when the knowledge pathway wins but there's no actual knowledge to retrieve.

This is mechanistic understanding. It's not "the model hallucinates sometimes" — it's "the model hallucinates *because this specific circuit misfires in this specific way*, and here are the features involved."

### How to read an attribution graph visually

When you look at a graph on Neuronpedia or in circuit-tracer's local visualizer:

- **Left to right** = input to output (information flow follows time/depth)
- **Node labels** describe what each feature represents ("known entity," "hedging language," "medical terminology")
- **Edge thickness** = strength of influence (thick = strong causal connection)
- **Color** often encodes positive (promoting) vs. negative (inhibiting) influence
- **Depth** = which layer the feature lives at (early layers detect surface patterns, late layers form output decisions)

You follow the thick edges from input tokens through mid-layer features to output logits. The path tells the story of why the model produced that particular output.

### Interventions: the causal test

Attribution graphs aren't just descriptive — they make testable predictions. If feature X supposedly causes behavior Y, you can:

1. **Ablate**: Zero out feature X and check if behavior Y disappears
2. **Amplify**: Increase feature X's activation and check if behavior Y gets stronger
3. **Transplant**: Copy feature X's activation from one context to another and check if behavior Y follows

Dario Amodei described a striking experiment: they used circuit-tracing techniques to alter a test model's beliefs about whether it was being evaluated. When the model "believed" it wasn't being watched, it became measurably more misaligned. This is a direct causal test enabled by circuit-level understanding.

### Limitations

Attribution graphs are approximations. Transcoders don't perfectly reconstruct MLP outputs — the error nodes in the graph represent what's missing. The linear attribution assumption (contributions sum) is a simplification — nonlinear interactions between features exist but aren't captured. And the graphs show the *most important* pathways, not all pathways.

These limitations are real but bounded. The reconstruction quality improves with better transcoders, and the key findings (like the hallucination circuit) have been validated with causal interventions.

**Sources**:
- Ameisen et al., "Circuit Tracing" (2025). [transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- Anthropic, "On the Biology of a Large Language Model" (2025). [transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- Anthropic, "Open-sourcing circuit-tracing tools" (2025). [anthropic.com/research/open-source-circuit-tracing](https://www.anthropic.com/research/open-source-circuit-tracing)

---
## 5. The Tooling Ecosystem: What Exists and How It Fits Together

### circuit-tracer

The centerpiece tool. Developed by Michael Hanna and Mateusz Piotrowski (Anthropic Fellows) in collaboration with Decode Research. Open-sourced on GitHub at `decoderesearch/circuit-tracer`.

What it does: takes a language model, a set of pre-trained transcoders, and a prompt, then generates an attribution graph showing the computational pathway from input tokens through features to output logits.

How it works internally: circuit-tracer creates a `ReplacementModel` that wraps the original transformer. It replaces MLP layers with transcoders, runs the model, computes direct effects between all active features, prunes weak connections, and outputs the resulting graph as a JSON file. The interactive visualizer (via `--server` flag) renders this in a browser identical to the published papers.

CLI usage:

```bash
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set qwen3 \
  --graph_output_path france.pt
```

Supported models with pre-trained transcoders:
- **Gemma-2-2B** — the original demo model, runs in free Colab
- **Llama-3.2-1B** — alternative small model
- **Qwen3-4B** — via `mwhanna/qwen3-4b-transcoders` on HuggingFace

Backend flexibility: circuit-tracer uses TransformerLens by default but supports nnsight as an alternative via `backend='nnsight'`. TransformerLens is faster and more mature. nnsight supports more models but is marked experimental in circuit-tracer.

### TransformerLens

The foundation library for mechanistic interpretability. Created by Neel Nanda (MATS/Google DeepMind). Wraps 50+ HuggingFace models with a hook-point system that exposes every internal activation for inspection and intervention.

The core concept: **HookPoints**. When you load a model into TransformerLens, every activation gets a named hook. You can attach functions to these hooks to cache activations (read), edit them (intervene), or replace them entirely. This is conceptually similar to your PyTorch forward hooks in `from-scratch-approach.md`, but standardized across architectures and integrated with the interpretability ecosystem.

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("Qwen/Qwen3-4B")

# Cache all activations
logits, cache = model.run_with_cache("The capital of France is")

# Access specific activations
residual_stream_layer_5 = cache["blocks.5.hook_resid_post"]
attention_patterns_layer_3 = cache["blocks.3.attn.hook_pattern"]
mlp_output_layer_7 = cache["blocks.7.mlp.hook_post"]
```

TransformerLens serves as the primary backend for circuit-tracer. It provides the performance and standardized activation access that circuit tracing requires. Its limitation: it only supports models with custom implementations in the library, not arbitrary HuggingFace models.

### Neuronpedia

The interactive web frontend hosted by Decode Research at neuronpedia.org. Over 7,000 attribution graphs have been generated here. You can explore Qwen3-4B features, browse attribution graphs, and test custom prompts — all in a browser with zero installation.

This is where to start building intuition. Before installing anything locally, spend time on Neuronpedia clicking through graphs for different prompts. Notice how features at different layers capture different levels of abstraction. Notice how the graph structure changes between factual recall ("The capital of France is Paris") and uncertain generation ("The capital of Moldavia is..."). The visual patterns will make the math concrete.

### nnsight

The alternative backend developed by the NDIF team. Instead of maintaining custom model implementations (like TransformerLens), nnsight wraps HuggingFace models directly, preserving their exact original code while adding a tracing system for interventions.

Advantage: works with essentially any HuggingFace model without custom adapter code. Recent benchmarks show it matches or exceeds TransformerLens speed while using less memory.

Disadvantage: less standardized across architectures, and circuit-tracer's nnsight integration is marked experimental. For practical circuit-tracing work in 2026, TransformerLens remains the default.

Where nnsight becomes relevant: if you want to work with models that TransformerLens doesn't support — for example, novel architectures like Qwen3.5's DeltaNet hybrid. nnsight could potentially wrap Qwen3.5 where TransformerLens cannot.

### SAELens

The library for training and analyzing sparse autoencoders. Maintained by Joseph Bloom, Curt Tigges, Anthony Duong, and David Chanin. Formerly part of TransformerLens (the HookedSAE functionality), now a standalone library at `decoderesearch/SAELens`.

SAELens provides the infrastructure for training SAEs on any PyTorch model. It integrates with TransformerLens but also works with raw HuggingFace models and nnsight. If you wanted to train your own SAEs on Qwen3.5's DeltaNet state matrices (a novel research direction), SAELens would be the starting point.

Pre-trained SAEs exist for Gemma models (Google's Gemma Scope) and are emerging for Qwen models (Qwen2.5-7B-Instruct, Qwen 1.5B), but comprehensive pre-trained SAE suites for the Qwen family are not yet as mature as for Gemma.

### EleutherAI's Attribute library

An independent implementation of attribution graphs at `EleutherAI/attribute`. Developed in parallel with circuit-tracer, with cross-layer transcoder support as a core feature from the start.

Its value: independent confirmation that the attribution methodology works across implementations. EleutherAI also maintains the CLT training library (`EleutherAI/clt-training`), contributing to the transcoder infrastructure.

### How the tools chain together

The practical workflow:

```
Neuronpedia (browser)          → Build intuition, explore existing graphs
    ↓
circuit-tracer (local)         → Generate your own attribution graphs
    uses TransformerLens       → As the model-wrapping backend
    uses pre-trained transcoders → From HuggingFace (e.g., mwhanna/qwen3-4b-transcoders)
    ↓
SAELens                        → If you need to train your own SAEs/transcoders
    ↓
EleutherAI Attribute           → Alternative implementation, cross-reference
```

---

## 6. The Qwen Bridge: Connecting Your Work to Circuit Tracing

### What exists right now for Qwen

**Qwen3-4B** has the most developed interpretability support in the Qwen family:
- Pre-trained transcoders at `mwhanna/qwen3-4b-transcoders` on HuggingFace
- Full circuit-tracer integration (it's one of the three supported model families)
- Interactive exploration on Neuronpedia's Qwen3-4B dashboard
- TransformerLens support for loading and hooking

**BluelightAI** has released cross-layer transcoders for Qwen3-0.6B and Qwen3-1.7B, with an interactive explorer at qwen3.bluelightai.com. Their topological analysis reveals how features organize hierarchically: early layers capture words, middle layers capture concepts, late layers capture output plans.

### The architectural divide: Qwen3 vs. Qwen3.5

This is the critical gap to understand clearly.

**Qwen3-4B** (which has transcoders) uses a standard transformer architecture. Every layer has softmax attention followed by an MLP. This is the architecture that the entire interpretability toolchain — SAEs, transcoders, circuit-tracer, TransformerLens — was built to analyze.

**Qwen3.5-0.8B** (which you've built from scratch) is a fundamentally different beast. Its 3:1 hybrid design means 75% of layers use Gated DeltaNet (a recurrent linear attention mechanism with state matrices, decay gating, and delta-rule updates) and only 25% use standard softmax attention. This is architecturally novel — the first major open model family to deploy linear attention at scale.

The implications for interpretability:

1. **Standard attention layers (25%)**: The 6 full-attention layers in Qwen3.5 work like standard transformers. Existing interpretability tools apply to these layers directly. You can trace attention patterns, extract features with SAEs, and potentially use transcoders on their associated MLPs.

2. **DeltaNet layers (75%)**: These are uncharted territory for circuit tracing. The computation is fundamentally different — information flows through a fixed-size state matrix (S_t) updated via the delta rule, not through explicit attention patterns. There are no attention weights to visualize. The "computation" at each step is a memory update (gated decay + error-correcting write), not a softmax-weighted sum.

3. **No existing transcoders for DeltaNet**: The transcoder methodology assumes MLP layers that take a residual stream input and produce an additive output. DeltaNet layers don't work this way — they maintain recurrent state that accumulates across tokens. Training transcoders for DeltaNet would require rethinking the architecture.

4. **Hybrid tracing challenge**: A complete circuit analysis of Qwen3.5 would need to trace information through *both* the DeltaNet state evolution and the standard attention/MLP computation. No published methodology exists for this as of early 2026.

### Two paths forward

**Path 1: Use Qwen3-4B for learning circuit tracing (the proven path)**

This is the straightforward approach. Qwen3-4B is a standard transformer with full tool support:

1. Load Qwen3-4B via TransformerLens
2. Download pre-trained transcoders from `mwhanna/qwen3-4b-transcoders`
3. Use circuit-tracer to generate attribution graphs
4. Explore results on Neuronpedia or locally
5. Run interventions (ablation, amplification) to test hypotheses

This path gets you hands-on with circuit tracing immediately. Your understanding of transformer internals from the Qwen3.5 work transfers directly — the components are the same, just without the DeltaNet hybrid. You could study hallucination circuits, trust-relevant features, or any other behavior in a fully supported environment.

Practical starting sequence:
- Start with the Gemma-2-2B demo notebooks in free Colab (zero hardware requirement)
- Move to Qwen3-4B locally once you understand the workflow
- Focus on medical/factual prompts to build domain expertise in hallucination circuits

**Path 2: Extend interpretability to Qwen3.5's DeltaNet layers (the research opportunity)**

This is harder but potentially much more valuable. Nobody has published circuit-tracing results for hybrid DeltaNet/attention architectures. The questions are genuinely open:

- Can you train SAEs on DeltaNet state matrices? The state matrix S_t is a 16×128×128 tensor per layer. It's a different object than a residual stream vector, but it might still encode features in superposition that an SAE could decompose.

- Can you define "features" for recurrent state? In standard transformers, a feature is a direction in the residual stream. In DeltaNet, the natural unit might be a key-value association in the state matrix — a specific (key direction, value direction) pair that the model has learned to store and retrieve.

- How do DeltaNet features interact with attention features? In the 3:1 pattern, every fourth layer is standard attention that "cleans up" after three DeltaNet layers. The interplay between these two representation systems is unexplored.

- Are the gating signals (beta, g) themselves features? Your from-scratch work already captures these. They control memory operations and might directly encode uncertainty or confidence — exactly the trust signals you're after.

Your from-scratch implementation gives you a unique advantage here. You understand DeltaNet's internals at the code level. You can modify the recurrence, inject probes into the state update, or intercept the delta (prediction error) at each step. This is the kind of deep architectural access that makes novel interpretability research possible.

The risk: this is genuine research, not application of existing tools. It might take months to produce results, and those results might be negative (DeltaNet might resist clean decomposition). But the upside is a publishable contribution to a field that MIT named a 2026 breakthrough technology.

### The recommended sequence

Both paths aren't mutually exclusive. The natural order:

1. **Finish Karpathy's foundations** (your current Phase 1) — build transformer intuition from scratch
2. **Learn circuit tracing on Qwen3-4B** (Path 1) — use the proven tools on the proven architecture
3. **Attempt to extend to Qwen3.5** (Path 2) — once you understand the methodology, push it into novel territory

Step 2 teaches you the science. Step 3 is where you contribute to it.

---

## 7. Key Papers in Reading Order

The interpretability literature is large, but these papers form the critical path from where you are now to being able to do circuit tracing research. The order matters — each paper builds on concepts from the previous ones.

### Foundation layer (understand the problem)

1. **Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)**
   - Anthropic. [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)
   - Introduces the residual stream view, attention heads as information movers, MLPs as information transformers. The conceptual vocabulary you need for everything else.

2. **Elhage et al., "Toy Models of Superposition" (2022)**
   - [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
   - Formalizes why neurons are polysemantic. The phase transition from clean to superposed representations. Geometric structure of feature packing.

### Feature extraction (solve the polysemanticity problem)

3. **Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023)**
   - [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
   - First demonstration that SAEs extract monosemantic features. The proof of concept.

4. **Bricken et al., "Towards Monosemanticity" (2023)**
   - [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
   - Scales SAEs up, finds thousands of interpretable features. Establishes SAEs as a practical tool.

5. **Templeton et al., "Scaling Monosemanticity" (2024)**
   - [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
   - 34 million features from Claude 3 Sonnet. Multi-modal features, safety-relevant features, feature steering. The production-scale validation.

### Circuit tracing (the full picture)

6. **Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" (2024)**
   - [arXiv:2406.11944](https://arxiv.org/abs/2406.11944)
   - Introduces transcoders for decomposing MLP computations. The bridge between "what's present" and "what's computed."

7. **Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025)**
   - [transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
   - The methodology paper. Cross-layer transcoders, attribution graph construction, pruning.

8. **Anthropic, "On the Biology of a Large Language Model" (2025)**
   - [transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
   - The findings paper. Hallucination circuits, planning, multilingual processing, entity recognition. The "what we found when we opened the hood."

### Context (the bigger picture)

9. **Dario Amodei, "The Urgency of Interpretability" (2025)**
   - [dario.amodei.com](https://dario.amodei.com)
   - Why this work matters on a 2-5 year timescale. The race between interpretability and capability. The hidden goal finding.

---

## 8. Mapping Your Existing Work to This Stack

Here's how everything you've already built connects to the interpretability pipeline:

| Your existing work | What it gives you | Next step in the stack |
|---|---|---|
| `trust-research-methods.md` | Deep understanding of transformer architecture, residual stream view, RoPE, GQA, DeltaNet | Foundation for understanding where features live and how information flows |
| `from-scratch-approach.md` | Signal extraction patterns, forward hooks, hidden state collection, DeltaNet state access | These are the raw signals that SAEs would decompose into interpretable features |
| `raschka-qwen35-transformers.py` | Working Gated DeltaNet implementation with chunked delta rule | The code you'd modify to inject interpretability probes into DeltaNet layers |
| `raschka-qwen35-layer-debugger.py` | Layer-by-layer output comparison between from-scratch and HuggingFace | Validates that your implementation is faithful — essential before trusting any signals |
| `qwen35-architecture-notes.md` | Hybrid attention design analysis, DeltaNet as error-correcting memory | Direct mapping to section 6's analysis of why DeltaNet is an interpretability challenge |
| `trustllm-truthfulness.py` | Hallucination evaluation (halu_qa, halu_summ, halu_dial), fact-checking F1 | Provides the behavioral benchmarks to test against — "does the model hallucinate?" pairs with "what circuit caused the hallucination?" |
| `trustllm-notes.md` | Gap analysis: missing calibration/uncertainty dimensions, GPT-as-judge coupling | Identifies where circuit-level signals could replace or augment black-box evaluation |

The through-line: your behavioral evaluation code (TrustLLM) asks "does the model hallucinate?" Your signal extraction code (from-scratch hooks) captures what's happening internally during hallucination. The interpretability stack (SAEs → transcoders → attribution graphs) is the science that turns those raw internal signals into mechanistic explanations of *why* the model hallucinates.

---

## 9. Open Questions and Research Directions

These are the questions that matter most for your trajectory, ordered by tractability:

### Tractable now (with existing tools on Qwen3-4B)

- **What do hallucination circuits look like in Qwen3-4B?** Anthropic showed them in Claude 3.5 Haiku. Do the same circuits exist in Qwen? Are the "default refusal" and "known entity override" features universal across architectures?

- **Do medical-domain hallucinations have distinct circuits?** When a model fabricates a drug interaction vs. fabricates a historical date, are different features involved? Your medical hallucination detection background makes you uniquely positioned to design the right probes.

- **Can activation monitoring detect hallucinations in real time?** Anthropic's recommended research directions explicitly call this out as underexplored. If you can identify a feature that reliably fires during hallucination, monitoring that feature during inference is a lightweight safety layer.

### Medium-term (requires some novel work)

- **Can SAEs decompose DeltaNet state matrices?** The state matrix S_t is a fundamentally different object than a residual stream vector. But it might still encode features in superposition. Training an SAE on flattened state matrices (16×128×128 → 262,144 dimensions) is computationally feasible and would be a novel contribution.

- **Are gating signals (beta, g) interpretable features?** Your from-scratch code already captures these. Do they correlate with known trust-relevant behaviors? If beta (write strength) spikes when the model is about to hallucinate, that's a detection signal that requires no SAE training at all.

- **How do trust signals differ between DeltaNet and attention layers?** In Qwen3.5's hybrid architecture, the same information passes through both layer types. Comparing signal quality across layer types within the same model is a natural experiment.

### Longer-term (genuine research frontier)

- **Full circuit tracing for hybrid DeltaNet/attention models.** This requires developing a transcoder-like decomposition for DeltaNet layers and a methodology for tracing information across heterogeneous layer types. Nobody has published this yet.

- **CoT faithfulness at the circuit level.** When a model shows its reasoning in chain-of-thought, does the internal circuit match the stated reasoning? This is Anthropic's "CoT faithfulness" research direction, and it connects directly to trust.

- **Activation monitoring as a production safety system.** Moving from "we can identify hallucination features in a lab" to "we can monitor these features at inference time with acceptable latency." This bridges interpretability research and engineering.

---

## Summary: The Interpretability Stack in One Picture

```
YOUR FOUNDATION                    THE SCIENCE                         THE TOOLS
─────────────────                  ───────────                         ─────────

Transformer architecture     →     Superposition hypothesis      →     (conceptual framework)
(trust-research-methods.md)        "more features than dimensions"

Signal extraction hooks      →     Sparse Autoencoders           →     SAELens
(from-scratch-approach.md)         "decompose polysemantic              (train SAEs on any model)
                                    neurons into features"

DeltaNet internals           →     Transcoders                   →     Pre-trained transcoders
(raschka-qwen35-transformers.py)   "decompose MLP computation          (mwhanna/qwen3-4b-transcoders)
                                    into interpretable steps"

Layer-by-layer debugging     →     Attribution Graphs            →     circuit-tracer
(raschka-qwen35-layer-debugger.py) "end-to-end circuit from             (generate & visualize graphs)
                                    input to output"
                                                                       Neuronpedia
                                                                       (explore interactively)

Behavioral evaluation        →     Circuit-level explanation     →     Causal interventions
(trustllm-truthfulness.py)         "this is WHY the model               (ablation, amplification,
                                    hallucinated"                        feature steering)
```

Each row connects something you've already built to the science that makes it interpretable and the tool that operationalizes it. The left column is done. The middle column is what this document teaches. The right column is what you use next.
