# Imagized Language Model (ILM)

Version: 0.1 (design draft)

Author: incubated by Codex CLI with user requirements

---

## Executive Summary

This document designs an Imagized Language Model (ILM): a language model that represents text as compact, image-like tensors and generates text via a diffusion-style denoising process in that image space. ILM encodes a sentence as a multi-channel 2D grid whose channels factor language into meta-elements (e.g., grammar, semantics, tone, emotion) and hierarchical, memory-like word codes. Generation proceeds coarse-to-fine, optionally with explicit conditioning on meta-elements. The goal is to enable efficient training and inference on normal computers while exploring controllable, interpretable superpositions of linguistic factors.

Key ideas:

- Superposition of language factors: explicit channels for meta-grammar, meta-semantics, tone, emotion.
- Memory-like, hierarchical embeddings: words/characters mapped to a multi-level categorical code (mixed-radix digits) so similar words are “close” in code space.
- Text-as-image: sequences are laid out on a 2D grid; each position contains multi-channel categorical planes. Diffusion operates over these image-like tensors (preferably in a compact latent space).
- Multiscript support: English via byte/character rows or small glyph renderings; CJK via font-rendered glyphs; hybrid pipelines unify both.
- Resource-aware: latent diffusion, compact DiT/UNet backbones, quantization, and compressed codebooks for embedding tables.

---

## Prior Work (context)

This design draws from established lines of research:

- Diffusion models and non-autoregressive generation: DDPM/score-based models; discrete diffusion (D3PM); masked iterative generation (e.g., Mask-Predict, MaskGIT); diffusion-style text generation and rectified/flow-matching approaches.
- Tokenization-free and character/byte models: ByT5, CANINE, Charformer, which show tokenization can be optional and downsampling can be learned.
- Glyph-aware language models for Chinese and other logographic scripts: models that incorporate rendered glyphs via CNNs to enrich character embeddings (e.g., ChineseBERT variants; Glyce-style representations).
- Efficient embeddings and memory: adaptive input representations, mixed-dimension embeddings, hash embeddings, product quantization, vector quantization (VQ-VAE) for compact codebooks.
- Vision backbones for dense generative modeling: UNet backbones and Diffusion Transformers (DiT) operating on latent image tokens.

We do not assume any single prior directly solves ILM, but each contributes proven components: discrete diffusion for categorical variables, glyph-aware char encoders, and compact latent diffusion.

---

## Mathematical Formulation

### 1) Hierarchical, memory-like vocabulary codes

Let the vocabulary be V (words or characters). Assign to each token v ∈ V a multi-level categorical code via a mapping

  φ(v) = (k₁, k₂, …, k_L), where k_ℓ ∈ {0, …, K_ℓ − 1}.

The number of levels is L, and level cardinalities are {K_ℓ}. This is a mixed-radix representation. The flat ID of v can be recovered by

  id(v) = Σ_{ℓ=1..L} k_ℓ · Π_{m<ℓ} K_m.

Interpretation: each level encodes a progressively finer category. For example, coarse semantic field (tools, animals), subcategory (birds), lemma cluster (sparrow-like), inflectional form, etc. For Chinese, levels can reflect radicals, components, frequency tiers, and character sets; for English, coarse POS, semantic cluster, lemma, and morphological variant.

This code supports:

- Spatial proximity: related words can share prefixes (k₁..k_r), making them close in code space.
- Memory mapping: each level can be stored in a compact table and looked up independently.
- Multichannel imaging: represent each level as a separate channel in an image.

### 2) Text-to-image rasterization

Given a sequence s = (v₁, v₂, …, v_n), define a fixed-width grid of width W and height H = ⌈n / W⌉. Define a rasterization π(i) = (row, col) with row = ⌊(i−1)/W⌋, col = (i−1) mod W.

For each position i, convert φ(v_i) into either:

- Multi-hot planes: for each level ℓ, a one-hot plane x^{(ℓ)} where x^{(ℓ)}[row, col, k] = 1 if k_ℓ = k, else 0. This yields Σ K_ℓ channels.
- Compact-coded planes: for each level ℓ, store the integer k_ℓ in a single-channel plane x^{(ℓ)} of shape H×W (integers, later embedded to continuous vectors when fed to the network).

In practice we prefer “compact-coded planes” then embed per-level integers to continuous feature maps via small learned embeddings E_ℓ ∈ ℝ^{K_ℓ×d_ℓ} and 1×1 convolutions to reach a common depth d.

### 3) Meta-element superposition channels

Introduce meta-channels for factors like grammar (g), semantics (s), tone (t), emotion (e). Two forms are supported:

1) Explicit conditioning vectors: z_g, z_s, z_t, z_e, either global or low-resolution spatial maps. At training, derive z_* from parsers/annotators (e.g., UD parses, SRL, sentiment), then predict/reconstruct them.
2) Latent factors: learn per-position latent planes m^{(f)} that the model can attend to; optionally add weak supervision.

Superposition: the per-position representation is the sum (or concatenation) of contributions from each factor after projections:

  h[row, col] = Σ_f P_f(m^{(f)}[row, col]) + Σ_ℓ P_ℓ(Embed_ℓ(k_ℓ[row, col])) + pos[row, col],

where P_* are 1×1 learned projections and pos is a 2D positional encoding.

### 4) Discrete diffusion in image space

Define x₀ as the full multi-level coded grid for a sentence. We apply a diffusion process over time steps t = 1..T.

Two options:

- Discrete diffusion (D3PM-style): for each level ℓ and pixel p, corrupt x₀^{(ℓ)} to x_t^{(ℓ)} via a categorical noising process q_t(k|k₀) with schedule β_t. The model learns p_θ(x_{t−1}|x_t, z) or equivalently predicts the posterior over clean symbols.
- Continuous relaxation: embed each level via E_ℓ(k) ∈ ℝ^{d_ℓ}, stack channels, and add Gaussian noise as in DDPM; the model predicts ε_θ and we use standard DDPM/score matching losses. Discrete decoding uses argmax over per-level posteriors from denoised embeddings.

Coarse-to-fine schedule: set β_t per-level so coarse levels (ℓ small) retain clean signal longer, while fine levels (ℓ large) corrupt earlier and more strongly. This realizes a principled coarse→fine generative trajectory.

### 5) Latent diffusion

To run on normal computers, encode the multi-channel grid x_t into a compact latent z_t via an autoencoder (VQ-VAE or continuous VAE). Diffusion operates in latent space, reducing memory and compute. Sampling uses the decoder to map denoised z_0 back to x_0, then to tokens.

---

## From Words to Images (and back)

### Building the hierarchical code φ

Construct φ by combining prior knowledge and data-driven clustering:

- Coarse levels: POS tags, WordNet supersenses, radicals/components (CJK), frequency buckets.
- Mid levels: semantic clusters from pretrained embeddings (e.g., clustering normalized word vectors).
- Fine levels: lemma/inflectional variants, subword or character n-grams.

Algorithm sketch:

1) Start with a base embedding e(v) from a pretrained model or FastText/GloVe/word2vec.
2) Hierarchical clustering (agglomerative or k-means at each level) with constraints so siblings remain balanced.
3) Assign stable codes ensuring K_ℓ is small (e.g., 8–64) and Π K_ℓ ≥ |V| with slack for growth.
4) Freeze φ and export level tables; store on disk as memory-mapped arrays for fast lookup.

### Encoding pipeline

- Tokenize to words or characters (or bytes for tokenization-free pipelines).
- Map each token to φ(v) and rasterize to the 2D grid.
- Optional glyph augmentation:
  - English: per-character glyph patches (e.g., 16×16 grayscale) composed to a small word glyph via text rendering.
  - Chinese/Japanese: render characters directly via fonts to small glyphs; pass through a tiny CNN to produce embeddings that inform clustering and/or serve as auxiliary channels.

### Decoding pipeline

- From denoised grid x̂_0, read per-level integers k̂_ℓ[row, col]; reconstruct id(v̂) via mixed-radix composition; map to tokens via inverse table.
- Optionally: refine with a small autoregressive corrector (e.g., a lightweight transformer) to repair local inconsistencies.

---

## Model Architecture

Two viable backbones:

1) UNet (2D convolutional, with attention in bottleneck): strong for image-like diffusion, simple and efficient on CPUs/low-end GPUs.
2) Diffusion Transformer (DiT): patchify the grid into tokens (e.g., 4×4 patches), add time embeddings, and run a transformer. Good for discrete/continuous hybrid setups and for conditioning.

Recommended minimal configuration (laptop-friendly):

- Latent autoencoder: 8× compression in H,W and 2–4× in channels.
- Backbone: DiT Base (≈150M params) or UNet Small (≈80–120M), with group norm, GELU, and optional low-rank adapters (LoRA) for specialization.
- Per-level embeddings: K_ℓ ∈ {16, 16, 32, 64} with d_ℓ ∈ {8, 8, 16, 32} then project to 128–256 channels total.
- Conditioning: classifier-free guidance for z_g, z_s, z_t, z_e; sinusoidal/learned 2D positions.

---

## Training Objectives

- Discrete diffusion (per-level): cross-entropy over k_{t−1} conditioned on (x_t, t, conditioning), minimizing the variational bound as in discrete DDPM.
- Continuous diffusion: MSE over ε_θ predictions in latent space; KL terms from the latent autoencoder.
- Auxiliary tasks (optional):
  - Predict UD dependencies or POS at each position (grammar supervision).
  - Predict sentiment or style attributes (tone/emotion supervision).
  - Contrastive alignment between glyph CNN embeddings and level-ℓ projections to keep φ semantically meaningful.

---

## Inference (Sampling)

- Coarse-to-fine schedule: sample from t = T→0 with larger β_t on fine levels; optionally first sample only coarse levels (ℓ=1..r) to obtain a “skeleton,” then condition fine-level denoising on the skeleton.
- Classifier-free guidance: mix conditional and unconditional predictions to steer towards specified grammar/semantics/tone/emotion.
- Temperature and nucleus filters operate per-level at the final denoise to avoid off-manifold symbols.

Pseudocode (discrete, per-level, coarse→fine):

```
for t in reversed(1..T):
  for level ℓ in 1..L:
    x̂_{t-1}^{(ℓ)} = model.sample_level(x_t, t, ℓ, cond)
  x_t ← combine({x̂_{t-1}^{(ℓ)}}_ℓ)
return x̂_0
```

---

## Efficiency on Normal Computers

- Latent diffusion: operate at 16–64× fewer spatial elements than pixel space.
- Mixed-radix codes: small K_ℓ keeps per-level embeddings tiny; only a few channels are active.
- Quantization: 8-bit weights/activations; 4-bit weight-only quantization where stable. Integer embeddings are naturally cheap.
- Sparse updates: when editing/continuation, diffuse only the region-of-interest on the grid.
- Streaming decoding: output lines/patches incrementally to keep memory bounded.
- Lightweight corrector: a 20–40M parameter autoregressive corrector can repair local grammar without heavy compute.

Rough budget for a laptop GPU/CPU:

- 150M-parameter DiT/UNet with 8× latent compression, 32×32 latent grids, T=16–32 steps can sample at interactive speeds on consumer GPUs; CPU-only is slower but usable for short sequences.

---

## English and Chinese Pipelines

### English (character/byte + words)

- Option A (tokenization-free): bytes/characters directly clustered to φ; no tokenizer needed.
- Option B (word-first): standard tokenizer → words; backoff to character φ for OOV.
- Optional glyph aid: render small character glyphs (e.g., 16×16 grayscale) and provide a 2–4 channel CNN feature map as auxiliary input.

### Chinese/Japanese (glyph-forward)

- Character-based with φ aligned to radicals/components/frequency tiers.
- Render glyphs at 16–32 px; pass through a tiny CNN for glyph-aware embeddings; concatenate with level embeddings.
- For mixed scripts, normalize width W and interleave per-line grids.

---

## Building φ and the Tables (practical)

1) Gather vocabulary V and pretrained embeddings e(v) (alternatively compute from scratch with a small encoder).
2) Decide levels L and K_ℓ to cover |V| with ≈20–30% slack.
3) For ℓ=1..L:
   - Cluster current groups to K_ℓ branches with balance constraints.
   - Re-embed group centroids for the next level.
4) Emit lookup tables: for each v, store k₁..k_L and id(v); store inverse maps.
5) Memory-map tables for O(1) lookup at preprocessing and decoding.

Code sketch (Python-like):

```python
class HierCode:
    def __init__(self, levels):
        self.levels = levels  # list of K_ℓ
        self.code = {}        # v -> [k1..kL]
        self.inverse = None

    def fit(self, vocab, embed):
        groups = {None: list(vocab)}
        for ℓ, K in enumerate(self.levels, start=1):
            new_groups = {}
            for parent, items in groups.items():
                X = np.stack([embed(v) for v in items])
                labels = balanced_kmeans(X, K)
                for k in range(K):
                    bucket = [items[i] for i in np.where(labels==k)[0]]
                    new_groups[(parent, k)] = bucket
                    for v in bucket:
                        self.code.setdefault(v, [None]*len(self.levels))
                        self.code[v][ℓ-1] = k
            groups = new_groups
        self._build_inverse()

    def encode(self, v):
        return self.code[v]

    def _build_inverse(self):
        self.inverse = {}
        for v, ks in self.code.items():
            idv = 0; mul = 1
            for k, K in zip(ks, self.levels):
                idv += k * mul
                mul *= K
            self.inverse[idv] = v
```

---

## Evaluation

- Intrinsic:
  - Negative log-likelihood / bits-per-char via importance or variational bounds in discrete diffusion.
  - Perplexity proxy from reconstructed sequences.
  - Edit distance and grammar accuracy (UD parsing of outputs).
  - Conditioning fidelity for tone/emotion/semantic constraints.
- Extrinsic:
  - Downstream task transfer: classification, QA with ILM as a generator.
  - Robustness: OOD scripts, noisy text, mixed languages.
- Efficiency:
  - Tokens/sec vs. steps T; memory footprint; CPU-only throughput.

---

## Risks, Pitfalls, and Mitigations

- Sequence structure in 2D: simple rasterization may blur long-range dependencies. Mitigate with strong 2D positional encodings and global attention in the backbone (or hybrid 1D-2D attention).
- Discrete decoding errors: per-level argmax can produce invalid combinations. Use constrained decoding (validity masks), small correctors, and per-level temperature control.
- Overly large glyph inputs: limit glyph resolution (16–32 px) and compress via tiny CNNs; keep glyph channels optional.
- Building φ poorly: bad clusters harm generalization. Use semantic priors (POS, radicals), validate with nearest-neighbor consistency, and allow periodic recoding.
- Training cost: use latent diffusion, fewer steps (e.g., T=16–32), and distillation/knowledge transfer from a small AR teacher.

---

## Minimal Viable Prototype (MVP)

- Data: Wikitext-103 (EN) + small Chinese corpus (e.g., news subset). Limit vocab to 50k EN words + 7k CJK chars.
- φ: L=4 with K = [16, 16, 32, 64] (capacity 524,288 ids); assign EN OOV to character backoff.
- Grid: W=64 columns; H variable; pad/truncate to 4096 positions per sample.
- Latent AE: 4× downscale; channels 64→16.
- Backbone: UNet-Small (~100M) with attention in bottleneck; T=24 steps; discrete or relaxed continuous diffusion.
- Conditioning: grammar (POS bigrams), coarse semantics (supersenses), simple sentiment for tone/emotion.

---

## Implementation Plan (phased)

1) Tables and φ
   - Build hierarchical codes and memory-mapped tables.
   - Export encode/decode utilities.
2) Rasterization + Latent AE
   - Implement grid layout, per-level embeddings, and a small latent autoencoder.
3) Diffusion backbone
   - Start with UNet-Small; implement discrete diffusion heads per level.
4) Conditioning
   - Add classifier-free guidance; optional grammar/tone supervision.
5) Training loop
   - Mixed precision where possible; gradient checkpointing; EMA weights.
6) Inference & tooling
   - Sampler, constraints, ROI editing; CPU-only mode; quantized checkpoints.
7) Evaluation
   - Intrinsic metrics, ablations (with/without glyphs; levels L; K sizes).

---

## Extensions and Open Questions

- Flow/ODE variants for fewer steps and faster sampling.
- Retrieval-augmented ILM: patch-wise retrieval from a memory bank of φ-coded snippets.
- Joint vision-text ILM: unify literal images with imagized text in a single latent space.
- Cross-lingual alignment: align φ across languages via shared meta-channels and contrastive training.
- Editing and controllability: structured masks over levels/channels to edit semantics or tone without affecting grammar.

---

## References (indicative)

- Denoising Diffusion Probabilistic Models (DDPM); score-based generative modeling.
- Discrete diffusion for categorical data (D3PM).
- Mask-Predict and non-autoregressive generation; MaskGIT.
- ByT5, CANINE, Charformer (tokenization-free/char-level modeling).
- ChineseBERT and glyph-aware character embeddings; Glyce-style approaches.
- Vector-quantized and product-quantized embeddings; VQ-VAE.
- Diffusion Transformers (DiT); latent diffusion.

These references are well-known families of work; select specific implementations based on your toolchain (PyTorch/JAX) and license requirements.

---

## Summary

ILM reframes language modeling as diffusion over a compact, image-like representation that factors language into explicit, compositional channels. Hierarchical, memory-like codes make embedding tables structured and compressible; meta-element channels enable controllability (grammar, semantics, tone, emotion). With latent diffusion and small backbones, the approach is practical on normal computers, and it opens avenues for multiscript and cross-modal modeling.

