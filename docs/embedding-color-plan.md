ILM Phase-1 Plan: Learning “Color” Codes for EN+ZH

Goal
- Learn a compact, gradient-learnable, product-code (“color”) embedding that maps each English word or Chinese character image to a 3-channel code c = (k1, k2, k3), with kℓ ∈ {0,…,31} (32 levels per channel), i.e., a 32×32×32 code space (32,768 codes).
- Train jointly on English + Chinese so semantically related units (synonyms, inflections, translations) share nearby codes or identical prefixes (e.g., do/did/done, make/做/作).
- Operate within a single RTX 4090 as default; second 4090 optional for concurrent eval.

Key Ingredients
- Visual encoders for inputs: English words as 128×128 binary matrices; Chinese characters rendered via a standard font (e.g., Noto Sans CJK) at 64×64.
- Differentiable product quantization for the 3-channel code using Gumbel-Softmax or VQ-VAE with straight-through (ST) estimator.
- Distributional InfoNCE to maximize MI between code and linguistic context (corpus structure).
- Cross-lingual alignment at sentence level (InfoNCE on EN↔ZH parallel pairs) to force code convergence across languages.
- Usage/independence regularizers to fill the 32×32×32 space and factorize channels into coarse/fine semantics.

0) Data: Sources and Commands
- English: Wikipedia (dump), Wikitext-103, CC-100 (en), OSCAR (filtered).
- Chinese: Wikipedia (zh), CC-100 (zh), OSCAR (zh).
- Parallel EN↔ZH: OPUS families (e.g., OpenSubtitles, TED), Tatoeba, WikiMatrix.
- Quick start (Hugging Face Datasets):
  - pip install datasets
  - Python:
    - from datasets import load_dataset
    - en = load_dataset('wikitext', 'wikitext-103-raw-v1')['train']
    - zh = load_dataset('mc4', 'zh', streaming=True)
    - opus = load_dataset('opus_books', 'en-zh', split='train')
- Preprocess: normalize Unicode; EN lowercase+lemmatize; ZH per-character. Save vocab lists: artifacts/vocab/en_top30k.txt, artifacts/vocab/zh_char.txt

1) Inputs as Images
- English → 128×128 matrix: rows=128 Latin symbols; columns=positions; set 1 at (row(char_i), col=i).
- Chinese → 64×64 glyphs via PIL and Noto/SimSun, grayscale normalized.
- Visual encoders: small CNNs (4 conv blocks) → z_vis ∈ R^d0. Shared head optional.

2) “Color” Codes via Product Quantization
- Three discrete channels, 32 choices each.
- Gumbel-Product: logits_ℓ ∈ R^32; sample with Gumbel-Softmax (τ annealed). Codebook C_ℓ ∈ R^{32×d}; embedding e = Σ_ℓ C_ℓ^T y_ℓ.
- VQ-Product (ST): hard argmax per channel; commitment loss L_commit = ||sg[z_vis]-e||^2 + β||z_vis-sg[e]||^2.
- Marginal usage: L_usage = Σ_ℓ KL(p(k_ℓ) || U(32)) over batch/EMA.
- Channel independence: L_indep = Σ_{ℓ≠m} ||Corr(y_ℓ, y_m) - I||_F^2 (Barlow Twins style), discouraging redundancy.

3) Learn from Corpus Structure (Distributional MI)
- Context encoder: mean-of-neighbors (CBOW) or tiny Transformer over local window → cxt_t ∈ R^d.
- InfoNCE (CPC): L_NCE = -E log exp(sim(e_t, cxt_t)/τ) / Σ_neg exp(sim(e_t, cxt_neg)/τ).
- Coarse-to-fine: auxiliary CE to predict k1 from context (L_coarse), higher noise for deeper channels.

4) Cross-Lingual Alignment (EN↔ZH)
- Sentence-level InfoNCE: pool token codes into s_en, s_zh and align with symmetric CLIP-style loss L_xling.
- Optional anchors: small seed dictionary; penalize Hamming distance between known translation pairs.

5) Total Objective (per batch)
- L_total = L_NCE + λ1 L_xling + λ2 L_commit + λ3 L_usage + λ4 L_indep + λ5 L_coarse
- Rationale:
  - InfoNCE maximizes a lower bound on I(code; context); synonyms/inflections must share similar codes.
  - Cross-lingual InfoNCE maximizes I(s_en; s_zh) → translations converge.
  - Usage/independence avoid collapse and factorize channels (coarse/fine semantics).
  - Product capacity (32^3) fits 30k EN lemmas + ~7k ZH chars with prefix sharing.

6) Efficient Training on 1×4090
- Params: visual CNNs ≤5M; context encoder ≤20M; codebooks negligible; total <30M.
- Batch: ~4096 tokens; in-batch negatives; fp16; AdamW lr=3e-4; τ anneal 0.2→0.07; Gumbel τ 1.0→0.3.
- Time: 12–48 GPU-hours on 50–100M tokens to converge a usable code.

7) Verification and Metrics
- Cluster purity (EN synonyms): P@1 ≥ 0.6.
- Lemma grouping: share k1 ≥ 0.7 for family.
- Cross-lingual retrieval (sentences): R@1 ≥ 0.4 on OPUS subset.
- Code usage entropy ≥ 90% of uniform; off-diagonal corr ≤ 0.05.
- Qualitative: 3D code cube visualization; do/did/done/make/做 nearby.

8) Executable Steps
- S1 Download & preprocess → artifacts/vocab/* (2 days)
- S2 Build images (EN 128×128, ZH 64×64) (1 day)
- S3 Implement encoders + product code layer; unit tests (1 day)
- S4 Train L_NCE only; inspect neighbors (1–2 days)
- S5 Add L_xling; measure en↔zh retrieval (1–2 days)
- S6 Export codes table: token → (k1,k2,k3) and embedding (0.5 day)

Defaults
- Channels=3, depth=32, d=256. EN vocab top-30k lemmas; ZH top-7k chars.
- Images: EN 128×128 binary; ZH 64×64 grayscale.
- Regularizers: λ1=1.0, λ2=0.25, λ3=0.1, λ4=0.1, λ5=0.5 (tune in pilot).

This plan is tailored to learn “colors” that compact meaning across English words and Chinese characters using only corpus structure and visuals, under objectives with theoretical grounding (InfoNCE MI maximization + alignment + anti-collapse regularization).
