# Imagized Diffusion Language Model — Code Plan and Metrics

> **Archived baseline (2026-08-12).** This product-code and discrete-diffusion
> plan is not the current ILM paradigm because it relies on a finite symbolic
> code space. See `docs/imagized-language-model.md` and
> `docs/first-imagized-language-model-goal.md` for the continuous retinal-flow
> model. The material below is retained to reproduce earlier experiments.

## Objectives
- Build a trainable, single‑GPU friendly pipeline that learns visual “color‑code” word embeddings and a diffusion‑like model over sentence frames.
- Achieve measurable semantic structure: similar meanings map to nearby embeddings, cross‑lingual alignment (EN↔ZH), and coherent denoising on masked frames.
- Produce publishable metrics (retrieval, clustering, probes) suitable for a CVPR‑style submission.

## Scope (MVP → Iteration)
- MVP (Weeks 1–2):
  - Train 3‑channel × 32‑state product “color code” for EN words + ZH characters from generated images (no external corpora needed initially).
  - Evaluate intrinsic metrics: code usage entropy, independence (HSIC), nearest‑neighbor purity on small synonym lists, EN↔ZH retrieval on curated pairs.
- Iteration 1 (Weeks 3–4):
  - Build sentence→frame converter and discrete diffusion (masked denoising) on 2D grids.
  - Evaluate masked‑LM‑style losses, infilling quality, and semantic stability under noise.
- Iteration 2 (Weeks 5–6):
  - Add cross‑lingual semantic constraints (InfoNCE with bilingual pairs; distributional contexts from sampled corpora).
  - Evaluate cross‑lingual retrieval Recall@K and cluster NMI/ARI improvements.

## Repository Additions
- Code modules
  - `ilm/data/loader.py`: dataset loaders for word/char images and sentence frames.
  - `ilm/code/product.py`: 3×32 differentiable product code layer (Gumbel‑Softmax or VQ‑ST).
  - `ilm/encoders/glyph_cnn.py`: lightweight CNN for glyph images → latent.
  - `ilm/encoders/context_encoder.py`: small Transformer/CBOW for context vectors.
  - `ilm/frames/pack.py`: sentence→frame packing (grid/tiles, padding, masks).
  - `ilm/diffusion/discrete.py`: masked discrete diffusion schedule + loss.
  - `ilm/diffusion/unet2d.py`: small 2D U‑Net with time conditioning.
  - `ilm/eval/metrics.py`: retrieval, clustering, HSIC, code‑usage, probes.
- Scripts/CLIs
  - `scripts/train_color_codes.py`: train product codes with InfoNCE + regularizers.
  - `scripts/eval_color_codes.py`: compute intrinsic metrics, NN retrieval, probes.
  - `scripts/build_sentence_frames.py`: convert JSONL paragraphs to frame datasets.
  - `scripts/train_diffusion.py`: train masked diffusion over frames.
  - `scripts/eval_diffusion.py`: compute denoising metrics and infilling quality.
- Config
  - `configs/color.yaml`, `configs/diffusion.yaml`: hyperparameters.
  - Makefile targets: `make images_common_freq`, `make train_color`, `make eval_color`, `make frames`, `make train_diffusion`, `make eval_diffusion`.

## Data Pipeline
- Inputs (already scaffolded):
  - Image sets of common EN words and ZH characters: `scripts/build_images_common*.py` produce `data/processed/images_common*` with `index.tsv`.
- Sentence frames:
  - Start from `scripts/sample_paragraphs.py` (100 paragraphs). New: `scripts/build_sentence_frames.py` packs sequences into grids.
  - Representation: for a sentence of length `n ≤ 64`, pack per‑token glyph images into an `H×W` grid (e.g., 8×8 tokens), each token cell is a 64×64 glyph image reduced via CNN to a `d`‑dim latent; concatenate to `8×8×d` frame.
  - Masks: binary mask for padded tokens; language flag per token (EN/ZN); punctuation markers as special token‑cells.

## Models
1) Color‑code embedding (words/chars)
   - Product codes: 3 channels, `K=32` values each (total states 32³ = 32768), temperature‑annealed Gumbel‑Softmax for differentiability.
   - Embedding composition: `e(w) = E1[z1] + E2[z2] + E3[z3]` with `E* ∈ R^{32×d}` (d=128..256). Optional orthogonality penalty between `E1,E2,E3`.
   - Glyph encoder: `glyph_cnn` maps 64×64 (grayscale) to `d`.
   - Losses:
     - InfoNCE(word): align glyph embedding `g(w)` and code embedding `e(w)` with in‑batch negatives.
     - Code usage: entropy maximization across batches per channel to avoid collapse.
     - Independence: HSIC(z1, z2), HSIC(z1, z3), HSIC(z2, z3) penalties.
     - Optional bilingual: InfoNCE between EN word and ZH char for curated pairs.

2) Sentence→frames diffusion
   - Frame representation: `X ∈ R^{H×W×d}` built from token glyph CNN features; optional addition of code embeddings when available.
   - Discrete diffusion (masked denoising):
     - Forward: random masking ratio `t ~ U(0,1)`, replace fraction `β_t` of token cells/features with `[MASK]`.
     - Reverse: U‑Net2D predicts clean features or mask logits; loss is masked cross‑entropy or MSE on clean latent features.
   - Conditioning: add global sentence context (mean of token features) and simple time embedding via FiLM/conv‑affine.
   - Inference: iterative unmasking/refinement over 8–16 steps.

## Training
Stage A: Color‑code learning (single 4090)
- Data: `data/processed/images_common_freq/index.tsv` (e.g., 3k EN + 3k ZH tokens).
- Batch size: 512 (mixed precision), epochs: 20, optimizer: AdamW.
- Temperature schedule: τ from 1.0 → 0.2 (cosine decay).
- Command: `python scripts/train_color_codes.py --config configs/color.yaml`

Stage B: Frames diffusion (single 4090)
- Build frames: `python scripts/build_sentence_frames.py --input data/processed/sample_100.jsonl --out data/frames/`.
- Model: U‑Net base channels 64, depth 3, 8–16 denoising steps.
- Loss: masked denoising (cross‑entropy for discrete masks + MSE on continuous latents).
- Command: `python scripts/train_diffusion.py --config configs/diffusion.yaml`

## Metrics (publishable)
Intrinsic (codes)
- Code usage entropy (per channel) and occupancy histogram; target close to uniform.
- Independence: HSIC between channels (lower is better).
- NN retrieval within language: top‑K synonyms by cosine; precision@K on curated synonym sets.
- Cross‑lingual retrieval (EN↔ZH): Recall@K using a bilingual list; report R@1/5/10.
- Clustering: KMeans over code embeddings; NMI/ARI vs coarse labels (POS, semantic category) from small lexicons.

Diffusion (frames)
- Masked infilling accuracy: token recovery rate at various mask ratios; negative log‑likelihood bound via RA‑MLM.
- Denoising MSE on latent features and structural similarity across frames.
- Stability: sentence representation cosine similarity before/after moderate noise.

Probes (semantic evidence)
- Linear probes on code embeddings for POS tags (EN UD subset) and simple SRL roles (small labeled set). Report accuracy/F1.
- Topographic similarity: Pearson correlation between distances in code space vs distances in contextual co‑occurrence space.

Efficiency
- Throughput (samples/s), VRAM usage at batch sizes; single 4090 feasibility.

## Definitions of Done (DoD)
- A. Color codes
  - Training converges; entropy per channel ∈ [2.8, 3.3] (for K=32; max=5) with no channel collapse.
  - EN↔ZH retrieval R@10 ≥ 0.25 on curated 300 pairs.
  - NN purity on 10 small synonym sets ≥ 0.6 at K=10.
- B. Diffusion
  - Masked infilling accuracy ≥ 0.7 at 30% mask on test frames.
  - Denoising steps ≤ 16 with non‑divergent loss curve; qualitative infills look plausible.
- C. Reproducibility
  - All results reproducible via configs, fixed seeds, and logged artifacts (wandb or local JSON logs).

## Risk Mitigation
- Font coverage for ZH: verify Noto Sans CJK; skip/flag missing glyphs.
- Channel collapse: entropy regularization and KL towards uniform prior.
- Training instability: gradient clipping, cosine LR, warmup, τ annealing.
- Data scarcity: start with curated pairs, then augment with weak bilingual dictionaries or mined pairs from corpora.

## Immediate Tasks (Executable)
1) Implement loaders: `ilm/data/loader.py` to read `index.tsv` and yield `(image, id, lang)`.
2) Implement `ilm/code/product.py` with Gumbel‑Softmax product codes and usage/independence losses.
3) Implement `ilm/encoders/glyph_cnn.py` (simple 3‑block CNN to `d=128`).
4) Train codes: `scripts/train_color_codes.py` MVP loop; log entropy/HSIC and save codes.
5) Eval codes: `scripts/eval_color_codes.py` for NN retrieval, R@K, purity, NMI/ARI.
6) Frame builder: `scripts/build_sentence_frames.py` packing sentences; save tensors + masks.
7) Diffusion core: `ilm/diffusion/discrete.py`, `ilm/diffusion/unet2d.py` + `scripts/train_diffusion.py`.
8) Eval diffusion: `scripts/eval_diffusion.py` infilling accuracy, MSE curves.

## Example Commands
```
# Build common images (already available)
python scripts/build_images_common_freq.py --out data/processed/images_common_freq

# Train color codes
python scripts/train_color_codes.py --config configs/color.yaml \
  --data-root data/processed/images_common_freq --batch-size 512 --epochs 20

# Evaluate codes
python scripts/eval_color_codes.py --codes artifacts/codes.pt --pairs assets/bilingual_pairs.tsv

# Build sentence frames from 100‑paragraph sample
python scripts/sample_paragraphs.py --out data/processed/sample_100.jsonl
python scripts/build_sentence_frames.py --input data/processed/sample_100.jsonl --out data/frames

# Train diffusion
python scripts/train_diffusion.py --config configs/diffusion.yaml --frames data/frames

# Evaluate diffusion
python scripts/eval_diffusion.py --frames data/frames --checkpoint artifacts/diffusion.ckpt
```
