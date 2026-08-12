# ILM Development Plan — Executable, Verifiable Steps

> **Historical plan.** This document predates the Retinal Flow Language Model
> and remains for reproducibility. The active milestones and acceptance gates
> are in `docs/first-imagized-language-model-goal.md`.

This plan breaks the Imagized Language Model (ILM) into concrete, sequential tasks. Every task declares: objective, actions to perform, artifacts produced, and verifiable checks (commands and quantitative acceptance criteria). The plan assumes a Linux environment with Python 3.10+, CUDA GPU optional, and XeLaTeX installed (already present here).

Conventions
- Workspace root: project repository root
- Virtual env: `.venv`
- Artifacts dir: `artifacts/`
- Data dir: `data/`
- Source tree (to be created): `ilm/`
- Scripts (to be created): `scripts/`
- Tests (to be created): `tests/`
- Configs: `configs/`

Legend
- Cmd indicates a shell command to run
- DoD indicates Definition of Done (quantitative acceptance)


## Phase 0 — Repository Bootstrap and CI

0.1 Project skeleton
- Objective: Create minimal, runnable project structure.
- Actions:
  - Create folders: `ilm/`, `scripts/`, `tests/`, `configs/`, `artifacts/`, `data/raw`, `data/processed`
  - Add `pyproject.toml` with dependencies (numpy, scipy, torch, torchvision, sentencepiece, transformers, einops, tqdm, pyyaml, rich, pandas, pillow, scikit-learn, faiss-cpu, regex, matplotlib; optional: torchmetrics, lightning) and dev (pytest, coverage, ruff or flake8, black, mypy).
  - Add `Makefile` with common targets (`env`, `lint`, `fmt`, `test`, `train`, `sample`, `pdf`, `clean`).
- Artifacts: `pyproject.toml`, `Makefile`, `ilm/__init__.py`
- Verify:
  - Cmd: `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .`
  - Cmd: `python -c "import ilm; print('OK')"` (expect OK)
- DoD: Install succeeds; module import prints OK.

0.2 Code style and CI
- Objective: Enforce style and tests automatically.
- Actions:
  - Add ruff/black config in `pyproject.toml`, `mypy.ini`, `.editorconfig`
  - Add `.github/workflows/ci.yml`: setup Python, install deps, run ruff/black --check, mypy, pytest
- Verify:
  - Cmd: `make lint` && `make test` (empty test passes)
- DoD: CI passes on push/PR (green check on GitHub).

0.3 Publication build target
- Objective: Rebuild PDFs for docs.
- Actions:
  - Add Makefile targets: `pdf` (xelatex `publication/ilm-structured.tex` twice), `clean-pdf`
- Verify:
  - Cmd: `make pdf` (produces `publication/ilm-structured.pdf`)
- DoD: PDF exists with timestamp updated.


## Phase 1 — Hierarchical Code Tables (φ) and Memory Maps

1.1 Tokenization strategy draft
- Objective: Choose initial pipeline (EN words + char backoff; ZH char + glyph).
- Actions:
  - Create `configs/tokenization.yaml` with language-specific settings
  - Implement `scripts/tokenize_sample.py` to tokenize a sample corpus
- Verify:
  - Cmd: `python scripts/tokenize_sample.py --text "Hello 世界" --lang en-zh`
  - Expect: JSON with tokens per language and backoff when needed
- DoD: Mixed input tokenizes without error.

1.2 Build base embeddings
- Objective: Compute or load base vectors for words/chars to drive clustering.
- Actions:
  - Implement `scripts/build_base_embeddings.py` reading a small corpus to train fastText-style or averaging contextual embeddings (e.g., MiniLM) to vectors; char n-gram vectors for EN characters; radical-informed features for ZH optional.
- Artifacts: `artifacts/emb/base_vectors.npz` (keys→float32 vectors)
- Verify:
  - Cmd: `python scripts/build_base_embeddings.py --input data/processed/sample.txt --out artifacts/emb/base_vectors.npz`
  - Cmd:
    ```bash
    python - <<'PY'
    import numpy as np; d=np.load('artifacts/emb/base_vectors.npz'); print(len(d.files)>100)
    PY
    ```
- DoD: ≥10k entries for EN words or ≥5k for ZH chars with finite vectors.

1.3 Hierarchical clustering to mixed‑radix codes
- Objective: Create φ(v)=(k1..kL) codes with balanced branching.
- Actions:
  - Implement `ilm/coding/hier_code.py` with balanced k-means per level or agglomerative with size caps; accept K per level via `configs/phi.yaml`
  - Provide deterministic seeding and export tables
- Artifacts: `artifacts/phi/code.jsonl` (v, [k1..kL], id), `artifacts/phi/inv.jsonl` (id→v)
- Verify:
  - Cmd: `python -m ilm.coding.hier_code --vectors artifacts/emb/base_vectors.npz --config configs/phi.yaml --out artifacts/phi`
  - Cmd: `jq 'length>0' artifacts/phi/code.jsonl >/dev/null`
  - Invariants: unique ids; product capacity ≥ |V|; ≤5% empty slots per level
- DoD: Uniqueness and capacity checks pass; script reproducible (same seed→same mapping).

1.4 Memory-mapped lookup
- Objective: Fast O(1) encode/decode.
- Actions:
  - Implement `ilm/coding/lookup.py` using numpy memmap or parquet; provide encode(v)→(k…), decode(id)→v; bulk APIs
- Verify:
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.coding.lookup import HierLookup
    L=HierLookup('artifacts/phi')
    ks=L.encode('hello'); v=L.decode(L.id_from_code(ks)); print(v=='hello')
    PY
    ```
- DoD: Round-trip correctness ≥ 99.99% for sampled 10k items; speed ≥ 1e6 lookups/sec on CPU (batched).


## Phase 2 — Rasterization to Image-Like Tensors

2.1 Grid layout and integer planes
- Objective: Map sequence to H×W planes for each level.
- Actions:
  - Implement `ilm/raster/grid.py`: layout (r,c) mapping; padding
  - Implement `ilm/raster/planes.py`: build integer planes X^(ℓ) ∈ [0..Kℓ−1]
- Verify:
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.raster.planes import to_planes
    print(to_planes(['a','b','c'],W=8) is not None)
    PY
    ```
- DoD: Variable length sequences map without overflow; padding masks exported.

2.2 Per-level embeddings and projection
- Objective: Lift integer planes to continuous feature maps.
- Actions:
  - Implement `ilm/raster/embed.py`: per-level embeddings Eℓ[Kℓ×dℓ], 1×1 projection to common D
- Verify:
  - Unit tests in `tests/test_embed.py` cover shapes and masking semantics
  - Cmd: `pytest -q`
- DoD: 100% tests in raster package green; coverage ≥ 85% for raster code.


## Phase 3 — Latent Autoencoder (Compression)

3.1 Tiny AE/VQ-VAE for text-grids
- Objective: 4–8× spatial downscale and 2–4× channel shrink.
- Actions:
  - Implement `ilm/ae/autoencoder.py` with Conv/ResBlocks; option VQ
  - Training script `scripts/train_ae.py`; config in `configs/ae.yaml`
- Verify:
  - Cmd: `python scripts/train_ae.py --config configs/ae.yaml --steps 10_000`
  - Metrics: val recon MSE ≤ 0.02 or token recon acc ≥ 0.98 for synthetic grids
- DoD: Achieves target on held-out synthetic; artifacts saved in `artifacts/ae/*.pt`.

3.2 AE export and on-the-fly encode/decode
- Objective: Productionize encoder/decoder for downstream diffusion.
- Actions:
  - Implement `ilm/ae/runtime.py`: Enc(x)→z, Dec(z)→x with autocast
- Verify:
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.ae.runtime import Enc, Dec; import torch
    x=torch.randn(1,128,32,32); z=Enc(x); x2=Dec(z); print(x2.shape)
    PY
    ```
- DoD: Decode(Encode(x)) L2 error within training stats; batch throughput ≥ 200 samples/s on RTX 4090.


## Phase 4 — Diffusion Backbone

4.1 UNet/DiT backbone on latent z
- Objective: Implement a compact diffusion model.
- Actions:
  - Implement `ilm/diffusion/backbone_unet.py` and/or `backbone_dit.py`
  - Time embedding, conditional injection hooks
- Verify:
  - Unit: forward on random z,t,c returns same shapes; params count ≤ 200M for base
  - Cmd: `pytest -k diffusion -q`
- DoD: Sanity forward works in fp16; memory footprint within spec.

4.2 Discrete diffusion heads (per-level) and continuous ε-head
- Objective: Support both discrete (D3PM-like) and continuous (DDPM-like) regimes.
- Actions:
  - Implement `ilm/diffusion/heads.py`: categorical logits per level; ε-prediction head
- Verify:
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.diffusion.heads import Heads; print(Heads(64,[16,16,32,64]))
    PY
    ```
- DoD: Head wiring passes shape tests; temperature scaling configurable.


## Phase 5 — Meta-Element Conditioning

5.1 Encoders for grammar/semantics/emotion/style
- Objective: Learn/fuse meta-elements as conditioning vectors/maps.
- Actions:
  - Implement `ilm/meta/encoders.py`; optional weak supervision (UD, SRL, sentiment)
- Verify:
  - Cmd: `pytest -k meta -q`
- DoD: Encoders produce fixed-size c; ablation (no-conditioning) vs conditioning improves validation loss ≥ 3%.

5.2 Classifier-free guidance & coarse-to-fine scheduling
- Objective: Steering and level-wise noise schedules.
- Actions:
  - Implement CFG mixing; per-level β^(ℓ) schedules in `configs/diffusion.yaml`
- Verify:
  - Cmd: `python scripts/sample.py --cfg 2.0 --levels coarse`
  - Qualitative: Skeleton-first then refine; quantitative: reduced edit distance vs. flat schedule by ≥ 5%.


## Phase 6 — Training Loop and Data

6.1 Data pipeline
- Objective: Stream text→planes→latents with augmentation.
- Actions:
  - Implement `ilm/data/dataset.py`; collate masks; optional glyph channels
- Verify:
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.data.dataset import SampleLoader; print(next(iter(SampleLoader('data/processed'))) is not None)
    PY
    ```
- DoD: Throughput ≥ 1k seq/s CPU preproc; deterministic sharding with seed.

6.2 Trainer
- Objective: Single/multi-GPU training, mixed precision, EMA.
- Actions:
  - Implement `scripts/train_diffusion.py` (Lightning or raw PyTorch)
  - Checkpointing, resume, logging (TensorBoard/W&B optional)
- Verify:
  - Cmd: `python scripts/train_diffusion.py --config configs/diffusion.yaml --steps 50_000`
  - Metrics: validation NLL proxy ↓ across epochs; early stopping works
- DoD: Reaches target loss on small corpus; produces checkpoint `artifacts/diffusion/*.pt`.


## Phase 7 — Inference and Editing

7.1 Sampler
- Objective: t=T→0 denoise with CFG and masks; ROI editing.
- Actions:
  - Implement `ilm/infer/sampler.py` with discrete/continuous paths; masks for inpainting and region-only sampling
- Verify:
  - Cmd: `python scripts/sample.py --checkpoint artifacts/diffusion/last.pt --prompt "test" --steps 24 --out artifacts/samples`
  - Output text decodes and respects masks (diff greater inside ROI than outside)
- DoD: Sampling stable with ≤ 2 GB VRAM for 512-token grid; per-sample time ≤ 1s on 4090 for base model.

7.2 Autoregressive corrector (optional)
- Objective: Small AR pass to patch local inconsistencies.
- Actions:
  - Implement `ilm/infer/corrector.py` (20–40M params)
- Verify:
  - Cmd: `python scripts/correct.py --in artifacts/samples --out artifacts/samples_fixed`
  - BLEU/grammar metrics improve ≥ 3% relative to raw samples
- DoD: Measurable improvement; overhead ≤ 20% latency.


## Phase 8 — English and Chinese Pipelines

8.1 English character/byte backoff
- Objective: Robust OOV handling via bytes/chars.
- Actions:
  - Implement `ilm/lang/en.py`: tokenizer + backoff; integrate planes for bytes
- Verify:
  - Cmd: `python scripts/tokenize_sample.py --text "antidisestablishmentarianism✨" --lang en`
  - Output uses byte backoff for emoji and rare tokens
- DoD: Zero failures on 10k random strings.

8.2 Chinese glyph encoder (optional)
- Objective: Render char glyphs and extract tiny CNN features.
- Actions:
  - Implement `scripts/render_glyphs.py`; `ilm/glyph/cnn.py`
- Verify:
  - Cmd: `python scripts/render_glyphs.py --text "你好世界" --font SimSun --out artifacts/glyphs`
  - Cmd:
    ```bash
    python - <<'PY'
    from ilm.glyph.cnn import GlyphCNN; import torch
    net=GlyphCNN(); x=torch.randn(4,1,32,32); print(net(x).shape)
    PY
    ```
- DoD: Features concatenate with level embeddings; ablation improves ZH intrinsic score ≥ 2%.


## Phase 9 — Evaluation Suite

9.1 Intrinsic metrics
- Objective: Reproducible, automated metrics.
- Actions:
  - Implement `scripts/eval_intrinsic.py` (NLL proxy, edit distance, UD parse quality, conditioning fidelity)
- Verify:
  - Cmd: `python scripts/eval_intrinsic.py --checkpoint artifacts/diffusion/last.pt --split valid`
  - Outputs JSON with metrics
- DoD: All metrics generated; regression thresholds documented in `configs/metrics.yaml`.

9.2 Extrinsic tasks
- Objective: Downstream sanity (classification, QA small sets).
- Actions:
  - Implement `scripts/eval_extrinsic.py` using small public datasets
- Verify:
  - Cmd: `python scripts/eval_extrinsic.py --checkpoint ...`
- DoD: Baselines established; ILM within 20% of comparable AR baseline or better where controllability matters.


## Phase 10 — Optimization for Consumer Hardware

10.1 Quantization
- Objective: 8-bit activations; 4/8-bit weights.
- Actions:
  - Integrate bitsandbytes/torchao for weight-only INT4/INT8; post-training calibration
- Verify:
  - Cmd: `python scripts/quantize.py --checkpoint ... --out ...`
  - Measure memory with `torch.cuda.memory_allocated()`
- DoD: Model RAM ≤ 2 GB for base; ≤ 1.5× latency increase.

10.2 LoRA adapters
- Objective: Efficient domain/style adaptation.
- Actions:
  - Implement LoRA injection for key layers; training script `scripts/train_lora.py`
- Verify:
  - Cmd: `python scripts/train_lora.py --lora-rank 8 --steps 5_000`
  - Evaluate intrinsic metrics
- DoD: Matching ≥ 95% of full finetune improvement with ≤ 1% trained params.


## Phase 11 — Packaging, CLI, and Docs

11.1 CLI tools
- Objective: ship runnable commands.
- Actions:
  - Implement console entry points: `ilmc` (codes), `ilmr` (raster), `ilmae`, `ilmd` (train), `ilms` (sample)
- Verify:
  - Cmd: `pip install -e . && ilms --help` (non-zero help)
- DoD: Each CLI provides expected options and runs.

11.2 User docs and tutorials
- Objective: Clear usage and examples.
- Actions:
  - Write `docs/quickstart.md`, `docs/pipelines.md`, `docs/conditioning.md` with copy-pastable commands
- Verify:
  - Follow docs to reproduce a minimal sample on a fresh machine
- DoD: Fresh run succeeds end-to-end with small data.


## Phase 12 — Release Management

12.1 Versioned artifacts
- Objective: Tag and publish model checkpoints and code.
- Actions:
  - Create `CHANGELOG.md`; tag `v0.1.0`; store artifacts in releases or model hub
- Verify:
  - Cmd: `git tag v0.1.0 && git push --tags`
- DoD: Release page contains checkpoints and instructions.


## Checklists — Per Milestone DoD Snapshot

- M1 φ tables: encode/decode round-trip ≥ 99.99%; ≥ 1e6 lookups/s batched CPU
- M2 raster: planes→embeddings shapes correct; coverage ≥ 85%
- M3 AE: recon MSE ≤ 0.02 or token acc ≥ 0.98; export works
- M4 backbone: forward works fp16; params ≤ 200M (base)
- M5 heads: categorical + ε heads verified
- M6 meta: conditioning improves val loss ≥ 3%
- M7 sampler: 512-token grid ≤ 1s/sample on 4090; ROI edits honored
- M8 EN/ZH: zero tokenization failures; glyph features improve ZH ≥ 2%
- M9 eval: metrics JSON reproducible; thresholds set
- M10 opt: RAM ≤ 2 GB; LoRA efficacy ≥ 95% of full finetune delta
- M11 CLI/docs: fresh run end-to-end success
- M12 release: tagged, artifacts published


## Getting Started — Minimal Commands

- Create env and install:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip && pip install -e .`
- Build φ on toy data:
  - `python scripts/build_base_embeddings.py --input data/processed/sample.txt --out artifacts/emb/base_vectors.npz`
  - `python -m ilm.coding.hier_code --vectors artifacts/emb/base_vectors.npz --config configs/phi.yaml --out artifacts/phi`
- Raster smoke test:
  - ```bash
    python - <<'PY'
    from ilm.raster.planes import to_planes
    print(to_planes(['hello','world'],W=16) is not None)
    PY
    ```
- AE pretrain (toy):
  - `python scripts/train_ae.py --config configs/ae.yaml --steps 2000`
- Diffusion pretrain (toy):
  - `python scripts/train_diffusion.py --config configs/diffusion.yaml --steps 5000`
- Sample:
  - `python scripts/sample.py --checkpoint artifacts/diffusion/last.pt --prompt "Test" --steps 24`

---

This plan is living: as modules land, update each task’s commands to the real entry points and tighten DoD thresholds based on observed baselines.
