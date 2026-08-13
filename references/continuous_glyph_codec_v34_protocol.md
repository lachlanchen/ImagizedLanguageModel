# Continuous Glyph Representation Codec V34: Preregistered Protocol

Date frozen: 2026-08-14

Status: frozen before V34 implementation and measurement

## Fixed Question

Can a compact, codebook-free convolutional autoencoder map arbitrary `32 x 32`
writing patches into a normalized 768-dimensional continuous space and recover
unseen Chinese font patches and unseen historical character families with
stroke-level fidelity on one RTX 4090?

This is a visual-interface experiment, not a language-understanding claim.

## Fixed Boundary

Allowed codec input and output:

- input `pixels: float[B,1,32,32]` in `[0,1]`;
- continuous latent `float[B,768]`;
- optional scalar latent-noise standard deviation during training; and
- direct output pixel logits `float[B,1,32,32]`.

Forbidden in codec methods, saved codec batches, and deployed reconstruction:

- strings, tokenizers, text/token/byte/Unicode/character IDs;
- OCR, character classification, vocabulary logits, glyph lookup, retrieval;
- vector quantization, codebooks, nearest-neighbor latent assignment;
- target images supplied to the decoder; and
- calls to an external model at runtime.

Text rendering, SVG rasterization, labels used for deterministic data splits,
and OCR used only by evaluation are outside the student boundary.

## Fixed Architecture

- patch size: `32 x 32`, one monochrome channel;
- latent width: `768`;
- encoder channels: `32, 64, 96, 192`;
- decoder channels: `192, 96, 64, 32`;
- one GroupNorm-SiLU two-convolution residual block at each scale;
- downsampling: `4 x 4`, stride-2 convolution;
- upsampling: `4 x 4`, stride-2 transposed convolution;
- encoder projection: flattened `192 x 4 x 4` to `768`;
- non-affine per-sample layer normalization on the latent;
- decoder projection: `768` to `192 x 4 x 4`;
- final `3 x 3` convolution to one-channel logits; and
- no adversarial discriminator, perceptual foundation model, VAE KL term, or
  diffusion process in this experiment.

## Fixed Data

Rendered stream:

- source: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- existing deterministic V33 train/development/sealed record partitions;
- train fonts: Noto Sans CJK Regular and Medium;
- development font: Noto Serif CJK Regular; and
- sealed font: Noto Serif CJK Light.

Historical stream:

- SQLite index:
  `/home/lachlan/ProjectsLFS/incoder/data/historic/etymology.sqlite3`;
- SQLite SHA-256:
  `c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738`;
- glyph root: `/home/lachlan/ProjectsLFS/incoder` plus each indexed local path;
- eligible input: the 84,626 indexed `.svg` glyph files only;
- ordered SVG-content manifest SHA-256:
  `3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c`;
- partition key: `sha256("v34:historic:" + modern_character)`;
- split: first 90% train, next 5% development, final 5% sealed; and
- all forms of a modern-character key belong to one split.

SVGs are rasterized onto white `32 x 32` canvases while preserving aspect ratio,
centered with a two-pixel nominal margin, then thresholded at `0.5`. The local
historical corpus is not copied or redistributed.

Each optimization update contains all active patches from eight rendered train
strips plus 128 historical-train glyphs. The two streams remain separately
reported even though their losses are optimized jointly.

## Fixed Objective and Optimization

- updates: `6,000`;
- optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `0.01`;
- peak learning rate: `2e-4`;
- warmup: `250` updates;
- cosine decay to `2e-5`;
- gradient norm clip: `1.0`;
- precision: BF16 on one RTX 4090;
- seed: `20_263_400`;
- EMA: `0.999`;
- checkpoint overwrite interval: `1,000` updates and clean signal;
- maximum allocated CUDA memory: below `12 GiB`; and
- training loss: boundary-weighted BCE plus `0.5` Sobel L1 plus `0.5` ink Dice.

For half of training patches, independently selected by the fixed RNG, decoder
input receives Gaussian latent noise with standard deviation sampled uniformly
from `[0,0.05]`. OCR and data labels never affect gradients.

## Development Evaluation

Evaluate raw and EMA routes after all updates. The preregistered selection route
is EMA. Use at least:

- 4,096 active patches from development-font rendered strips;
- 4,096 development historical glyphs, or the full split if smaller;
- the same rendered set with latent Gaussian noise `sigma=0.03`; and
- 512 all-white patches.

Fixed binarization threshold: `0.5`. Report ink F1, Sobel edge F1, exact-patch
rate, blank false ink, latent per-dimension standard deviation, clean/noisy
metrics, parameter count, throughput, peak VRAM, hashes, and deterministic
galleries. On rendered strips, also report paired `chi_tra` OCR retention using
the target-normalized V33.1 evaluator.

## Mandatory Development Gate

All conditions must hold for EMA:

- finite model, latent, and checkpoint;
- clean rendered ink F1 at least `0.985`;
- clean rendered edge F1 at least `0.980`;
- clean rendered OCR retention at least `0.95`;
- noisy-latent rendered ink F1 at least `0.970`;
- noisy-latent rendered edge F1 at least `0.950`;
- historical ink F1 at least `0.960`;
- historical edge F1 at least `0.940`;
- all-white false-ink rate below `0.005`;
- mean latent per-dimension standard deviation at least `0.10`;
- all 6,000 updates complete; and
- peak allocated CUDA memory below `12 GiB`.

Failure stops V34. Gate changes require a new protocol.

## Sealed Evaluation

Open the sealed rendered font and historical-character split once, only after
the development gate passes. Acceptance additionally requires every sealed F1
and OCR-retention metric to be at least `0.97` of its corresponding development
value. No tuning follows sealed evaluation.

## Decision Labels

- `continuous-codec-qualified`: development and sealed conditions pass;
- `development-qualified-sealed-failure`: development passes but sealed
  transfer fails;
- `continuous-codec-failure`: the completed route fails development fidelity;
  and
- `invalid-run`: boundary, source hash, split, finite-state, update, evaluator,
  checkpoint, or resource integrity fails.

A qualified codec permits a separately preregistered causal continuous-latent
language experiment. It does not itself establish language modeling.
