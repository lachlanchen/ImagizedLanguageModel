# The First Imagized Language Model: Engineering Goal

## North star

Build an independent language model whose native interface is visible writing:

```text
typed text -> deterministic rasterizer --+
                                         +-> writing image -> ILM -> answer image
uploaded page / glyph / handwriting -----+
```

The rasterizer is an input adapter, not a tokenizer. The learned student receives
RGB canvases and continuous spatial latents only. Its primary output is an image
of writing. OCR may add a searchable sidecar after generation, but OCR text is
not the model output and cannot represent every historical form.

## Required capabilities

1. Accept a question typed in Chinese or English by rasterizing it before the
   model boundary.
2. Accept an image containing printed, handwritten, damaged, or historical
   writing without first converting it to Unicode.
3. Return a readable answer as one or more page/line images.
4. Answer bounded instruction-following questions learned from rasterized
   Alpaca-style instruction/response pairs.
5. Answer questions such as “What is the origin of the Kanji 言?” or
   “说明言字的演变” with real visual forms from available evidence.
6. Represent oracle-bone, bronze, seal, clerical/manuscript, traditional,
   simplified, Japanese Kanji, variant, damaged, and non-Unicode forms as
   pixels rather than forcing them through a character vocabulary.
7. Attach provenance metadata to retrieved historical forms. A generated form
   must not be presented as an attested specimen unless it is linked to a
   source asset.
8. Run training and inference on one 24 GiB RTX 4090.
9. Be independent at inference. Qwen 4B/8B may create offline curricula,
   answers, semantic plans, critiques, and preference pairs, but the deployed
   ILM must not call Qwen or require its weights.

## Model contract

The first production-shaped implementation uses:

- a KL-regularized visual VAE that compresses writing canvases by 8x per axis;
- a conditional latent rectified-flow U-Net with multi-scale image condition;
- classifier-free guidance and exponential-moving-average weights;
- 8-16-step Heun sampling for lower latency than long DDPM trajectories;
- page continuation tiles for responses that do not fit one canvas;
- retrieval of attested historical glyph images as visual evidence; and
- optional OCR only for evaluation and Unicode-representable sidecars.

There is no BPE vocabulary, character embedding table, token ID tensor, or
cross-entropy language head in the independent model.

## Dataset workaround

Large text instruction datasets do not need to be discarded. They are converted
at the offline data boundary:

1. Read an openly licensed instruction/response record.
2. Render question and answer into paired page or line images using open fonts.
3. Randomize font family, simplified/traditional form, layout, paper, blur,
   scan noise, and writing direction while preserving the semantic pair.
4. Split long responses into ordered continuation images.
5. Mix in scanned pages, handwriting, Kanji vectors, and provenance-linked
   historical glyph panels.
6. Train the student only on the resulting image tensors.

This gives the visual model broad semantic supervision without pretending that
all useful open datasets were originally distributed as images.

## Proof levels

### P0: mathematical and software proof

- Real VAE reconstruction optimization and latent calibration.
- Real flow-matching loss from data latents to Gaussian noise.
- Iterative sampling from noise, not direct RGB regression or a fixed template.
- Resumable checkpoints, EMA, held-out data, and deterministic inference.

### P1: first trained visual instruction proof

- Train on a license-recorded subset of rasterized Chinese and English Alpaca.
- Mix real historical Chinese glyph assets into origin/evolution tasks.
- Demonstrate typed Chinese and English questions becoming image answers.
- Save generated outputs, loss curves, model hashes, hardware use, and metrics.

### P2: bounded Qwen-8B parity

Define a fixed benchmark containing short visual instruction following,
bilingual questions, historical-form recognition, and glyph-origin answers.
Compare answer correctness, visual legibility, provenance accuracy, latency,
VRAM, parameter count, and throughput. “Parity” may be claimed only for metrics
on this named benchmark, never as general equivalence to Qwen 8B.

### P3: broad image-native language model

Scale pretraining to open Chinese/English corpora rendered as visual tiles,
scanned books, handwriting, and more writing systems. Add recurrent visual
memory for multi-page dialogue and distill a larger teacher curriculum. This is
the stage at which broad language competence can be tested credibly.

## Efficiency claim

Image-native representation is a hypothesis, not automatically more efficient
than tokens. The project must report measured values. The intended advantages
are parallel page generation, compact spatial latents, preservation of visual
forms, and a much smaller independent student. The costs include VAE compute,
multiple flow evaluations, and difficulty rendering exact small text. Negative
results remain valid evidence.

## Non-negotiable evaluation rules

- Do not call a one-step smoke run “trained.”
- Do not call direct image reconstruction “language generation.”
- Do not claim an unattested generated glyph is historical evidence.
- Do not claim Qwen-8B parity without a named benchmark and measurements.
- Keep training sources, licenses, transformations, splits, and hashes in a
  machine-readable manifest.
