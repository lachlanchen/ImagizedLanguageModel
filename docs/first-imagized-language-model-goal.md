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

The main implementation is a **Visual Folio Machine**, not a page diffusion
model and not a next-patch clone. It separates visual reading, semantic field,
memory, evidence, and writing:

- an ordered axial retina reads full-resolution writing lines without OCR;
- offline teacher fields are centered and distilled into a continuous student
  field; the teacher is absent from the checkpoint and runtime;
- a content-addressed folio stores image-derived keys and exact answer pages;
- a provenance gate copies attested historical forms instead of hallucinating
  them;
- a whole-answer masked canvas revises visible regions in parallel rather than
  feeding its own fragile pixel columns back forever;
- optional OCR is restricted to evaluation and representable sidecars.

There is no BPE vocabulary, character embedding table, Unicode-ID tensor,
language-token codebook, or cross-entropy language head in the independent
model. Continuous foveal states and canvas regions are spatial observations,
not renamed linguistic tokens.

The earlier whole-page KL-VAE plus rectified-flow U-Net remains in the repo as
a negative baseline. On the first measured run its held-out flow velocity MSE
fell from 1.780 at step 200 to 0.851 at step 1,200, yet its output remained
illegible page texture. That result rejects the assumption that learning page
appearance from noise is sufficient to learn visible language.

The causal InkStream experiment is a second negative baseline. A 7.46M model
trained for 2,500 updates reached teacher-forced validation ink F1 0.639 using
0.61 GB peak VRAM, yet autonomous output became repeated grey column stamps.
That rejects teacher-forced next-column accuracy as evidence of visual language
generation. The project now treats held-out semantic retrieval and autonomous
page legibility as separate mandatory tests.

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

- Image-only model calls after the raster boundary.
- Multi-view retinal invariance under unseen font, layout, and scan damage.
- Content-addressed answer-image memory with insertion without retraining.
- Held-out retrieval, provenance, latency, memory-size, and VRAM measurements.
- Resumable checkpoints and deterministic independent inference.

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
are sparse computation over occupied ink, one visual memory entry per answer
rather than repeated parametric memorization, exact preservation of unusual
forms, and a much smaller independent student. Costs include high-resolution
visual encoding, memory storage, and the difficulty of composing genuinely new
readable writing. Negative results remain valid evidence.

## Non-negotiable evaluation rules

- Do not call a one-step smoke run “trained.”
- Do not call direct image reconstruction “language generation.”
- Do not claim an unattested generated glyph is historical evidence.
- Do not claim Qwen-8B parity without a named benchmark and measurements.
- Keep training sources, licenses, transformations, splits, and hashes in a
  machine-readable manifest.
