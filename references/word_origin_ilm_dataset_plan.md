# Visual Word-Origin ILM: Dataset And Learning Plan

Date: 2026-08-12

Status: goal-defining dataset contract; training results are not claimed

## 1. Product Question

Can a compact model learn enough language directly from visible writing to read
an English or Chinese question and generate a useful answer page, while also
preserving historical or unencoded forms that a token vocabulary cannot emit?

Transcription is included but is not the endpoint. Given a page image, the ILM
should be able to reproduce or normalize its writing visually and support an
optional coded transcript where a codec exists. The same model must also
continue, translate, explain, compare, answer, and generate visual language.
Unlike an OCR-only pipeline, it retains form, spatial relation, material trace,
and variation through time as potentially meaningful evidence.

The target is not OCR plus a hidden LLM plus a renderer. The student contract is:

```text
input writing pixels -> continuous retinal fields -> causal visual state
                     -> continuous answer plan -> answer pixels
```

A typed prompt is allowed only because the UI rasterizes it before the student.
OCR, Unicode, and a text model may be used to prepare or evaluate data offline,
but they are removed before the student batch is constructed and absent at
inference.

## 2. Three Data Layers

### A. Releasable training corpus

Use public-domain or openly licensed sources for any released dataset or
checkpoint:

- public-domain Chinese and English books;
- rights-documented Wikisource packages;
- openly licensed dictionaries and etymology records;
- permissively licensed fonts and handwriting datasets;
- public historical-glyph collections with per-image provenance; and
- synthetic page renderings whose source text and fonts permit redistribution.

Every artifact records source URI/path, rights statement, source hash, page or
record locator, crop coordinates, rendering transform, and output hash.

### B. Private reference and evaluation library

The 11 local works in `references/source_books/manifest.json` total
963,986,234 bytes. They are hash-registered and symlinked locally, not committed.
Their rights are unverified. Until rights are established, use them only to:

- study page and glyph-evolution layouts;
- compare factual claims during private research;
- construct nonredistributed evaluation prompts where lawful; and
- audit whether an answer copied a source passage or image.

They must not silently enter a released training mixture or checkpoint.

### C. Historical visual evidence

The local historic-glyph collection contains attested oracle, bronze, seal, and
Liushutong images. These are evidence-bearing visual objects, not decorative
styles. Each crop needs a source ID, stage, associated modern form when known,
and source/license status. An answer must mark one of two modes:

- **attested**: source pixels are retrieved and cited; or
- **synthesized**: pixels are generated and explicitly labeled as a hypothesis.

The model may learn a stage manifold, but generated historical-looking marks
must never be presented as archaeological evidence.

## 3. Reproducible Preparation Pipeline

1. Register every source by SHA-256 before extraction.
2. Render PDF/EPUB pages to lossless page images with page locators.
3. Extract embedded text or run OCR only into an offline sidecar. The sibling
   `ZhJpBook` tools provide a starting point for source-auditable extraction.
4. Detect text lines, figures, glyph panels, captions, and reading order.
5. Retain original crops. Never reconstruct an ancient glyph from OCR text.
6. Align sidecar spans to image rectangles for curriculum construction.
7. Create multiple visual views: font, spacing, direction, scale, blur, damage,
   paper, camera perspective, and handwriting.
8. Serialize student examples with image tensors, masks, coordinates, and
   continuous temporal order only.
9. Run a boundary audit that rejects strings, character IDs, token IDs, Unicode
   IDs, OCR arrays, source-answer IDs, and teacher outputs in the batch.
10. Split by source work, lexical headword, visual exemplar, and font before
    augmentation so neighboring pages or alternate renders cannot leak.

OCR sidecars remain useful for quality control and post-hoc scoring. They are
not a legal route around the image-only student boundary.

## 4. Efficient Visual Language Units

Full pages are too expensive to make the only random variable. Use a hierarchy
of continuous visual units:

- **glyph/stroke field**: `32x32` or `64x64` crops for identity and motor control;
- **fixation/word field**: short horizontal or vertical regions for local syntax;
- **line field**: a line image plus neighboring context for language prediction;
- **page field**: low-resolution page state plus selected high-resolution
  foveae; and
- **answer canvas**: a planned sequence of regions rendered into one page.

No level is a discrete codebook entry. The hierarchy changes spatial scale, not
the ontology of the model.

## 5. Learning Objectives

### 5.1 Visual alphabet and motor control

Train multiview retinal agreement, variance preservation, continuous next-state
prediction, and a topology-first writer. The current V18 result proves this is
feasible for many simple and medium Chinese forms. V19 tests whether a retained
spatial retinal field repairs dense forms.

### 5.2 Local visual language

For ordered fixation fields `x_1 ... x_T`, train:

```text
past visual fields -> next visual field distribution
masked line regions -> missing visual fields
full history -> target state better than last fixation only
```

Use shuffled-line, shuffled-page, same-layout, unigram, and symbolic bigram
controls. A model has not learned useful language merely because it reconstructs
font or paper texture.

### 5.3 Book pretraining

Adjacent-page prediction is an auxiliary long-context task, not the sole
objective. Construct overlapping trajectories:

```text
line 1...n       -> line n+1
fixations 1...t  -> fixation t+1
page n summary   -> selected fields on page n+1
question region  -> answer-region trajectory
```

Sample semantic regions more heavily than margins/backgrounds. Compare against
a same-book but wrong-next-page control, because page style alone can make a
false success.

### 5.4 Visual instruction following

Offline teachers may draft candidate question/answer text from openly licensed
facts. Render both sides, delete the strings from student records, and train the
ILM to predict an answer-region trajectory. Mix:

- direct factual questions;
- explanation and comparison;
- continuation and correction;
- English question to Chinese answer and the reverse;
- photographed prompt to clean answer page; and
- queries requiring an attested historical image panel.

Teacher provenance belongs in the dataset receipt. No teacher is called during
student inference.

### 5.5 Answer-page generation

Factor generation into continuous stages:

1. predict a low-resolution semantic/layout field;
2. choose foveal regions through learned continuous attention;
3. generate line and glyph motor plans;
4. optionally add stochastic paper/style residuals; and
5. reread the generated pixels before advancing.

This avoids spending a diffusion model's full capacity on simultaneous factual
reasoning, typography, and page texture.

## 6. First Named Benchmark

Freeze 200 word-origin questions before instruction training:

- 50 Chinese questions with modern encoded forms;
- 50 Chinese questions requiring historical or unencoded visual evidence;
- 50 English word-origin questions;
- 50 cross-language comparison or photographed-prompt questions.

Half use deterministic rendered prompts and half use held-out scans, camera
distortion, handwriting, or unfamiliar fonts. Split all headwords and source
pages from training.

Report:

- factual proposition precision and recall from human annotation;
- modern answer-text character/word error rate from an evaluator only;
- blinded readability and usefulness;
- attested-glyph stage retrieval and source-ID accuracy;
- false-attestation rate for synthesized forms;
- preservation of unencoded image regions;
- page overflow, overlap, reading order, and blank-output rates;
- autonomous reread stability across answer regions;
- parameters, training hours, peak VRAM, latency, and energy; and
- an inference boundary receipt.

Compare with retrieval-only, fixed-template, OCR-plus-small-LLM, local Qwen 8B,
and current multimodal baselines. Claims are task-specific. "On par with Qwen"
is permitted only if the fixed benchmark supports it.

## 7. Single-4090 Scaling Strategy

Do not begin with a 4B full-page diffusion model. First prove each bottleneck:

1. `20-40M`: retina, causal field, and topology writer at fixation scale.
2. `100-300M`: multiscale line model with cached frozen retina features during
   training, while raw pixels remain the deployed input.
3. `0.5-1.5B`: sparse page memory and answer planner, using BF16, checkpointing,
   fused attention, gradient accumulation, and curriculum mixing.
4. Train high-resolution surface refinement separately and invoke it only after
   the answer plan is stable.

Parameter count is not the objective. The efficient hypothesis is that sparse
foveation, continuous fields, and factorized planning reduce sequence length and
avoid a giant token vocabulary. This must be measured at matched answer quality.

## 8. Promotion Rules

A stage advances only when its own causal test passes:

- perception: held-out cross-font/script retrieval;
- language: full visual history beats last-only, unigram, and fixed bigram;
- writing: correct continuous state beats shuffled/zero-state branches and
  passes prospective blinded readability;
- instruction: answer pages beat retrieval/templates on held-out questions;
- independence: no symbolic or external-model channel appears at inference; and
- evidence: attested and synthesized historical forms are never conflated.

This sequence keeps the project centered on the core claim: language must be
learned and emitted through visible writing, not merely wrapped in images.

## 9. General Sensory Extension

The visual model is one instance of a possible sensory-language architecture:

```text
continuous sensory observation -> causal sensory field -> predicted state
                               -> motor generator -> self-perceived signal
```

For writing, observations and outputs are retinal image fields. A later speech
model could use waveform or time-frequency fields and a vocal motor generator,
with no requirement that an intermediate transcript exist. Cross-modal training
could align writing and voice through continuous predictive consequences rather
than a shared token ID. This is a research direction only; the visual model must
first pass its own language and answer-page gates.
