# Image-Native Language Model Research Dossier

Date: 2026-08-12

## 2026-08-12 experimental revision: Predictive Visual Field

The project now has a measured bridge from Retinal Flow Language Modeling to a
new **Predictive Visual Field (PVF)** hypothesis. The implemented RFLM remains a
causal model over continuous image fixations:

```text
ordered ink images -> foveal retina -> recurrent visual field
                   -> image energy + conditional pixel flow
                   -> candidate ink images -> reread -> visual feedback
```

Its student boundary excludes strings, token IDs, Unicode IDs, OCR,
character-label supervision, external language models, and discrete visual
codebooks. Typed text is rasterized before the boundary. At inference, generated
pixels are reread and become the next input.

The implemented system has four trainable parts:

1. A convolutional `32x32` foveal retina with a 192-dimensional continuous
   output and an EMA target branch.
2. A three-layer, 384-dimensional GRU visual field.
3. A continuous compatibility energy that can score any candidate image
   embedding without a vocabulary-sized head.
4. A conditional pixel-space rectified-flow writer with high-noise training and
   a differentiable write-read cycle.

The model contains 11,690,244 parameters. V7 resumed the closed-loop V6
checkpoint for 800 updates on one RTX 4090. It added independent image-anchor
context contrast, normalized target likelihood, and differentiable
sampled-endpoint identity. A fixed evaluation over 512 common Han characters,
four prototype views, and 2,423 eligible held-out contexts produced:

| Measurement | V6 closed loop | V7 step 5,800 |
|---|---:|---:|
| Retina oracle top-1 | 98.184% | 98.267% |
| RFLM full-context top-1 | 1.197% | **2.311%** |
| RFLM full-context top-5 | 3.219% | **5.613%** |
| Last-fixation-only top-1 | 1.692% | 2.022% |
| Unigram top-1 | 1.857% | 1.857% |
| Symbolic bigram top-1 | 13.578% | 13.578% |
| Raw target-score gain | +2.806 | +2.463 |
| Normalized target-log-probability gain | -0.9066 | **-0.2155** |
| Generated context cosine gain | +0.0077 | **+0.0303** |
| Generated sample target hit | 1.0417% | 0.5208% |
| Autonomous late/early ink | 1.168 | **1.050** |
| Autonomous sparse cells | 18.75% | **15.63%** |

V6 samples its deployed pixel writer, rereads selected bitmap candidates, and
trains state consistency, next-image energy, and recovery flow. V7 retains that
stability regularizer and adds two corrections. First, it compares full and
last-only states using normalized visual target probability rather than raw
energy. Second, it differentiates through a two-step pixel-flow endpoint and
aligns the reread sample with independent target images. The training image
anchors are selected offline, use pixels disjoint from the evaluator views, do
not expose IDs or target indices to the student, and are not deployed.

The acceptance result remains false. V7 more than doubles top-1, crosses the
unigram baseline, restores generated context signal, and closes about 76% of
V6's calibrated context deficit. Nevertheless, normalized full-history gain is
still negative, top-1 is far below the 13.578% bigram, and matched 32-cell
generation remains unreadable. Raw energy remains positive in both V6 and V7,
which proves why it cannot serve as an acceptance metric.

This result motivates a structural split:

```text
writing pixels -> image-derived retinal states -> causal visual state flow
               -> sampled intended state -> pixel actuator -> writing pixels
```

PVF first learns the conditional distribution of the next retinal state with
flow matching. A separate pixel writer then renders a sampled state and is
audited by rereading. There is no nearest-character projection or token
unembedding at deployment. This makes the state-flow branch falsifiable before
typographic generation is allowed to obscure its language accuracy.

Three primary research threads make this intervention technically plausible
without establishing it in advance:

- [Recurrent JEPA](https://arxiv.org/abs/2411.16695) explicitly predicts the
  representation of the next visual fixation from past fixations and analyzes
  collapse avoidance.
- [D-JEPA](https://arxiv.org/abs/2410.03755) separates target-embedding
  prediction from a compact conditional generative model, avoiding the need to
  denoise the complete predictor.
- [Embedded Language Flows](https://arxiv.org/abs/2605.10938) reports effective
  language dynamics in continuous embedding flow. ELF still starts and ends
  with discrete tokens, so it does not satisfy the ILM boundary; PVF replaces
  its token embedding/unembedding with an image retina and pixel actuator.

The RFLM does not replace the need for provenance-bearing folio memory. It
replaces that memory as the definition of the language model. Later etymology
answers should combine an autonomous visual writer for connective language with
a separate evidence gate that inserts attested historical pixels.

The complete V7 measured receipt is in
[`docs/retinal-flow-v7-anchor-identity-result.md`](../docs/retinal-flow-v7-anchor-identity-result.md).
The V6 precursor remains in
[`docs/retinal-flow-v6-closed-loop-result.md`](../docs/retinal-flow-v6-closed-loop-result.md).

## 2026 architecture revision: Visual Folio Machine

Three measured results changed the project direction:

1. Whole-page conditional flow reduced validation velocity loss but generated
   only page-like texture. Page statistics are not language.
2. A global visual associative encoder beat random retrieval but failed all
   historical paraphrases. One pooled vector trained only from paired images
   did not acquire enough semantics.
3. Causal 8-pixel InkStream reached teacher-forced validation F1 0.639 after
   2,500 updates, but every autonomous sample collapsed to repetitive grey
   columns. Local pixel likelihood and scheduled sampling did not solve
   exposure bias or semantic generation.

That earlier implementation therefore separated four operations:

- **ordered visual reading:** convolutional stroke retina plus axial line/page
  attention;
- **semantic field distillation:** a local open embedding teacher supplies
  centered continuous targets offline, then is removed;
- **visual folio recall:** image-derived keys retrieve exact image-valued answer
  pages, with no answer strings in runtime memory;
- **evidence-aware composition:** attested historical pixels are copied with
  provenance; novel connective writing will use masked whole-field revision.

This is different from PIXEL, which is primarily a masked visual encoder, and
from PIXAR/PixelGPT, which use autoregressive patch prediction. It is also not
OCR-RAG: no OCR transcript reaches the student or deployed memory lookup. The
first bounded proof is semantic image-to-image retrieval under unseen fonts and
validated paraphrases. Open-ended image generation remains a later, separately
measured proof.

The BGE-M3 cache is a training workaround, not part of the claimed independent
model. Raw BGE fields on a 10,000-document Chinese corpus had unrelated-pair
mean cosine about 0.360 and mean-vector norm 0.597. Centering reduced the
unrelated-pair mean to approximately zero, preventing a student from earning a
high cosine merely by learning the corpus-wide direction.

Goal: design an Imagized Language Model (ILM) whose primary input and output are images of writing, not linguistic tokens. The model should read book pages, handwritten scripts, oracle-bone forms, cuneiform-like signs, and other historical writing as visual evidence; it should answer by producing readable rendered images, including modern-language explanations and historically faithful glyph forms.

## 1. Core Position

The project should not be framed as "OCR plus LLM plus renderer". That pipeline can be useful as a teacher and evaluator, but it violates the research goal: the model should learn language through visible marks the way a human reader is exposed to pages.

The proposed object is an **image-native language model**:

```text
image/page/glyph input -> visual latent language state -> image/page/glyph output
```

The implemented RFLM uses continuous retinal vectors and pixel-space flow. The
next PVF separates a continuous retinal-state flow from a conditioned pixel
actuator. Neither uses VAE IDs or a learned visual codebook: a finite glyph-code
vocabulary would weaken the open-form claim and could become a renamed
character vocabulary. Earlier latent/codebook proposals are retained below as
historical design alternatives.

The minimum successful system should do three things:

1. Read an input image containing language and condition on it.
2. Generate an output image containing coherent language.
3. Generate or transform character shapes across scripts and historical time.

## 2. What Existing Models Do Not Yet Do

Current large language models usually use text tokens as the central representation. GPT-3 is a canonical large autoregressive token model, trained as a next-token predictor at large scale. GPT-4 introduced strong multimodal input capability, but its reported interface is still image/text input to text output. OpenAI's GPT-4o API documentation likewise describes text and image input with text output for that model. Llama 3 is a dense Transformer family with large token context windows and compositional multimodal extensions; it is still primarily a token-language foundation model.

Modern product systems now expose image understanding and image generation endpoints. OpenAI's image/vision documentation describes APIs for processing images as input and generating images as output, and Google's Gemini API documentation describes image generation from text and image prompts. These systems are important baselines, but the ILM research gap remains: the visible page should be the **native language substrate**, not only an input modality, a tool call, or a display layer around a token-language core.

The gap for ILM is therefore precise: build a language model whose **visible writing image is both the input substrate and the output substrate**, rather than only an input modality or a display layer.

## 3. Relevant Research Threads

### 3.1 OCR-Free Document Understanding

Donut (OCR-free Document Understanding Transformer) showed that document images can be processed without an explicit OCR stage, using an encoder-decoder Transformer directly over document pixels. Pix2Struct showed that screenshot parsing can be used as a pretraining signal for visual language understanding, and it renders text prompts into the image itself. These are important because they prove that visual text can be learned directly from pixels.

Limitation for ILM: Donut/Pix2Struct typically decode into text strings, not into output page images. They are strong encoders/teachers but do not satisfy the image-output requirement.

### 3.2 Token-Free / Byte-Level Language Modeling

CANINE and ByT5 avoid wordpiece tokenization and operate at character/byte level. MEGABYTE uses multiscale modeling to handle very long byte sequences across text, images, audio, and other modalities. These works support the intuition that fixed subword tokens are not necessary for language learning.

Limitation for ILM: bytes/characters are still symbolic sequences. They do not preserve the visual form of oracle bone inscriptions, calligraphy, damaged scans, or cuneiform wedges.

### 3.3 Image Generative Models as Language Generators

Latent diffusion models and Diffusion Transformers (DiT) make high-resolution image generation computationally practical by denoising compact latent grids. MaskGIT shows iterative masked generation over image tokens. These are natural candidates for producing output pages and glyph panels.

Limitation for ILM: generic image generators rarely learn precise long-form legible language without a strong text model. ILM needs training objectives that make written language structure central, not incidental.

### 3.4 Discrete Diffusion and Diffusion Language Models

D3PM, Diffusion-LM, SEDD, and LLaDA-style models show that language generation need not be strictly left-to-right. This matters because visual reading and page synthesis are not inherently left-to-right at the global level: a page has layout, paragraphs, tables, annotations, and marginalia.

Limitation for ILM: most diffusion language models still operate over text tokens or discrete vocabulary symbols. ILM should move the diffusion state into image/patch/latent space.

### 3.5 Historical Glyph and Script Modeling

Chinese etymology data, oracle-bone glyphs, bronze inscriptions, seal forms, Liushutong forms, cuneiform signs, hieroglyphs, and manuscript scans are high-value data because they expose language as evolving visual shape, not only as modern Unicode.

Available local data snapshot (external to this repo, git-ignored):

- SQLite DB: `/home/lachlan/ProjectsLFS/incoder/data/historic/etymology.sqlite3`
- Glyph files: `/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs`
- Observed counts from the current local snapshot:
  - 9,055 characters
  - 84,642 glyph records
  - oracle: 32,216
  - bronze: 23,500
  - liushutong: 21,923
  - seal: 6,996
  - unknown: 7

This dataset is enough for a first glyph-evolution generator and a glyph-aware visual language pretraining task.

## 4. Archived Proposal: ILM-V Latent Page Model

This section records the pre-RFLM proposal. Whole-page flow experiments learned
page texture without readable language, so the latent page generator is no
longer the primary model. It remains useful as a renderer and baseline.

### 4.1 Data Representation

Use **visual canvases**, not text token sequences.

Recommended canvas types:

1. Glyph tile: 128x128 or 256x256 grayscale/RGB image.
2. Line strip: 1024x128 image representing one rendered line.
3. Page tile: 768x1024 or 1024x1024 image representing a document page crop.
4. Multi-panel answer sheet: a rendered output canvas with modern text plus historical glyph panels.

Each sample can be represented at multiple levels:

- Pixel image `x`.
- VAE latent grid `z = E_img(x)`.
- Optional visual code grid `c = Q(z)` from a VQ tokenizer trained on glyph/page images.
- Layout mask `m` for regions: question, answer, glyph panel, timeline, citation strip.

Important: `c` is a **visual code**, not a language token vocabulary.

### 4.2 Model Blocks

1. **Visual encoder**
   - Patch or convolutional encoder for page/glyph images.
   - Multi-resolution: glyph detail at high resolution; page layout at lower resolution.
   - Candidate: small ViT/Swin/ConvNeXt hybrid.

2. **Visual memory transformer**
   - Maintains spatial language state across panels/pages.
   - Uses 2D relative position, block-sparse/window attention, and cross-panel attention.
   - Designed to run on 24GB GPUs.

3. **Image latent diffusion / masked latent prior**
   - Denoises output image latents conditioned on input image latents.
   - Supports inpainting: ask for a blank answer region and let the model fill it.
   - Supports glyph evolution: condition on modern character image and output historical forms.

4. **Renderer/decoder**
   - Decodes generated latents back to image.
   - Optional super-resolution head for strokes.

5. **Auxiliary readers**
   - OCR or transcription heads can be used for training/evaluation only.
   - They should not become the primary generation path.

### 4.3 Training Objectives

Use a mixture of losses:

1. **Masked visual language modeling**
   - Mask patches in page/glyph images; predict missing visual latents.
   - Objective: learn written structure without OCR.

2. **Denoising page reconstruction**
   - Add noise/blur/dropout to glyphs and scanned pages; reconstruct.
   - Objective: robust reading of damaged books/manuscripts.

3. **Image-to-image instruction following**
   - Input: rendered question image.
   - Output: rendered answer image.
   - Generate synthetic pairs from text corpora by rendering both prompt and answer into images.

4. **Glyph evolution modeling**
   - Input: modern character tile, optional stage label rendered as image.
   - Output: one or more historical glyph forms.
   - Also train inverse tasks: historical glyph -> modern character explanation sheet.

5. **Contrastive semantic alignment**
   - Same content rendered in different fonts/languages/scripts should align in latent space.
   - Different content should separate.

6. **Readability and structure critics**
   - Auxiliary OCR/readability model scores whether output is legible.
   - Layout critic checks no overlap, correct ordering, and consistent panels.

## 5. Training Plan for One or Two RTX 3090 GPUs

Assumption: 24GB VRAM per GPU. Use mixed precision, gradient checkpointing, xformers/FlashAttention where possible, and image latents rather than full-resolution pixels.

### Phase 0: Data Pipeline (1-2 weeks)

Outputs:

- Unified image manifest for modern rendered corpora and historic glyphs.
- Glyph evolution manifest from local SQLite DB.
- Rendered instruction image pairs.

Practical settings:

- Glyph tiles: 128x128 grayscale.
- Line strips: 1024x128.
- Page crops: 512x512 initially.
- Store manifests as Parquet/SQLite/JSONL with paths and metadata.

### Phase 1: Glyph Autoencoder and Codebook (1 GPU)

Model:

- VAE or VQ-VAE, 30M-100M params.
- Input: glyph/page tiles.
- Latent: 16x16 or 32x32 grid.

Historical goal:

- Reconstruct stroke shape faithfully.
- Learn compact visual units for writing.

### Phase 2: Glyph Evolution Generator (1 GPU)

Model:

- Conditional latent diffusion or MaskGIT over glyph latents, 100M-250M params.

Tasks:

- modern char -> oracle/bronze/seal forms.
- historical glyph -> modern char answer sheet.
- stage interpolation: oracle -> bronze -> seal -> modern.

This is the first useful demo because the local dataset is already strong.

### Phase 3: Visual Page Language Model (1-2 GPUs)

Model:

- 200M-600M parameter visual transformer/diffusion prior.
- Context: several page/line panels in latent form.

Tasks:

- Continue the next line/page as image.
- Fill missing text region as image.
- Translate rendered input image into rendered output image.
- Answer a rendered question with a rendered answer.

Feasible constraints:

- Start at 256x256/512x512 crops.
- Use accumulation to simulate larger batch.
- Use LoRA/adapters for instruction tuning.

### Phase 4: Multiscript Expansion (1-2 GPUs)

Data:

- Chinese historical glyphs.
- Printed books in English/Chinese/Japanese/Korean/Arabic/French/Spanish/Vietnamese.
- Cuneiform/hieroglyph datasets where licensing permits.

Tasks:

- Script identification as image.
- Cross-script answer sheets.
- Glyph synthesis for missing ancient forms.

### Phase 5: Evaluation

Use both visual and linguistic metrics:

- OCR accuracy on generated modern text images (only as an evaluator).
- Human readability rating.
- Glyph retrieval: generated ancient form should retrieve the correct character/stage.
- Historical plausibility: nearest-neighbor distance in stage-specific glyph embedding space.
- Layout validity: no overlapping text/panels.
- Task success: image question -> image answer judged by OCR + model + human.

## 6. First Demonstration to Build

Prompt image:

```text
Explain the evolution of the Chinese character 中.
Show oracle, bronze, seal, and modern forms.
Use modern English and Chinese labels.
```

Target output image:

- Title line.
- Modern explanation paragraph.
- Four-panel glyph evolution timeline.
- Stage labels: Oracle, Bronze, Seal, Modern.
- Optional source/citation strip.

This demonstration is strong because it directly shows why image-output matters: the answer is not just text; it includes shape evolution.

## 7. Risks and Design Decisions

### Risk: output text may be illegible

Mitigation: train at line-strip resolution first; add readability critic; use render-supervised synthetic pairs before free generation.

### Risk: model cheats through OCR labels

Mitigation: keep OCR as auxiliary/evaluation, not as primary hidden state. Ablate by training image-only paths and testing on glyph forms without Unicode labels.

### Risk: ancient glyph hallucination

Mitigation: retrieval-augmented glyph memory. For factual answers, generate from retrieved real glyph exemplars or condition on them.

### Risk: 3090 compute is insufficient

Mitigation: stage the project. Start with glyph autoencoding/evolution, then line-level visual LM, then page-level instruction tuning.

### Risk: licenses and scraping

Mitigation: keep downloaded data local/git-ignored; cite sources; prefer public/open datasets for papers; use local data for experiments where permitted.

## 8. Archived Implementation Direction

These were the original proposed modules; the active implementation is now
`ilm/visual_lm/retinal_flow_lm.py` with the retinal train, evaluation, and
inference scripts documented in the root README.

1. `ilm/visual_lm/autoencoder.py`
   - VAE/VQ-VAE for glyph and page tiles.

2. `ilm/visual_lm/dataset.py`
   - Reads glyph DB manifests and rendered page manifests.

3. `scripts/build_glyph_evolution_manifest.py`
   - Exports stage-grouped glyph pairs from SQLite.

4. `scripts/train_visual_glyph_autoencoder.py`
   - First 3090-friendly training loop.

5. `scripts/train_glyph_evolution_diffusion.py`
   - Conditional generation over glyph latents.

6. `scripts/render_image_instruction_pairs.py`
   - Converts text instruction data into prompt/answer images.

7. `publication/ilm-image-native/`
   - Paper draft and diagrams.

## 9. Image-First Training And Inference Contract

The updated ILM-V contract should be strict enough to distinguish the project from a normal OCR + LLM + renderer stack.

### 9.1 Training Figure: Visual Corpus To Image Targets

Training samples should be built as paired or self-supervised image canvases:

```text
text corpus -> page rasterizer -> page image
scanned book/manuscript -> crop/deskew/tile -> page image
historical glyph SVG/bitmap -> normalized glyph tile -> glyph image
unknown or unencoded marks -> raw visual patch -> image region
```

The model sees these as visual latents. Metadata such as source URL, crop box, character ID, stage label, and license can remain in the training manifest, but it should not become a hidden text-language path. The core losses should remain visual:

1. Mask page/glyph patches and predict visual latents.
2. Reconstruct damaged or incomplete page images.
3. Map prompt images to answer images.
4. Generate historical glyph panels from visual examples.
5. Use OCR/readability only as an auxiliary critic or evaluation metric.

This matters for scripts that are missing from font tables or computer codecs. A bronze inscription, cuneiform wedge, or damaged manuscript mark can still be a valid training unit because it is an image patch.

### 9.2 Inference Figure: Text Or Image Prompt To Page Image

The user interface may look like a chat system, but the model contract is image-native:

```text
typed prompt -> page rasterizer -> prompt image
uploaded page/glyph/photo -> visual cropper -> prompt image
prompt image -> ILM-V visual latent state -> rendered answer image
answer image -> optional OCR/transcoder -> text layer only where representable
```

The output is first a PNG/page image, like a section of a book. It can contain English, modern Chinese, classical Chinese, and glyph regions that have no Unicode representation. If a span is representable in computer codecs, a post-processing reader may attach a searchable text layer. If not, the correct representation is still the image region plus coordinates, provenance, and nearest known glyph references.

For the `言` demonstration, a target answer should look like a page excerpt:

- English paragraph: "YAN is the written idea of speech."
- Chinese paragraph: "言，本義為言語、說話、辭令。"
- Classical-style line: "言者，心聲見於簡冊也。"
- Timeline panels: oracle `J04903`, bronze `B02975`, seal `S01648`, modern `言`.

The critical point is that the page image is not a rendering after a token answer. It is the model's primary answer object.

## 10. Literature and Sources

- OpenAI Images and Vision API documentation: https://platform.openai.com/docs/guides/images-vision
- OpenAI Image generation guide: https://platform.openai.com/docs/guides/image-generation
- Google Gemini API image generation documentation: https://ai.google.dev/gemini-api/docs/image-generation
- OpenAI, GPT-4 Technical Report: https://arxiv.org/abs/2303.08774
- OpenAI GPT-4o API model documentation: https://developers.openai.com/api/docs/models/gpt-4o
- Brown et al., Language Models are Few-Shot Learners (GPT-3): https://arxiv.org/abs/2005.14165
- Meta AI, The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Kim et al., OCR-free Document Understanding Transformer (Donut): https://arxiv.org/abs/2111.15664
- Lee et al., Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding: https://arxiv.org/abs/2210.03347
- Clark et al., CANINE: Pre-training an Efficient Tokenization-Free Encoder: https://arxiv.org/abs/2103.06874
- Xue et al., ByT5: Towards a Token-Free Future: https://arxiv.org/abs/2105.13626
- Yu et al., MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers: https://arxiv.org/abs/2305.07185
- Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM): https://arxiv.org/abs/2107.03006
- Li et al., Diffusion-LM Improves Controllable Text Generation: https://arxiv.org/abs/2205.14217
- Lou et al., Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD): https://arxiv.org/abs/2310.16834
- Nie et al., Large Language Diffusion Models (LLaDA): https://arxiv.org/abs/2502.09992
- Chang et al., MaskGIT: Masked Generative Image Transformer: https://arxiv.org/abs/2202.04200
- Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models: https://arxiv.org/abs/2112.10752
- Peebles and Xie, Scalable Diffusion Models with Transformers (DiT): https://arxiv.org/abs/2212.09748
- Lipman et al., Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA): https://arxiv.org/abs/2301.08243
- Ross, Gordon, and Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger): https://proceedings.mlr.press/v15/ross11a.html
- Bengio et al., Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks: https://papers.nips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html
- Ning et al., Input Perturbation Reduces Exposure Bias in Diffusion Models: https://proceedings.mlr.press/v202/ning23a.html
- Yan et al., Eye movement guidance in Chinese reading: Is there a preferred viewing location?: https://doi.org/10.1016/j.visres.2011.03.004
