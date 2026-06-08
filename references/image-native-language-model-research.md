# Image-Native Language Model Research Dossier

Date: 2026-06-08

Goal: design an Imagized Language Model (ILM) whose primary input and output are images of writing, not linguistic tokens. The model should read book pages, handwritten scripts, oracle-bone forms, cuneiform-like signs, and other historical writing as visual evidence; it should answer by producing readable rendered images, including modern-language explanations and historically faithful glyph forms.

## 1. Core Position

The project should not be framed as "OCR plus LLM plus renderer". That pipeline can be useful as a teacher and evaluator, but it violates the research goal: the model should learn language through visible marks the way a human reader is exposed to pages.

The proposed object is an **image-native language model**:

```text
image/page/glyph input -> visual latent language state -> image/page/glyph output
```

Internally, it may use patches, continuous VAE latents, or learned visual codebook entries. These are not word/subword tokens. They are spatial visual units, closer to retinal/text-line patches than to BPE.

The minimum successful system should do three things:

1. Read an input image containing language and condition on it.
2. Generate an output image containing coherent language.
3. Generate or transform character shapes across scripts and historical time.

## 2. What Existing Models Do Not Yet Do

Current large language models usually use text tokens as the central representation. GPT-3 is a canonical large autoregressive token model, trained as a next-token predictor at large scale. GPT-4 introduced strong multimodal input capability, but its reported interface is still image/text input to text output. OpenAI's GPT-4o API documentation likewise describes text and image input with text output for that model. Llama 3 is a dense Transformer family with large token context windows and compositional multimodal extensions; it is still primarily a token-language foundation model.

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

## 4. Proposed Architecture: ILM-V (Image-Native Visual Language Model)

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

Goal:

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

## 8. Implementation Direction in This Repo

Immediate files/modules to add next:

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

## 9. Literature and Sources

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

